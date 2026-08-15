"""Model access, with instrumentation and guards.

Everything that talks to a model goes through here. That is deliberate: the
pipeline reports per-stage latency, and a stage that made its own HTTP call
would be invisible in those numbers.

Two backends
------------
``OllamaClient`` talks to a real server. ``MockClient`` produces deterministic
output with no model at all, so the whole pipeline runs in about a second in
tests and in continuous integration. Every stage after this one is written
against the interface rather than the implementation, which means a retrieval
bug can be found without waiting a minute per query on a Raspberry Pi.

Mock embeddings are a hashing vectoriser rather than random noise. Random
vectors would make retrieval tests meaningless: every query would match
arbitrarily. Hashing tokens into buckets gives vectors whose similarity tracks
lexical overlap, so a mock retrieval test genuinely exercises ranking, and a
chunk about annual leave really does score higher for a question about annual
leave. It is not semantic, and it is not meant to be. It is enough to make
retrieval logic testable without a model.

Endpoint guard
--------------
``describe_endpoint()`` records the responding server's version and a hash of
its model store. A development tool once forwarded the Raspberry Pi's Ollama
port to the laptop's localhost, so measurements believed to be from the laptop
were executing on the Pi and looked entirely plausible. Recording what actually
answered turns that from an invisible failure into a visible one.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .config import Config, load_config
from .hostinfo import arm_frequency_hz, cpu_temperature_c, ollama_info, throttle_state


class LLMError(RuntimeError):
    """Raised when a model call fails or returns something unusable."""


@dataclass(frozen=True)
class Generation:
    """One model response, with everything needed to account for its cost."""

    text: str
    model: str
    prompt_tokens: int
    eval_tokens: int
    prompt_seconds: float
    eval_seconds: float
    wall_seconds: float
    load_seconds: float = 0.0
    cpu_temp_c: float | None = None
    arm_frequency_hz: int | None = None
    throttled: bool | None = None
    # The merged options as sent, so a record shows what ran rather than what
    # a config file said afterwards. Claiming "effective options are recorded"
    # while this was absent made verification_options serialise as null.
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens_per_second(self) -> float | None:
        return round(self.prompt_tokens / self.prompt_seconds, 2) if self.prompt_seconds else None

    @property
    def eval_tokens_per_second(self) -> float | None:
        return round(self.eval_tokens / self.eval_seconds, 2) if self.eval_seconds else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
            "prompt_seconds": round(self.prompt_seconds, 4),
            "eval_seconds": round(self.eval_seconds, 4),
            "wall_seconds": round(self.wall_seconds, 4),
            "load_seconds": round(self.load_seconds, 4),
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "eval_tokens_per_second": self.eval_tokens_per_second,
            "cpu_temp_c": self.cpu_temp_c,
            "arm_frequency_hz": self.arm_frequency_hz,
            "throttled": self.throttled,
            # The options as posted. Declaring the field without writing it
            # here left every record carrying an empty dictionary while the
            # code claimed effective options were captured.
            "options": dict(self.options),
        }


class LLMClient(Protocol):
    """The interface every pipeline stage depends on."""

    backend: str

    def generate(self, prompt: str, *, model: str | None = None,
                 options: dict[str, Any] | None = None) -> Generation: ...

    def embed(self, text: str, *, model: str | None = None) -> list[float]: ...

    def embed_batch(self, texts: Sequence[str], *,
                    model: str | None = None) -> list[list[float]]: ...

    def describe_endpoint(self) -> dict[str, Any]: ...


# --- real backend -----------------------------------------------------------


def canonical_model_name(name: str | None) -> str:
    """Compare model names the way Ollama actually reports them.

    ``config.json`` names the embedding model ``nomic-embed-text``. ``/api/ps``
    reports it as ``nomic-embed-text:latest``, because Ollama fills in the
    implicit tag. An exact string comparison therefore fails in **both**
    directions, and both are dangerous: after an eviction the configured name is
    absent from the reported list whether or not the model actually went, so a
    failed eviction reads as a success; and after a load the same mismatch
    rejects a model that is correctly resident.

    Amendment 1.21. Normalising the implicit tag is the whole fix.
    """
    if not name:
        return ""
    text = str(name).strip()
    return text[: -len(":latest")] if text.endswith(":latest") else text


class OllamaClient:
    """HTTP client for a local Ollama server."""

    backend = "ollama"

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or load_config()
        self.base_url = self.config.require("llm.ollama_url").rstrip("/")
        self.timeout = self.config.require("llm.timeout_seconds")
        self.default_model = self.config.require("llm.generation_model")
        self.embedding_model = self.config.require("llm.embedding_model")
        self.call_count = 0

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise LLMError(f"{endpoint} failed against {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"{endpoint} returned invalid JSON: {exc}") from exc

    def generate(self, prompt: str, *, model: str | None = None,
                 options: dict[str, Any] | None = None) -> Generation:
        chosen = model or self.default_model
        merged = dict(self.config.require("generation"))
        if options:
            merged.update(options)
        # Underscore-prefixed keys are documentation for a human reader. They
        # were being posted to Ollama as model options, which is silently
        # tolerated and silently meaningless.
        merged = {k: v for k, v in merged.items() if not k.startswith("_")}
        # Ollama only makes a generation reproducible when `seed` is passed in
        # options. It was previously recorded in the provenance block and never
        # sent, which made every result look controlled while being subject to
        # sampling variation. Fail loudly rather than repeat that.
        if "seed" not in merged:
            raise LLMError(
                "No seed in generation options. Results would not be reproducible. "
                "Set generation.seed in config.json."
            )

        temp_before = cpu_temperature_c()
        started = time.perf_counter()
        response = self._post("/api/generate", {
            "model": chosen, "prompt": prompt, "stream": False, "options": merged,
        })
        wall = time.perf_counter() - started
        self.call_count += 1

        if "response" not in response:
            raise LLMError(f"generate returned no 'response' field: {sorted(response)}")

        throttle = throttle_state()
        return Generation(
            options=dict(merged),
            text=response["response"].strip(),
            model=chosen,
            prompt_tokens=response.get("prompt_eval_count") or 0,
            eval_tokens=response.get("eval_count") or 0,
            prompt_seconds=(response.get("prompt_eval_duration") or 0) / 1e9,
            eval_seconds=(response.get("eval_duration") or 0) / 1e9,
            wall_seconds=wall,
            load_seconds=(response.get("load_duration") or 0) / 1e9,
            cpu_temp_c=cpu_temperature_c() or temp_before,
            arm_frequency_hz=arm_frequency_hz(),
            throttled=(throttle or {}).get("now", {}).get("throttled") if throttle else None,
        )

    #: Embeddings stay on the CPU. The project has always described them that
    #: way, but until amendment 1.17 the embedding call posted no options at
    #: all, so the device was whatever Ollama chose. On a performance run that
    #: is not a detail: a "CPU-only" condition whose query embedding was
    #: silently offloaded to the GPU is not the condition it claims to be.
    EMBEDDING_OPTIONS: dict[str, Any] = {"num_gpu": 0}

    def embed(self, text: str, *, model: str | None = None,
              options: dict[str, Any] | None = None) -> list[float]:
        chosen = model or self.embedding_model
        merged = dict(self.EMBEDDING_OPTIONS)
        if options:
            merged.update(options)
        response = self._post(
            "/api/embeddings",
            {"model": chosen, "prompt": text, "options": merged},
        )
        vector = response.get("embedding")
        if not vector:
            raise LLMError(f"embeddings returned nothing usable: {sorted(response)}")
        self.call_count += 1
        return [float(v) for v in vector]

    def embed_batch(self, texts: Sequence[str], *,
                    model: str | None = None,
                    options: dict[str, Any] | None = None) -> list[list[float]]:
        """Ollama has no batch embedding endpoint, so this is a sequential loop.

        Kept as a method anyway so callers do not have to care, and so a future
        backend with real batching can be swapped in without touching them.
        """
        return [self.embed(text, model=model, options=options) for text in texts]

    # --- placement control, for performance runs ---------------------------

    def unload(self, model: str | None = None, *, embedding: bool = False) -> dict[str, Any]:
        """Evict a model from memory so the next call reloads it where asked.

        Ollama keeps a model resident after a call, and a resident model keeps
        the placement it was loaded with. Switching a run from GPU to CPU
        without an eviction therefore measures the *previous* placement, which
        is the failure this repository has already documented once. Posting
        ``keep_alive: 0`` is the documented way to unload.

        ``embedding`` selects the endpoint. An embedding model is not served by
        ``/api/generate``, so addressing it there was simply wrong. Amendment
        1.19.8: what the server did with that request was never observed, so it
        is not claimed that the eviction definitely failed, only that the wrong
        endpoint was used and the outcome was never checked. That model is the
        one whose placement this project pins explicitly, which is why
        ``scripts/preflight_placement.py`` exercises it against a live server.
        """
        chosen = model or (self.embedding_model if embedding else self.default_model)
        if embedding:
            return self._post(
                "/api/embeddings",
                {"model": chosen, "prompt": "", "keep_alive": 0},
            )
        return self._post(
            "/api/generate", {"model": chosen, "prompt": "", "keep_alive": 0}
        )

    def residency(self) -> list[dict[str, Any]]:
        """What Ollama currently has loaded, and where.

        ``/api/ps`` reports ``size`` and ``size_vram`` per loaded model. A model
        with non-zero ``size_vram`` is on the GPU whatever the request asked
        for, so this is the check that turns a requested placement into an
        observed one.
        """
        request = urllib.request.Request(f"{self.base_url}/api/ps")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            # Amendment 1.19. Returning [] here conflated "nothing is loaded"
            # with "I could not ask", and the caller read the empty list as
            # evidence of CPU placement. An unanswered question is not an
            # answer of no.
            raise LLMError(
                f"/api/ps failed against {self.base_url}: {exc}. Residency is "
                "unknown, which is not the same as nothing being resident."
            ) from exc
        return list(payload.get("models") or [])

    #: Pre-declared, so the wait is a stated rule rather than a number tuned
    #: until the check passed. Amendment 1.22.
    UNLOAD_TIMEOUT_SECONDS = 30.0
    UNLOAD_POLL_INTERVAL_SECONDS = 0.25

    def wait_until_unloaded(
        self,
        models,
        *,
        timeout: float | None = None,
        interval: float | None = None,
        now=None,
        sleep=None,
    ) -> dict[str, Any]:
        """Poll ``/api/ps`` until the named models are gone, or give up.

        An unload is a *request*. Disappearance from ``/api/ps`` is the
        *observation*, and the two are not simultaneous. Checking once,
        immediately, conflates "the server has not caught up" with "the server
        ignored me", and those need different responses.

        **The elapsed time is recorded, and it is the point.** Ollama's default
        retention is five minutes. If a model disappears within this timeout the
        unload was effective; if it only goes at around 300 seconds it expired
        on its own and the unload did nothing. A single immediate check cannot
        tell those apart, and neither can a poll that does not report how long
        it waited.

        Transient ``/api/ps`` failures are tolerated *during* the wait and
        recorded, because a momentary refusal is not evidence either way. If the
        window closes with no successful empty observation, the result is
        ``cleared: False`` and the caller fails closed.
        """
        import time as _time

        now = now or _time.monotonic
        sleep = sleep or _time.sleep
        timeout = self.UNLOAD_TIMEOUT_SECONDS if timeout is None else timeout
        interval = self.UNLOAD_POLL_INTERVAL_SECONDS if interval is None else interval

        wanted = {canonical_model_name(m) for m in models if m}
        started = now()
        observations = 0
        transient: list[str] = []
        remaining: list[str] = sorted(wanted)

        while True:
            try:
                observed = self.observed_placement()
                observations += 1
                resident = {
                    canonical_model_name(n) for n in observed["models_loaded"]
                }
                remaining = sorted(wanted & resident)
                if not remaining:
                    # At least one *successful* observation showing them gone.
                    return {
                        "cleared": True,
                        "elapsed_seconds": round(now() - started, 3),
                        "successful_observations": observations,
                        "transient_errors": transient,
                        "remaining": [],
                        "timeout_seconds": timeout,
                        "poll_interval_seconds": interval,
                    }
            except LLMError as exc:
                transient.append(str(exc))

            if now() - started >= timeout:
                return {
                    "cleared": False,
                    "elapsed_seconds": round(now() - started, 3),
                    "successful_observations": observations,
                    "transient_errors": transient,
                    "remaining": remaining,
                    "timeout_seconds": timeout,
                    "poll_interval_seconds": interval,
                }
            sleep(interval)

    def observed_placement(self) -> dict[str, Any]:
        """Requested placement is a hope; this is what is actually loaded.

        Raises ``LLMError`` if ``/api/ps`` cannot be reached. Amendment 1.19:
        the caller must not be able to mistake an unreachable endpoint for a
        CPU-resident model.

        ``size_vram`` is carried through verbatim, including when it is absent,
        so a downstream check can distinguish zero VRAM from unreported VRAM.
        """
        loaded = self.residency()
        on_gpu = [
            m for m in loaded
            if isinstance(m.get("size_vram"), (int, float)) and m["size_vram"] > 0
        ]
        return {
            "models_loaded": [m.get("name") or m.get("model") for m in loaded],
            "any_on_gpu": bool(on_gpu),
            "vram_bytes": {
                (m.get("name") or m.get("model")): m.get("size_vram")
                for m in loaded
            },
            "sizes": {
                (m.get("name") or m.get("model")): m.get("size") for m in loaded
            },
            "complete": all(
                isinstance(m.get("size_vram"), (int, float)) for m in loaded
            ),
        }


# --- mock backend -----------------------------------------------------------

MOCK_DIMENSIONS = 256


def _hashing_vector(text: str, dimensions: int = MOCK_DIMENSIONS) -> list[float]:
    """Deterministic pseudo-embedding based on lexical overlap.

    Each token is hashed into a bucket and counted, then the vector is L2
    normalised. Two texts sharing vocabulary land close together; two texts
    sharing nothing land far apart. That is all a retrieval test needs, and it
    is reproducible across machines and Python versions because it uses
    hashlib rather than the salted built-in ``hash``.
    """
    buckets = [0.0] * dimensions
    tokens = [t for t in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split() if len(t) > 2]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        buckets[int.from_bytes(digest, "big") % dimensions] += 1.0
    norm = math.sqrt(sum(v * v for v in buckets))
    if norm == 0:
        buckets[0] = 1.0
        return buckets
    return [v / norm for v in buckets]


class MockClient:
    """Deterministic backend for tests and development without a model."""

    backend = "mock"

    def __init__(self, config: Config | None = None,
                 responses: dict[str, str] | None = None) -> None:
        self.config = config or load_config()
        self.default_model = "mock-generate"
        self.embedding_model = "mock-embed"
        self.responses = responses or {}
        self.prompts: list[str] = []
        self.call_count = 0
        self.last_options: dict[str, Any] = {}

    def generate(self, prompt: str, *, model: str | None = None,
                 options: dict[str, Any] | None = None) -> Generation:
        self.prompts.append(prompt)
        self.call_count += 1
        # The same merge and filter the real client performs, so option
        # recording is testable without a running backend.
        merged = dict(self.config.require("generation"))
        if options:
            merged.update(options)
        merged = {k: v for k, v in merged.items() if not k.startswith("_")}
        self.last_options = dict(merged)
        for trigger, canned in self.responses.items():
            if trigger in prompt:
                text = canned
                break
        else:
            digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=6).hexdigest()
            text = f"MOCK ANSWER {digest}"
        return Generation(
            text=text,
            model=model or self.default_model,
            prompt_tokens=len(prompt.split()),
            eval_tokens=len(text.split()),
            prompt_seconds=0.0,
            eval_seconds=0.0,
            wall_seconds=0.0,
            options=dict(merged),
        )

    EMBEDDING_OPTIONS: dict[str, Any] = {"num_gpu": 0}

    def unload(self, model: str | None = None, *, embedding: bool = False) -> dict[str, Any]:
        chosen = model or (self.embedding_model if embedding else self.default_model)
        return {"mock": True, "unloaded": chosen, "embedding": embedding}

    def residency(self) -> list[dict[str, Any]]:
        return []

    def observed_placement(self) -> dict[str, Any]:
        return {"models_loaded": [], "any_on_gpu": False, "vram_bytes": {},
                "sizes": {}, "complete": True}

    def wait_until_unloaded(self, models, **kwargs) -> dict[str, Any]:
        return {"cleared": True, "elapsed_seconds": 0.0,
                "successful_observations": 1, "transient_errors": [],
                "remaining": [], "timeout_seconds": 0.0,
                "poll_interval_seconds": 0.0}

    def embed(self, text: str, *, model: str | None = None,
              options: dict[str, Any] | None = None) -> list[float]:
        self.call_count += 1
        return _hashing_vector(text)

    def embed_batch(self, texts: Sequence[str], *,
                    model: str | None = None,
                    options: dict[str, Any] | None = None) -> list[list[float]]:
        return [self.embed(text, model=model) for text in texts]

    def describe_endpoint(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "base_url": None,
            "version": "mock",
            "reachable": True,
            "model_store_fingerprint": "mock",
        }


def build_client(config: Config | None = None, *, mock: bool = False) -> LLMClient:
    """Return the configured backend.

    ``mock`` is an explicit argument rather than a config value on purpose: an
    experimental run must never fall back to the mock backend because a server
    was unreachable. A failed run is recoverable; a run silently scored against
    fabricated output is not.
    """
    config = config or load_config()
    if mock:
        return MockClient(config)
    backend = config.require("llm.backend")
    if backend != "ollama":
        raise LLMError(f"Unknown llm.backend {backend!r}")
    return OllamaClient(config)
