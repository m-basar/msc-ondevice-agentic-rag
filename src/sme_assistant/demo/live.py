"""Run Arm D live, for demonstration only.

Amendment 1.27. Nothing this module produces is scored, and nothing it produces
is written into a frozen run directory. It builds the same pipeline the
experiment used, so what is demonstrated is the artefact rather than a
reimplementation of it, but it calls ``score_answer`` nowhere: there is no
quality metric in the output to be mistaken for a result later.

Only Arm D runs here. A live four-arm comparison would produce something that
looks exactly like the reported experiment and is not it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..common.config import load_config
from ..common.hostinfo import ollama_info
from ..common.llm_client import build_client, canonical_model_name
from ..generate.generator import Generator
from ..ingest.index import Index, index_path_for
from ..retrieve.retriever import EvidenceFormat, RetrievalMode, Retriever
from ..verify import Verifier
from .replay import ArmAnswer, _arm_answer

#: Arm D, restated here rather than imported from ``scripts/run_arms.py``.
#: The dashboard must not import the experiment runner: that dependency would
#: run the wrong way and would put a demonstrator on the experiment's path.
LIVE_ARM = "D"
LIVE_RETRIEVAL = RetrievalMode.ALL
LIVE_EVIDENCE = EvidenceFormat.WITH_STATUS


def _file_sha256(path: Path) -> str | None:
    """Restated here rather than imported from ``evaluation.run_writer``.

    Two lines of hashlib against a dependency from the demonstrator onto the
    experiment runner, which is the direction this module exists to avoid.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


class LiveUnavailable(RuntimeError):
    """The models or the index needed to answer are not present."""


@dataclass
class LiveAssistant:
    """Arm D over the frozen index, for questions typed at the dashboard."""

    config: Any
    client: Any
    retriever: Retriever
    generator: Generator
    verifier: Verifier

    @classmethod
    def build(cls, root: Path | str | None = None) -> "LiveAssistant":
        config = load_config() if root is None else load_config(Path(root) / "config.json")
        index = Index.load(index_path_for(config))
        client = build_client(config)
        return cls(
            config=config, client=client,
            retriever=Retriever(index, client, config),
            generator=Generator(client, config),
            verifier=Verifier(client, config),
        )

    def model_status(self) -> dict[str, Any]:
        """Check before accepting a question, not after failing to answer one.

        A dashboard that takes a question and then dies on a connection error
        has wasted the user's attention and told them nothing useful. This
        reports what is missing while the input is still disabled.
        """
        generation = self.config.require("llm.generation_model")
        verification = self.config.get("llm.verification_model")
        base_url = self.config.get("llm.ollama_url") or ""
        status: dict[str, Any] = {
            "generation": generation,
            "verification": verification,
            "base_url": base_url,
            "ready": False,
            "detail": "",
        }
        try:
            info = ollama_info(base_url)
        except Exception as exc:  # noqa: BLE001 - reported, never raised onward
            status["detail"] = f"could not reach Ollama at {base_url}: {exc}"
            return status
        if not info.get("reachable"):
            status["detail"] = (
                f"Ollama is not reachable at {base_url}. Start it with "
                "'ollama serve', or use Frozen Study Replay, which needs no "
                "models at all."
            )
            return status
        present = {canonical_model_name(m.get("name"))
                   for m in info.get("models") or []}
        missing = [m for m in (generation, verification)
                   if m and canonical_model_name(m) not in present]
        if missing:
            status["detail"] = (
                "Ollama is running but these models are not pulled: "
                + ", ".join(missing)
                + ". Pull them with 'ollama pull <model>'."
            )
            return status
        status["ready"] = True
        status["version"] = info.get("version")
        return status

    def answer(self, question: str) -> ArmAnswer:
        """Retrieve, draft, verify. One question, Arm D, nothing recorded."""
        text = (question or "").strip()
        if not text:
            raise LiveUnavailable("no question was given")
        retrieval = self.retriever.retrieve(text, mode=LIVE_RETRIEVAL)
        draft = self.generator.answer(text, retrieval,
                                      evidence_format=LIVE_EVIDENCE)
        verified = self.verifier.verify(draft)
        payload = (verified.to_dict() if hasattr(verified, "to_dict")
                   else dict(verified))
        payload.update({
            "arm": LIVE_ARM,
            "question_id": "live",
            "question": text,
            "category": "live demonstration",
            "family_id": None,
        })
        payload.setdefault("answer", getattr(verified, "final_answer", "") or "")
        return _arm_answer(payload)

    def frozen_arm_d_agreement(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Compare this live pipeline with the frozen Arm D manifest.

        Amendment 1.32.4. The previous version put every field into one verdict
        and, when only ``index_file_sha256`` differed, returned ``matches=True``
        with ``index_rebuilt=True``. **A differing hash does not establish that a
        file was rebuilt from the same recipe.** A reviewer swapped the index
        metadata to a different 768-dimensional embedding model and the
        comparison still reported a match and a rebuild, because the only
        index-side fields checked were the corpus and chunk-set hashes, and
        those describe the *input* to the build rather than the build.

        Three questions are now answered separately and named separately,
        because they have different answers and different consequences.

        ``configuration_matches``
            Is this the same pipeline the frozen run used - the arm, the two
            modes, the two models, the configuration fingerprint, the sampling
            options and the retrieval parameters? Fixed by the commit, so any
            checkout must agree.
        ``source_matches``
            Is it over the same material and built by the same recipe - corpus,
            chunk set, embedding model, dimensions, chunk count, chunking
            parameters, backend and model-store fingerprint? A wrong embedding
            model fails here, loudly.
        ``frozen_index_identical``
            Is it the *same index file* the frozen run used? Only ever true on
            the machine that produced it, because the index is a build artefact
            and is not in the repository.

        ``matches`` is configuration and source. The index file's identity is
        reported as its own fact and never described as a rebuild, which is a
        claim about history this code cannot make.

        Reports rather than raises: an index can legitimately be rebuilt, and a
        demonstrator that refuses to start is less honest than one that says
        which fields differ. What must not happen is silence.
        """
        arm = dict(manifest.get("arm") or {})
        provenance = dict(manifest.get("provenance") or {})
        frozen_index = dict(provenance.get("index_metadata") or {})
        clean = lambda d: {k: v for k, v in (d or {}).items()
                           if not str(k).startswith("_")}
        index_metadata = dict(getattr(self.retriever.index, "metadata", {}) or {})

        def endpoint(meta):
            return clean(meta.get("endpoint") or {}).get("model_store_fingerprint")

        configuration = {
            "arm": (LIVE_ARM, arm.get("arm")),
            "retrieval_mode": (LIVE_RETRIEVAL.value, arm.get("retrieval_mode")),
            "evidence_format": (LIVE_EVIDENCE.value, arm.get("evidence_format")),
            "verification": (True, arm.get("verification")),
            "generation_model": (
                canonical_model_name(self.config.require("llm.generation_model")),
                canonical_model_name(arm.get("generation_model") or "")),
            "verification_model": (
                canonical_model_name(self.config.get("llm.verification_model") or ""),
                canonical_model_name(arm.get("verification_model") or "")),
            "config_sha256": (self.config.fingerprint(),
                              provenance.get("config_sha256")),
            "generation_options": (clean(self.config.get("generation")),
                                   clean(provenance.get("generation_options"))),
            "retrieval": ({"top_k": self.config.get("retrieval.top_k"),
                           "min_similarity": self.config.get("retrieval.min_similarity")},
                          clean(provenance.get("retrieval"))),
        }
        # The build recipe, not just its input. Amendment 1.32.4: an index over
        # the same corpus built by a different embedding model is a different
        # index, and comparing corpus hashes alone cannot see that.
        source = {
            "corpus_sha256": (index_metadata.get("corpus_sha256"),
                              provenance.get("corpus_sha256")),
            "chunk_set_sha256": (index_metadata.get("chunk_set_sha256"),
                                 provenance.get("chunk_set_sha256")),
            "embedding_model": (
                canonical_model_name(index_metadata.get("embedding_model") or ""),
                canonical_model_name(frozen_index.get("embedding_model") or "")),
            "dimensions": (index_metadata.get("dimensions"),
                           frozen_index.get("dimensions")),
            "chunk_count": (index_metadata.get("chunk_count"),
                            frozen_index.get("chunk_count")),
            "chunking": (clean(index_metadata.get("chunking")),
                         clean(frozen_index.get("chunking"))),
            "index_backend": (index_metadata.get("backend"),
                              frozen_index.get("backend")),
            "model_store_fingerprint": (endpoint(index_metadata),
                                        endpoint(frozen_index)),
        }
        identity = {
            "index_file_sha256": (_file_sha256(index_path_for(self.config)),
                                  provenance.get("index_file_sha256")),
        }

        def disagreeing(group):
            return sorted(k for k, (live, frozen) in group.items() if live != frozen)

        configuration_differs = disagreeing(configuration)
        source_differs = disagreeing(source)
        index_differs = disagreeing(identity)
        fields = {**configuration, **source, **identity}
        return {
            "matches": not configuration_differs and not source_differs,
            "differs": configuration_differs + source_differs,
            "configuration_matches": not configuration_differs,
            "configuration_differs": configuration_differs,
            "source_matches": not source_differs,
            "source_differs": source_differs,
            "frozen_index_identical": not index_differs,
            "local_index_file_differs": bool(index_differs),
            "compared": sorted(configuration) + sorted(source),
            "fields": {k: {"live": live, "frozen": frozen}
                       for k, (live, frozen) in fields.items()},
        }
