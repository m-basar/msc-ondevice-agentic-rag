"""Prove eviction and placement work, against a live Ollama, before a timed run.

    python scripts/preflight_placement.py --placement cpu
    python scripts/preflight_placement.py --placement gpu

Amendment 1.20. This is a **standalone** command and nothing calls it
automatically. That is the point of it existing separately.

Two reasons it cannot live inside `run_arms.py`.

**It would warm the run it is meant to protect.** A preflight loads a model.
Loading a model immediately before Arm B turns a cold start into a warm one and
changes the very quantity H5 measures. Amendment 1.19's version did exactly
that, and it was invoked automatically.

**It could not otherwise be exercised.** If the only real invocation is inside a
performance run, then proving eviction works live requires starting another
performance execution, which is the thing there is no budget for. As a separate
command it can be run any number of times, on either machine, without producing
a run directory or touching a result.

What it checks, for **all three** models rather than the generator alone:

* the generation model, through `/api/generate`
* the verification model, through `/api/generate`
* the embedding model, through `/api/embeddings`, which is the one whose
  eviction was got wrong in amendment 1.17 and the reason a preflight was added

For each: evict, confirm it is gone from `/api/ps`, load it with a synthetic
prompt that is **not a test question**, confirm it is resident on the intended
device, then evict it again. Generation and verification follow the requested
placement. The embedding model is always expected on the CPU, because the
project pins it there.

**It leaves nothing loaded.** Everything is unloaded before exit, so running it
does not warm a subsequent run either.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import (  # noqa: E402
    LLMError,
    OllamaClient,
    canonical_model_name,
)

#: A prompt that is not a test question and asserts nothing about the corpus.
SYNTHETIC_PROMPT = "ping"


def resident_names(observed: dict) -> set[str]:
    """Loaded model names, canonicalised. Amendment 1.21."""
    return {canonical_model_name(n) for n in (observed.get("models_loaded") or [])}


def vram_for(observed: dict, model: str):
    """VRAM for a model, matched canonically rather than by exact string."""
    wanted = canonical_model_name(model)
    for name, value in (observed.get("vram_bytes") or {}).items():
        if canonical_model_name(name) == wanted:
            return value
    return None


def check_model(client: OllamaClient, model: str, *, embedding: bool,
                expected: str, placement: str) -> dict:
    """One evict, load, observe cycle for a single model.

    The caller is responsible for the final eviction, which happens in a
    ``finally`` block so that a failure part way through still cleans up.
    Amendment 1.21: the previous version unloaded at the end of this function,
    so any raise above that line left the model resident, contradicting 1.20's
    promise that nothing is left loaded.
    """
    stage: dict = {"model": model, "endpoint":
                   "/api/embeddings" if embedding else "/api/generate",
                   "expected_placement": expected}

    client.unload(model, embedding=embedding)
    # Amendment 1.22. An unload is a request; disappearance is the
    # observation, and the two are not simultaneous. Checking once immediately
    # cannot tell "the server has not caught up" from "the server ignored me".
    waited = client.wait_until_unloaded([model])
    stage["unload_wait"] = waited
    stage["loaded_after_eviction"] = waited["remaining"]
    if not waited["cleared"]:
        raise LLMError(
            f"{model} was still resident {waited['elapsed_seconds']}s after an "
            f"eviction through {stage['endpoint']}, over "
            f"{waited['timeout_seconds']}s of polling. Eviction is not taking "
            "effect, so a run would measure whatever placement the model "
            "already had."
        )

    if embedding:
        # EMBEDDING_OPTIONS already pins num_gpu to 0.
        client.embed(SYNTHETIC_PROMPT)
    else:
        # Amendment 1.21. The placement must be *applied*, not merely observed.
        # Without num_gpu the server picks a device and the check becomes a
        # report of whatever Ollama chose rather than a test of the condition.
        client.generate(
            SYNTHETIC_PROMPT, model=model,
            options={"num_predict": 1,
                     "num_gpu": 0 if placement == "cpu" else -1},
        )
    stage["options_applied"] = (
        {"num_gpu": 0} if embedding
        else {"num_gpu": 0 if placement == "cpu" else -1}
    )

    after_load = client.observed_placement()
    stage["loaded_after_synthetic_call"] = after_load["models_loaded"]
    stage["vram_bytes"] = after_load["vram_bytes"]

    if canonical_model_name(model) not in resident_names(after_load):
        raise LLMError(
            f"{model} is not resident after a synthetic call, so its placement "
            "cannot be observed."
        )
    vram = vram_for(after_load, model)
    if not isinstance(vram, (int, float)):
        raise LLMError(
            f"/api/ps reported {model} without a numeric size_vram, so its "
            "placement cannot be confirmed."
        )
    seen = "gpu" if vram > 0 else "cpu"
    stage["observed_placement"] = seen
    if seen != expected:
        raise LLMError(
            f"{model} loaded onto {seen} but {expected} was expected "
            f"(size_vram {vram})."
        )
    return stage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--placement", choices=["gpu", "cpu"], required=True,
                        help="intended device for the generation and "
                             "verification models; embeddings are always CPU")
    parser.add_argument("--out", default=str(ROOT / "results" / "diagnostics"))
    args = parser.parse_args(argv)

    config = load_config()
    client = OllamaClient(config)

    models = [
        (config.require("llm.generation_model"), False, args.placement),
        (config.require("llm.verification_model"), False, args.placement),
        # Pinned to the CPU by EMBEDDING_OPTIONS whatever the run's placement.
        (config.require("llm.embedding_model"), True, "cpu"),
    ]

    report: dict = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "requested_placement": args.placement,
        "synthetic_prompt": SYNTHETIC_PROMPT,
        "uses_test_questions": False,
        "stages": [],
    }

    try:
        for model, embedding, expected in models:
            print(f"  checking {model} ({'embedding' if embedding else 'generate'}) "
                  f"-> expecting {expected}")
            report["stages"].append(
                check_model(client, model, embedding=embedding, expected=expected,
                            placement=args.placement)
            )
            print(f"    confirmed on {report['stages'][-1]['observed_placement']}")
        report["result"] = "pass"
    except LLMError as exc:
        report["result"] = "fail"
        report["error"] = str(exc)
        print(f"\n  PREFLIGHT FAILED: {exc}", file=sys.stderr)
    finally:
        # Amendment 1.21. Cleanup is unconditional. A failure part way through
        # previously jumped straight to the handler with a model still loaded,
        # which is exactly the warm state 1.20 promised never to leave behind.
        cleanup_errors = []
        for model, embedding, _ in models:
            try:
                client.unload(model, embedding=embedding)
            except Exception as exc:  # pragma: no cover - network dependent
                cleanup_errors.append(f"{model}: {exc}")
        try:
            # Poll for all three together rather than reading once. The
            # elapsed figure is what distinguishes an effective unload from
            # Ollama's own five-minute expiry on the next run.
            waited = client.wait_until_unloaded([m for m, _, _ in models])
            report["cleanup_wait"] = waited
            report["loaded_at_exit"] = waited["remaining"]
            report["clean_exit"] = waited["cleared"]
            if waited["transient_errors"]:
                cleanup_errors.extend(waited["transient_errors"])
        except Exception as exc:  # pragma: no cover - network dependent
            report["loaded_at_exit"] = None
            report["clean_exit"] = False
            cleanup_errors.append(f"residency unreadable at exit: {exc}")
        report["cleanup_errors"] = cleanup_errors
        # Exit residency is recorded whatever happened, and a dirty exit is a
        # failure in its own right even when every model check passed.
        if report.get("result") == "pass" and not report.get("clean_exit"):
            report["result"] = "fail"
            waited = report.get("cleanup_wait") or {}
            report["error"] = (
                f"models remain loaded at exit: {report['loaded_at_exit']}, "
                f"after {waited.get('elapsed_seconds')}s of polling against a "
                f"{waited.get('timeout_seconds')}s budget. A preflight that "
                "leaves a model resident warms the next run."
            )
            print(f"\n  PREFLIGHT FAILED: {report['error']}", file=sys.stderr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = out / f"{stamp}_preflight_{args.placement}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(f"\n  written to {target}")

    if report["result"] == "pass":
        print("  all three models evicted, loaded on the intended device, and "
              "evicted again. Nothing is left loaded.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
