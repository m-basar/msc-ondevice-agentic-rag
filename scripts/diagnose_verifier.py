"""Why did the verifier miss every conflict? Model, window, or task?

Pilot 02 gave detection of zero across eighteen genuine-conflict questions. On
CONF-01-Q1 the verifier received both HR-13#001 and HR-03#001, with the
SUPERSEDED marker present, and answered "the evidence does not mention mileage
rates for business travel". That is not a parsing failure or a prompt-ordering
failure. Something upstream of wording is wrong.

Three explanations fit, and they call for entirely different responses:

``model``    llama3.2:3b cannot do structured cross-passage reasoning. Response:
             report it as an on-device feasibility finding, and try a different
             model for the same prompt.
``window``   six chunks and 3,000 characters bury the two that matter. Response:
             give the verifier a narrower window, which is a design change
             rather than a prompt change.
``task``     the task is beyond a small model however it is framed. Response:
             a negative result, honestly reported.

This runs one prompt across those axes on a handful of development conflicts
and prints what each combination inferred. It is a diagnostic, not an
experiment: nothing here is reported, and it touches only tuning families.

    python scripts/diagnose_verifier.py
    python scripts/diagnose_verifier.py --models llama3.2:3b qwen2.5:3b
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
from sme_assistant.common.llm_client import build_client  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.generate.generator import Generator  # noqa: E402
from sme_assistant.ingest.index import Index, index_path_for  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402
from sme_assistant.retrieve.retriever import Retriever  # noqa: E402
from sme_assistant.verify import schema  # noqa: E402
from sme_assistant.verify.verifier import build_verification_prompt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                        default=["llama3.2:3b", "qwen2.5:3b", "phi3:mini"])
    parser.add_argument("--windows", nargs="+", type=int, default=[6, 2])
    parser.add_argument("--questions", type=int, default=4)
    args = parser.parse_args()

    config = load_config()
    evaluation = load_evaluation_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(evaluation.path("conflicts"))
    question_set = load_question_set(evaluation.path("question_set"))
    client = build_client(config)
    index = Index.load(index_path_for(config), kb=kb, config=config)
    retriever = Retriever(index, client, config)
    generator = Generator(client, config)

    # Development conflict families only, one question each, so the sample
    # spans subtypes rather than repeating one family's paraphrases.
    seen, chosen = set(), []
    for question in question_set.split("dev"):
        if not question.family_id or question.family_id in seen:
            continue
        family = registry.by_id(question.family_id)
        if not family.is_conflict:
            continue
        seen.add(question.family_id)
        chosen.append((question, family))
        if len(chosen) >= args.questions:
            break

    print(f"{len(chosen)} development conflict questions, "
          f"{len(args.models)} models, windows {args.windows}\n")
    print(f"{'question':<14}{'declared':<22}{'model':<14}{'k':>3}  inferred")
    print("-" * 76)

    rows = []
    for question, family in chosen:
        for k in args.windows:
            retrieval = retriever.retrieve(question.text, top_k=k)
            draft = generator.answer(question.text, retrieval)
            prompt = build_verification_prompt(
                question.text, draft.answer, retrieval.evidence_text()
            )
            for model in args.models:
                options = {k2: v for k2, v in config.require("verification").items()
                           if k2 != "confidence"}
                generation = client.generate(prompt, model=model, options=options)
                result = schema.parse(
                    generation.text, [s.chunk_id for s in retrieval], {}, draft.answer
                )
                flag = "  <-- detected" if result.conflict_detected else ""
                print(f"{question.question_id:<14}{family.conflict_type:<22}"
                      f"{model:<14}{k:>3}  {result.relationship}{flag}")
                rows.append({
                    "question_id": question.question_id,
                    "declared": family.conflict_type,
                    "model": model, "top_k": k,
                    "inferred": result.relationship,
                    "detected": result.conflict_detected,
                    "parse_failed": result.parse_failed,
                    "rationale": result.rationale,
                    "seconds": generation.wall_seconds,
                })
        print()

    out = Path(config.path("paths.results")).parent / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"{stamp}_verifier_diagnosis.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("Detection rate by model and window:")
    for model in args.models:
        for k in args.windows:
            subset = [r for r in rows if r["model"] == model and r["top_k"] == k]
            hits = sum(r["detected"] for r in subset)
            print(f"  {model:<14} k={k}  {hits}/{len(subset)}")

    print(f"\nWritten to {path}")
    print("\nReading it:")
    print("  detection improves at k=2 only    -> the window buries the conflict;")
    print("                                       narrow the verifier's evidence")
    print("  another model detects at k=6      -> llama3.2:3b is the limit;")
    print("                                       an on-device capability finding")
    print("  nothing detects anywhere          -> the task is beyond a 3B model;")
    print("                                       report the null result honestly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
