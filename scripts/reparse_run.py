"""Re-read a recorded run's raw verifier output under the current validation.

    python scripts/reparse_run.py --run results/runs/..._D_dev_pilot05

**No model is called.** Every response is already on disk; this parses the
stored text again and reports what the pipeline would serve today.

Why this is not a re-run
------------------------
A validation change alters what the pipeline does with an answer, not what the
model said. Re-running would draw fresh samples and mix a model difference into
a measurement of a code change. Reparsing holds the model output exactly fixed
and varies only the rules, which is the only way to attribute the difference.

Why it does not write a run directory
-------------------------------------
Deliberately. A reparsed result is a counterfactual - what *would* have been
served - and a directory beside the real runs would eventually be read as one.
The summariser continues to report pilot 05 as it was actually served, because
that is what happened. This produces a diagnostic alongside it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.evaluation.run_writer import read_run  # noqa: E402
from sme_assistant.evaluation.stopping_gate import (  # noqa: E402
    evaluate_gate,
    format_gate,
)
from sme_assistant.generate.generator import extract_citations  # noqa: E402
from sme_assistant.verify import schema  # noqa: E402


def reparse(record: dict) -> dict:
    """One record as the current validation would have produced it."""
    updated = copy.deepcopy(record)
    raw = (record.get("verification") or {}).get("raw") or ""
    available = [s["chunk_id"] for s in record["retrieval"]["results"]]
    draft = record.get("draft_answer") or record.get("answer") or ""

    result = schema.parse(raw, available, {}, draft=draft)
    served = result.final_answer or draft
    citations, documents = extract_citations(served)

    updated["verification"] = result.to_dict()
    updated["answer"] = served
    updated["answer_revised"] = bool(result.revised)
    updated["revision_rejected"] = bool(result.revision_rejected)
    updated["citations"] = list(citations)
    updated["document_citations"] = list(documents)
    updated["hallucinated_citations"] = [
        c for c in citations if c not in available
    ]
    updated["has_valid_citation_ids"] = bool(citations) and not updated[
        "hallucinated_citations"
    ]
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    config = load_config()
    evaluation = load_evaluation_config()
    registry = load_conflicts(evaluation.path("conflicts"))
    question_set = load_question_set(evaluation.path("question_set"))

    directory = Path(args.run)
    if not directory.is_absolute():
        directory = ROOT / directory
    manifest, records = read_run(directory)
    updated = [reparse(r) for r in records]

    def summary(rows):
        return {
            "revisions_served": sum(
                bool(r.get("answer_revised") and not r.get("revision_rejected"))
                for r in rows
            ),
            "revisions_rejected": sum(bool(r.get("revision_rejected")) for r in rows),
            "conflicts_detected": sum(
                r["verification"]["conflict_detected"] for r in rows
            ),
            "claim_audit_complete": sum(
                bool(r["verification"].get("claim_audit_complete")) for r in rows
            ),
            "parse_failed": sum(r["verification"]["parse_failed"] for r in rows),
            "relationships": dict(Counter(
                r["verification"]["relationship"] for r in rows
            )),
        }

    before, after = summary(records), summary(updated)
    print(f"Run       {directory.name}")
    print(f"Model     {(manifest.get('arm') or {}).get('verification_model')}")
    print(f"Records   {len(records)}   (no model called)\n")
    print(f"{'quantity':<26}{'as served':>12}{'reparsed':>12}")
    print("-" * 50)
    for key in ("revisions_served", "revisions_rejected", "conflicts_detected",
                "claim_audit_complete", "parse_failed"):
        mark = "" if before[key] == after[key] else "   <--"
        print(f"{key:<26}{before[key]:>12}{after[key]:>12}{mark}")
    print(f"\n  relationships as served: {before['relationships']}")
    print(f"  relationships reparsed : {after['relationships']}")
    if before["relationships"] == after["relationships"]:
        print("\n  Inference is unchanged, as it must be: the validation "
              "governs what is done\n  with the verifier's finding, not what "
              "the finding was.")

    print("\n" + "=" * 68)
    print("The gate as it would read under the current validation.")
    print("This is a counterfactual. The run as served is what happened.")
    print("=" * 68 + "\n")
    expected = {q.question_id: tuple(q.expected_chunks or ())
                for q in question_set.split(manifest.get("split", "dev"))}
    for line in format_gate(evaluate_gate(
        updated, registry, arm=(manifest.get("arm") or {}).get("arm", "D"),
        expected_chunks=expected,
    )):
        print(line)

    out = Path(config.path("paths.results")).parent / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"{stamp}_reparse_{directory.name}.json"
    path.write_text(json.dumps({
        "what_this_is": (
            "A counterfactual. The stored raw verifier output of "
            f"{directory.name}, parsed again under the validation current at "
            "the commit below. No model was called. The run as served remains "
            "the record of what happened."
        ),
        "source_run": directory.name,
        "source_manifest": manifest,
        "as_served": before,
        "reparsed": after,
        "per_question": [
            {
                "question_id": r["question_id"],
                "served_before": o.get("answer"),
                "served_after": r.get("answer"),
                "revised_before": bool(o.get("answer_revised")),
                "revised_after": bool(r.get("answer_revised")),
                "validation_failures": r["verification"].get("validation_failures"),
                "claim_audit_complete": r["verification"].get("claim_audit_complete"),
            }
            for o, r in zip(records, updated)
            if o.get("answer") != r.get("answer")
            or bool(o.get("answer_revised")) != bool(r.get("answer_revised"))
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
