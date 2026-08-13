"""Summarise a set of arm runs, at both levels and by conflict subtype.

This reads runs and the question set. It is evaluation code, so unlike the
verifier it is allowed to know which family a question belongs to; that is the
whole point of scoring.

Every figure appears twice, per the protocol: a plain mean over questions for
readability, and a mean of per-group means for inference. The sample size cited
is the number of groups, never the number of questions.

Run:

    python scripts/summarise_arms.py --split dev

What this does not do
---------------------
It does not score conflict handling or answer correctness. Those are manual and
blinded, against the rubric written into the question set, and no automatic
proxy for them is reported here. What it does report is everything that can be
counted without judgement: detection rates, citation behaviour, abstention,
latency, and the failure modes the verifier records about itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.evaluation.aggregate import aggregate  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.evaluation.run_writer import read_run  # noqa: E402


def load_runs(config, split: str, mock: bool) -> dict[str, list[dict]]:
    index_path = Path(config.path("paths.results")) / (
        f"latest_{split}{'_mock' if mock else ''}.json"
    )
    if not index_path.exists():
        raise SystemExit(f"No run index at {index_path}. Run scripts/run_arms.py first.")
    directories = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        arm: read_run(path if Path(path).is_absolute() else ROOT / path)[1]
        for arm, path in sorted(directories.items())
    }


def subtype_of(record, registry) -> str:
    """The declared subtype, used only for reporting, never at runtime."""
    if not record.get("family_id"):
        return record.get("category") or "other"
    try:
        return registry.by_id(record["family_id"]).conflict_type
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    config = load_config()
    evaluation = load_evaluation_config()
    registry = load_conflicts(evaluation.path("conflicts"))
    question_set = load_question_set(evaluation.path("question_set"))
    runs = load_runs(config, args.split, args.mock)

    print(f"Split {args.split}{'  (MOCK)' if args.mock else ''}   "
          f"arms {', '.join(runs)}   "
          f"{len(question_set.split(args.split))} questions in "
          f"{len(question_set.split(args.split).groups)} groups\n")

    # --- countable metrics, both levels -------------------------------------
    metrics = [
        ("has_valid_citation_ids", "citation validity"),
        ("scoring.citation_support", "citation support"),
        ("scoring.citation_completeness", "citation completeness"),
        ("refusal_heuristic", "refusal (heuristic)"),
    ]
    print(f"{'metric':<26} {'arm':<4} {'question':>9} {'group':>8} {'n groups':>9}")
    for key, label in metrics:
        for arm, records in runs.items():
            result = aggregate(records, key)
            q = "     -   " if result.question_level is None else f"{result.question_level:9.3f}"
            g = "    -   " if result.group_level is None else f"{result.group_level:8.3f}"
            print(f"{label:<26} {arm:<4} {q} {g} {result.group_count:9}")
        print()

    # --- what the verifier said about itself --------------------------------
    for arm, records in runs.items():
        verifications = [r["verification"] for r in records if "verification" in r]
        if not verifications:
            continue
        n = len(verifications)
        print(f"Arm {arm} verification behaviour, over {n} answers:")
        print(f"  conflict detected        {sum(v['conflict_detected'] for v in verifications):3}")
        print(f"    resolved supersession  {sum(v['is_resolved'] for v in verifications):3}")
        print(f"    unresolved             {sum(v['is_unresolved'] for v in verifications):3}")
        print(f"  escalated                {sum(v['escalate'] for v in verifications):3}")
        print(f"  answer revised           {sum(r.get('answer_revised', False) for r in records):3}")
        print(f"  revision rejected        {sum(v.get('revision_rejected', False) for v in verifications):3}")
        print(f"  invented evidence        {sum(bool(v['invented_ids']) for v in verifications):3}")
        print(f"  parse failed             {sum(v['parse_failed'] for v in verifications):3}")

        relationships = defaultdict(int)
        for v in verifications:
            relationships[v["relationship"]] += 1
        print("  relationships inferred   "
              + ", ".join(f"{k} {n}" for k, n in sorted(relationships.items())))
        print()

    # --- detection against the declared subtype ------------------------------
    # The reported comparison. A verifier that flags everything scores well on
    # the conflict families and badly here, which is the point of the controls.
    print("Conflict detection by declared subtype "
          "(compatible families are negative controls, where detection is a false positive):")
    header = f"  {'subtype':<24}" + "".join(f"{a:>10}" for a in runs)
    print(header)
    subtypes = defaultdict(lambda: defaultdict(list))
    for arm, records in runs.items():
        for record in records:
            if not record.get("family_id") or "verification" not in record:
                continue
            subtypes[subtype_of(record, registry)][arm].append(
                record["verification"]["conflict_detected"]
            )
    for subtype, per_arm in sorted(subtypes.items()):
        row = f"  {subtype:<24}"
        for arm in runs:
            values = per_arm.get(arm)
            # Arms without a verification layer do not produce a detection at
            # all. Showing 0.00 read as "detected nothing", which is a claim
            # about their behaviour rather than the absence of the measurement.
            # Their conflict handling comes from blinded manual scoring.
            row += f"{'      N/A' if not values else f'{sum(values) / len(values):>10.2f}'}"
        print(row)
    print()

    # --- cost ----------------------------------------------------------------
    print("Latency, seconds per question:")
    for arm, records in runs.items():
        wall = [r["wall_seconds"] for r in records if "wall_seconds" in r]
        verify = [r.get("verification_seconds", 0.0) for r in records]
        mean = sum(wall) / len(wall) if wall else 0.0
        vmean = sum(verify) / len(verify) if verify else 0.0
        print(f"  {arm}  total {mean:7.2f}   of which verification {vmean:6.2f}")

    baseline = runs.get("B")
    if baseline and "D" in runs:
        b = sum(r["wall_seconds"] for r in baseline) / len(baseline)
        d = sum(r["wall_seconds"] for r in runs["D"]) / len(runs["D"])
        if b:
            import platform
            on_pi = platform.machine().lower().startswith("aarch")
            note = ("H5 predicts 1.5 to 2.5" if on_pi else
                    "H5 is a Pi-only prediction; this is a laptop figure")
            print(f"\n  D/B ratio {d / b:.2f}   ({note})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
