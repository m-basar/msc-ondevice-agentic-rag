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

It also prints the stopping gate, which is a computed decision rather than a
metric. See ``evaluation/stopping_gate.py`` for the rule and ``docs/
VERIFIER_PROTOCOL.md`` for why it was committed before the run it judges.
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
from sme_assistant.verify.schema import ABSTENTION_TEXT  # noqa: E402
from sme_assistant.evaluation.stopping_gate import (  # noqa: E402
    evaluate_gate,
    format_gate,
)


def load_runs(config, split: str, mock: bool) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Return answers and manifests, keyed by arm.

    The manifest used to be discarded here. That is how the H5 latency note
    came to describe the machine doing the summarising rather than the machine
    that produced the numbers: a Pi run summarised on a laptop was labelled a
    laptop figure. What produced a measurement is recorded in the run, so it
    should be read from the run.
    """
    index_path = Path(config.path("paths.results")) / (
        f"latest_{split}{'_mock' if mock else ''}.json"
    )
    if not index_path.exists():
        raise SystemExit(f"No run index at {index_path}. Run scripts/run_arms.py first.")
    directories = json.loads(index_path.read_text(encoding="utf-8"))
    loaded = {
        arm: read_run(path if Path(path).is_absolute() else ROOT / path)
        for arm, path in sorted(directories.items())
    }
    return ({arm: answers for arm, (_, answers) in loaded.items()},
            {arm: manifest for arm, (manifest, _) in loaded.items()})


def hardware_of(manifest: dict) -> tuple[str, str]:
    """``(machine, hostname)`` as recorded when the run was made."""
    host = ((manifest or {}).get("environment") or {}).get("host") or {}
    return (str(host.get("machine") or "unknown"),
            str(host.get("hostname") or manifest.get("host") or "unknown"))


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
    runs, manifests = load_runs(config, args.split, args.mock)

    print(f"Split {args.split}{'  (MOCK)' if args.mock else ''}   "
          f"arms {', '.join(runs)}   "
          f"{len(question_set.split(args.split))} questions in "
          f"{len(question_set.split(args.split).groups)} groups\n")

    # --- countable metrics, both levels -------------------------------------
    # Citation metrics are computed over answers that make a claim. An
    # abstention cites nothing by design, so including them measures how often
    # the verifier refused and calls it citation validity: pilot 04's figure
    # fell from 0.78 to 0.34 with no change in citation behaviour at all,
    # because 25 answers had become the template.
    def makes_a_claim(record):
        verification = record.get("verification") or {}
        if "served_abstention" in verification:
            return not verification["served_abstention"]
        return str(record.get("answer") or "").strip() != ABSTENTION_TEXT

    claiming = {arm: [r for r in records if makes_a_claim(r)]
                for arm, records in runs.items()}
    abstained = {arm: len(records) - len(claiming[arm])
                 for arm, records in runs.items()}
    if any(abstained.values()):
        print("Abstentions served (excluded from the citation metrics below, "
              "since an abstention cites nothing by design):")
        for arm, n in abstained.items():
            print(f"  {arm}  {n}/{len(runs[arm])}")
        print()

    metrics = [
        ("has_valid_citation_ids", "citation validity"),
        ("scoring.citation_support", "citation support"),
        ("scoring.citation_completeness", "citation completeness"),
        # Kept, and no longer the abstention figure. It reads prose for
        # refusal phrasing, and the template does not phrase itself the way a
        # model does, so it reported 2 of 41 while 25 abstentions were served.
        # The structural count in the gate below is the one to cite.
        ("refusal_heuristic", "refusal (prose heuristic)"),
    ]
    print(f"{'metric':<26} {'arm':<4} {'question':>9} {'group':>8} {'n groups':>9}"
          "   (claim-making answers only)")
    for key, label in metrics:
        for arm, records in runs.items():
            scope = claiming[arm] if key != "refusal_heuristic" else records
            result = aggregate(scope, key)
            q = "     -   " if result.question_level is None else f"{result.question_level:9.3f}"
            g = "    -   " if result.group_level is None else f"{result.group_level:8.3f}"
            print(f"{label:<26} {arm:<4} {q} {g} {result.group_count:9}")
        print()

    if any(abstained.values()):
        print("  The prose heuristic above under-counts by design: it matches "
              "refusal wording,\n  and the abstention template does not phrase "
              "itself as a model would. Cite the\n  structural abstention "
              "count in the gate below instead.\n")

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
            # The hardware that produced the runs, not the hardware reading
            # them. Summarising a Pi run on the laptop previously printed
            # "this is a laptop figure" against Pi measurements.
            b_machine, b_host = hardware_of(manifests.get("B", {}))
            d_machine, d_host = hardware_of(manifests.get("D", {}))
            if (b_machine, b_host) != (d_machine, d_host):
                note = (f"NOT COMPARABLE: B ran on {b_host} ({b_machine}), "
                        f"D on {d_host} ({d_machine})")
            elif d_machine.lower().startswith("aarch"):
                note = f"H5 predicts 1.5 to 2.5; measured on {d_host}"
            else:
                note = (f"H5 is a Pi-only prediction; measured on {d_host} "
                        f"({d_machine})")
            print(f"\n  D/B ratio {d / b:.2f}   ({note})")

    # --- the stopping gate ---------------------------------------------------
    # Written and committed before the run it judges. The point is that the
    # decision is computed, not formed by looking at a table.
    if "D" in runs:
        print()
        for line in format_gate(evaluate_gate(
            runs["D"], registry, arm="D", baseline=runs.get("B"),
        )):
            print(line)
        if "B" not in runs:
            print("\n  Note: Arm B is absent, so the citation-completeness "
                  "contrast is unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
