"""Emit the amendment appendix table for the dissertation.

    python scripts/make_amendment_table.py > docs/dissertation/appendix_amendments.md

Amendment numbers, dates and sub-entry counts are read from
``docs/PREREGISTRATION.md`` so they cannot drift from it. The one-line summary
for each amendment is authored here, keyed by number: prose cannot be derived
from a heading, and pretending otherwise would be the sort of provenance claim
amendment 1.26.6 was written about.

The script fails if the document contains an amendment this table has no
summary for, so adding an amendment without describing it is an error rather
than a silently shorter table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Phase boundaries, taken from this repository's commit history. The frozen
#: four-arm test run is 4ba79da, 14 August 07:13; the key was unsealed at
#: ed65f22, 15 August 01:42; the quality analysis was signed off at be55077.
PHASES = {
    "1.1": ("A", "Development, before the frozen confirmatory runs"),
    "1.13": ("B", "Manual scoring, after the runs and before unsealing"),
    "1.15": ("C", "Hardware boundary and execution, after the quality analysis"),
    "1.25": ("D", "Write-up corrections"),
}

SUMMARY = {
    "1.1": "Pilot-contaminated families excluded; per-question rubrics; the arms "
           "restated as a tree rather than a ladder; a correctness metric added; "
           "blinding and provenance defects recorded.",
    "1.2": "The conflict taxonomy replaced one type with four; three compatible "
           "families added as negative controls; eight further families planted; "
           "H2 split into H2a, H2b and H2c.",
    "1.3": "Retrieval calibrated on the development split alone: top_k 4 to 6, "
           "min_similarity 0.32 to 0.30. No threshold separates answerable from "
           "unanswerable questions, and that is reported as a finding.",
    "1.4": "Four families had been typed by intuition and typed wrongly. "
           "Reclassified, the enabling condition named, conflict-pair recall "
           "corrected, and the overclaims it had supported withdrawn.",
    "1.5": "CONF-12 reclassified; H2a and H2b merged into a single pooled H2 "
           "because the subtype judgement was not reliable enough to condition a "
           "threshold on; rubric drift repaired; gold data frozen.",
    "1.6": "Arm D was rewriting answers it had no complaint about. A serving rule "
           "was implemented so that a verifier finding nothing is a no-op, every "
           "served revision names its warrant, and the stopping rule became code.",
    "1.7": "A reproducibility claim made and withdrawn; the corrected instrument "
           "and its decision rule fixed before measurement; the abstention text "
           "moved into the source so no claim could be smuggled into free prose.",
    "1.8": "The pre-declared development gate restored and made to fire; "
           "decorative safeguards made real; the abstention template applied "
           "wherever the verifier abstains.",
    "1.9": "Pilot 04 passed containment and exposed a measurement defect that "
           "amendment 1.8 had introduced. Coverage is now reported beside every "
           "conditional figure and false refusals are attributed.",
    "1.10": "The verifier diagnostic protocol selected Qwen2.5 3B over Llama 3.2 "
            "3B on conflict detection. What the result does not establish is "
            "recorded alongside it.",
    "1.11": "Every Qwen response had omitted the claim audit because the prompt "
            "taught the omission and the validated helper was never wired in. "
            "Both fixed; the system now fails closed on a missing audit.",
    "1.12": "Pilot 06 against the declared gate. The gate fails, the budget is "
            "spent, and the defensible statement about the verifier is written at "
            "the width the evidence supports.",
    "1.13": "The blinding was defeated on thirteen items carrying the abstention "
            "template verbatim, exposing all 68 items of one arm. Recorded rather "
            "than repaired, with the unblinding rate measured rather than assumed.",
    "1.14": "Manual scoring completed. The rubric agreed on 58 of 58 duplicate "
            "groups; the abstention flag drifted with position, so a re-pass was "
            "run under a rule fixed in advance and became the reported value.",
    "1.15": "The hardware runs declared performance-only, before any were "
            "executed, so that a timing run could not produce a quality figure.",
    "1.16": "The enforcement 1.15 claimed did not exist; the four frozen quality "
            "runs are now a closed list. Four analysis corrections, including a "
            "post-unsealing denominator reverted and 'equivalent' withdrawn.",
    "1.17": "The runner gate, placement enforcement and observation, Arm D's "
            "stage reporting, and H5 restricted to the Raspberry Pi 5, all "
            "corrected and tested before any hardware execution.",
    "1.18": "Validation hardened before any latency value was read, with the "
            "reason for hardening after the runs rather than before stated "
            "explicitly.",
    "1.19": "Seven fail-open paths closed before reading latency, including a "
            "rejected run whose latency was computed anyway and a rejection "
            "record that invited destroying the evidence.",
    "1.20": "The preflight was testing the wrong model, warming the run it "
            "protected, and could not be exercised without spending a run. Made "
            "standalone and per timed stage.",
    "1.21": "Model names canonicalised against what Ollama reports, placement "
            "actually applied rather than only observed, cleanup run on the path "
            "where it mattered, and magnitudes stripped from rejection records.",
    "1.22": "The first live preflight passed all three placement checks and its "
            "cleanup check failed. The failure is retained, and the reading that "
            "would explain it away is declined for want of evidence.",
    "1.23": "The unload budget had assumed a retention it never stated, and one "
            "field was doing three jobs. Both corrected before the instrument was "
            "rerun.",
    "1.24": "The unload question closed by measurement. The analyser's success "
            "path had transposed two arguments and no test had ever reached it; "
            "fixed, made keyword-only, and covered end to end.",
    "1.25": "Two pre-registered primary metrics, answer correctness and "
            "superseded citation rate, had been scored and frozen but never "
            "aggregated. The rule was fixed before either figure was computed.",
    "1.26": "Cohen's kappa was quoted in working notes but existed in no file; "
            "implemented and tested. Figure-generation rules declared, then "
            "seven reporting defects corrected following independent review.",
}


def main() -> int:
    text = (ROOT / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    lines = text.splitlines()

    found: list[dict] = []
    current: dict | None = None
    for line in lines:
        head = re.match(r"^# Amendment (1\.\d+) - (.+)$", line)
        if head:
            current = {"id": head.group(1), "date": head.group(2), "subs": 0}
            found.append(current)
        elif current is not None and re.match(r"^## 1\.\d+\.\d+ ", line):
            current["subs"] += 1

    missing = [a["id"] for a in found if a["id"] not in SUMMARY]
    if missing:
        print(f"no summary written for amendment(s): {', '.join(missing)}",
              file=sys.stderr)
        return 1

    print("# Appendix B: Pre-registration amendment record")
    print()
    print(f"All {len(found)} amendments to `docs/PREREGISTRATION.md`, in order. "
          "Numbers, dates and\nsub-entry counts are read from that document by "
          "`scripts/make_amendment_table.py`;\nthe summaries are written for this "
          "appendix. Each amendment in the source carries\nits own reason, its "
          "evidence, and a statement of what it did **not** change.")
    print()
    print("Phases are taken from the commit history. The frozen four-arm test run "
          "is\n`4ba79da`, 14 August 07:13; the blinding key was opened at "
          "`ed65f22`, 15 August\n01:42; the quality analysis was signed off at "
          "`be55077`.")
    print()
    print("| # | Date | Entries | What it changed |")
    print("|---|---|---:|---|")
    for amendment in found:
        if amendment["id"] in PHASES:
            letter, label = PHASES[amendment["id"]]
            print(f"| | | | **Phase {letter}: {label}** |")
        date = amendment["date"].replace(" 2026", "")
        print(f"| {amendment['id']} | {date} | {amendment['subs']} | "
              f"{SUMMARY[amendment['id']]} |")
    print()
    total = sum(a["subs"] for a in found)
    print(f"**{len(found)} amendments, {total} numbered sub-entries.** Phase A "
          f"amendments precede the\nfrozen confirmatory runs and could and did "
          "change the design. Phase B onwards\ncould not: the runs were complete, "
          "and every later amendment either governs how\nthe existing data are "
          "scored and analysed, or concerns the separate hardware\nexperiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
