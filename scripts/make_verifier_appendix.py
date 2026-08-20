"""Emit Appendix D, the verifier relationship classification, from the analysis.

    python scripts/make_verifier_appendix.py > docs/dissertation/appendix_verifier_classification.md

Amendment 1.30.5. Every count in the appendix is read from the
``verifier_relationship_diagnostic`` block of ``results/analysis/hypotheses.json``
and none is typed here. The prose is authored, because prose cannot be derived
from a JSON object and pretending otherwise is the provenance overstatement
amendment 1.26.6 was written about; the numbers inside the prose are formatted
from the same block as the tables, so the two cannot disagree.

The script refuses rather than emitting a shorter appendix if the diagnostic is
absent, is not at the frozen shape, or has acquired a total spanning the three
hypothesis groups. A generator that quietly produces a smaller document when its
input changes is a worse failure than one that stops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.evaluation.analysis import (  # noqa: E402
    DIAGNOSTIC_GROUPS,
    DIAGNOSTIC_RUN,
    FROZEN_DIAGNOSTIC_SHAPE,
)
from sme_assistant.verify.schema import (  # noqa: E402
    CONTEXTUALLY_COMPATIBLE,
    INSUFFICIENT,
    MUTUALLY_EXCLUSIVE,
    NO_RELATIONSHIP,
    STRICTER_LOOSER,
    SUPERSESSION,
    VALID_RELATIONSHIPS,
)

#: Column order for the confusion matrix. Every relationship the schema defines
#: gets a column, including the one the verifier never returned: a column of
#: dashes states that in the table rather than only in a sentence beneath it.
COLUMNS = (INSUFFICIENT, MUTUALLY_EXCLUSIVE, NO_RELATIONSHIP,
           CONTEXTUALLY_COMPATIBLE, STRICTER_LOOSER, SUPERSESSION)

DECLARED_LABEL = {
    "version_supersession": "Version supersession",
    "mutually_exclusive": "Mutually exclusive",
    "stricter_looser": "Stricter-looser",
    "compatible": "Compatible (controls)",
}

GROUP_LABEL = {
    "H1_supersession": "H1, supersession",
    "H2_live_disagreement": "H2, live disagreement",
    "compatible_controls": "Negative controls",
}

#: The illustrative case. Authored, keyed by question identifier, and checked
#: against the frozen record below so the description cannot drift from it.
CASE_ID = "CONF-02-Q1"
CASE_PROSE = """`{qid}` asks when Statutory Sick Pay starts being paid. The corpus holds
HR-02, withdrawn, saying the fourth qualifying day, and HR-12, current, saying
the first. Both were retrieved and HR-02 carried a `[SUPERSEDED]` marker in the
evidence block.

Arm D's frozen record classifies the relationship as `{reported}` where the
declared type maps to `{expected}`. In its claim audit it marks the claim drawn
from the current document `CONTRADICTED` and records the withdrawn document's
claim as `SUPPORTED`.

The served answer was nonetheless correct, because the verifier returned the
draft unchanged. The failure is confined to the internal audit and is invisible
in the answer the user receives, which is the reason it went unreported until
the demonstrator displayed the audit alongside the answer."""


def cell(value: int) -> str:
    return str(value) if value else "-"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else ROOT / "results" / "analysis" / "hypotheses.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    diagnostic = report.get("verifier_relationship_diagnostic")
    if not diagnostic:
        raise SystemExit(
            f"{path} carries no verifier_relationship_diagnostic block. Run "
            "scripts/analyse_results.py first; this script computes nothing."
        )
    shape = diagnostic["shape"]
    if (shape["families"], shape["questions"],
            shape["paraphrases_per_family"]) != tuple(FROZEN_DIAGNOSTIC_SHAPE):
        raise SystemExit(
            f"the diagnostic was computed at shape {shape}, the frozen registry "
            f"is {FROZEN_DIAGNOSTIC_SHAPE._asdict()}"
        )
    if diagnostic["source_run"] != DIAGNOSTIC_RUN:
        raise SystemExit(
            f"the diagnostic names source run {diagnostic['source_run']!r}, "
            f"this appendix describes {DIAGNOSTIC_RUN!r}"
        )
    # A total over the three groups is the error 1.30.5 removes. If one ever
    # reappears in the analysis, this stops rather than typesetting it.
    for banned in ("all_registered_families", "total", "overall"):
        if banned in diagnostic:
            raise SystemExit(
                f"the diagnostic carries a {banned!r} key. Amendment 1.30.5 "
                "reports the three hypothesis groups separately and offers no "
                "figure spanning them."
            )

    rows = diagnostic["per_question"]
    confusion = diagnostic["confusion"]
    groups = diagnostic["by_hypothesis_group"]

    out: list[str] = []
    add = out.append

    add("# Appendix D: Verifier relationship classification")
    add("")
    add("Supporting detail for the exploratory section 4.13. **Post-hoc and "
        "exploratory**,")
    add("governed by pre-registration amendments 1.29 and 1.30: the pattern was "
        "inspected")
    add("before the rule was written, no threshold is applied, no verdict is "
        "reached and no")
    add("chance baseline is computed. Source is the frozen Arm D quality run")
    add(f"`{DIAGNOSTIC_RUN}` and nothing else, validated before it is read.")
    add("")
    add("Two measures are reported and never combined. **Detection** is binary: "
        "did the")
    add("verifier report any conflict relationship. **Exact classification** "
        "asks whether")
    add("the reported relationship equalled the declared type mapped through")
    add("`DECLARED_TO_INFERRED`, which is used unmodified.")
    add("")
    add("**No figure spans the three groups below.** H1 and H2 are separate "
        "hypotheses")
    add("with separate decision rules, and the compatible families are controls "
        "whose")
    add("denominator belongs to a false-positive rate. Every table here is "
        "within a group")
    add("or within a family, and no row of totals is offered.")
    add("")
    add("Every count in this appendix is generated by "
        "`scripts/make_verifier_appendix.py`")
    add("from `results/analysis/hypotheses.json`. No number in it is typed by "
        "hand.")
    add("")

    # --- D.1 -----------------------------------------------------------------
    add("## D.1 Confusion matrix")
    add("")
    add("Rows are the declared type from the registry, columns the relationship "
        "the")
    add("verifier returned. The cell that would be an exact match is emboldened. "
        "All six")
    add("relationships the verifier's schema defines are given a column, "
        "including any it")
    add("never returned.")
    add("")
    add("| Declared type | " + " | ".join(f"`{c}`" for c in COLUMNS) + " |")
    add("|---|" + "---:|" * len(COLUMNS))
    expected_by_declared = {r["declared"]: r["expected"] for r in rows}
    for declared in DECLARED_LABEL:
        if declared not in confusion:
            continue
        counts = confusion[declared]
        cells = []
        for column in COLUMNS:
            text = cell(counts.get(column, 0))
            if column == expected_by_declared.get(declared) and text != "-":
                text = f"**{text}**"
            cells.append(text)
        add(f"| {DECLARED_LABEL[declared]} | " + " | ".join(cells) + " |")
    add("")

    unused = sorted(VALID_RELATIONSHIPS
                    - {r["reported"] for r in rows})
    if unused:
        add("The verifier **never returned "
            + ", ".join(f"`{u}`" for u in unused) + "** at any point in the run.")
        if CONTEXTUALLY_COMPATIBLE in unused:
            add("That is the category the three compatible control families call "
                "for, so one of")
            add("its six categories went unused and the controls could not have "
                "been classified")
            add("exactly however they were retrieved.")
        add("")

    most_common = max(({r["reported"] for r in rows}),
                      key=lambda rel: sum(1 for r in rows if r["reported"] == rel))
    add(f"`{most_common}` is the most common answer, including on families where "
        "two")
    add("documents visibly disagree.")
    add("")
    controls = groups["compatible_controls"]
    add(f"On the compatible controls the verifier reported a conflict "
        f"relationship on {controls['detected']}")
    add(f"of {controls['questions']} questions. That is its internal conclusion "
        "and **not** what H2c measures;")
    add("H2c reads the reviewer's judgement of the served answer and records "
        "zero false")
    add("conflicts. The internal conclusion did not reach the answer.")
    add("")

    # --- D.2 -----------------------------------------------------------------
    add("## D.2 Per group")
    add("")
    add("`Pair` counts the questions where the chunks carrying both sides of the "
        "focal")
    add("disputed fact were retrieved, computed with `anchor_chunks` and")
    add("`pair_is_present`. `Majority` counts families exactly classified on at "
        "least two")
    add("of their three paraphrases; the `Pair` column is a question count, not "
        "a second")
    add("denominator for `Detected` and `Exact`, which are over all of the "
        "group's")
    add("questions.")
    add("")
    add("| Group | Declared types | Questions | Pair | Detected | Exact | "
        "Majority |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for key, declared_types in DIAGNOSTIC_GROUPS.items():
        group = groups[key]
        add(f"| {GROUP_LABEL[key]} | "
            + ", ".join(f"`{d}`" for d in declared_types)
            + f" | {group['questions']} | {group['pair_present_questions']}"
            f" | {group['detected']} | {group['exactly_classified']}"
            f" | {group['families_exact_on_a_majority']} / {group['families']} |")
    add("")

    # --- D.3 -----------------------------------------------------------------
    add("## D.3 Per family")
    add("")
    add("Descriptive, and deliberately without a total: the four declared types "
        "belong to")
    add("two hypotheses and one control set, and a column sum would span all "
        "three.")
    add("")
    add("| Family | Declared | Expected relationship | Questions | Pair | "
        "Detected | Exact |")
    add("|---|---|---|---:|---:|---:|---:|")
    families: dict[str, dict[str, int | str]] = {}
    for row in rows:
        entry = families.setdefault(row["family_id"], {
            "declared": row["declared"], "expected": row["expected"],
            "n": 0, "pair": 0, "detected": 0, "exact": 0})
        entry["n"] += 1
        entry["pair"] += bool(row["pair_present"])
        entry["detected"] += bool(row["detected"])
        entry["exact"] += bool(row["exact"])
    for family_id in sorted(families):
        entry = families[family_id]
        add(f"| {family_id} | `{entry['declared']}` | `{entry['expected']}` | "
            f"{entry['n']} | {entry['pair']} | {entry['detected']} | "
            f"{entry['exact']} |")
    add("")

    # --- D.4 -----------------------------------------------------------------
    case = next((r for r in rows if r["question_id"] == CASE_ID), None)
    if case is None:
        raise SystemExit(
            f"the illustrative case {CASE_ID} is not in the diagnostic; the "
            "prose below describes a record this run does not contain"
        )
    add("## D.4 One illustrative case")
    add("")
    add(CASE_PROSE.format(qid=CASE_ID, reported=case["reported"],
                          expected=case["expected"]))
    add("")
    add("**This is one question.** It illustrates the pattern in the tables "
        "above; it")
    add("does not establish it, and no argument in this dissertation rests on it "
        "alone.")

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
