"""Emit Appendix D, the verifier relationship classification, from the analysis.

    python scripts/make_verifier_appendix.py > docs/dissertation/appendix_verifier_classification.md

Amendment 1.30.5. Every count in the appendix is read from the
``verifier_relationship_diagnostic`` block of ``results/analysis/hypotheses.json``
and none is typed here. The prose is authored, because prose cannot be derived
from a JSON object and pretending otherwise is the provenance overstatement
amendment 1.26.6 was written about; the numbers inside the prose are formatted
from the same block as the tables, so the two cannot disagree.

Section D.4 is the exception and is derived in full. Amendment 1.33 records why:
it was a prose template describing a frozen record - which document was
withdrawn, what the claim audit returned, whether the draft was served unchanged
- and every one of those facts is in the record. It now reads them from the
authenticated Arm D run rather than restating them.

The script refuses rather than emitting a shorter appendix if the diagnostic is
absent, is not at the frozen shape, or has acquired a total spanning the three
hypothesis groups. A generator that quietly produces a smaller document when its
input changes is a worse failure than one that stops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.evaluation.analysis import (  # noqa: E402
    DIAGNOSTIC_GROUPS,
    DIAGNOSTIC_RUN,
    FROZEN_DIAGNOSTIC_SHAPE,
    load_diagnostic_source,
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

#: The illustrative case, named here and described nowhere. Amendment 1.33.
#:
#: The previous version held a prose template with two substituted fields and
#: the rest typed: which document was withdrawn, which figure each carried, that
#: both were retrieved, that one bore a superseded marker, what the claim audit
#: returned and that the draft came back unchanged. Every one of those is in the
#: frozen record, in a document whose header says every count in it is generated
#: and no number typed. The description was accurate when written and nothing
#: checked it, which is the defect this project has now corrected nine times.
#:
#: D.4 is now derived from the authenticated Arm D record. Only the question
#: identifier is chosen here.
CASE_ID = "CONF-02-Q1"


def describe_case(record: Mapping[str, Any], reported: str,
                  expected: str) -> list[str]:
    """Narrate one frozen record from the record itself.

    Introduces no metric, denominator, threshold or verdict: it reports what a
    single stored record contains, which is what an illustration is for. The
    counts that carry weight are in D.1 to D.3 and are unaffected by anything
    here.
    """
    lines: list[str] = []
    add = lines.append

    retrieval = (record.get("retrieval") or {}).get("results") or []
    by_status: dict[str, list[dict]] = {}
    for result in retrieval:
        by_status.setdefault(result.get("status", "unknown"), []).append(result)
    superseded = by_status.get("superseded", [])
    current = by_status.get("current", [])
    verification = record.get("verification") or {}
    verdicts = verification.get("verdicts") or []

    add(f"`{record['question_id']}` asks: *{record['question'].strip()}*")
    add("")
    add(f"The retrieval returned {len(retrieval)} chunks, of which "
        f"{len(superseded)} carried a `[SUPERSEDED]` marker in the evidence "
        f"block and {len(current)} did not. The two highest ranked are the two "
        "sides of the disputed fact:")
    add("")
    add("| Rank | Chunk | Status | Source |")
    add("|---:|---|---|---|")
    for result in retrieval[:2]:
        marker = ("**superseded**" if result.get("status") == "superseded"
                  else result.get("status", "unknown"))
        add(f"| {result.get('rank')} | `{result['chunk_id']}` | {marker} | "
            f"{result.get('citation', '')} |")
    add("")
    add(f"The verifier classified the relationship as `{reported}`, where the "
        f"registry's declared type maps to `{expected}`.")
    add("")
    if verdicts:
        add("Its claim audit, reproduced exactly as the record stores it:")
        add("")
        add("| Claim | Verdict | Supporting | Contradicting |")
        add("|---|---|---|---|")
        for verdict in verdicts:
            supporting = ", ".join(f"`{c}`" for c in verdict.get("supporting") or [])
            contradicting = ", ".join(f"`{c}`" for c in
                                      verdict.get("contradicting") or [])
            add(f"| {verdict.get('claim', '').strip()} | "
                f"`{verdict.get('verdict')}` | {supporting or '-'} | "
                f"{contradicting or '-'} |")
        add("")
    revised = bool(verification.get("revised"))
    served = (record.get("answer") or "").strip()
    serving = "replaced the draft" if revised else "returned the draft unchanged"
    add(f"The verifier **{serving}**, so the answer served was:")
    add("")
    add(f"> {served}")
    add("")
    add("Whatever the audit above records, that is the sentence the user saw. "
        "The audit is the verifier's own working, and it is not what any "
        "reported metric reads: H2c is scored on the reviewer's judgement of "
        "the served answer, which is why this record contributes no false "
        "conflict anywhere in Chapter 4.")
    return lines


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
            "section below describes a record this run does not contain"
        )
    source = load_diagnostic_source(ROOT / "results" / "runs")
    record = next((r for r in source.records if r["question_id"] == CASE_ID), None)
    if record is None:
        raise SystemExit(
            f"{CASE_ID} is not in the authenticated source run; D.4 is derived "
            "from that record and will not be written without it"
        )
    add("## D.4 One illustrative case")
    add("")
    add("Derived from the authenticated Arm D record, not written about it.")
    add("Amendment 1.33 records why: the previous version of this section was a")
    add("prose template whose description of the retrieval, the claim audit and")
    add("the serving decision was typed rather than read, in an appendix whose")
    add("header says every count in it is generated.")
    add("")
    add("**Nothing here is a live demonstration.** Appendix C shows the same")
    add("question asked live on the Raspberry Pi 5; that is a separate unscored")
    add("execution whose recorded output differs, and none of it is read here.")
    add("")
    out.extend(describe_case(record, case["reported"], case["expected"]))
    add("")
    add("**This is one question.** It illustrates the pattern in the tables "
        "above; it")
    add("does not establish it, and no argument in this dissertation rests on it "
        "alone.")

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
