"""Produce the reported results from the frozen test run and the manual scores.

    python scripts/analyse_results.py

Runs after unsealing. Writes item-level joined data, family tables, the
hypothesis decisions under the section 5 rule, leave-one-family-out sensitivity
and the CONF-04 two-way sensitivity, to ``results/analysis/``.

Two things this script will not do. It does not re-score anything: the judgement
logs are read-only. And it does not choose a threshold, a denominator or a
direction criterion; those come from section 5 as corrected by amendment 1.5.3,
and each decision states which rule produced it so a reader can check.

H5 needs the hardware runs and is reported as pending until those exist. The
frozen laptop quality run remains the sole evidential source for H1 to H4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.evaluation.analysis import (  # noqa: E402
    COMPATIBLE_CONTROL,
    CORRECTNESS,
    DIRECTION_REQUIRED,
    GAPS,
    LIVE_DISAGREEMENT,
    SUPERSESSION,
    AnalysisError,
    contrast,
    citation_metrics,
    common_eligibility_variant,
    decide,
    decide_within_margin,
    family_table,
    join,
    leave_one_family_out,
    primary_metrics,
    question_table,
    quality_run_directories,
    rate_by_arm,
    select,
)
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402

MANUAL = ROOT / "results" / "manual"
OUT = ROOT / "results" / "analysis"


def test_runs() -> list[Path]:
    """Quality runs only. Performance runs are tagged and excluded by name."""
    return quality_run_directories(ROOT / "results" / "runs")


def both_levels(rows, *, field_name="score") -> dict:
    return {
        "by_question": question_table(rows, field_name=field_name),
        "by_family": family_table(rows, field_name=field_name),
        "families": len({r.group_id for r in rows}),
        "questions": len(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", default=str(MANUAL / "review_sheet.jsonl"))
    parser.add_argument("--key", default=str(MANUAL / "review_sheet_key.json"))
    parser.add_argument("--judgements", default=str(MANUAL / "judgements.jsonl"))
    parser.add_argument("--abstention", default=str(MANUAL / "abstention_pass.jsonl"))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    question_set = load_question_set(load_evaluation_config().path("question_set"))

    rows = join(
        sheet=args.sheet,
        key=args.key,
        judgements=args.judgements,
        abstention=args.abstention if Path(args.abstention).exists() else None,
        question_set=question_set,
        runs=test_runs(),
    )
    # newline="\n" rather than the platform default. .gitattributes already
    # pins these to LF in the repository, but a file *generated* on Windows gets
    # CRLF, so a provenance hash taken on the generated file would not match one
    # taken on a checkout. Since results here are read back and hashed, the two
    # should be the same bytes on every platform.
    (out / "joined.jsonl").write_text(
        "\n".join(json.dumps(r.to_dict()) for r in rows) + "\n",
        encoding="utf-8", newline="\n",
    )

    report: dict = {"schema_version": "1.0", "items": len(rows), "hypotheses": {}}

    # --- H1 -----------------------------------------------------------------
    sup = select(rows, SUPERSESSION)
    table = family_table(sup)
    need, _ = DIRECTION_REQUIRED["H1"]
    # H1 makes two different kinds of claim and they need two different rules.
    # "A < B" is superiority. "B ~ C ~ D" is a claim that the arms sit within
    # the pre-specified margin, and running that through the superiority rule
    # reports a large difference against the contribution as "direction 0/4,
    # not supported", which reads as no difference found. The margin comparison
    # is operational: no equivalence test is performed anywhere in this study.
    h1 = {
        "statement": "A < B ~ C ~ D on the supersession families",
        "levels": both_levels(sup),
        "superiority_leg": {"claim": "A scores below B, C and D",
                            "contrasts": {}, "decisions": {}},
        "within_margin_leg": {"claim": "B, C and D differ by no more than the pre-specified 0.25 margin", 
                            "contrasts": {}, "decisions": {}},
    }
    for treatment, baseline in (("B", "A"), ("C", "A"), ("D", "A")):
        c = contrast(table, treatment, baseline)
        h1["superiority_leg"]["contrasts"][f"{treatment}_vs_{baseline}"] = c.to_dict()
        h1["superiority_leg"]["decisions"][f"{treatment}_vs_{baseline}"] = decide(
            c, direction_required=need
        )
    for treatment, baseline in (("C", "B"), ("D", "B"), ("D", "C")):
        c = contrast(table, treatment, baseline)
        h1["within_margin_leg"]["contrasts"][f"{treatment}_vs_{baseline}"] = c.to_dict()
        h1["within_margin_leg"]["decisions"][f"{treatment}_vs_{baseline}"] = (
            decide_within_margin(c)
        )

    superiority_ok = all(
        d["verdict"] == "supported"
        for d in h1["superiority_leg"]["decisions"].values()
    )
    outside_margin = [
        name for name, d in h1["within_margin_leg"]["decisions"].items()
        if d["verdict"] != "within margin"
    ]
    h1["verdict"] = (
        "supported" if superiority_ok and not outside_margin else "not supported"
    )
    h1["reading"] = (
        "Both legs fail. A is not below B by the declared margin: the difference "
        "is exactly 0.250, which does not exceed 0.25, and the direction holds "
        "in 2 of 4 families against the 3 of 4 required. And B, C and D are not "
        "all within the 0.25 margin: "
        + "; ".join(
            f"{name} differs by "
            f"{h1['within_margin_leg']['decisions'][name]['magnitude']:.4f} in "
            f"{h1['within_margin_leg']['decisions'][name]['higher_arm']}'s favour"
            for name in outside_margin
        )
        + ". B, C and D are therefore not described as equivalent, and no "
        "equivalence test was performed."
    ) if not superiority_ok and outside_margin else "See per-leg decisions."
    h1["leave_one_family_out"] = {
        "B_vs_A": leave_one_family_out(table, "B", "A"),
        "D_vs_B": leave_one_family_out(table, "D", "B"),
        "D_vs_C": leave_one_family_out(table, "D", "C"),
    }
    report["hypotheses"]["H1"] = h1

    # --- H2 -----------------------------------------------------------------
    live = select(rows, LIVE_DISAGREEMENT)
    table2 = family_table(live)
    need2, _ = DIRECTION_REQUIRED["H2"]
    h2 = {
        "statement": "A ~ B ~ C < D on the eight pooled live-disagreement families",
        "confirmatory_contrast": "D_vs_B",
        "levels": both_levels(live),
        "contrasts": {},
        "decisions": {},
        "by_subtype": {
            name: both_levels(select(rows, (behaviour,)))
            for name, behaviour in (
                ("mutually_exclusive", "surface_both_and_escalate"),
                ("stricter_looser", "prefer_stricter_and_escalate"),
            )
        },
    }
    for treatment, baseline in (("D", "B"), ("D", "A"), ("D", "C"), ("B", "A")):
        c = contrast(table2, treatment, baseline)
        h2["contrasts"][f"{treatment}_vs_{baseline}"] = c.to_dict()
        h2["decisions"][f"{treatment}_vs_{baseline}"] = decide(
            c, direction_required=need2
        )
    # The confirmatory contrast decides H2. Storing only the per-contrast
    # decisions left the hypothesis itself without a verdict, so a reader had to
    # know which contrast was confirmatory to work out the answer.
    h2["verdict"] = h2["decisions"][h2["confirmatory_contrast"]]["verdict"]
    h2["verdict_basis"] = (
        f"Inherited from {h2['confirmatory_contrast']}, the pre-registered "
        "confirmatory contrast for H2 (section 2). The other contrasts are "
        "descriptive."
    )
    h2["leave_one_family_out"] = {"D_vs_B": leave_one_family_out(table2, "D", "B")}
    h2["sensitivity_reading"] = (
        "Removing any single family leaves the paired difference negative, with "
        "a spread of "
        f"{h2['leave_one_family_out']['D_vs_B']['range']:.4f} across the eight "
        "folds. This shows the point estimate is stable and that no one family "
        "is driving it. It is not evidence for the null: with eight families the "
        "study could not have detected a small true effect, and a stable "
        "estimate of a small negative number is not a demonstration that the "
        "true difference is zero."
    )
    report["hypotheses"]["H2"] = h2

    # --- H2c ----------------------------------------------------------------
    controls = select(rows, COMPATIBLE_CONTROL)
    false_conflict = rate_by_arm(controls, "asserts_conflict")
    report["hypotheses"]["H2c"] = {
        "statement": "false-conflict rate(D) > false-conflict rate(B) on the controls",
        "control_families": sorted({r.group_id for r in controls}),
        "false_conflict_rate": false_conflict,
        "families_with_any_false_conflict": {
            arm: f"{v['groups_any_hit']}/{v['groups']}"
            for arm, v in false_conflict.items()
        },
        "questions_with_a_false_conflict": {
            arm: f"{v['hits']}/{v['questions']}" for arm, v in false_conflict.items()
        },
        # Section 5 makes the family the decision unit. Comparing question
        # counts would let an arm with more questions in a family carry the
        # verdict.
        "verdict": (
            "not supported"
            if false_conflict["D"]["groups_any_hit"]
            <= false_conflict["B"]["groups_any_hit"]
            else "supported"
        ),
        "verdict_unit": "control families with any false conflict",
        "reading": (
            "The directional prediction is falsified: D did not over-detect "
            "relative to B. Both are at zero, on families and on questions."
        ),
        "power_limitation": (
            f"Zero events observed in every arm across "
            f"{len({r.group_id for r in controls})} control families and "
            f"{len(controls) // 4} questions per arm. This is a floor, not a "
            "measurement: the design could not have detected moderate "
            "over-detection. The pre-registration states that a null H2c is a "
            "genuinely strong result and should be reported as such, and that "
            "holds, but it is reported with this limitation attached rather "
            "than as a demonstration that over-detection does not occur."
        ),
    }

    # --- H3 -----------------------------------------------------------------
    # The convention from summarise_arms.py, which predates unsealing: claim-
    # making answers only, both levels, group level for inference. An earlier
    # version of this script used a different denominator chosen after the key
    # was opened. It did not change the verdict, which is not a defence.
    established = citation_metrics(test_runs())
    report["hypotheses"]["H3"] = {
        "statement": "citation support < citation validity, in every arm",
        "primary": {
            "basis": (
                "scripts/summarise_arms.py convention, established before "
                "unsealing: restricted to claim-making answers, because an "
                "abstention cites nothing by design. Group level is the unit "
                "for inference."
            ),
            "by_arm": established,
        },
        "sensitivity_common_eligibility": {
            "basis": (
                "Answers where citation support is defined, so validity and "
                "support share a denominator. A narrower question, reported so "
                "the choice of denominator is visible. Not the headline figure."
            ),
            "by_arm": common_eligibility_variant(rows),
        },
        "verdict": (
            "supported"
            if all(v["support_below_validity_group_level"] for v in established.values())
            else "not supported"
        ),
        "verdict_basis": "group level, primary convention, all four arms",
        "note": (
            "Citation support is a lower bound by construction, per section 4. "
            "The verdict is the same under both calculations."
        ),
    }

    # --- H4 -----------------------------------------------------------------
    gaps = select(rows, GAPS)
    abstention = rate_by_arm(gaps, "abstained")
    # Gap topics, not questions. Section 5 again.
    beats = [
        a for a in ("A", "B", "C")
        if abstention["D"]["groups_all_hit"] > abstention[a]["groups_all_hit"]
    ]
    report["hypotheses"]["H4"] = {
        "statement": "D abstains appropriately more often than A, B and C",
        "gap_topics": len({r.group_id for r in gaps}),
        "appropriate_abstention": abstention,
        "abstained_field_source": "second pass, per amendment 1.14.4 rule 1",
        "verdict": "supported" if len(beats) == 3 else "not supported",
        "verdict_unit": "gap topics abstained on throughout",
        "reading": (
            f"D exceeds {beats or 'no arm'} and does not exceed "
            f"{[a for a in ('A','B','C') if a not in beats]}. D ties B at "
            f"{abstention['D']['groups_all_hit']}/{abstention['D']['groups']} gap "
            f"topics, both at ceiling. A and C are descriptively lower and that "
            "difference is reported, but H4 as stated requires D to exceed all "
            "three and it does not."
        ),
    }

    # --- H5 -----------------------------------------------------------------
    report["hypotheses"]["H5"] = {
        "statement": "latency(D) is between 1.5x and 2.5x latency(B) on the Pi 5",
        "verdict": "pending",
        "note": (
            "Requires the performance-only hardware executions, which are not "
            "part of the frozen laptop quality run and have not been run."
        ),
    }

    # --- CONF-04 sensitivity -------------------------------------------------
    # The one unreconciled divergence: items 27 and 221 are the same answer to
    # CONF-04-Q2 and differ only on asserts_conflict. It cannot be settled by
    # judgement, the reviewer being unblinded, so any figure that depends on it
    # is reported both ways.
    conf04 = [r for r in rows if r.group_id == "CONF-04"]
    sensitivity = {}
    for assumption, value in (("as_recorded", None), ("both_true", True),
                              ("both_false", False)):
        flags = [
            value if (value is not None and r.item in (27, 221)) else r.asserts_conflict
            for r in conf04
        ]
        sensitivity[assumption] = {
            "asserts_conflict_in_CONF_04": sum(1 for f in flags if f),
            "of_questions": len(conf04),
        }
    report["sensitivity_conf04"] = {
        "divergent_items": [27, 221],
        "field": "asserts_conflict",
        "in_H2c_denominator": "CONF-04" in {r.group_id for r in controls},
        "variants": sensitivity,
        "effect_on_hypotheses": (
            "None. CONF-04 is a version_supersession family, so it is in H1's "
            "denominator and not in H2c's, and H1 is scored on the three-point "
            "rubric rather than on asserts_conflict. The divergence moves only "
            "the descriptive false-conflict count outside the controls."
        ),
    }

    # --- blinding ------------------------------------------------------------
    asked = [r for r in rows if r.arm_identified is not None]
    report["blinding"] = {
        "self_reported_unblinding": sum(1 for r in asked if r.arm_identified),
        "items_asked": len(asked),
        "note": (
            "Not evidence that the blinding held; see amendments 1.13.4 and "
            "1.14.5. Thirteen items were structurally identifiable regardless."
        ),
    }

    # --- primary metrics carrying no hypothesis ------------------------------
    # Amendment 1.25. Section 4 names five primary metrics. Three reach the
    # hypotheses above; these two were scored, frozen and then never counted.
    # The rule lives in analysis.primary_metrics, where it is tested, and
    # neither metric takes a verdict.
    report["primary_metrics"] = primary_metrics(rows)

    (out / "hypotheses.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8", newline="\n"
    )

    # --- terminal summary ----------------------------------------------------
    print(f"  joined {len(rows)} items from {len(test_runs())} test runs\n")
    def show_superiority(label, decision):
        flag = "  [confounded]" if decision["confounded"] else ""
        print(f"    {label:<10} d={decision['paired_mean_difference']:+.4f}"
              f"  direction {decision['direction']}"
              f"  -> {decision['verdict']}{flag}")

    def show_equivalence(label, decision):
        # Report the direction the data actually runs in. Printing "direction
        # 0/4" for a difference that favours the baseline reads as no
        # difference found, which is the opposite of what was observed.
        higher = decision["higher_arm"]
        where = (
            f"{higher} higher in {decision['higher_in_families']}"
            f", {decision['tied_families']} tied"
            if higher else "no difference"
        )
        flag = "  [practical, confounded]" if decision["confounded"] else ""
        print(f"    {label:<10} d={decision['paired_mean_difference']:+.4f}"
              f"; {where}")
        print(f"    {'':<10} -> {decision['verdict']}{flag}")

    for name in ("H1", "H2", "H2c", "H3", "H4", "H5"):
        block = report["hypotheses"][name]
        print(f"  {name}: {block['statement']}")
        if name == "H1":
            print("    superiority leg, A below the rest:")
            for label, decision in block["superiority_leg"]["decisions"].items():
                show_superiority(label, decision)
            print("    within-margin leg, B ~ C ~ D (operational, not a statistical test):")
            for label, decision in block["within_margin_leg"]["decisions"].items():
                show_equivalence(label, decision)
            print(f"    OVERALL -> predicted A < B ~ C ~ D pattern "
                  f"{block['verdict']}")
        elif "decisions" in block:
            for label, decision in block["decisions"].items():
                show_superiority(label, decision)
        else:
            print(f"    -> {block['verdict']}")
        print()

    primary = report["primary_metrics"]
    print("  Primary metrics carrying no hypothesis (descriptive, amendment 1.25):")
    accuracy = primary["answer_correctness"]
    print(f"    answer correctness, {accuracy['questions_per_arm']} questions in "
          f"{accuracy['groups']} groups per arm, three-point scale:")
    for arm, value in sorted(accuracy["by_question"].items()):
        group_value = mean(
            [t[arm] for t in accuracy["by_group"].values() if arm in t]
        )
        print(f"      {arm}  question {value:.4f}   group {group_value:.4f}")
    superseded = primary["superseded_citation_rate"]
    print(f"    superseded citation rate, {superseded['families']} "
          "supersession families per arm:")
    for arm, value in sorted(superseded["by_arm"].items()):
        print(f"      {arm}  {value['hits']}/{value['questions']} answers "
              f"({value['rate']:.4f})   families with any "
              f"{value['groups_any_hit']}/{value['groups']}")
    print()

    print(f"  written to {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)
