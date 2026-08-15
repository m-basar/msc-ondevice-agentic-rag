"""Tests for the post-unsealing analysis.

This is the stage that turns judgements into claims, so the arithmetic here is
the arithmetic the dissertation reports. Three things are easy to get wrong and
are pinned down:

- pairing within family before averaging, which is not a difference of means
  once families differ in size
- the three-outcome decision rule, especially at the threshold boundary, where
  "exceeds 0.25" must not quietly become "reaches 0.25"
- the confounded contrast being labelled as confounded rather than reported as
  an ablation
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sme_assistant.evaluation.analysis import (
    AnalysisError,
    Joined,
    contrast,
    decide,
    decide_equivalence,
    family_table,
    join,
    leave_one_family_out,
    question_table,
    quality_run_directories,
    rate_by_arm,
    select,
)
from sme_assistant.evaluation.manual_scoring import Judgement, append_judgement
from sme_assistant.evaluation.question_set import Question, QuestionSet


def row(arm, group, score, *, behaviour="prefer_stricter_and_escalate",
        qid=None, abstained=False, conflict=False, item=1):
    return Joined(
        item=item, arm=arm, question_id=qid or f"{group}-Q1", group_id=group,
        category="conflict", expected_behaviour=behaviour, score=score,
        asserts_conflict=conflict, abstained=abstained, uncertain=False,
        arm_identified=False,
    )


# --- pairing -----------------------------------------------------------------


def test_the_contrast_pairs_within_family_rather_than_differencing_means():
    """With unequal family sizes these are different numbers.

    Only the paired one answers "did D beat B on the same material", which is
    what section 5 asks for.
    """
    rows = [
        # FAM-1 has three questions for each arm, FAM-2 has one.
        *[row("D", "FAM-1", 0, item=i) for i in (1, 2, 3)],
        *[row("B", "FAM-1", 0, item=i) for i in (4, 5, 6)],
        row("D", "FAM-2", 2, item=7),
        row("B", "FAM-2", 0, item=8),
    ]
    table = family_table(rows)
    result = contrast(table, "D", "B")

    # Paired: (0 - 0) and (2 - 0), averaged, is 1.0.
    assert result.paired_mean_difference == pytest.approx(1.0)
    # Pooling every question instead would give 2/4 - 0/4 = 0.5.
    flat = question_table(rows)
    assert flat["D"] - flat["B"] == pytest.approx(0.5)
    assert result.better == 1 and result.tied == 1 and result.worse == 0


def test_family_means_average_paraphrases_before_the_family_enters():
    rows = [
        row("D", "FAM-1", 2, item=1), row("D", "FAM-1", 0, item=2),
        row("B", "FAM-1", 0, item=3), row("B", "FAM-1", 0, item=4),
    ]
    assert family_table(rows)["FAM-1"]["D"] == pytest.approx(1.0)


# --- the decision rule -------------------------------------------------------


def make_contrast(diffs: dict[str, float]):
    table = {f: {"D": d, "B": 0.0} for f, d in diffs.items()}
    return contrast(table, "D", "B")


def test_exactly_at_the_threshold_does_not_count_as_exceeding_it():
    """Section 5 says the difference must *exceed* 0.25.

    H1's B versus A lands on 0.250 exactly, so this boundary decides a reported
    verdict rather than being hypothetical.
    """
    result = make_contrast({f"F{i}": 0.25 for i in range(4)})
    decision = decide(result, direction_required=3)
    assert decision["effect_criterion_met"] is False
    assert decision["direction_criterion_met"] is True
    assert decision["verdict"] == "suggestive"


def test_both_criteria_gives_supported():
    result = make_contrast({f"F{i}": 0.5 for i in range(4)})
    assert decide(result, direction_required=3)["verdict"] == "supported"


def test_a_large_effect_carried_by_one_family_is_only_suggestive():
    """A point estimate favouring the contribution is not enough on its own.

    This is the case the three-outcome rule exists for: one family moves the
    mean past the threshold while the direction fails.
    """
    result = make_contrast({"F0": 2.0, "F1": -0.1, "F2": -0.1, "F3": -0.1})
    decision = decide(result, direction_required=3)
    assert decision["effect_criterion_met"] is True
    assert decision["direction"] == "1/4"
    assert decision["verdict"] == "suggestive"


def test_neither_criterion_gives_not_supported():
    result = make_contrast({f"F{i}": -0.3 for i in range(8)})
    assert decide(result, direction_required=6)["verdict"] == "not supported"


def test_the_c_to_d_contrast_is_labelled_confounded():
    """Section 2: C to D changes retrieval mode and verification together."""
    table = {"F0": {"D": 1.0, "C": 0.0, "B": 0.0}}
    assert contrast(table, "D", "C").confounded is True
    assert "not as an ablation" in contrast(table, "D", "C").note
    assert contrast(table, "D", "B").confounded is False


# --- sensitivity -------------------------------------------------------------


def test_leave_one_family_out_names_the_family_carrying_the_result():
    table = {"F0": {"D": 2.0, "B": 0.0}, "F1": {"D": 0.0, "B": 0.0},
             "F2": {"D": 0.0, "B": 0.0}, "F3": {"D": 0.0, "B": 0.0}}
    report = leave_one_family_out(table, "D", "B")
    assert report["full"] == pytest.approx(0.5)
    assert report["most_influential_family"] == "F0"
    assert report["folds"]["F0"] == pytest.approx(0.0)
    assert report["range"] > 0


# --- rates -------------------------------------------------------------------


def test_rate_reports_the_group_count_alongside_the_question_count():
    """A reader shown 10/10 must also be told it rests on five gap topics."""
    rows = [
        row("D", "GAP-1", 2, behaviour="abstain", abstained=True, item=1),
        row("D", "GAP-1", 2, behaviour="abstain", abstained=True, item=2),
        row("D", "GAP-2", 2, behaviour="abstain", abstained=False, item=3),
    ]
    result = rate_by_arm(rows, "abstained")["D"]
    assert result["hits"] == 2
    assert result["questions"] == 3
    assert result["groups"] == 2
    assert result["groups_all_hit"] == 1


def test_select_filters_to_the_behaviours_a_hypothesis_is_stated_over():
    rows = [row("D", "F0", 2), row("D", "F1", 2, behaviour="abstain")]
    assert len(select(rows, ("abstain",))) == 1


# --- the join ----------------------------------------------------------------


def question(qid, group, behaviour="abstain"):
    return Question(question_id=qid, text="q?", category="unanswerable",
                    group_id=group, split="test", answerability="unanswerable",
                    expected_behaviour=behaviour)


def build_inputs(tmp_path, *, abstained_first, abstained_second):
    sheet = tmp_path / "sheet.jsonl"
    sheet.write_text(json.dumps({
        "item": 1, "question_id": "GAP-1-Q1", "group_id": "GAP-1",
        "question": "q?", "system": "system_A", "answer": "a",
        "scoring_criteria": {"0": "", "1": "", "2": ""},
    }) + "\n", encoding="utf-8")

    key = tmp_path / "key.json"
    key.write_text(json.dumps({"mapping": {"D": "system_A"}}), encoding="utf-8")

    judgements = tmp_path / "j.jsonl"
    append_judgement(judgements, Judgement(item=1, question_id="GAP-1-Q1", score=2,
                                           abstained=abstained_first))

    abstention = tmp_path / "a.jsonl"
    abstention.write_text(json.dumps({
        "item": 1, "question_id": "GAP-1-Q1", "abstained": abstained_second,
    }) + "\n", encoding="utf-8")

    qs = QuestionSet((question("GAP-1-Q1", "GAP-1"),))
    return sheet, key, judgements, abstention, qs


def test_the_join_takes_abstained_from_the_second_pass(tmp_path):
    """Amendment 1.14.4 rule 1 makes the re-pass the reported value."""
    sheet, key, judgements, abstention, qs = build_inputs(
        tmp_path, abstained_first=False, abstained_second=True
    )
    rows = join(sheet=sheet, key=key, judgements=judgements,
                abstention=abstention, question_set=qs)
    assert rows[0].abstained is True
    assert rows[0].arm == "D"


def test_the_join_falls_back_to_the_first_pass_only_when_asked(tmp_path):
    sheet, key, judgements, abstention, qs = build_inputs(
        tmp_path, abstained_first=False, abstained_second=True
    )
    rows = join(sheet=sheet, key=key, judgements=judgements,
                abstention=None, question_set=qs)
    assert rows[0].abstained is False


def test_the_join_refuses_an_incomplete_scoring_pass(tmp_path):
    """Analysis on a partial pass would report a subset as if it were the whole."""
    sheet, key, judgements, abstention, qs = build_inputs(
        tmp_path, abstained_first=True, abstained_second=True
    )
    Path(judgements).write_text("", encoding="utf-8")
    with pytest.raises(AnalysisError, match="no judgement"):
        join(sheet=sheet, key=key, judgements=judgements,
             abstention=abstention, question_set=qs)


def test_the_join_refuses_an_opaque_code_the_key_does_not_cover(tmp_path):
    sheet, key, judgements, abstention, qs = build_inputs(
        tmp_path, abstained_first=True, abstained_second=True
    )
    Path(key).write_text(json.dumps({"mapping": {"D": "system_Z"}}), encoding="utf-8")
    with pytest.raises(AnalysisError, match="No arm for opaque code"):
        join(sheet=sheet, key=key, judgements=judgements,
             abstention=abstention, question_set=qs)


# --- the quality / performance boundary --------------------------------------


def make_run_dir(root: Path, name: str, *, split: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"split": split, "arm": {"arm": "B"}}), encoding="utf-8"
    )
    return directory


def test_performance_runs_cannot_enter_the_quality_analysis(tmp_path):
    """Amendment 1.15 declares the boundary; this is what enforces it.

    A tagged run directory does not end in ``_test``, so pointing the analysis
    at the whole results tree still cannot pull a performance execution into a
    hypothesis about answer quality.
    """
    make_run_dir(tmp_path, "20260814_054606_B_test", split="test")
    make_run_dir(tmp_path, "20260816_101010_B_test_pi5perf", split="test")
    make_run_dir(tmp_path, "20260813_102840_B_dev", split="dev")

    found = [p.name for p in quality_run_directories(tmp_path)]
    assert found == ["20260814_054606_B_test"]


def test_a_renamed_directory_is_still_checked_against_its_manifest(tmp_path):
    """A directory can be renamed by hand; a manifest cannot be renamed by
    accident."""
    make_run_dir(tmp_path, "20260813_102840_B_test", split="dev")
    assert quality_run_directories(tmp_path) == []


# --- equivalence is not superiority ------------------------------------------


def test_a_difference_against_the_treatment_is_not_reported_as_no_difference():
    """The regression this exists to prevent.

    H1's D versus C is a difference of 0.333 running consistently in C's
    favour. Passing it through the superiority rule returns "direction 0/4, not
    supported", which a reader takes as no difference found. The equivalence
    rule has to say the opposite: equivalence refuted, C higher.
    """
    table = {
        "CONF-02": {"C": 1.333, "D": 0.667},
        "CONF-03": {"C": 2.0, "D": 1.667},
        "CONF-04": {"C": 1.0, "D": 1.0},
        "CONF-10": {"C": 2.0, "D": 1.667},
    }
    result = contrast(table, "D", "C")
    assert result.paired_mean_difference == pytest.approx(-0.3333, abs=1e-3)

    # What the superiority rule says, and why it is the wrong instrument here.
    superiority = decide(result, direction_required=3)
    assert superiority["direction"] == "0/4"
    assert superiority["verdict"] == "not supported"

    # What the equivalence rule says.
    equivalence = decide_equivalence(result)
    assert equivalence["verdict"] == "not equivalent"
    assert equivalence["higher_arm"] == "C"
    assert equivalence["higher_in_families"] == "3/4"
    assert equivalence["tied_families"] == 1
    assert equivalence["magnitude"] > 0.25


def test_confounding_limits_attribution_without_erasing_the_difference():
    """A limit on causal attribution must not be used to make an inconvenient
    observation disappear."""
    table = {f"F{i}": {"C": 1.0, "D": 0.0} for i in range(4)}
    decision = decide_equivalence(contrast(table, "D", "C"))
    assert decision["confounded"] is True
    assert decision["verdict"] == "not equivalent"
    assert "cannot be attributed to either alone" in decision["reading"]
    assert "rather than erasing it" in decision["reading"]


def test_a_small_difference_is_equivalent():
    table = {f"F{i}": {"B": 1.0, "D": 1.1} for i in range(4)}
    decision = decide_equivalence(contrast(table, "D", "B"))
    assert decision["verdict"] == "equivalent"
    assert decision["magnitude"] == pytest.approx(0.1)


def test_a_difference_exactly_at_the_threshold_still_counts_as_equivalent():
    """Equivalence is refuted only by *exceeding* the margin, matching the
    superiority rule's use of the same 0.25."""
    table = {f"F{i}": {"B": 0.0, "D": 0.25} for i in range(4)}
    assert decide_equivalence(contrast(table, "D", "B"))["verdict"] == "equivalent"


def test_family_level_hit_counts_answer_different_questions():
    """all_hit is the family-level success for abstention; any_hit is the
    family-level failure for false conflicts. Reporting only all_hit would show
    zero for an arm that flagged a false conflict on a third of its questions.
    """
    rows = [
        row("D", "CONF-07", 2, behaviour="answer_without_flagging_conflict",
            conflict=True, item=1),
        row("D", "CONF-07", 2, behaviour="answer_without_flagging_conflict",
            conflict=False, item=2),
        row("D", "CONF-09", 2, behaviour="answer_without_flagging_conflict",
            conflict=False, item=3),
    ]
    result = rate_by_arm(rows, "asserts_conflict")["D"]
    assert result["groups_all_hit"] == 0
    assert result["groups_any_hit"] == 1
    assert result["hits"] == 1 and result["questions"] == 3
