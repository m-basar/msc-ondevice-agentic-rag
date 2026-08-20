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
    FROZEN_QUALITY_RUNS,
    cohens_kappa,
    contrast,
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
from sme_assistant.evaluation.manual_scoring import Judgement, append_judgement
from sme_assistant.evaluation.question_set import Question, QuestionSet


def row(arm, group, score, *, behaviour="prefer_stricter_and_escalate",
        qid=None, abstained=False, conflict=False, item=1, superseded=0):
    return Joined(
        item=item, arm=arm, question_id=qid or f"{group}-Q1", group_id=group,
        category="conflict", expected_behaviour=behaviour, score=score,
        asserts_conflict=conflict, abstained=abstained, uncertain=False,
        arm_identified=False, cited_superseded=superseded,
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
    """Amendment 1.15 declares the boundary; amendment 1.16 enforces it.

    A tagged performance run and a development run both sit in the same tree
    and neither is admitted. The stronger case, an *untagged* performance run,
    is covered separately below.
    """
    for name in FROZEN_QUALITY_RUNS:
        make_run_dir(tmp_path, name, split="test")
    make_run_dir(tmp_path, "20260816_101010_B_test_pi5perf", split="test")
    make_run_dir(tmp_path, "20260813_102840_B_dev", split="dev")

    found = [p.name for p in quality_run_directories(tmp_path)]
    assert found == list(FROZEN_QUALITY_RUNS)


# --- the margin comparison is not superiority ------------------------------------------


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

    # What the margin rule says.
    equivalence = decide_within_margin(result)
    assert equivalence["verdict"] == "outside margin"
    assert equivalence["higher_arm"] == "C"
    assert equivalence["higher_in_families"] == "3/4"
    assert equivalence["tied_families"] == 1
    assert equivalence["magnitude"] > 0.25


def test_confounding_limits_attribution_without_erasing_the_difference():
    """A limit on causal attribution must not be used to make an inconvenient
    observation disappear."""
    table = {f"F{i}": {"C": 1.0, "D": 0.0} for i in range(4)}
    decision = decide_within_margin(contrast(table, "D", "C"))
    assert decision["confounded"] is True
    assert decision["verdict"] == "outside margin"
    assert "cannot be attributed to either alone" in decision["reading"]
    assert "rather than erasing it" in decision["reading"]


def test_a_small_difference_is_equivalent():
    table = {f"F{i}": {"B": 1.0, "D": 1.1} for i in range(4)}
    decision = decide_within_margin(contrast(table, "D", "B"))
    assert decision["verdict"] == "within margin"
    assert decision["magnitude"] == pytest.approx(0.1)


def test_a_difference_exactly_at_the_threshold_still_counts_as_equivalent():
    """Equivalence is refuted only by *exceeding* the margin, matching the
    superiority rule's use of the same 0.25."""
    table = {f"F{i}": {"B": 0.0, "D": 0.25} for i in range(4)}
    assert decide_within_margin(contrast(table, "D", "B"))["verdict"] == "within margin"


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


# --- the hardware boundary, amendment 1.16 -----------------------------------


def test_an_untagged_extra_test_run_cannot_enter_the_quality_analysis(tmp_path):
    """The hole amendment 1.15 left open.

    The first implementation accepted any untagged ``*_test`` directory whose
    manifest said ``split == "test"``. A performance run satisfies both, so the
    enforcement 1.15.3 claimed did not exist. A closed list cannot be satisfied
    by a run created afterwards, tagged or not.
    """
    for name in FROZEN_QUALITY_RUNS:
        make_run_dir(tmp_path, name, split="test")
    make_run_dir(tmp_path, "20260816_101010_B_test", split="test")   # untagged
    make_run_dir(tmp_path, "20260816_101011_D_test_pi5", split="test")  # tagged

    found = [p.name for p in quality_run_directories(tmp_path)]
    assert found == list(FROZEN_QUALITY_RUNS)


def test_a_frozen_run_marked_performance_is_refused(tmp_path):
    for name in FROZEN_QUALITY_RUNS:
        make_run_dir(tmp_path, name, split="test")
    manifest = tmp_path / FROZEN_QUALITY_RUNS[0] / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["purpose"] = "performance"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnalysisError, match="purpose=performance"):
        quality_run_directories(tmp_path)


def test_a_missing_frozen_run_is_an_error_not_a_smaller_analysis(tmp_path):
    """Three arms would still produce numbers, and they would not be the
    pre-registered ones."""
    for name in FROZEN_QUALITY_RUNS[:3]:
        make_run_dir(tmp_path, name, split="test")
    with pytest.raises(AnalysisError, match="missing"):
        quality_run_directories(tmp_path)


def test_a_renamed_directory_cannot_impersonate_a_frozen_run(tmp_path):
    for name in FROZEN_QUALITY_RUNS:
        make_run_dir(tmp_path, name, split="dev")
    with pytest.raises(AnalysisError, match="split="):
        quality_run_directories(tmp_path)


def test_two_runs_supplying_the_same_arm_and_question_are_refused(tmp_path):
    """A duplicate run would overwrite the automatic metrics silently while the
    manual scores stayed with the frozen run."""
    sheet, key, judgements, abstention, qs = build_inputs(
        tmp_path, abstained_first=True, abstained_second=True
    )
    runs = []
    for name in ("run_one", "run_two"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"split": "test", "arm": {"arm": "D"}}), encoding="utf-8"
        )
        (directory / "answers.jsonl").write_text(
            json.dumps({"question_id": "GAP-1-Q1", "wall_seconds": 1.0}) + "\n",
            encoding="utf-8",
        )
        runs.append(directory)

    with pytest.raises(AnalysisError, match="Two runs supply"):
        join(sheet=sheet, key=key, judgements=judgements, abstention=abstention,
             question_set=qs, runs=runs)


# --- margin language is operational, not statistical -------------------------


def test_the_margin_verdict_does_not_claim_equivalence():
    """No interval is computed anywhere in this study, so nothing here may say
    the arms were shown to be the same."""
    table = {f"F{i}": {"B": 1.0, "D": 1.1} for i in range(4)}
    decision = decide_within_margin(contrast(table, "D", "B"))
    assert decision["verdict"] == "within margin"
    assert "equivalen" not in decision["reading"].lower()
    assert "not a statistical equivalence test" in decision["basis"].lower()
    assert "not the same as having been shown to be equal" in decision["reading"]


# --- the two primary metrics that carry no hypothesis, amendment 1.25 --------


def correctness_and_supersession_rows():
    """A minimal set spanning both denominators plus material in neither."""
    rows = []
    for arm in ("A", "D"):
        # Correctness: two behaviours, one of them a two-question group.
        rows.append(row(arm, "FACT-1", 2, behaviour="answer_directly"))
        rows.append(row(arm, "FACT-2", 0, behaviour="answer_directly"))
        rows.append(row(arm, "PART-1", 2, behaviour="answer_and_flag_gap",
                        qid="PART-1-Q1"))
        rows.append(row(arm, "PART-1", 0, behaviour="answer_and_flag_gap",
                        qid="PART-1-Q2"))
        # Supersession, scored on the rubric and carrying withdrawn citations.
        rows.append(row(arm, "CONF-02", 1, behaviour="cite_current_only",
                        superseded=3 if arm == "A" else 0))
        rows.append(row(arm, "CONF-03", 1, behaviour="cite_current_only",
                        superseded=1 if arm == "A" else 0))
        # In neither denominator, and must not leak into either.
        rows.append(row(arm, "GAP-1", 2, behaviour="abstain"))
        rows.append(row(arm, "CONF-11", 1))
    return rows


def test_neither_primary_metric_carries_a_verdict():
    """Section 3 states no prediction over either, so the section 5 rule does
    not apply. A threshold bolted on later would be a test invented after the
    data, which is what the pre-registration exists to prevent.

    The check is over keys rather than over the serialised text, because the
    prose in this block has to be free to say the word "verdict" in order to
    state that there is not one.
    """
    report = primary_metrics(correctness_and_supersession_rows())

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)

    found = {k for k in keys(report)}
    for banned in ("verdict", "threshold", "direction", "direction_required",
                   "effect_criterion_met", "decision", "decisions"):
        assert banned not in found, banned
    assert "no verdict, no threshold, no direction criterion" in report["basis"]


def test_answer_correctness_uses_only_its_two_behaviours():
    """Abstention and conflict questions are scored on the same three-point
    scale, so a denominator taken from the score field rather than the
    behaviour would silently swallow them."""
    report = primary_metrics(correctness_and_supersession_rows())
    accuracy = report["answer_correctness"]
    assert accuracy["questions_per_arm"] == 4
    assert accuracy["groups"] == 3
    assert set(accuracy["by_group"]) == {"FACT-1", "FACT-2", "PART-1"}
    # A: 2, 0, 2, 0 over questions; the two-question group averages to 1.
    assert accuracy["by_question"]["A"] == pytest.approx(1.0)
    assert accuracy["by_group"]["PART-1"]["A"] == pytest.approx(1.0)


def test_correctness_is_reported_at_both_levels_because_they_differ():
    """Eleven of the twelve real groups are single questions, so the two levels
    nearly coincide. Reporting both is what makes that visible rather than
    assumed."""
    rows = [
        row("A", "FACT-1", 2, behaviour="answer_directly"),
        row("A", "PART-1", 0, behaviour="answer_and_flag_gap", qid="PART-1-Q1"),
        row("A", "PART-1", 0, behaviour="answer_and_flag_gap", qid="PART-1-Q2"),
    ]
    accuracy = primary_metrics(rows + [
        row("A", "CONF-02", 1, behaviour="cite_current_only"),
    ])["answer_correctness"]
    # Question level 2/3 = 0.667; group level (2 + 0) / 2 = 1.0.
    assert accuracy["by_question"]["A"] == pytest.approx(2 / 3)
    group_mean = sum(t["A"] for t in accuracy["by_group"].values()) / 2
    assert group_mean == pytest.approx(1.0)


def test_a_superseded_citation_counts_the_answer_once_not_the_citations():
    """Section 4 states the metric over answers. Counting citations would let
    one badly cited answer outweigh three clean ones."""
    report = primary_metrics(correctness_and_supersession_rows())
    superseded = report["superseded_citation_rate"]["by_arm"]
    # Arm A cited withdrawn documents in both answers, three times in one.
    assert superseded["A"]["hits"] == 2
    assert superseded["A"]["questions"] == 2
    assert superseded["A"]["rate"] == pytest.approx(1.0)
    assert superseded["D"]["hits"] == 0


def test_the_family_figure_is_families_with_any_false_citation():
    """A family reported as clean because only two of its three questions cited
    a withdrawn document would be misleading. This is the correction amendment
    1.16.2 made to H2c, applied here for the same reason."""
    rows = [
        row("A", "CONF-02", 1, behaviour="cite_current_only", qid="CONF-02-Q1",
            superseded=1),
        row("A", "CONF-02", 1, behaviour="cite_current_only", qid="CONF-02-Q2",
            superseded=0),
        row("A", "FACT-1", 2, behaviour="answer_directly"),
    ]
    superseded = primary_metrics(rows)["superseded_citation_rate"]["by_arm"]["A"]
    assert superseded["groups_any_hit"] == 1
    assert superseded["groups_all_hit"] == 0


def test_a_missing_denominator_is_an_error_not_an_empty_metric():
    """An empty table printed as a result reads as a measurement of zero."""
    with pytest.raises(AnalysisError, match="answer-correctness"):
        primary_metrics([row("A", "CONF-02", 1, behaviour="cite_current_only")])
    with pytest.raises(AnalysisError, match="supersession"):
        primary_metrics([row("A", "FACT-1", 2, behaviour="answer_directly")])


# --- chance-corrected agreement, amendment 1.26 ------------------------------


def test_kappa_matches_a_hand_worked_table():
    """The 2x2 is reported with the coefficient so it can be checked by hand.
    This is that check, written against the arithmetic rather than the code."""
    pairs = ([(True, True)] * 62 + [(True, False)] * 9
             + [(False, True)] * 10 + [(False, False)] * 191)
    result = cohens_kappa(pairs)
    assert result["n"] == 272
    assert result["table"] == {
        "both_marked": 62, "first_pass_only": 9,
        "second_pass_only": 10, "neither_marked": 191,
    }
    # po = 253/272; pe = (71*72 + 201*200) / 272**2
    assert result["observed_agreement"] == pytest.approx(253 / 272)
    expected = (71 * 72 + 201 * 200) / (272 ** 2)
    assert result["expected_agreement"] == pytest.approx(expected)
    assert result["cohens_kappa"] == pytest.approx(
        ((253 / 272) - expected) / (1 - expected)
    )


def test_kappa_is_below_raw_agreement_on_an_imbalanced_field():
    """The reason for reporting it at all. A field where one outcome dominates
    scores high on raw agreement for doing very little."""
    pairs = [(False, False)] * 95 + [(True, True)] * 2 + [(True, False)] * 3
    result = cohens_kappa(pairs)
    assert result["observed_agreement"] == pytest.approx(0.97)
    assert result["cohens_kappa"] < result["observed_agreement"]


def test_kappa_is_zero_when_agreement_is_only_what_chance_predicts():
    pairs = ([(True, True)] * 25 + [(True, False)] * 25
             + [(False, True)] * 25 + [(False, False)] * 25)
    assert cohens_kappa(pairs)["cohens_kappa"] == pytest.approx(0.0)


def test_kappa_is_undefined_rather_than_one_when_nothing_varies():
    """Two passes that both marked every item agree perfectly and establish
    nothing. Returning 1.0 there would report a reliability that was never
    tested."""
    result = cohens_kappa([(True, True)] * 40)
    assert result["observed_agreement"] == 1.0
    assert result["cohens_kappa"] is None


def test_kappa_refuses_an_empty_comparison():
    with pytest.raises(AnalysisError, match="no paired judgements"):
        cohens_kappa([])


# --- group-level means live in the analysis, not the plotting layer ----------


def test_answer_correctness_carries_its_group_level_mean():
    """The figure reads this rather than averaging at plot time. A number
    derived in the plotting layer has no test behind it."""
    report = primary_metrics(correctness_and_supersession_rows())
    accuracy = report["answer_correctness"]
    for arm in ("A", "D"):
        expected = sum(
            row[arm] for row in accuracy["by_group"].values() if arm in row
        ) / len(accuracy["by_group"])
        assert accuracy["group_level"][arm] == pytest.approx(expected)


def test_the_group_level_mean_differs_from_the_question_level_one():
    """If they always coincided there would be no reason to report both. They
    diverge as soon as one group carries more questions than another."""
    rows = [
        row("A", "FACT-1", 2, behaviour="answer_directly"),
        row("A", "PART-1", 0, behaviour="answer_and_flag_gap", qid="PART-1-Q1"),
        row("A", "PART-1", 0, behaviour="answer_and_flag_gap", qid="PART-1-Q2"),
        row("A", "CONF-02", 1, behaviour="cite_current_only"),
    ]
    accuracy = primary_metrics(rows)["answer_correctness"]
    assert accuracy["by_question"]["A"] == pytest.approx(2 / 3)
    assert accuracy["group_level"]["A"] == pytest.approx(1.0)
