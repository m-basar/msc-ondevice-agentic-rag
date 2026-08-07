"""Tests for two-level aggregation.

The first test is the whole argument in numbers: a metric where the
question-level mean and the family-level mean differ, because one family
happened to receive more paraphrases than another. Reporting only the first
would credit the system for being good at whichever family was asked about most
often.
"""

from __future__ import annotations

import pytest

from sme_assistant.evaluation.aggregate import (
    AggregationError,
    aggregate,
    compare_arms,
)


def record(question_id, group_id, value, **extra):
    return {"question_id": question_id, "group_id": group_id, "score": value, **extra}


# --- why both levels are reported -------------------------------------------


def test_paraphrase_heavy_families_skew_the_question_level_mean():
    """Four questions on a family the system handles, one on a family it fails.

    Question level says 0.8. Family level says 0.5. The second is the honest
    figure: the system succeeds on one family out of two.
    """
    records = [
        record("Q1", "CONF-05", 1.0),
        record("Q2", "CONF-05", 1.0),
        record("Q3", "CONF-05", 1.0),
        record("Q4", "CONF-05", 1.0),
        record("Q5", "CONF-06", 0.0),
    ]
    result = aggregate(records, "score")
    assert result.question_level == 0.8
    assert result.group_level == 0.5
    assert result.question_count == 5
    assert result.group_count == 2
    assert result.disagreement == 0.3


def test_both_levels_are_present_in_the_saved_output():
    """Neither figure may be dropped: one is readable, the other is defensible."""
    payload = aggregate([record("Q1", "CONF-05", 1.0)], "score").to_dict()
    assert "question_level" in payload
    assert "group_level" in payload
    assert payload["inference_unit"] == "group"
    assert "not independent observations" in payload["note"]


def test_the_output_states_the_sample_size_to_cite():
    payload = aggregate(
        [record("Q1", "CONF-05", 1.0), record("Q2", "CONF-05", 1.0)], "score"
    ).to_dict()
    assert payload["question_count"] == 2
    assert payload["group_count"] == 1
    assert "group_count, not question_count" in payload["note"]


# --- inference over groups ---------------------------------------------------


def test_standard_error_is_computed_over_families_not_questions():
    """Ten identical questions in one family carry no more information than one.

    Computing the error over questions would report a tiny interval from what
    is effectively a single observation repeated.
    """
    one_family = [record(f"Q{i}", "CONF-05", 1.0) for i in range(10)]
    assert aggregate(one_family, "score").group_standard_error is None, (
        "a single group cannot support an interval, however many questions it has"
    )

    two_families = one_family + [record("Q10", "CONF-06", 0.0)]
    assert aggregate(two_families, "score").group_standard_error is not None


def test_the_interval_widens_when_families_disagree():
    agreeing = aggregate(
        [record("Q1", "A", 1.0), record("Q2", "B", 1.0), record("Q3", "C", 1.0)], "score"
    )
    disagreeing = aggregate(
        [record("Q1", "A", 1.0), record("Q2", "B", 0.0), record("Q3", "C", 1.0)], "score"
    )
    assert agreeing.group_standard_error == 0.0
    assert disagreeing.group_standard_error > agreeing.group_standard_error


# --- what must not be silently coerced --------------------------------------


def test_a_none_metric_is_skipped_not_read_as_zero():
    """Citation support is undefined when nothing was cited.

    Reading that as zero would make an abstaining system look dishonest rather
    than cautious, and would drag the mean down for a behaviour that is
    correct.
    """
    records = [
        record("Q1", "CONF-05", 1.0),
        {"question_id": "Q2", "group_id": "CONF-05", "score": None},
    ]
    result = aggregate(records, "score")
    assert result.question_level == 1.0
    assert result.skipped == 1
    assert result.to_dict()["skipped_records"] == 1


def test_booleans_are_counted_as_rates():
    result = aggregate(
        [record("Q1", "A", True), record("Q2", "B", False)], "score"
    )
    assert result.group_level == 0.5


def test_a_nested_metric_can_be_addressed_by_a_dotted_key():
    records = [
        {"question_id": "Q1", "group_id": "A", "scoring": {"citation_support": 0.5}},
        {"question_id": "Q2", "group_id": "B", "scoring": {"citation_support": 1.0}},
    ]
    assert aggregate(records, "scoring.citation_support").group_level == 0.75


def test_a_record_with_no_grouping_information_is_refused():
    """Silently pooling ungrouped records would defeat the entire mechanism."""
    with pytest.raises(AggregationError, match="unit of independence"):
        aggregate([{"score": 1.0}], "score")


def test_family_id_serves_as_the_group_when_group_id_is_absent():
    result = aggregate(
        [
            {"question_id": "Q1", "family_id": "CONF-05", "score": 1.0},
            {"question_id": "Q2", "family_id": "CONF-05", "score": 0.0},
        ],
        "score",
    )
    assert result.group_count == 1


# --- comparing arms ----------------------------------------------------------


def test_arms_are_compared_paired_by_family():
    """Pairing removes between-family variance, which dominates at nine families."""
    baseline = [record("Q1", "CONF-05", 0.0), record("Q2", "CONF-06", 1.0)]
    verified = [record("Q1", "CONF-05", 1.0), record("Q2", "CONF-06", 1.0)]

    comparison = compare_arms({"A_baseline": baseline, "D_verified": verified}, "score")
    difference = comparison["paired_differences"]["D_verified_minus_A_baseline"]

    assert difference["mean_difference"] == 0.5
    assert difference["paired_groups"] == 2
    assert difference["improved"] == 1
    assert difference["unchanged"] == 1
    assert difference["worsened"] == 0


def test_a_comparison_reports_the_family_count_not_the_question_count():
    baseline = [record(f"Q{i}", "CONF-05", 0.0) for i in range(6)]
    verified = [record(f"Q{i}", "CONF-05", 1.0) for i in range(6)]
    comparison = compare_arms({"A": baseline, "D": verified}, "score")
    assert comparison["paired_differences"]["D_minus_A"]["paired_groups"] == 1, (
        "six paraphrases of one family are one paired observation"
    )
