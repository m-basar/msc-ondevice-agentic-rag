"""Tests for question grouping and split integrity.

The first two tests are the reason this module exists. A question set whose
paraphrases straddle the dev/test boundary looks perfectly ordinary: the
question strings differ, no identifier repeats, and every automated check on
content passes. What has actually happened is that the system was tuned against
the same document pair, the same conflicting figures and the same retrieval
behaviour it is later scored on. The contamination is invisible in the results.
"""

from __future__ import annotations

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.evaluation.config import load_evaluation_config
from sme_assistant.evaluation.conflicts import load_conflicts
from sme_assistant.evaluation.question_set import (
    Question,
    QuestionSet,
    QuestionSetError,
    assign_splits,
    validate_question_set,
)
from sme_assistant.kb.loader import load_knowledge_base


@pytest.fixture(scope="module")
def registry():
    return load_conflicts(load_evaluation_config().path("conflicts"))


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base(load_config().path("paths.kb_docs"))


def conflict_question(qid: str, family: str, split: str, **overrides) -> Question:
    defaults = dict(
        question_id=qid,
        text=f"question {qid}",
        category="conflict",
        group_id=family,
        split=split,
        answerability="answerable",
        expected_behaviour="surface_both_and_qualify",
        risk_level="medium",
        family_id=family,
        gold_answer="Both documents are live and they disagree.",
    )
    defaults.update(overrides)
    return Question(**defaults)


# --- the leak this module prevents ------------------------------------------


def test_paraphrases_split_across_dev_and_test_are_rejected():
    """The contamination that no content check would catch."""
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev"),
        conflict_question("Q2", "CONF-05", "test", paraphrase_of="Q1"),
    ))
    with pytest.raises(QuestionSetError, match="more than one split"):
        validate_question_set(question_set)


def test_the_error_names_the_offending_group():
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev"),
        conflict_question("Q2", "CONF-05", "test"),
        conflict_question("Q3", "CONF-06", "dev"),
    ))
    with pytest.raises(QuestionSetError) as exc:
        validate_question_set(question_set)
    assert "CONF-05" in str(exc.value)
    assert "CONF-06" not in str(exc.value), "a sound group must not be blamed"


def test_a_family_kept_whole_is_accepted():
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "test"),
        conflict_question("Q2", "CONF-05", "test", paraphrase_of="Q1"),
        conflict_question("Q3", "CONF-05", "test", paraphrase_of="Q1"),
        conflict_question("Q4", "CONF-06", "dev"),
    ))
    validate_question_set(question_set)
    assert question_set.group_split_map() == {"CONF-05": {"test"}, "CONF-06": {"dev"}}


def test_a_paraphrase_cannot_belong_to_a_different_group():
    """Regrouping a paraphrase would evade the split check entirely."""
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev"),
        conflict_question("Q2", "CONF-06", "test", paraphrase_of="Q1"),
    ))
    with pytest.raises(QuestionSetError, match="same observation"):
        validate_question_set(question_set)


def test_splits_can_only_be_assigned_by_group():
    """The API makes per-question assignment unavailable, not merely discouraged."""
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev"),
        conflict_question("Q2", "CONF-05", "dev"),
        conflict_question("Q3", "CONF-06", "dev"),
    ))
    reassigned = assign_splits(question_set, test_groups=["CONF-05"])
    assert [q.split for q in reassigned] == ["test", "test", "dev"]
    validate_question_set(reassigned)


def test_unknown_group_cannot_be_assigned():
    question_set = QuestionSet((conflict_question("Q1", "CONF-05", "dev"),))
    with pytest.raises(QuestionSetError, match="Unknown groups"):
        assign_splits(question_set, test_groups=["CONF-99"])


# --- grouping ---------------------------------------------------------------


def test_every_question_declares_a_group():
    question_set = QuestionSet((conflict_question("Q1", "CONF-05", "dev", group_id=""),))
    with pytest.raises(QuestionSetError, match="unit of independence"):
        validate_question_set(question_set)


def test_a_conflict_question_groups_by_its_family():
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev", group_id="something-else"),
    ))
    with pytest.raises(QuestionSetError, match="group by family"):
        validate_question_set(question_set)


def test_effective_sample_size_is_reported_alongside_question_count():
    """A reader must not be able to mistake 48 questions for 48 observations."""
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev"),
        conflict_question("Q2", "CONF-05", "dev"),
        conflict_question("Q3", "CONF-06", "dev"),
    ))
    summary = question_set.summary()
    assert summary["question_count"] == 3
    assert summary["group_count"] == 2
    assert "independent groups" in summary["effective_sample_size_note"]


# --- cross-validation folds -------------------------------------------------


def test_folds_are_held_out_by_family_not_by_question():
    """A fold that held out one paraphrase would train on its siblings."""
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev"),
        conflict_question("Q2", "CONF-05", "dev"),
        conflict_question("Q3", "CONF-06", "dev"),
    ))
    folds = list(question_set.leave_one_family_out())
    assert len(folds) == 2, "one fold per family, not per question"

    for group, held, rest in folds:
        held_groups = {q.group_id for q in held}
        rest_groups = {q.group_id for q in rest}
        assert held_groups == {group}
        assert not (held_groups & rest_groups), (
            f"{group} appears in both the held-out fold and the remainder"
        )


# --- agreement with the registry and the corpus ------------------------------


def test_expected_behaviour_must_match_the_conflict_type(registry):
    """A supersession family is resolvable; a current_current family is not."""
    question_set = QuestionSet((
        conflict_question(
            "Q1", "CONF-01", "dev", expected_behaviour="surface_both_and_qualify"
        ),
    ))
    with pytest.raises(QuestionSetError, match="cite_current_only"):
        validate_question_set(question_set, registry=registry)


def test_an_unregistered_family_is_rejected(registry):
    question_set = QuestionSet((conflict_question("Q1", "CONF-99", "dev"),))
    with pytest.raises(QuestionSetError, match="not in the conflict registry"):
        validate_question_set(question_set, registry=registry)


def test_every_registered_family_must_be_exercised(registry):
    """A family with no question inflates the apparent scope of the evaluation."""
    question_set = QuestionSet((conflict_question("Q1", "CONF-05", "dev"),))
    with pytest.raises(QuestionSetError, match="No question exercises"):
        validate_question_set(question_set, registry=registry)


def test_an_undeclared_gap_topic_is_rejected(registry):
    question_set = QuestionSet((
        Question(
            question_id="Q1",
            text="What is the cycle to work scheme?",
            category="unanswerable",
            group_id="cycle to work",
            split="dev",
            answerability="unanswerable",
            expected_behaviour="abstain",
            gap_topic="cycle to work",
        ),
    ))
    with pytest.raises(QuestionSetError, match="not a declared gap"):
        validate_question_set(question_set, registry=registry)


def test_an_unanswerable_question_cannot_name_expected_chunks():
    question_set = QuestionSet((
        Question(
            question_id="Q1",
            text="Does the company offer a pension?",
            category="unanswerable",
            group_id="pensions and auto-enrolment",
            split="dev",
            answerability="unanswerable",
            expected_behaviour="abstain",
            gap_topic="pensions and auto-enrolment",
            expected_chunks=("HR-01#001",),
        ),
    ))
    with pytest.raises(QuestionSetError, match="unanswerable but names expected chunks"):
        validate_question_set(question_set)


def test_a_reference_to_an_unknown_document_is_rejected(kb):
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev", expected_documents=("ZZ-99",)),
    ))
    with pytest.raises(QuestionSetError, match="unknown document"):
        validate_question_set(question_set, kb=kb)


def test_an_answerable_question_needs_a_gold_answer():
    question_set = QuestionSet((conflict_question("Q1", "CONF-05", "dev", gold_answer=" "),))
    with pytest.raises(QuestionSetError, match="no gold answer"):
        validate_question_set(question_set)


def test_duplicate_question_ids_are_rejected():
    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev"),
        conflict_question("Q1", "CONF-06", "dev"),
    ))
    with pytest.raises(QuestionSetError, match="Duplicate question id"):
        validate_question_set(question_set)


# --- round trip -------------------------------------------------------------


def test_a_set_with_straddling_groups_cannot_even_be_written(tmp_path):
    """The guard sits on the write path too, not only on load."""
    from sme_assistant.evaluation.question_set import write_question_set

    question_set = QuestionSet((
        conflict_question("Q1", "CONF-05", "dev"),
        conflict_question("Q2", "CONF-05", "test"),
    ))
    with pytest.raises(QuestionSetError, match="more than one split"):
        write_question_set(question_set, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_round_trip_preserves_grouping(tmp_path):
    from sme_assistant.evaluation.question_set import (
        load_question_set,
        write_question_set,
    )

    original = QuestionSet((
        conflict_question("Q1", "CONF-05", "test"),
        conflict_question("Q2", "CONF-05", "test", paraphrase_of="Q1"),
        conflict_question("Q3", "CONF-06", "dev"),
    ))
    path = write_question_set(original, tmp_path / "set.json", corpus_sha256="abc")
    reloaded = load_question_set(path)

    assert len(reloaded) == 3
    assert reloaded.groups == ("CONF-05", "CONF-06")
    assert reloaded.by_id("Q2").paraphrase_of == "Q1"
    assert reloaded.metadata["corpus_sha256"] == "abc"
    assert reloaded.split("test").groups == ("CONF-05",)
