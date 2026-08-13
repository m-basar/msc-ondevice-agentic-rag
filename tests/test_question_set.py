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
        expected_behaviour="surface_both_and_escalate",
        risk_level="medium",
        family_id=family,
        gold_answer="Both documents are live and they disagree.",
        required_claims=("the two documents disagree",),
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
    supersession = next(f for f in registry.families if f.is_filter_resolvable)
    question_set = QuestionSet((
        conflict_question(
            "Q1", supersession.family_id, "test",
            expected_behaviour="surface_both_and_escalate",
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
    one = registry.families[0]
    question_set = QuestionSet((
        conflict_question(
            "Q1", one.family_id, "test",
            expected_behaviour=(
                "cite_current_only" if one.is_filter_resolvable
                else "surface_both_and_qualify"
            ),
        ),
    ))
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


# --- the tuning boundary ----------------------------------------------------


def test_a_reported_family_cannot_appear_in_the_development_split(registry):
    """Tuning against a family that is later scored contaminates the result."""
    reported = next(f for f in registry.families if not f.is_filter_resolvable)
    question_set = QuestionSet((conflict_question("Q1", reported.family_id, "dev"),))
    with pytest.raises(QuestionSetError, match="reported family"):
        validate_question_set(question_set, registry=registry)


def test_a_tuning_family_cannot_appear_in_the_test_split(registry):
    """The boundary runs both ways.

    A tuning family was inspected during development, so reporting it would be
    reporting a result the design was fitted to.
    """
    question_set = QuestionSet((
        conflict_question("Q1", "TUNE-01", "test"),
    ))
    with pytest.raises(QuestionSetError, match="tuning family"):
        validate_question_set(question_set, registry=registry)


def test_a_tuning_family_in_the_development_split_is_accepted(registry):
    def behaviour(family):
        return (
            "cite_current_only"
            if family.is_filter_resolvable
            else "surface_both_and_qualify"
        )

    question_set = QuestionSet(
        tuple(
            conflict_question(f"D{i}", f.family_id, "dev", expected_behaviour=f.expected_behaviour)
            for i, f in enumerate(registry.tuning_families)
        )
        + tuple(
            conflict_question(f"T{i}", f.family_id, "test", expected_behaviour=f.expected_behaviour)
            for i, f in enumerate(registry.families)
        )
    )
    validate_question_set(question_set, registry=registry)


def test_tuning_families_are_not_counted_as_reported(registry):
    """A reader must not be able to mistake eleven families for the sample size."""
    assert len(registry.families) == 15
    assert len(registry.tuning_families) == 8
    assert len(registry.all_families) == 23
    assert {f.family_id for f in registry.families}.isdisjoint(
        {f.family_id for f in registry.tuning_families}
    )


# --- the real question set ---------------------------------------------------


@pytest.fixture(scope="module")
def real_question_set():
    from sme_assistant.evaluation.question_set import load_question_set

    return load_question_set(load_evaluation_config().path("question_set"))


def test_the_written_question_set_is_valid(real_question_set, registry, kb):
    """Guards the frozen artefact against corpus drift.

    If a document is edited such that an expected chunk no longer exists, or a
    family stops being current_current, this fails here rather than silently
    producing a run scored against stale gold data.
    """
    validate_question_set(real_question_set, registry=registry, kb=kb)


def test_every_expected_chunk_still_exists(real_question_set, kb):
    from sme_assistant.common.config import load_config
    from sme_assistant.ingest.chunker import chunk_corpus

    config = load_config()
    known = {
        c.chunk_id
        for c in chunk_corpus(
            kb,
            config.require("chunking.max_words"),
            config.require("chunking.overlap_sentences"),
            config.require("chunking.min_words"),
        )
    }
    missing = [
        (q.question_id, c)
        for q in real_question_set
        for c in q.expected_chunks
        if c not in known
    ]
    assert not missing, f"expected chunks no longer in the chunk set: {missing}"


def test_no_reported_family_appears_in_the_development_split(real_question_set, registry):
    reported = {f.family_id for f in registry.families}
    leaked = [
        q.question_id
        for q in real_question_set.split("dev")
        if q.family_id in reported
    ]
    assert not leaked, f"reported families in the development split: {leaked}"


def test_the_conflict_families_are_the_largest_category(real_question_set):
    """The set is conflict-weighted by design; a drift away from that is a
    change to the study, not a tidy-up."""
    summary = real_question_set.summary()
    assert summary["by_category"]["conflict"] >= summary["by_category"]["factual"]
    assert summary["family_group_count"] == 23


def test_the_test_split_covers_every_reported_family(real_question_set, registry):
    covered = {q.family_id for q in real_question_set.split("test") if q.family_id}
    assert covered == {f.family_id for f in registry.families}


def test_paraphrases_number_three_per_conflict_family(real_question_set):
    for group in real_question_set.family_groups:
        assert len(real_question_set.of_group(group)) == 3, (
            f"{group} does not have three paraphrases, so families contribute "
            "unequal weight to the question-level figure"
        )


# --- provenance is checked, not merely recorded ------------------------------


def test_a_question_set_from_a_different_chunk_set_is_refused(real_question_set):
    """Recording a hash and checking it are different things.

    The question set names specific chunk identifiers as evidence. If the
    chunker changes, those identifiers keep resolving but point at different
    text, and every scored result is quietly wrong. The corpus hash cannot see
    that, which is why this one matters most.
    """
    from sme_assistant.evaluation.question_set import check_provenance

    with pytest.raises(QuestionSetError, match="chunk_set_sha256 recorded"):
        check_provenance(real_question_set, chunk_set_sha256="0" * 64)


def test_a_question_set_from_a_different_corpus_is_refused(real_question_set):
    from sme_assistant.evaluation.question_set import check_provenance

    with pytest.raises(QuestionSetError, match="corpus_sha256 recorded"):
        check_provenance(real_question_set, corpus_sha256="0" * 64)


def test_the_current_state_passes_provenance(real_question_set, kb, registry):
    from sme_assistant.common.config import load_config
    from sme_assistant.evaluation.question_set import check_provenance
    from sme_assistant.ingest.chunker import chunk_corpus
    from sme_assistant.ingest.index import chunk_set_fingerprint

    config = load_config()
    chunks = chunk_corpus(
        kb,
        config.require("chunking.max_words"),
        config.require("chunking.overlap_sentences"),
        config.require("chunking.min_words"),
    )
    check_provenance(
        real_question_set,
        corpus_sha256=kb.fingerprint(),
        chunk_set_sha256=chunk_set_fingerprint(chunks),
        registry_sha256=registry.fingerprint(),
    )


def test_a_set_with_no_recorded_hash_is_refused_not_skipped():
    """A set written before this check existed cannot be trusted to be current."""
    from sme_assistant.evaluation.question_set import QuestionSet, check_provenance

    with pytest.raises(QuestionSetError, match="not recorded"):
        check_provenance(QuestionSet((), None, {}), chunk_set_sha256="a" * 64)


# --- per-question rubrics ----------------------------------------------------


def test_every_answerable_question_carries_its_own_required_claims(real_question_set):
    """A family-level fact list marks a concise correct answer wrong and passes
    an incomplete one."""
    for question in real_question_set:
        if question.answerability == "unanswerable":
            assert not question.required_claims
        else:
            assert question.required_claims, f"{question.question_id} has no rubric"


def test_the_rubric_that_was_in_force_is_written_into_the_artefact(real_question_set):
    rubrics = real_question_set.metadata["scoring_rubrics"]
    for behaviour in {q.expected_behaviour for q in real_question_set}:
        assert behaviour in rubrics
        assert set(rubrics[behaviour]) == {"0", "1", "2"}


def test_paraphrases_of_a_family_share_one_focal_claim(real_question_set):
    """Three questions asking different things are not paraphrases, however
    much they share a document pair."""
    for group in real_question_set.family_groups:
        questions = real_question_set.of_group(group)
        claims = {q.required_claims for q in questions}
        assert len(claims) == 1, (
            f"{group} has questions with different required claims, so they are "
            "not paraphrases of one focal claim"
        )


# --- the conflict-handling rule stays one rule -------------------------------


def test_no_unresolvable_conflict_gold_answer_recommends_an_action(real_question_set):
    """Surface and escalate, never recommend, under amendment 1.1.12.

    The rule is uniform because a conservative reading exists for only one of
    the five reported current_current families. A rule that fires on one family
    and not the others cannot be scored on a single rubric, and Arm D's score
    would depend on how many risk-asymmetric conflicts happened to be in the set
    rather than on how well it handles conflict.
    """
    recommending = (
        "the safe course", "we recommend", "you should follow", "it is safer to",
        "err on the side", "the safer option is", "in the meantime, follow",
        "until this is resolved, apply",
    )
    for question in real_question_set:
        if question.expected_behaviour != "surface_both_and_qualify":
            continue
        lowered = question.gold_answer.lower()
        leaked = [phrase for phrase in recommending if phrase in lowered]
        assert not leaked, (
            f"{question.question_id} recommends an action ({leaked}). The rule is "
            "to surface both positions and escalate, applied uniformly so that "
            "one rubric can score every family."
        )


def test_every_unresolvable_conflict_requires_the_disagreement_to_be_stated(real_question_set):
    """Giving both figures without saying they conflict is a lesser behaviour
    and is scored as 1, not 2."""
    for question in real_question_set:
        if question.expected_behaviour != "surface_both_and_qualify":
            continue
        assert any("disagree" in claim for claim in question.required_claims), (
            f"{question.question_id} does not require the disagreement to be stated"
        )
