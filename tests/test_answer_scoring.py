"""Tests for citation support and completeness.

The first test is the reason this module exists: an answer from the Stage 4
pilot that every metric of the time scored as a success, and which cited the
wrong passage.
"""

from __future__ import annotations

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.evaluation.answer_scoring import (
    extract_quantities,
    score_answer,
    split_claims,
)
from sme_assistant.ingest.chunker import chunk_corpus
from sme_assistant.kb.loader import load_knowledge_base


@pytest.fixture(scope="module")
def chunk_texts():
    config = load_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    return {c.chunk_id: c.text for c in chunk_corpus(
        kb,
        config.require("chunking.max_words"),
        config.require("chunking.overlap_sentences"),
        config.require("chunking.min_words"),
    )}


# --- the failure this module was built to catch -----------------------------


def test_the_pilot_miscitation_is_detected(chunk_texts):
    """llama3.2:3b, Stage 4 pilot, commit f274f60.

    The answer is factually correct. "Within 1 hour" appears in IT-03#002, the
    Timescales table, not in IT-03#001, which lists what counts as an incident.
    Under the Stage 4 metrics this passed everything: the identifier was real,
    retrieved, and nothing was invented.
    """
    assert "1 hour" not in chunk_texts["IT-03#001"], (
        "the corpus has changed and this test no longer reproduces the miscitation"
    )
    assert "1 hour" in chunk_texts["IT-03#002"]

    score = score_answer(
        "You must report a lost laptop to the IT helpdesk within 1 hour of "
        "becoming aware [IT-03#001].",
        chunk_texts,
    )
    assert score.citation_completeness == 1.0, "the claim was cited"
    assert score.citation_support == 0.0, "but the citation does not support it"
    assert "1 hour" in score.unsupported_claims[0].unsupported_quantities


def test_the_same_claim_cited_correctly_scores_full_support(chunk_texts):
    score = score_answer(
        "You must report a lost laptop within 1 hour [IT-03#002].", chunk_texts
    )
    assert score.citation_support == 1.0


def test_the_pilot_mileage_answer_is_fully_supported(chunk_texts):
    """The pilot's other answer was correctly attributed, and must not be
    penalised by a metric designed to catch the first one."""
    score = score_answer(
        "According to [HR-13#001], the mileage rate for business travel is as follows:\n"
        "* 55 pence per mile for the first 10,000 business miles in a tax year\n"
        "* 25 pence per mile thereafter.",
        chunk_texts,
    )
    assert score.citation_completeness == 1.0
    assert score.citation_support == 1.0


def test_a_claim_cited_to_an_unrelated_chunk_is_unsupported(chunk_texts):
    score = score_answer("The mileage rate is 55 pence per mile [HR-01#001].", chunk_texts)
    assert score.citation_support == 0.0
    assert "55 pence" in score.unsupported_claims[0].unsupported_quantities


# --- completeness is not the same as support --------------------------------


def test_an_uncited_claim_lowers_completeness_not_support(chunk_texts):
    score = score_answer("The mileage rate is 55 pence per mile.", chunk_texts)
    assert score.citation_completeness == 0.0
    assert score.citation_support is None, (
        "support is undefined when nothing was cited; reporting zero would "
        "conflate not citing with citing badly"
    )


def test_completeness_counts_claims_not_retrieved_chunks(chunk_texts):
    """A generator is not expected to cite every passage it was given."""
    score = score_answer(
        "Leave is 25 days [HR-01#001]. The rate is 55 pence per mile [HR-13#001].",
        chunk_texts,
    )
    assert score.citation_completeness == 1.0


# --- quantity extraction ----------------------------------------------------


def test_citation_markers_are_not_treated_as_quantities():
    """[IT-03#001] would otherwise contribute the quantities 03 and 001.

    No chunk contains those, so every cited claim would score unsupported and
    the metric would read zero everywhere while appearing to work.
    """
    quantities = extract_quantities("The rate is 55 pence per mile [HR-13#001].")
    values = {value for value, _ in quantities}
    assert "55" in values
    assert "13" not in values
    assert "001" not in values


@pytest.mark.parametrize("text,expected_value", [
    ("within 1 hour", "1"),
    ("55 pence per mile", "55"),
    ("capped at £130 per night", "£130"),
    ("at least 14 characters", "14"),
    ("up to 15%", "15%"),
    ("the first 10,000 business miles", "10,000"),
])
def test_quantities_are_extracted_with_their_units(text, expected_value):
    assert expected_value in {value for value, _ in extract_quantities(text)}


def test_a_qualitative_claim_is_not_checkable():
    """Absence of a quantity means this method cannot judge the claim.

    Counting it as supported would overstate the metric, which is why such
    claims are excluded from both denominators rather than assumed correct.
    """
    score = score_answer("Alcohol is not reimbursed [HR-13#002].", {"HR-13#002": "x"})
    assert score.checkable == ()
    assert score.citation_support is None


# --- claim splitting --------------------------------------------------------


def test_bullets_inherit_the_lead_sentence_citation():
    """Models commonly cite once then list figures beneath.

    Treating the block as one claim would hide which figure was attributed to
    what; treating the bullets as uncited would understate completeness.
    """
    claims = split_claims(
        "According to [HR-13#001], the rates are:\n* 55 pence per mile\n* 25 pence thereafter."
    )
    bullets = [c for c in claims if c.text.startswith("*")]
    assert len(bullets) == 2
    assert all(c.citations == ("HR-13#001",) for c in bullets)


def test_an_independent_sentence_does_not_inherit_a_citation():
    claims = split_claims("The rate is 55 pence [HR-13#001]. Leave is 25 days.")
    assert claims[0].citations == ("HR-13#001",)
    assert claims[1].citations == (), (
        "a following sentence that is not a bullet must not inherit the "
        "previous sentence's citation, or completeness would be overstated"
    )


# --- honesty about the method -----------------------------------------------


def test_the_result_declares_itself_a_lower_bound(chunk_texts):
    """A citation containing the right figure may still be the wrong source.

    The saved result must say so, so that nobody reads this number as a
    complete measure of citation correctness.
    """
    payload = score_answer("Leave is 25 days [HR-01#001].", chunk_texts).to_dict()
    assert payload["is_lower_bound"] is True
    assert payload["method"] == "quantity_overlap"
    assert "manual review" in payload["note"]
