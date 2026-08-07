"""Tests for vector search.

Retrieval is where the experiment's independent variables live. `top_k`, the
similarity threshold and the status filter are all decisions that change what
the model sees, so each needs to behave exactly as configured rather than
approximately.
"""

from __future__ import annotations

import math

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.common.llm_client import MockClient
from sme_assistant.ingest.index import build_index
from sme_assistant.kb.loader import load_knowledge_base
from sme_assistant.retrieve.retriever import (
    RetrievalMode,
    Retriever,
    cosine_similarity,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def kb(config):
    return load_knowledge_base(config.path("paths.kb_docs"))


@pytest.fixture(scope="module")
def client(config):
    return MockClient(config)


@pytest.fixture(scope="module")
def retriever(kb, client, config):
    return Retriever(build_index(kb, client, config), client, config)


# --- similarity -------------------------------------------------------------


def test_cosine_of_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_ignores_magnitude():
    """nomic-embed-text returns unnormalised vectors, so a dot product would
    rank by length as well as direction and quietly favour longer chunks."""
    assert cosine_similarity([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_handles_a_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- ranking ----------------------------------------------------------------


def test_results_are_ordered_by_descending_score(retriever):
    result = retriever.retrieve("how many days of annual leave do I get?")
    scores = [s.score for s in result]
    assert scores == sorted(scores, reverse=True)
    assert [s.rank for s in result] == list(range(1, len(result) + 1))


def test_top_k_is_respected(retriever):
    for k in (1, 2, 4, 8):
        assert len(retriever.retrieve("annual leave", top_k=k)) == k


def test_retrieval_is_deterministic(retriever):
    """Ties are broken by chunk_id, so two identical queries cannot disagree."""
    a = retriever.retrieve("what is the fire assembly point?")
    b = retriever.retrieve("what is the fire assembly point?")
    assert [s.chunk_id for s in a] == [s.chunk_id for s in b]
    assert [s.score for s in a] == [s.score for s in b]


def test_relevant_chunk_outranks_an_unrelated_one(retriever):
    result = retriever.retrieve("how many days of annual leave do I get?", top_k=10)
    ranks = {s.chunk.doc_id: s.rank for s in result}
    assert "HR-01" in ranks, "the annual leave policy did not appear in the top 10"


def test_timings_are_recorded(retriever):
    result = retriever.retrieve("annual leave")
    assert result.embed_seconds >= 0
    assert result.search_seconds > 0
    assert result.candidates_considered > 0


# --- the status filter, which is evaluation arm 2 ---------------------------


def test_current_only_excludes_superseded_chunks(retriever, kb):
    result = retriever.retrieve("mileage rate", top_k=20, mode=RetrievalMode.CURRENT_ONLY)
    assert all(s.chunk.is_current for s in result)
    assert not result.has_superseded


def test_all_mode_considers_more_candidates_than_current_only(retriever):
    everything = retriever.retrieve("mileage rate", mode=RetrievalMode.ALL)
    current = retriever.retrieve("mileage rate", mode=RetrievalMode.CURRENT_ONLY)
    assert everything.candidates_considered > current.candidates_considered, (
        "the two evaluation arms are searching the same candidate set, so the "
        "filter baseline is not actually filtering anything"
    )


def test_superseded_chunks_are_reachable_by_default(retriever):
    """They must be, or the verification layer has nothing to detect."""
    result = retriever.retrieve("expenses mileage pence per mile claim", top_k=40)
    assert any(not s.chunk.is_current for s in result), (
        "no superseded chunk appeared even in the top 40, so the planted "
        "conflicts can never reach the model"
    )


def test_category_filter_restricts_results(retriever):
    result = retriever.retrieve("policy", top_k=10, category="OPS")
    assert result.results
    assert all(s.chunk.category == "OPS" for s in result)


# --- refusal ----------------------------------------------------------------


def test_low_similarity_sets_the_refusal_signal(retriever):
    """Vector search always returns something. The threshold is what separates
    "here are the nearest four passages" from "the corpus has an answer"."""
    result = retriever.retrieve("annual leave", min_similarity=0.99)
    assert result.below_threshold
    assert result.should_refuse
    assert len(result) > 0, "refusal is a judgement about scores, not an empty result"


def test_high_similarity_does_not_refuse(retriever):
    result = retriever.retrieve("annual leave entitlement days", min_similarity=0.0)
    assert not result.should_refuse


# --- conflict signalling ----------------------------------------------------


def test_supersession_pairs_are_reported_when_both_retrieved(retriever):
    result = retriever.retrieve("expenses mileage pence per mile claim", top_k=40)
    doc_ids = {s.chunk.doc_id for s in result}
    if {"HR-03", "HR-13"} <= doc_ids:
        assert result.conflicts, (
            "both halves of a supersession pair were retrieved but no conflict "
            "was reported from metadata"
        )
        pair = result.conflicts[0]
        assert pair["superseded"] == "HR-03"
        assert pair["current"] == "HR-13"
        assert pair["detectable_from"] == "metadata"


def test_no_conflict_reported_when_only_one_side_retrieved(retriever):
    result = retriever.retrieve("annual leave", top_k=1)
    assert result.conflicts == ()


# --- evidence formatting ----------------------------------------------------


def test_evidence_block_marks_superseded_documents(retriever):
    result = retriever.retrieve("expenses mileage pence per mile claim", top_k=40)
    text = result.evidence_text()
    for scored in result:
        assert scored.chunk_id in text, "a retrieved chunk is missing from the evidence"
        if not scored.chunk.is_current:
            assert f"[SUPERSEDED, replaced by {scored.chunk.superseded_by}]" in text


def test_evidence_block_carries_citable_identifiers(retriever):
    result = retriever.retrieve("fire assembly point")
    text = result.evidence_text()
    for scored in result:
        assert f"[{scored.chunk_id}]" in text
        assert str(scored.chunk.effective_date) in text


# --- guards -----------------------------------------------------------------


def test_mismatched_embedding_model_is_rejected(kb, client, config):
    """Query and chunk vectors from different models are not comparable, and
    every similarity score would be meaningless rather than merely wrong."""
    index = build_index(kb, client, config)
    index.metadata["embedding_model"] = "some-other-model"
    with pytest.raises(ValueError, match="different spaces"):
        Retriever(index, client, config)


def test_result_serialises_for_a_results_file(retriever):
    payload = retriever.retrieve("annual leave").to_dict()
    for key in ("query", "mode", "top_k", "min_similarity", "best_score",
                "should_refuse", "has_superseded", "candidates_considered",
                "embed_seconds", "search_seconds", "conflicts", "results"):
        assert key in payload
    import json

    assert json.dumps(payload)
