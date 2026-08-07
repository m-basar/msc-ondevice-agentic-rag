"""Tests for baseline grounded generation.

The most important test in this file is `test_baseline_prompt_gives_no_conflict_guidance`.
The ablation in Chapter 4 compares this pipeline against the same pipeline with
verification added. If conflict handling leaked into the baseline prompt, that
comparison would measure prompt engineering rather than the contribution, and
the result would be worthless while looking fine.
"""

from __future__ import annotations

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.common.llm_client import MockClient
from sme_assistant.generate.generator import (
    Generator,
    extract_citations,
    looks_like_refusal,
)
from sme_assistant.generate.prompts import BASELINE_SYSTEM, build_baseline_prompt
from sme_assistant.ingest.index import build_index
from sme_assistant.kb.loader import load_knowledge_base
from sme_assistant.retrieve.retriever import Retriever


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def kb(config):
    return load_knowledge_base(config.path("paths.kb_docs"))


@pytest.fixture(scope="module")
def retriever(kb, config):
    client = MockClient(config)
    return Retriever(build_index(kb, client, config), client, config)


# --- the baseline must stay a baseline ---------------------------------------


def test_baseline_prompt_gives_no_conflict_guidance():
    """A baseline that has been secretly helped is not a baseline.

    If any of these words appeared, the ablation would be comparing two
    conflict-aware systems and would understate the contribution to zero.
    """
    prompt = build_baseline_prompt("q", "some evidence").lower()
    forbidden = (
        "superseded", "withdraw", "most recent", "latest version", "current version",
        "effective date", "conflict", "contradict", "precedence", "prefer the",
        "out of date", "no longer in force",
    )
    leaked = [word for word in forbidden if word in prompt]
    assert not leaked, (
        f"the baseline prompt mentions {leaked}, which is guidance the "
        "verification layer is supposed to provide. The ablation would measure "
        "prompt wording rather than the contribution."
    )


def test_baseline_prompt_asks_for_citations_and_admits_ignorance():
    lowered = BASELINE_SYSTEM.lower()
    assert "cite" in lowered
    assert "only the evidence" in lowered
    assert "does not contain the answer" in lowered


def test_prompt_contains_the_evidence_and_the_question():
    prompt = build_baseline_prompt("How much leave?", "[HR-01#001] 25 days")
    assert "How much leave?" in prompt
    assert "[HR-01#001] 25 days" in prompt


def test_empty_evidence_uses_the_no_evidence_template():
    prompt = build_baseline_prompt("Anything about pensions?", "")
    assert "No relevant evidence was retrieved" in prompt
    assert "EVIDENCE:" not in prompt


# --- citation extraction ----------------------------------------------------


def test_extracts_chunk_and_document_citations():
    chunks, documents = extract_citations(
        "The rate is 55p [HR-13#001] and claims go in within 30 days [HR-13]."
    )
    assert chunks == ("HR-13#001",)
    assert documents == ("HR-13",)


def test_citations_are_deduplicated_but_keep_first_appearance_order():
    """Order matters: whether the model led with the current policy or the
    withdrawn one is part of the analysis."""
    chunks, _ = extract_citations("[HR-13#001] then [HR-03#001] then [HR-13#001] again")
    assert chunks == ("HR-13#001", "HR-03#001")


def test_ordinals_are_normalised():
    chunks, _ = extract_citations("[HR-13#1] and [HR-13#001]")
    assert chunks == ("HR-13#001",)


def test_text_without_citations_yields_nothing():
    assert extract_citations("The mileage rate is 55 pence per mile.") == ((), ())


@pytest.mark.parametrize("text,expected", [
    ("The evidence does not contain the answer.", True),
    ("This is not covered by the documents provided.", True),
    ("I cannot answer from the evidence given.", True),
    ("The rate is 55 pence per mile [HR-13#001].", False),
])
def test_refusal_detection(text, expected):
    assert looks_like_refusal(text) is expected


# --- the audit that makes the baseline measurable ---------------------------


def test_hallucinated_citation_is_detected(retriever, config):
    """The model invented an identifier that was never in its evidence."""
    retrieval = retriever.retrieve("annual leave entitlement")
    client = MockClient(config, responses={"EVIDENCE": "Leave is 25 days [ZZ-99#001]."})
    result = Generator(client, config).answer("How much leave?", retrieval)
    assert "ZZ-99#001" in result.hallucinated_citations
    assert not result.has_valid_citation_ids


def test_citing_a_withdrawn_document_is_flagged(retriever, config):
    """The headline failure the baseline is expected to make."""
    # min_similarity=0 so the evidence block is definitely built: with mock
    # lexical embeddings this broad query can otherwise fall below the refusal
    # threshold, and the test would silently exercise the no-evidence path.
    retrieval = retriever.retrieve(
        "expenses mileage pence per mile claim", top_k=40, min_similarity=0.0
    )
    assert not retrieval.should_refuse
    superseded = [s for s in retrieval if not s.chunk.is_current]
    assert superseded, "no superseded chunk retrieved, so this test cannot run"

    target = superseded[0].chunk_id
    client = MockClient(config, responses={"EVIDENCE": f"The rate is 40 pence [{target}]."})
    result = Generator(client, config).answer("What is the mileage rate?", retrieval)

    assert target in result.cited_superseded
    assert not result.hallucinated_citations, (
        "citing a retrieved but withdrawn document is a different failure from "
        "inventing an identifier, and the two must not be conflated"
    )


def test_uncited_retrieved_chunks_are_reported(retriever, config):
    retrieval = retriever.retrieve("annual leave entitlement")
    cited = retrieval.results[0].chunk_id
    client = MockClient(config, responses={"EVIDENCE": f"Answer [{cited}]."})
    result = Generator(client, config).answer("How much leave?", retrieval)
    assert cited not in result.uncited_chunks
    assert len(result.uncited_chunks) == len(retrieval) - 1


def test_a_well_cited_answer_is_grounded(retriever, config):
    retrieval = retriever.retrieve("annual leave entitlement")
    cited = retrieval.results[0].chunk_id
    client = MockClient(config, responses={"EVIDENCE": f"25 days [{cited}]."})
    result = Generator(client, config).answer("How much leave?", retrieval)
    assert result.has_valid_citation_ids
    assert result.citations == (cited,)


def test_an_uncited_answer_is_not_grounded(retriever, config):
    """Not necessarily wrong, but unverifiable, which here amounts to the same."""
    retrieval = retriever.retrieve("annual leave entitlement")
    client = MockClient(config, responses={"EVIDENCE": "Twenty five days."})
    result = Generator(client, config).answer("How much leave?", retrieval)
    assert not result.has_valid_citation_ids
    assert result.citations == ()


# --- refusal path -----------------------------------------------------------


def test_no_evidence_is_sent_when_retrieval_falls_below_threshold(retriever, config):
    retrieval = retriever.retrieve("annual leave", min_similarity=0.99)
    assert retrieval.should_refuse
    result = Generator(MockClient(config), config).answer("How much leave?", retrieval)
    assert "No relevant evidence was retrieved" in result.prompt
    assert "EVIDENCE:" not in result.prompt


def test_result_serialises_for_a_results_file(retriever, config):
    retrieval = retriever.retrieve("annual leave")
    result = Generator(MockClient(config), config).answer("How much leave?", retrieval)
    payload = result.to_dict()
    for key in ("question", "answer", "citations", "hallucinated_citations",
                "cited_superseded", "has_valid_citation_ids", "refusal_heuristic",
                "generation", "retrieval"):
        assert key in payload
    import json

    assert json.dumps(payload)


def test_wall_time_includes_retrieval_and_generation(retriever, config):
    retrieval = retriever.retrieve("annual leave")
    result = Generator(MockClient(config), config).answer("How much leave?", retrieval)
    assert result.wall_seconds == pytest.approx(
        retrieval.embed_seconds + retrieval.search_seconds + result.generation.wall_seconds
    ), "end-to-end time must include vector search, not only embedding and generation"


# --- the refusal the heuristic missed ---------------------------------------


def test_the_pilot_refusal_is_now_detected():
    """Regression test for a real miss.

    During the Stage 4 pilot llama3.2:3b produced this exact sentence in
    response to a question about pensions, a topic the corpus does not cover.
    The model was right and the evaluator was wrong: none of the markers
    matched, so a correct refusal was scored as a non-refusal.
    """
    answer = (
        "I cannot provide an answer about the company pension scheme "
        "as it is not present in the provided evidence."
    )
    assert looks_like_refusal(answer), (
        "the pilot refusal is still not detected; extending the marker list "
        "did not cover the phrasing that actually occurred"
    )


def test_refusal_heuristic_is_labelled_as_diagnostic_in_saved_results(retriever, config):
    """Anyone reading a results file must see that this number is not scoring.

    Keyword matching over free text cannot be trusted for a reported metric.
    The flag stays as a diagnostic; final refusal scoring uses a predefined
    rubric or structured output from the verification layer.
    """
    retrieval = retriever.retrieve("annual leave")
    payload = Generator(MockClient(config), config).answer("q", retrieval).to_dict()
    assert payload["refusal_heuristic_is_diagnostic_only"] is True


# --- experimental arm A ------------------------------------------------------


def test_plain_evidence_withholds_status_and_dates(retriever, config):
    """Arm A against arm B, the only difference being the metadata.

    The pilot showed a 3B model resolving a supersession conflict correctly
    when the marker was present. Without a no-marker arm there is no way to
    tell whether the model reasoned or simply read the label.
    """
    from sme_assistant.retrieve.retriever import EvidenceFormat

    retrieval = retriever.retrieve(
        "expenses mileage pence per mile claim", top_k=40, min_similarity=0.0
    )
    assert retrieval.has_superseded, "this test needs a superseded chunk retrieved"

    plain = retrieval.evidence_text(EvidenceFormat.PLAIN)
    marked = retrieval.evidence_text(EvidenceFormat.WITH_STATUS)

    assert "SUPERSEDED" in marked
    assert "SUPERSEDED" not in plain
    assert "effective" not in plain
    for scored in retrieval:
        assert f"[{scored.chunk_id}]" in plain, "arm A must keep citable identifiers"
        assert scored.chunk.text in plain, "arm A must present the same text"


def test_the_two_arms_differ_only_in_metadata(retriever, config):
    """Same chunks, same order, same text. Any difference in answers is
    attributable to the metadata rather than to what was retrieved."""
    from sme_assistant.retrieve.retriever import EvidenceFormat

    retrieval = retriever.retrieve("annual leave entitlement", min_similarity=0.0)
    plain = retrieval.evidence_text(EvidenceFormat.PLAIN)
    marked = retrieval.evidence_text(EvidenceFormat.WITH_STATUS)
    assert plain.count("---") == marked.count("---")
    assert len(plain) < len(marked)


def test_generator_passes_the_evidence_format_through(retriever, config):
    from sme_assistant.retrieve.retriever import EvidenceFormat

    retrieval = retriever.retrieve("mileage", top_k=40, min_similarity=0.0)
    generator = Generator(MockClient(config), config)
    plain = generator.answer("q", retrieval, evidence_format=EvidenceFormat.PLAIN)
    marked = generator.answer("q", retrieval, evidence_format=EvidenceFormat.WITH_STATUS)
    assert "SUPERSEDED" not in plain.prompt
    assert "SUPERSEDED" in marked.prompt
