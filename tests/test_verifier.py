"""Tests for the verification layer.

The first section is the most important in the file. If the verifier can reach
gold data, the conflict-handling result is a dictionary lookup dressed as a
finding, and it would look excellent while measuring nothing.
"""

from __future__ import annotations

import json

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.common.llm_client import MockClient
from sme_assistant.generate.generator import Generator
from sme_assistant.ingest.index import build_index
from sme_assistant.kb.loader import load_knowledge_base
from sme_assistant.retrieve.retriever import Retriever
from sme_assistant.verify import Verifier, schema
from sme_assistant.verify.schema import (
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE,
    SUPPORTED,
    Verification,
)


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


POLICY = {"on_supported": "medium", "on_resolved_conflict": "medium",
          "on_unresolved_conflict": "low", "on_insufficient": "low",
          "on_contradiction": "low", "on_invented_evidence": "low",
          "on_parse_failure": "low"}
TWO = {"HR-01#001", "IT-03#002"}


# --- the boundary that makes the result mean anything ------------------------


def test_the_verify_package_cannot_reach_gold_data():
    """Arm D must infer conflict, not look it up.

    A verifier that read the declared family type would produce excellent
    output and measure a dictionary. The failure would be invisible in every
    answer, which is why this is enforced in the import graph rather than by
    intention.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "sme_assistant" / "verify"
    forbidden = (
        "evaluation.conflicts", "load_conflicts", "conflicts.json",
        "question_set", "SCORING_RUBRICS", "evaluation.config",
        "gold/", "family_id", "satisfying_action",
    )
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        # Strip docstrings and comments: this file discusses gold data at
        # length, and the point is that it never reads it.
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        for token in forbidden:
            in_prose = code.count(token) and all(
                token in block for block in ()
            )
            _ = in_prose
        stripped = _strip_docstrings(code)
        for token in forbidden:
            assert token not in stripped, (
                f"{path.name} references {token!r} outside documentation, which "
                "would let Arm D read the answer key"
            )


def _strip_docstrings(source: str) -> str:
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover
        return source
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body[0].value.value = ""
    return ast.unparse(tree)


def test_the_verifier_prompt_never_names_a_conflict_family():
    """The prompt is where leakage would be easiest and least visible."""
    from sme_assistant.verify.verifier import VERIFIER_SYSTEM, build_verification_prompt

    prompt = build_verification_prompt("q", "a", "[HR-01#001] text").upper()
    for token in ("CONF-", "TUNE-", "FAMILY", "GOLD", "EXPECTED BEHAVIOUR"):
        assert token not in prompt, f"the verifier prompt mentions {token}"
    assert "SUPPORTED" in VERIFIER_SYSTEM


def test_the_relationship_vocabulary_is_offered_not_answered():
    """Naming the categories is a briefing; supplying the answer is leakage."""
    from sme_assistant.verify.verifier import VERIFIER_SYSTEM

    for name in ("mutually_exclusive", "stricter_looser", "contextually_compatible"):
        assert name in VERIFIER_SYSTEM
    assert "the correct relationship is" not in VERIFIER_SYSTEM.lower()


# --- evidence identifiers ----------------------------------------------------


def test_an_invented_chunk_id_is_discarded_and_recorded():
    """A verifier citing evidence it was not given is the same failure as a
    generator doing it, and must be equally visible."""
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-01#001", "ZZ-99#001"]}],
        "relationship": "no_relationship",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY)
    assert result.verdicts[0].supporting == ("HR-01#001",)
    assert result.invented_ids == ("ZZ-99#001",)
    assert result.confidence == "low", "invented evidence must not leave confidence high"


def test_a_conflict_left_with_one_document_is_downgraded():
    """Stripping the invented identifier is not enough on its own.

    An earlier version removed the bad chunk and left the mutually_exclusive
    verdict standing on a single valid one. A disagreement needs two sides, so
    a conflict that survives on one document is not evidenced and the finding
    is withdrawn rather than repaired.
    """
    payload = json.dumps({
        "claims": [], "relationship": "mutually_exclusive",
        "conflicting_chunks": ["HR-01#001", "QQ-11#003"],
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY)
    assert "QQ-11#003" in result.invented_ids
    assert result.relationship == "insufficient"
    assert not result.conflict_detected
    assert result.conflicting_chunks == ()


def test_two_chunks_from_one_document_are_one_side_not_two():
    payload = json.dumps({
        "claims": [], "relationship": "stricter_looser",
        "conflicting_chunks": ["HR-01#001", "HR-01#002"],
    })
    result = schema.parse(payload, {"HR-01#001", "HR-01#002"}, POLICY)
    assert result.relationship == "insufficient", (
        "two passages of the same document are one position, not a disagreement"
    )


def test_a_conflict_across_two_documents_stands():
    payload = json.dumps({
        "claims": [], "relationship": "mutually_exclusive",
        "conflicting_chunks": sorted(TWO),
    })
    assert schema.parse(payload, TWO, POLICY).conflict_detected


# --- verdicts cannot outlive their evidence ----------------------------------


def test_supported_without_supporting_evidence_is_downgraded():
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED", "supporting": ["ZZ-99#001"]}],
        "relationship": "no_relationship",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY)
    assert result.verdicts[0].verdict == INSUFFICIENT_EVIDENCE


def test_contradicted_without_contradicting_evidence_is_downgraded():
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "CONTRADICTED", "contradicting": []}],
        "relationship": "no_relationship",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY)
    assert result.verdicts[0].verdict == INSUFFICIENT_EVIDENCE
    assert not result.any_contradiction


@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False), ("true", True), ("false", False),
    ("TRUE", True), ("no", False), (1, True), (0, False), (None, False),
])
def test_escalate_is_parsed_not_coerced(raw, expected):
    """bool("false") is True, so a verifier writing the string would have
    escalated everything."""
    payload = json.dumps({"claims": [], "relationship": "no_relationship",
                          "escalate": raw})
    assert schema.parse(payload, set(), POLICY).escalate is expected


# --- a resolved supersession is not an unresolved conflict -------------------


def test_supersession_is_detected_but_not_treated_as_unresolved():
    """It has a right answer: the current document governs.

    Capping its confidence with the unresolvable kinds would punish the system
    for noticing the withdrawn document rather than for mishandling it.
    """
    payload = json.dumps({"claims": [], "relationship": "supersession",
                          "conflicting_chunks": sorted(TWO)})
    result = schema.parse(payload, TWO, POLICY)
    assert result.conflict_detected
    assert result.is_resolved and not result.is_unresolved
    assert result.confidence == "medium"


def test_an_unresolvable_conflict_still_caps_confidence():
    payload = json.dumps({"claims": [], "relationship": "mutually_exclusive",
                          "conflicting_chunks": sorted(TWO)})
    result = schema.parse(payload, TWO, POLICY)
    assert result.is_unresolved and not result.is_resolved
    assert result.confidence == "low"


# --- parsing what small models actually emit ---------------------------------


@pytest.mark.parametrize("wrapper", [
    '{body}',
    'Here is my analysis:\n{body}\nI hope that helps.',
    '```json\n{body}\n```',
])
def test_json_survives_the_prose_small_models_wrap_it_in(wrapper):
    body = '{"claims": [], "relationship": "no_relationship"}'
    result = schema.parse(wrapper.format(body=body), set(), POLICY)
    assert not result.parse_failed
    assert result.relationship == "no_relationship"


def test_an_unparseable_response_fails_loudly_but_safely():
    """A formatting slip must not become a silent success."""
    result = schema.parse("I am not going to answer that.", set(), POLICY)
    assert result.parse_failed
    assert result.relationship == "insufficient"
    assert result.confidence == "low"
    assert not result.conflict_detected


def test_an_unknown_verdict_degrades_to_insufficient():
    payload = json.dumps({"claims": [{"claim": "x", "verdict": "PROBABLY_FINE"}],
                          "relationship": "no_relationship"})
    result = schema.parse(payload, set(), POLICY)
    assert result.verdicts[0].verdict == INSUFFICIENT_EVIDENCE


def test_an_unknown_relationship_degrades_to_insufficient():
    payload = json.dumps({"claims": [], "relationship": "sort of contradictory"})
    assert schema.parse(payload, set(), POLICY).relationship == "insufficient"


# --- what counts as a detected conflict --------------------------------------


@pytest.mark.parametrize("relationship,detected", [
    ("supersession", True),
    ("mutually_exclusive", True),
    ("stricter_looser", True),
    ("contextually_compatible", False),
    ("no_relationship", False),
    ("insufficient", False),
])
def test_contextually_compatible_is_not_a_detected_conflict(relationship, detected):
    """H2c depends on this distinction.

    Inferring that two passages apply in different circumstances is the
    verifier declining to flag. Counting it as a detection would make the
    false-conflict rate unmeasurable, which is the metric that keeps the
    contribution honest.
    """
    payload = json.dumps({"claims": [], "relationship": relationship,
                          "conflicting_chunks": sorted(TWO)})
    assert schema.parse(payload, TWO, POLICY).conflict_detected is detected


# --- confidence comes from runtime policy, not gold --------------------------


def test_a_detected_conflict_caps_confidence():
    payload = json.dumps({"claims": [], "relationship": "mutually_exclusive",
                          "conflicting_chunks": sorted(TWO)})
    assert schema.parse(payload, TWO, POLICY).confidence == "low"


def test_a_clean_supported_answer_may_hold_medium_confidence():
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED", "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
    })
    assert schema.parse(payload, {"HR-01#001"}, POLICY).confidence == "medium"


def test_the_runtime_policy_is_configured_outside_the_registry(config):
    """Two confidence policies exist and only one is readable at runtime."""
    policy = config.get("verification.confidence")
    assert policy, "no runtime confidence policy in config.json"
    for outcome in ("on_resolved_conflict", "on_unresolved_conflict",
                    "on_insufficient", "on_supported"):
        assert outcome in policy


# --- the layer end to end ----------------------------------------------------


def test_verification_runs_and_records_its_own_cost(retriever, config):
    """H5 measures this ratio, so the verification pass must be timed apart."""
    retrieval = retriever.retrieve("annual leave entitlement", min_similarity=0.0)
    cited = retrieval.results[0].chunk_id
    client = MockClient(config, responses={"EVIDENCE": f"Leave is 25 days [{cited}]."})
    answer = Generator(client, config).answer("How much leave?", retrieval)

    verifier_client = MockClient(config, responses={"policy auditor": json.dumps({
        "claims": [{"claim": "Leave is 25 days", "verdict": "SUPPORTED",
                    "supporting": [cited]}],
        "relationship": "no_relationship", "escalate": False,
    })})
    verified = Verifier(verifier_client, config).verify(answer)

    assert verified.verification.verdicts[0].verdict == SUPPORTED
    assert not verified.verification.conflict_detected
    assert verified.wall_seconds >= answer.wall_seconds
    payload = verified.to_dict()
    assert payload["arm_has_verification"] is True
    assert "verification_seconds" in payload


def test_the_verifier_sees_the_answer_it_is_auditing(retriever, config):
    """A second pass that cannot see the answer is not auditing anything."""
    retrieval = retriever.retrieve("annual leave", min_similarity=0.0)
    client = MockClient(config, responses={"EVIDENCE": "Leave is 25 days."})
    answer = Generator(client, config).answer("How much leave?", retrieval)
    verified = Verifier(client, config).verify(answer)
    assert "Leave is 25 days" in verified.prompt
    assert "How much leave?" in verified.prompt


def test_a_contradiction_is_carried_through_to_the_record(retriever, config):
    retrieval = retriever.retrieve("mileage expenses", top_k=6, min_similarity=0.0)
    ids = [s.chunk_id for s in retrieval]
    client = MockClient(config, responses={"EVIDENCE": "The rate is 40 pence."})
    answer = Generator(client, config).answer("What is the mileage rate?", retrieval)

    verifier_client = MockClient(config, responses={"policy auditor": json.dumps({
        "claims": [{"claim": "The rate is 40 pence", "verdict": "CONTRADICTED",
                    "contradicting": [ids[0]]}],
        "relationship": "supersession", "conflicting_chunks": ids[:2],
        "escalate": True, "rationale": "one passage replaces the other",
    })})
    verified = Verifier(verifier_client, config).verify(answer)

    assert verified.verification.any_contradiction
    assert verified.verification.conflict_detected
    assert verified.verification.escalate
    assert verified.to_dict()["verification"]["verdicts"][0]["verdict"] == CONTRADICTED


# --- the verifier must be able to change the answer --------------------------


def test_arm_d_is_scored_on_the_revised_answer_not_the_draft(retriever, config):
    """Without this, Arm D cannot beat Arm B by construction.

    The blinded scorer reads record["answer"]. An earlier version returned the
    audit and left the draft untouched, so a verifier could detect a conflict
    perfectly and still serve the wrong text.
    """
    retrieval = retriever.retrieve("mileage expenses", top_k=6, min_similarity=0.0)
    ids = [s.chunk_id for s in retrieval]
    docs = sorted({i.split("#")[0] for i in ids})
    pair = [next(i for i in ids if i.startswith(d)) for d in docs[:2]]

    client = MockClient(config, responses={"EVIDENCE": "The rate is 40 pence."})
    answer = Generator(client, config).answer("What is the mileage rate?", retrieval)

    corrected = f"The current rate is 55 pence per mile [{pair[1]}]."
    verifier_client = MockClient(config, responses={"policy auditor": json.dumps({
        "claims": [{"claim": "The rate is 40 pence", "verdict": "CONTRADICTED",
                    "contradicting": [pair[0]]}],
        "relationship": "supersession", "conflicting_chunks": pair,
        "escalate": False, "final_answer": corrected,
    })})
    verified = Verifier(verifier_client, config).verify(answer)
    payload = verified.to_dict()

    assert verified.final_answer == corrected
    assert payload["answer"] == corrected, "the scorer would read the stale draft"
    assert payload["draft_answer"] == "The rate is 40 pence."
    assert payload["answer_revised"] is True


def test_a_parse_failure_degrades_to_the_draft_not_to_silence(retriever, config):
    retrieval = retriever.retrieve("annual leave", min_similarity=0.0)
    client = MockClient(config, responses={"EVIDENCE": "Leave is 25 days."})
    answer = Generator(client, config).answer("How much leave?", retrieval)
    verifier_client = MockClient(config, responses={"policy auditor": "I cannot do that."})
    verified = Verifier(verifier_client, config).verify(answer)

    assert verified.verification.parse_failed
    assert verified.final_answer == "Leave is 25 days."
    assert verified.to_dict()["answer_revised"] is False


def test_the_prompt_asks_for_a_final_answer():
    from sme_assistant.verify.verifier import VERIFIER_SYSTEM

    assert "final_answer" in VERIFIER_SYSTEM
    assert "correct it" in VERIFIER_SYSTEM


# --- validation and the served answer must not come apart --------------------


def test_a_rejected_finding_cannot_reach_the_user_through_the_prose():
    """The relationship was downgraded; the answer written from it must go too.

    Otherwise a verifier alleges a conflict on invented evidence, has the
    finding correctly withdrawn, and still serves prose asserting it.
    """
    payload = json.dumps({
        "claims": [], "relationship": "mutually_exclusive",
        "conflicting_chunks": ["HR-01#001"],
        "final_answer": "These two policies contradict each other [HR-01#001].",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY, draft="Leave is 25 days.")
    assert result.relationship == "insufficient"
    assert result.final_answer == "Leave is 25 days."
    assert result.revision_rejected
    assert "two sides" in result.revision_rejected_reason
    assert not result.revised


def test_a_revised_answer_citing_unretrieved_evidence_is_rejected():
    payload = json.dumps({
        "claims": [], "relationship": "no_relationship",
        "final_answer": "The rate is 55 pence [ZZ-99#001].",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY, draft="draft text")
    assert result.final_answer == "draft text"
    assert result.revision_rejected
    assert "ZZ-99#001" in result.invented_ids
    assert result.confidence == "low"


def test_invented_evidence_outranks_a_favourable_relationship():
    """A supersession whose audit invented evidence is not a confident one."""
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-13#001", "ZZ-99#001"]}],
        "relationship": "supersession",
        "conflicting_chunks": ["HR-03#001", "HR-13#001"],
    })
    result = schema.parse(payload, {"HR-03#001", "HR-13#001"}, POLICY)
    assert result.is_resolved
    assert result.confidence == "low", (
        "an evidence problem must outrank the relationship label"
    )


def test_citation_metrics_describe_the_served_answer_not_the_draft(retriever, config):
    """These were inherited from a string that had been thrown away.

    A revision that fixed a miscitation would have been recorded as still
    carrying it, so the citation metrics would have measured Arm B's mistakes
    while the reader saw Arm D's corrections.
    """
    retrieval = retriever.retrieve("annual leave entitlement", min_similarity=0.0)
    real = retrieval.results[0].chunk_id

    client = MockClient(config, responses={"EVIDENCE": "Leave is 25 days [ZZ-99#001]."})
    answer = Generator(client, config).answer("How much leave?", retrieval)
    assert answer.hallucinated_citations, "the draft must start out miscited"

    corrected = f"Leave is 25 days [{real}]."
    verifier_client = MockClient(config, responses={"policy auditor": json.dumps({
        "claims": [{"claim": "Leave is 25 days", "verdict": "SUPPORTED",
                    "supporting": [real]}],
        "relationship": "no_relationship", "final_answer": corrected,
    })})
    payload = Verifier(verifier_client, config).verify(answer).to_dict()

    assert payload["citations"] == [real]
    assert payload["hallucinated_citations"] == []
    assert payload["has_valid_citation_ids"] is True
    # The draft's figures survive for comparison rather than being overwritten.
    assert payload["draft_hallucinated_citations"] == ["ZZ-99#001"]
    assert payload["draft_has_valid_citation_ids"] is False


def test_a_failed_claim_verdict_also_blocks_the_revision():
    """The gate began at the relationship check, so this slipped through.

    A CONTRADICTED verdict with nothing contradicting it was downgraded, but
    because the relationship was benign no failure was recorded, and the
    revision written from that failed verdict was served.
    """
    payload = json.dumps({
        "claims": [{"claim": "The rate is 40 pence", "verdict": "CONTRADICTED",
                    "contradicting": []}],
        "relationship": "no_relationship",
        "final_answer": "The rate is actually 55 pence [HR-01#001].",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY, draft="The rate is 40 pence.")

    assert result.verdicts[0].verdict == INSUFFICIENT_EVIDENCE
    assert result.validation_failures, "the downgrade was not recorded as a failure"
    assert result.revision_rejected
    assert result.final_answer == "The rate is 40 pence."
    assert result.confidence == "low"


def test_a_document_only_citation_in_the_revision_is_rejected():
    """Arm D must cite passages. A reader cannot check a claim against a whole
    document, and the citation metrics run after the answer has been served."""
    payload = json.dumps({
        "claims": [], "relationship": "no_relationship",
        "final_answer": "The rate is 55 pence per mile [HR-13].",
    })
    result = schema.parse(payload, {"HR-13#001"}, POLICY, draft="draft text")

    assert result.revision_rejected
    assert "documents rather than passages" in result.revision_rejected_reason
    assert result.final_answer == "draft text"
    assert result.confidence == "low"


def test_a_clean_revision_is_still_served():
    """The gate must reject bad revisions, not all of them."""
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED", "supporting": ["HR-13#001"]}],
        "relationship": "no_relationship",
        "final_answer": "The rate is 55 pence per mile [HR-13#001].",
    })
    result = schema.parse(payload, {"HR-13#001"}, POLICY, draft="The rate is 40 pence.")

    assert not result.validation_failures
    assert not result.revision_rejected
    assert result.revised
    assert result.final_answer == "The rate is 55 pence per mile [HR-13#001]."
    assert result.confidence == "medium"


# --- a silent normalisation is still a repair --------------------------------


def test_a_misspelled_verdict_blocks_the_revision():
    """Normalising an unrecognised enum is a repair, and a repair is a failure.

    A 3B model can misspell the verdict, cite a real passage, write a plausible
    revision, and have the revision served on the strength of a conclusion
    nobody could read.
    """
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTTED",
                    "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "Leave is 25 days [HR-01#001].",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY, draft="draft text")

    assert result.verdicts[0].verdict == INSUFFICIENT_EVIDENCE
    assert result.validation_failures
    assert result.revision_rejected
    assert result.final_answer == "draft text"
    assert result.confidence == "low"


def test_a_misspelled_relationship_blocks_the_revision():
    payload = json.dumps({
        "claims": [], "relationship": "mutualy_exclusive",
        "conflicting_chunks": sorted(TWO),
        "final_answer": "These conflict [HR-01#001] [IT-03#002].",
    })
    result = schema.parse(payload, TWO, POLICY, draft="draft text")

    assert result.relationship == "insufficient"
    assert result.validation_failures
    assert result.final_answer == "draft text"


def test_a_missing_relationship_blocks_the_revision():
    """Silence is not a conclusion."""
    payload = json.dumps({
        "claims": [], "final_answer": "Leave is 25 days [HR-01#001].",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY, draft="draft text")

    assert result.relationship == "insufficient"
    assert any("missing" in f for f in result.validation_failures)
    assert result.final_answer == "draft text"


def test_a_correctly_spelled_response_is_untouched():
    """The gate must not fire on well-formed output."""
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED", "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "Leave is 25 days [HR-01#001].",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY, draft="draft text")
    assert not result.validation_failures
    assert result.final_answer == "Leave is 25 days [HR-01#001]."
    assert result.confidence == "medium"
