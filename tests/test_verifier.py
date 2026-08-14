"""Tests for the verification layer.

The first section is the most important in the file. If the verifier can reach
gold data, the conflict-handling result is a dictionary lookup dressed as a
finding, and it would look excellent while measuring nothing.
"""

from __future__ import annotations

import json
from typing import Any

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

    verifier_client = MockClient(config, responses={"You audit answers": json.dumps({
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

    verifier_client = MockClient(config, responses={"You audit answers": json.dumps({
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
    verifier_client = MockClient(config, responses={"You audit answers": json.dumps({
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
    verifier_client = MockClient(config, responses={"You audit answers": "I cannot do that."})
    verified = Verifier(verifier_client, config).verify(answer)

    assert verified.verification.parse_failed
    assert verified.final_answer == "Leave is 25 days."
    assert verified.to_dict()["answer_revised"] is False


def test_the_prompt_asks_for_a_final_answer():
    from sme_assistant.verify.verifier import VERIFIER_SYSTEM

    assert "final_answer" in VERIFIER_SYSTEM
    assert "rewrite it" in VERIFIER_SYSTEM
    assert "null when the answer under review is already correct" in VERIFIER_SYSTEM


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
    verifier_client = MockClient(config, responses={"You audit answers": json.dumps({
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
    """The gate must reject bad revisions, not all of them.

    Written as the case the layer exists for: a superseded rate found
    alongside the current one, so the revision has something to correct.
    """
    payload = json.dumps({
        "claims": [{"claim": "the current rate is 55 pence", "verdict": "SUPPORTED",
                    "supporting": ["HR-13#001"]}],
        "relationship": "supersession",
        "conflicting_chunks": ["HR-13#001", "HR-03#001"],
        "final_answer": "The current rate is 55 pence per mile [HR-13#001].",
    })
    result = schema.parse(
        payload, {"HR-13#001", "HR-03#001"}, POLICY,
        draft="The rate is 40 pence per mile [HR-03#001].",
    )

    assert not result.validation_failures
    assert not result.revision_rejected
    assert result.revised
    assert result.final_answer == "The current rate is 55 pence per mile [HR-13#001]."
    assert result.confidence == "medium"


# --- a verifier that finds nothing must change nothing ------------------------
# Pilot 02 rewrote six answers on a verdict of no_relationship with every claim
# SUPPORTED. Two of those rewrites were served and were worse than the drafts
# they replaced. The cases below are the actual strings from that run.


def test_commentary_on_the_review_is_never_served_as_the_answer():
    """The exact failure from CONF-01-Q3.

    A correct, cited answer about mileage rates was replaced by a sentence
    about whether the passages conflict. The user asked what the rate is.
    """
    payload = json.dumps({
        "claims": [{"claim": "55 pence per mile", "verdict": "SUPPORTED",
                    "supporting": ["HR-13#001"]}],
        "relationship": "no_relationship",
        "final_answer": "The answer under review does not address a conflict "
                        "between passages.",
    })
    draft = "You may claim 55 pence per mile [HR-13#001]."
    result = schema.parse(payload, {"HR-13#001"}, POLICY, draft=draft)

    assert result.final_answer == draft
    assert not result.revised
    assert result.revision_rejected
    assert "nothing for a revision to fix" in result.revision_rejected_reason


def test_a_cosmetic_rewrite_of_an_unchallenged_answer_is_not_served():
    """Moving a citation is not verification, and it contaminates B versus D.

    The two arms share a draft so that any difference between them is caused
    by the verification layer. A reworded but equivalent answer puts a
    difference into that contrast which verification did not cause.
    """
    payload = json.dumps({
        "claims": [{"claim": "25 days", "verdict": "SUPPORTED",
                    "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "Full-time employees receive 25 days [HR-01#001].",
    })
    draft = "According to [HR-01#001], full-time employees receive 25 days."
    result = schema.parse(payload, {"HR-01#001"}, POLICY, draft=draft)

    assert result.final_answer == draft
    assert result.revision_rejected


def test_stripping_every_citation_from_a_supported_answer_is_rejected():
    """The exact failure from TUNE-06-Q1, minus its abstention.

    Checking that cited identifiers resolve passes vacuously on an answer that
    cites nothing, so the original guard let this through.
    """
    payload = json.dumps({
        "claims": [{"claim": "valid for 14 days", "verdict": "CONTRADICTED",
                    "contradicting": ["OPS-03#002"]}],
        "relationship": "no_relationship",
        "final_answer": "The validity period is not stated in the evidence.",
    })
    draft = "A Returns Authorisation number is valid for 14 days [OPS-03#002]."
    result = schema.parse(payload, {"OPS-03#002"}, POLICY, draft=draft)

    assert result.final_answer == draft
    assert result.revision_rejected
    assert "cites no retrievable passage" in result.revision_rejected_reason
    # A policy refusal and a validation failure are different things and are
    # recorded separately, or validation_failures stops measuring bad output.
    assert result.validation_failures


PILOT_03_SERVED = (
    "The validity period of a Returns Authorisation number is not explicitly "
    "stated in the provided evidence. However, it can be inferred that the "
    "14-day validity period mentioned in OPS-03#002 may not be applicable to "
    "all situations, as indicated by the different refund policies and "
    "procedures outlined in other documents."
)


def test_a_bare_identifier_in_prose_is_not_a_citation():
    """The near-miss that would have silenced the alarm without fixing anything.

    The first version of this check used the bare-identifier pattern, so it
    accepted "mentioned in OPS-03#002" as a citation. The pipeline's own
    extractor requires brackets and recorded that same answer as having no
    citations at all, so the check passed on an answer the system considered
    ungrounded. Two definitions of "cites" in one codebase is the bug.
    """
    assert not schema.cites_a_passage("the 14-day period mentioned in OPS-03#002")
    assert not schema.cites_a_passage("see HR-13 for details")
    assert not schema.cites_a_passage("see [HR-13] for details"), "document, not passage"
    assert schema.cites_a_passage("the rate is 55 pence [HR-13#001]")


def test_an_abstention_may_not_reassert_the_figure_it_withdrew():
    """The exact answer pilot 03 served, verbatim.

    Its first clause abstains and its second states the figure anyway with no
    citation the pipeline recognises. That is worse than either the draft or a
    clean abstention, and the abstention exception is what let it through.
    """
    assert schema.asserts_uncited_quantity(PILOT_03_SERVED)

    payload = json.dumps({
        "claims": [{"claim": "valid for 14 days", "verdict": "INSUFFICIENT_EVIDENCE"}],
        "relationship": "no_relationship",
        "final_answer": PILOT_03_SERVED,
    })
    draft = "A Returns Authorisation number is valid for 14 days [OPS-03#002]."
    result = schema.parse(payload, {"OPS-03#002"}, POLICY, draft=draft)

    # The model's wording is discarded entirely rather than inspected.
    assert result.final_answer == schema.ABSTENTION_TEXT
    assert "14-day" not in result.final_answer
    assert "OPS-03#002" not in result.final_answer


def test_an_assertion_with_no_digit_in_it_is_also_discarded():
    """The case a digit-hunting rule could never have caught.

    "valid for two weeks" asserts exactly what "valid for 14 days" asserts and
    contains no digit. Detecting assertions inside free prose is unbounded
    work, so the abstention wording is not the model's to write.
    """
    payload = json.dumps({
        "claims": [{"claim": "valid for 14 days", "verdict": "INSUFFICIENT_EVIDENCE"}],
        "relationship": "no_relationship",
        "final_answer": "The authorisation remains valid for two weeks.",
    })
    draft = "A Returns Authorisation number is valid for 14 days [OPS-03#002]."
    result = schema.parse(payload, {"OPS-03#002"}, POLICY, draft=draft)

    assert result.final_answer == schema.ABSTENTION_TEXT
    assert "two weeks" not in result.final_answer


def test_the_abstention_text_asserts_nothing():
    """It is served verbatim, so what it says is part of the system."""
    assert not schema.cites_a_passage(schema.ABSTENTION_TEXT)
    assert not schema.asserts_uncited_quantity(schema.ABSTENTION_TEXT)
    assert "does not support an answer" in schema.ABSTENTION_TEXT


def test_an_abstention_may_legitimately_cite_nothing():
    """Withdrawing an unsupported claim leaves nothing to cite.

    This is the case the decitation rule must not block, and it is why the
    rule keys on the verdicts rather than on the citation count alone.
    """
    payload = json.dumps({
        "claims": [{"claim": "valid for 14 days", "verdict": "INSUFFICIENT_EVIDENCE"}],
        "relationship": "no_relationship",
        "final_answer": "The evidence does not state a validity period.",
    })
    draft = "A Returns Authorisation number is valid for 14 days [OPS-03#002]."
    result = schema.parse(payload, {"OPS-03#002"}, POLICY, draft=draft)

    # The finding is respected and the wording is not the model's to choose.
    assert result.final_answer == schema.ABSTENTION_TEXT
    assert result.revised
    assert not result.revision_rejected


def test_a_revision_that_repairs_a_miscitation_is_warranted():
    """Correcting a wrong citation is the layer's job, not a cosmetic rewrite.

    The first version of the warrant rule blocked this, which would have
    stopped the verifier making the one correction it is best placed to make.
    """
    payload = json.dumps({
        "claims": [{"claim": "25 days", "verdict": "SUPPORTED",
                    "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "Employees receive 25 days [HR-01#001].",
    })
    result = schema.parse(
        payload, {"HR-01#001"}, POLICY,
        draft="Employees receive 25 days [ZZ-99#001].",
    )

    assert result.revised
    assert not result.revision_rejected
    assert result.final_answer == "Employees receive 25 days [HR-01#001]."


def test_a_detected_conflict_warrants_a_revision():
    payload = json.dumps({
        "claims": [{"claim": "the stricter limit applies", "verdict": "SUPPORTED",
                    "supporting": ["FIN-02#001"]}],
        "relationship": "stricter_looser",
        "conflicting_chunks": ["FIN-02#001", "OPS-01#003"],
        "final_answer": "Two limits apply; follow the stricter [FIN-02#001].",
    })
    result = schema.parse(
        payload, {"FIN-02#001", "OPS-01#003"}, POLICY, draft="The limit is 5,000.",
    )

    assert result.revised
    assert not result.revision_rejected


# --- every served revision names its warrant ---------------------------------


@pytest.mark.parametrize("payload_extra,expected_warrant,draft", [
    ({"relationship": "supersession",
      "conflicting_chunks": ["HR-13#001", "HR-03#001"]},
     "conflict_detected", "The rate is 40 pence [HR-03#001]."),
    ({"claims": [{"claim": "x", "verdict": "CONTRADICTED",
                  "contradicting": ["HR-13#001"]}]},
     "claim_contradicted", "The rate is 40 pence [HR-03#001]."),
    ({"claims": [{"claim": "x", "verdict": "INSUFFICIENT_EVIDENCE"}]},
     "claim_insufficient", "The rate is 40 pence [HR-03#001]."),
    ({}, "citation_repair", "The rate is 55 pence per mile [ZZ-99#001]."),
])
def test_a_served_revision_always_names_its_warrant(
    payload_extra, expected_warrant, draft
):
    """An unaudited rewrite is the thing the warrant rule exists to prevent.

    "It must have had a reason" is not an audit trail. If the revision was
    served, the record says why.
    """
    payload = {
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-13#001"]}],
        "relationship": "no_relationship",
        "final_answer": "The rate is 55 pence per mile [HR-13#001].",
    }
    payload.update(payload_extra)
    result = schema.parse(
        json.dumps(payload), {"HR-13#001", "HR-03#001"}, POLICY, draft=draft
    )

    assert result.revised, "this case is meant to produce a served revision"
    assert expected_warrant in result.revision_warrant
    assert result.to_dict()["revision_warrant"] == list(result.revision_warrant)


def test_an_unserved_revision_records_no_warrant():
    """Recording one would imply a licence had been exercised."""
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "Reworded but unchallenged [HR-01#001].",
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY,
                          draft="Unchallenged [HR-01#001].")

    assert not result.revised
    assert result.revision_warrant == ()


# --- a narrow warrant does not license a wide change --------------------------


def test_citation_repair_does_not_license_a_prose_rewrite():
    """Otherwise any content change launders through a misplaced identifier.

    The draft cites a chunk that was never retrieved, which warrants repair.
    That licence is to fix the citation, not to replace what the answer says.
    """
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "Leave must be booked six weeks ahead and cannot be "
                        "carried over [HR-01#001].",
    })
    result = schema.parse(
        payload, {"HR-01#001"}, POLICY,
        draft="Full-time employees receive 25 days of leave [ZZ-99#001].",
    )

    assert not result.revised
    assert result.revision_rejected
    assert "citation repair" in result.revision_rejected_reason
    assert result.final_answer.endswith("[ZZ-99#001].")


def test_citation_repair_must_leave_the_claim_exactly_as_it_was():
    """Exact content equality, not a similarity tolerance.

    0.90 permitted some content change with no principle behind the number.
    A model that cannot reproduce the sentence has its revision refused and
    the draft stands, which is what Arm B would have served anyway.
    """
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "Employees receive 25 days of leave each year [HR-01#001].",
    })
    result = schema.parse(
        payload, {"HR-01#001"}, POLICY,
        draft="Employees receive 25 days of leave [ZZ-99#001].",
    )

    assert not result.revised, "'each year' is a content change"
    assert result.revision_rejected
    assert "change the citations rather than the claim" in result.revision_rejected_reason


def test_citation_repair_that_only_moves_the_identifier_is_allowed():
    payload = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "Full-time employees receive 25 days of leave [HR-01#001].",
    })
    result = schema.parse(
        payload, {"HR-01#001"}, POLICY,
        draft="According to [ZZ-99#001], full-time employees receive 25 days of leave.",
    )

    assert result.revised
    assert result.revision_warrant == ("citation_repair",)


def test_a_wide_change_is_allowed_when_a_claim_warrant_supports_it():
    """The containment rule binds only when citation repair is the sole reason."""
    payload = json.dumps({
        "claims": [{"claim": "25 days", "verdict": "CONTRADICTED",
                    "contradicting": ["HR-01#001"]}],
        "relationship": "no_relationship",
        "final_answer": "The entitlement is 28 days, not 25 [HR-01#001].",
    })
    result = schema.parse(
        payload, {"HR-01#001"}, POLICY,
        draft="Employees receive 25 days of paid annual leave [ZZ-99#001].",
    )

    assert result.revised
    assert set(result.revision_warrant) == {"claim_contradicted", "citation_repair"}


def test_prose_similarity_ignores_citations_and_wording():
    same = schema.prose_similarity(
        "According to [HR-01#001], leave is 25 days.",
        "Leave is 25 days [HR-01#001].",
    )
    different = schema.prose_similarity(
        "You may claim 55 pence per mile [HR-13#001].",
        "The answer under review does not address a conflict between passages.",
    )
    assert same == 1.0
    assert different < 0.5
    assert different < schema.CITATION_REPAIR_MIN_SIMILARITY < same


def test_the_measure_is_not_sensitive_to_answer_length():
    """A character measure made the same edit pass or fail on length alone.

    Deleting "According to" scored 0.71 on a one-line answer and 0.96 on a
    paragraph, so a rule with one threshold would have refused the short one
    and allowed the long one for no reason connected to what changed.
    """
    short = schema.prose_similarity(
        "According to [HR-01#001], leave is 25 days.",
        "Leave is 25 days [HR-01#001].",
    )
    long = schema.prose_similarity(
        "According to [HR-01#001], full-time employees receive 25 days of paid "
        "annual leave per leave year, in addition to the 8 English bank holidays.",
        "Full-time employees receive 25 days of paid annual leave per leave "
        "year, in addition to the 8 English bank holidays [HR-01#001].",
    )
    assert short == long == 1.0


def test_the_threshold_separates_the_pilot_02_revisions():
    """The calibration is stated in the source, so it is checked here.

    A threshold fitted to a boundary case would be a different claim from one
    sitting in a gap, so the gap is asserted rather than described.
    """
    observed = [1.000, 1.000, 1.000, 0.696, 0.211, 0.000]
    threshold = schema.CITATION_REPAIR_MIN_SIMILARITY
    above = [v for v in observed if v >= threshold]
    below = [v for v in observed if v < threshold]

    assert above == [1.000, 1.000, 1.000]
    assert below == [0.696, 0.211, 0.000]
    assert min(above) - max(below) > 0.25, "the threshold sits in a wide gap"


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
    assert result.confidence == "medium"
    # The revision is withheld, but for the separate reason that the
    # verification found nothing to fix. No validation failure was recorded,
    # which is what this test is about.
    assert result.final_answer == "draft text"
    assert result.revision_rejected


# --- prompt revision 2, after dev-pilot-01 -----------------------------------


def test_the_relationship_is_asked_for_before_the_claims():
    """Revision 1 put it fifth and got no_relationship 35 times in 41.

    A model that spends its budget on the claim list arrives at the field the
    whole layer exists for as an afterthought.
    """
    from sme_assistant.verify.verifier import VERIFIER_SYSTEM

    assert VERIFIER_SYSTEM.index('"relationship"') < VERIFIER_SYSTEM.index('"claims"')
    assert VERIFIER_SYSTEM.index("STEP 1") < VERIFIER_SYSTEM.index("STEP 3")


def test_the_prompt_does_not_contradict_itself_about_answering():
    """Revision 1 said "your job is NOT to answer" and then asked for an answer.

    The model resolved that by rewriting answers and skipping the audit.
    """
    from sme_assistant.verify.verifier import VERIFIER_SYSTEM

    assert "NOT to answer" not in VERIFIER_SYSTEM


def test_the_examples_use_invented_documents_only():
    """Examples teach the shape without leaking corpus content.

    A worked example drawn from the real corpus would hand the verifier an
    answer it is supposed to derive.
    """
    import re

    from sme_assistant.verify.verifier import VERIFIER_SYSTEM
    from sme_assistant.common.config import load_config
    from sme_assistant.kb.loader import load_knowledge_base

    real = {d.doc_id for d in load_knowledge_base(load_config().path("paths.kb_docs"))}
    cited = set(re.findall(r"\b([A-Z]{2,4}-\d{2})#", VERIFIER_SYSTEM))
    assert cited, "the examples cite no identifiers, so they teach no shape"
    assert not (cited & real), f"the prompt cites real documents: {sorted(cited & real)}"


# --- effective options are recorded, not merely declared ---------------------


def test_generation_records_the_options_it_posted(config):
    """The field existed and nothing wrote to it.

    Every record carried an empty dictionary while the code claimed effective
    options were captured, which is worse than not claiming it.
    """
    client = MockClient(config)
    generation = client.generate("hello", options={"num_predict": 700})

    assert generation.options == client.last_options
    assert generation.to_dict()["options"] == client.last_options
    assert generation.options["num_predict"] == 700
    assert generation.options["num_ctx"] == 4096, (
        "num_ctx reaches the model by inheritance from the generation block"
    )


def test_underscore_metadata_never_reaches_the_model(config):
    """A _note key was being posted to Ollama as a model option."""
    client = MockClient(config)
    generation = client.generate("hello", options={"_note": "documentation"})
    assert not any(k.startswith("_") for k in generation.options)
    assert not any(k.startswith("_") for k in client.last_options)


def test_the_real_client_records_exactly_what_it_posted(config):
    """The recorded options are the posted options, on the real code path.

    MockClient mirrors the merge and filter, which makes the behaviour
    testable without a server but proves nothing about OllamaClient: the two
    implementations are duplicated and could drift apart silently, leaving the
    mock green while real runs recorded something other than what they sent.

    Stubbing the transport rather than the client keeps the whole of
    ``generate`` under test and captures the actual HTTP payload.
    """
    from sme_assistant.common.llm_client import OllamaClient

    client = OllamaClient(config)
    captured: dict[str, Any] = {}

    def fake_post(endpoint: str, payload: dict) -> dict:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"response": " grounded answer ", "prompt_eval_count": 11,
                "eval_count": 5, "prompt_eval_duration": 2_000_000_000,
                "eval_duration": 1_000_000_000, "load_duration": 0}

    client._post = fake_post  # type: ignore[method-assign]
    generation = client.generate(
        "hello", options={"num_predict": 700, "num_ctx": 4096, "_note": "docs"}
    )

    assert captured["endpoint"] == "/api/generate"
    assert generation.options == captured["payload"]["options"]
    assert generation.to_dict()["options"] == captured["payload"]["options"]
    # The filter has to happen before the wire, not merely before the record.
    assert not any(k.startswith("_") for k in captured["payload"]["options"])
    assert captured["payload"]["options"]["num_ctx"] == 4096
    assert captured["payload"]["options"]["seed"] == 42
    assert generation.text == "grounded answer"


def test_the_two_clients_merge_options_identically(config):
    """The mock is only a stand-in if it produces the same merge."""
    from sme_assistant.common.llm_client import OllamaClient

    real = OllamaClient(config)
    captured: dict[str, Any] = {}
    real._post = lambda endpoint, payload: (  # type: ignore[method-assign]
        captured.update(payload) or {"response": "x"}
    )
    overrides = {"temperature": 0.0, "num_predict": 700, "_note": "ignored"}
    real.generate("hello", options=overrides)

    mock = MockClient(config)
    mock.generate("hello", options=overrides)

    assert captured["options"] == mock.last_options


def test_the_verified_record_carries_the_verifier_options(retriever, config):
    retrieval = retriever.retrieve("annual leave", min_similarity=0.0)
    client = MockClient(config)
    answer = Generator(client, config).answer("How much leave?", retrieval)
    payload = Verifier(client, config).verify(answer).to_dict()

    assert payload["verification_options"]["num_ctx"] == 4096
    assert payload["verification_options"]["num_predict"] == 700
    assert not any(k.startswith("_") for k in payload["verification_options"])


# --- pilot 05: a relationship label is not a claim audit ----------------------


def test_a_missing_claims_key_is_a_validation_failure():
    """All 41 pilot 05 responses omitted it and nothing noticed.

    `payload.get("claims") or []` collapsed "audited nothing" together with
    "never performed the audit". They are different failures and only the
    second is a schema violation.
    """
    payload = json.dumps({
        "relationship": "mutually_exclusive",
        "conflicting_chunks": ["HR-13#001", "HR-03#001"],
    })
    result = schema.parse(payload, {"HR-13#001", "HR-03#001"}, POLICY, draft="x")

    assert not result.claim_audit_complete
    assert any("omitted the required claims audit" in f
               for f in result.validation_failures)


def test_an_empty_claims_list_is_not_the_same_as_a_missing_one():
    payload = json.dumps({"claims": [], "relationship": "no_relationship"})
    result = schema.parse(payload, {"HR-13#001"}, POLICY, draft="x")

    assert not result.claim_audit_complete
    assert not any("omitted" in f for f in result.validation_failures), (
        "an explicit empty list is a verifier that audited nothing, not a "
        "response that skipped the audit"
    )


def test_a_revision_is_not_served_on_a_relationship_label_alone():
    """Pilot 05 rewrote three answers with no per-claim finding behind them."""
    payload = json.dumps({
        "relationship": "mutually_exclusive",
        "conflicting_chunks": ["CS-03#001", "OPS-02#001"],
        "final_answer": "The answer under review is incorrect [CS-03#001].",
    })
    draft = "You can return an item within 28 days [CS-03#001]."
    result = schema.parse(payload, {"CS-03#001", "OPS-02#001"}, POLICY, draft=draft)

    assert result.final_answer == draft
    assert result.revision_rejected
    assert "no claim audit" in result.revision_rejected_reason


def test_a_bare_identifier_in_a_revision_is_not_a_citation():
    """The exact pilot 05 answer. cites_a_passage was tested and never wired in.

    "as outlined in both CS-03#001 and OPS-02#001" reads as cited and the
    pipeline recorded zero citations for it, because extract_citations
    requires brackets and the serving check used the bare-identifier pattern.
    """
    payload = json.dumps({
        "claims": [{"claim": "28 days", "verdict": "CONTRADICTED",
                    "contradicting": ["OPS-02#001"]}],
        "relationship": "mutually_exclusive",
        "conflicting_chunks": ["CS-03#001", "OPS-02#001"],
        "final_answer": ("The answer under review is incorrect. The correct "
                         "timeframe depends on the circumstances, as outlined "
                         "in both CS-03#001 and OPS-02#001."),
    })
    draft = "You can return an item within 28 days [CS-03#001]."
    result = schema.parse(payload, {"CS-03#001", "OPS-02#001"}, POLICY, draft=draft)

    assert result.final_answer == draft, "the bare-identifier revision is refused"
    assert "bare identifier" in result.revision_rejected_reason


def test_a_properly_cited_revision_with_an_audit_is_still_served():
    """The rules must reject the bad cases, not every case."""
    payload = json.dumps({
        "claims": [{"claim": "28 days", "verdict": "CONTRADICTED",
                    "contradicting": ["OPS-02#001"]}],
        "relationship": "mutually_exclusive",
        "conflicting_chunks": ["CS-03#001", "OPS-02#001"],
        "final_answer": ("Two live documents disagree: 28 days [CS-03#001] "
                         "and 14 days [OPS-02#001]."),
    })
    draft = "You can return an item within 28 days [CS-03#001]."
    result = schema.parse(payload, {"CS-03#001", "OPS-02#001"}, POLICY, draft=draft)

    assert result.claim_audit_complete
    assert result.revised
    assert not result.revision_rejected


# --- prompt revision 3, the last one -----------------------------------------


def test_every_worked_example_demonstrates_the_claims_audit():
    """The defect that produced pilot 05.

    The schema required `claims` in one place and all three worked examples
    omitted it, so the prompt demonstrated skipping the audit three times and
    required it once. Qwen omitted it on all 41 development calls.
    """
    import re

    from sme_assistant.verify.verifier import VERIFIER_SYSTEM

    blocks = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", VERIFIER_SYSTEM)
    assert blocks, "the prompt shows no JSON at all"
    missing = [b for b in blocks if '"claims"' not in b]
    assert not missing, (
        f"{len(missing)} of {len(blocks)} JSON blocks omit claims; an example "
        "that skips a required field teaches that skipping it is allowed"
    )


def test_the_claims_requirement_is_stated_as_a_rule():
    """Present in the schema is not the same as required in the rules."""
    from sme_assistant.verify.verifier import VERIFIER_SYSTEM

    assert '"claims" is required in every response' in VERIFIER_SYSTEM
    assert "even when you are not rewriting it" in VERIFIER_SYSTEM


def test_revision_3_changed_the_examples_and_nothing_else():
    """The budget bought a defect repair, not a redesign.

    Revision 3 is the last one available. If it had also reworded the steps or
    the relationship definitions, the run that follows would measure a mixture
    of the repair and whatever else was changed, and nothing would attribute
    either.
    """
    from sme_assistant.verify.verifier import VERIFIER_SYSTEM

    for anchor in (
        "STEP 1. Compare the passages with each other, before you look at the answer.",
        "STEP 3. Check each claim in the answer against a named passage.",
        "STEP 4. Only if the answer is wrong or omits a disagreement, rewrite it.",
        "stricter_looser         - both are in force and one is stricter.",
        "Use only identifiers shown in the evidence. Never invent one.",
        "Cite passages, not documents: [AA-11#001], never [AA-11].",
    ):
        assert anchor in VERIFIER_SYSTEM, f"revision 3 disturbed: {anchor!r}"
