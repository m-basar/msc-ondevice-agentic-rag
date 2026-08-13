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


POLICY = {"on_supported": "medium", "on_conflict": "low", "on_insufficient": "low",
          "on_contradiction": "low", "on_invented_evidence": "low",
          "on_parse_failure": "low"}


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


def test_conflicting_chunks_are_validated_too():
    payload = json.dumps({
        "claims": [], "relationship": "mutually_exclusive",
        "conflicting_chunks": ["HR-01#001", "QQ-11#003"],
    })
    result = schema.parse(payload, {"HR-01#001"}, POLICY)
    assert result.conflicting_chunks == ("HR-01#001",)
    assert "QQ-11#003" in result.invented_ids


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
    payload = json.dumps({"claims": [], "relationship": relationship})
    assert schema.parse(payload, set(), POLICY).conflict_detected is detected


# --- confidence comes from runtime policy, not gold --------------------------


def test_a_detected_conflict_caps_confidence():
    payload = json.dumps({"claims": [], "relationship": "mutually_exclusive"})
    assert schema.parse(payload, set(), POLICY).confidence == "low"


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
    for outcome in ("on_conflict", "on_insufficient", "on_supported"):
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
