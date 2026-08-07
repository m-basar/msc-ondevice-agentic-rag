"""Tests for the conflict registry and its agreement with the corpus.

The point of these tests is to make one specific mistake impossible: an
accidental contradiction between two documents being mistaken for a planted
one, and silently ending up in a gold answer. Every superseded document must
be accounted for, and every topic declared missing must genuinely be missing.
"""

from __future__ import annotations

import json

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.kb.conflicts import (
    ConflictRegistryError,
    load_conflicts,
    validate_against_corpus,
)
from sme_assistant.kb.loader import load_knowledge_base


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def kb(config):
    return load_knowledge_base(config.path("paths.kb_docs"))


@pytest.fixture(scope="module")
def registry(config):
    return load_conflicts(config.path("paths.conflicts"))


# --- the registry itself ----------------------------------------------------


def test_registry_loads(registry):
    assert len(registry) >= 4, "too few conflict families to report a rate"


def test_every_family_declares_facts_and_expectations(registry):
    for family in registry.families:
        assert family.conflicting_facts, f"{family.family_id} declares no facts"
        assert family.must_cite, f"{family.family_id} declares no required citation"
        for fact in family.conflicting_facts:
            assert set(fact) == {"fact", "superseded_value", "current_value"}
            assert fact["superseded_value"] != fact["current_value"]


def test_families_have_unique_ids(registry):
    ids = [f.family_id for f in registry.families]
    assert len(ids) == len(set(ids))


def test_families_span_more_than_one_domain(registry):
    """A single-domain conflict set would confound conflict type with subject."""
    assert len({f.domain for f in registry.families}) >= 3


def test_at_least_one_high_risk_family(registry):
    assert any(f.risk_level == "high" for f in registry.families)


def test_lookup_by_document(registry):
    family = registry.for_document("HR-03")
    assert family is not None
    assert family.current_document == "HR-13"
    assert registry.for_document("GEN-01") is None


def test_expected_answer_policy_is_declared(registry):
    policy = registry.expected_answer_policy
    for key in ("answer_with", "cite", "flag", "confidence_ceiling", "rationale"):
        assert key in policy, f"expected_answer_policy is missing {key!r}"


# --- registry against corpus ------------------------------------------------


def test_registry_matches_corpus(registry, kb):
    validate_against_corpus(registry, kb)


def test_every_superseded_document_is_registered(registry, kb):
    """The check that stops an accidental contradiction becoming a test case."""
    registered = {d for f in registry.families for d in f.documents}
    for doc in kb.superseded():
        assert doc.doc_id in registered, (
            f"{doc.doc_id} is superseded but no conflict family accounts for it"
        )


def test_declared_gaps_are_genuinely_absent(registry, kb):
    """Guards the unanswerable question set against corpus drift."""
    validate_against_corpus(registry, kb)  # gap probes run inside
    assert len(registry.fully_absent_topics) >= 5


def test_unregistered_superseded_document_is_rejected(registry, kb, monkeypatch):
    """Simulate the failure this module exists to prevent."""
    trimmed = type(registry)(
        {
            "families": [],
            "expected_answer_policy": registry.expected_answer_policy,
            "deliberate_gaps": {"fully_absent": [], "partially_present": []},
        },
        registry.source,
    )
    with pytest.raises(ConflictRegistryError, match="not declared in the conflict registry"):
        validate_against_corpus(trimmed, kb)


# --- corpus fingerprinting --------------------------------------------------


def test_corpus_fingerprint_is_stable(config):
    a = load_knowledge_base(config.path("paths.kb_docs")).fingerprint()
    b = load_knowledge_base(config.path("paths.kb_docs")).fingerprint()
    assert a == b


def test_manifest_records_every_document(kb):
    manifest = kb.manifest()
    assert manifest["document_count"] == len(kb)
    assert len(manifest["documents"]) == len(kb)
    for entry in manifest["documents"]:
        assert len(entry["sha256"]) == 64
    assert json.dumps(manifest)  # must be serialisable into a results file


def test_summary_reports_superseded_share(kb):
    """The share must be reported, because it is a limitation on external validity."""
    summary = kb.summary()
    assert "superseded_share" in summary
    assert 0 < summary["superseded_share"] < 0.25


# --- loader hardening -------------------------------------------------------


def test_duplicate_front_matter_key_is_rejected(tmp_path):
    from sme_assistant.kb.loader import KnowledgeBaseError, load_document

    text = (
        "---\nid: TEST-01\ntitle: A\ncategory: T\nversion: 1.0\n"
        "effective_date: 2026-01-01\nstatus: current\nstatus: superseded\n---\n\nBody text.\n"
    )
    path = tmp_path / "a.md"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="duplicate front matter key"):
        load_document(path)


def test_malformed_document_id_is_rejected(tmp_path):
    from sme_assistant.kb.loader import KnowledgeBaseError, load_document

    text = (
        "---\nid: hr1\ntitle: A\ncategory: T\nversion: 1.0\n"
        "effective_date: 2026-01-01\nstatus: current\n---\n\nBody text.\n"
    )
    path = tmp_path / "a.md"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="must match CATEGORY-NN"):
        load_document(path)


def test_half_wired_supersession_is_rejected(tmp_path):
    from sme_assistant.kb.loader import KnowledgeBaseError, load_knowledge_base

    old = (
        "---\nid: TST-01\ntitle: A\ncategory: TST\nversion: 1.0\n"
        "effective_date: 2025-01-01\nstatus: superseded\nsuperseded_by: TST-02\n---\n\nOld body.\n"
    )
    new = (
        "---\nid: TST-02\ntitle: A\ncategory: TST\nversion: 2.0\n"
        "effective_date: 2026-01-01\nstatus: current\n---\n\nNew body.\n"
    )
    (tmp_path / "a.md").write_text(old, encoding="utf-8")
    (tmp_path / "b.md").write_text(new, encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="does not declare 'supersedes"):
        load_knowledge_base(tmp_path)


def test_unresolvable_cross_reference_is_rejected(tmp_path):
    from sme_assistant.kb.loader import KnowledgeBaseError, load_knowledge_base

    text = (
        "---\nid: TST-01\ntitle: A\ncategory: TST\nversion: 1.0\n"
        "effective_date: 2026-01-01\nstatus: current\n---\n\n"
        "For details of the escalation route see TST-99.\n"
    )
    (tmp_path / "a.md").write_text(text, encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="unknown document 'TST-99'"):
        load_knowledge_base(tmp_path)


def test_metadata_mapping_is_read_only(kb):
    doc = next(iter(kb))
    with pytest.raises(TypeError):
        doc.metadata["injected"] = "value"  # type: ignore[index]


def test_document_collection_is_a_tuple(kb):
    assert isinstance(kb.documents, tuple)
