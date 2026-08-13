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
from sme_assistant.evaluation.config import load_evaluation_config
from sme_assistant.evaluation.conflicts import (
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
    return load_conflicts(load_evaluation_config().path("conflicts"))


# --- the registry itself ----------------------------------------------------


def test_registry_loads(registry):
    assert len(registry) >= 4, "too few conflict families to report a rate"


def test_every_family_declares_facts_and_expectations(registry):
    for family in registry.families:
        assert family.conflicting_facts, f"{family.family_id} declares no facts"
        assert family.must_cite, f"{family.family_id} declares no required citation"
        for fact in family.conflicting_facts:
            assert set(fact.values) == set(fact.anchors)
            assert set(fact.values) <= set(family.documents)
            distinct = {v.strip().lower() for v in fact.values.values()}
            assert len(distinct) >= 2, f"{family.family_id}: {fact.fact} has no disagreement"


def test_families_have_unique_ids(registry):
    ids = [f.family_id for f in registry.families]
    assert len(ids) == len(set(ids))


def test_families_span_more_than_one_domain(registry):
    """A single-domain conflict set would confound conflict type with subject."""
    assert len({f.domain for f in registry.families}) >= 3


def test_at_least_one_high_risk_family(registry):
    assert any(f.risk_level == "high" for f in registry.families)


def test_registry_contains_conflicts_a_metadata_filter_cannot_solve(registry):
    """The finding that justifies the whole contribution.

    Every version_supersession family can be resolved by filtering on the
    status field, with no reasoning about claims at all. If the registry
    contained only those, an examiner could reasonably ask why the
    verification layer exists. At least two current_current families must be
    present, where both documents are live and no metadata distinguishes them.
    """
    unresolvable = registry.of_type("mutually_exclusive")
    assert len(unresolvable) >= 4, (
        "Fewer than four mutually_exclusive families. Every conflict would be "
        "solvable by a three-line metadata filter, and the verification layer "
        "would have nothing to demonstrate."
    )
    for family in unresolvable:
        assert family.authoritative is None
        assert family.resolution == "escalate_unresolved"
        assert not family.is_filter_resolvable

    # Negative controls. Without them nothing measures over-detection, and a
    # verifier that shouted "conflict" at every document pair would score
    # perfectly on every other family.
    controls = registry.of_type("compatible")
    assert len(controls) >= 3, (
        "Fewer than three compatible families. Nothing would measure the "
        "false-conflict rate, which is the failure a conflict detector most "
        "needs to avoid."
    )
    for family in controls:
        assert not family.is_conflict
        assert family.reconciliation, f"{family.family_id} does not say why it is not a conflict"
        assert family.reconciliation_anchors, (
            f"{family.family_id}: the reconciliation is asserted but not evidenced "
            "in the corpus, which makes it a trick question rather than a fair test"
        )

    # An action satisfies both, but a naive answer quoting the looser figure is
    # still unsafe, so these need their own behaviour.
    for family in registry.of_type("stricter_looser"):
        assert family.stricter in family.documents
        # The check four misclassifications would have failed. If the action can
        # be written, the family is not mutually exclusive.
        assert family.satisfying_action, family.family_id
    for family in unresolvable:
        assert family.no_satisfying_action_because, family.family_id
        assert not family.satisfying_action, family.family_id


def test_current_current_families_involve_only_live_documents(registry, kb):
    for family in registry.families:
        if family.conflict_type == "version_supersession":
            continue
        for doc_id in family.documents:
            assert kb.by_id(doc_id).is_current, (
                f"{family.family_id}: {doc_id} is not current, so a status filter "
                "would resolve this conflict after all"
            )


def test_lookup_by_document(registry, kb):
    families = registry.for_document("HR-03")
    assert len(families) == 1
    assert families[0].authoritative == "HR-13"

    # A document can be involved in more than one disagreement, which is
    # realistic. The lookup returns all of them rather than the first.
    assert len(registry.for_document("OPS-05")) >= 2
    # A document in no family at all. GEN-01 was used here until it joined
    # TUNE-05, which is why this now derives rather than hard-codes.
    unused = {d.doc_id for d in kb} - {
        doc for family in registry.all_families for doc in family.documents
    }
    assert unused, "every document is in a family, so this check is vacuous"
    assert registry.for_document(sorted(unused)[0]) == []


def test_confidence_policy_covers_every_outcome(registry):
    policy = registry.confidence_policy
    for outcome in ("resolved_supersession", "unresolved_conflict", "unanswerable"):
        assert outcome in policy, f"confidence_policy is missing {outcome!r}"
        assert "rationale" in policy[outcome]
    assert policy["unresolved_conflict"]["confidence"] == "capped at low"
    assert "high" in policy["resolved_supersession"]["confidence"], (
        "A conflict the system genuinely resolved should be allowed high confidence, "
        "otherwise the confidence signal carries no information"
    )


# --- registry against corpus ------------------------------------------------


def test_registry_matches_corpus(registry, kb):
    validate_against_corpus(registry, kb)


def test_every_superseded_document_is_registered(registry, kb):
    """The check that stops an accidental contradiction becoming a test case."""
    registered = {d for f in registry.all_families for d in f.documents}
    for doc in kb.superseded():
        assert doc.doc_id in registered, (
            f"{doc.doc_id} is superseded but no conflict family accounts for it"
        )


def test_declared_gaps_are_genuinely_absent(registry, kb):
    """Guards the unanswerable question set against corpus drift."""
    validate_against_corpus(registry, kb)  # gap probes run inside
    assert len(registry.fully_absent) >= 5


def test_partial_gaps_have_their_near_miss_evidence(registry, kb):
    """A partial gap needs a document that looks relevant and is not.

    If the near-miss text disappears the question stops being partial and
    becomes plainly unanswerable, which is a different and easier category.
    """
    for partial in registry.partially_present:
        body = kb.by_id(partial.mentioned_in).body.lower()
        assert partial.must_contain.lower() in body
        assert partial.missing_detail_probes, (
            f"{partial.topic}: no probes, so nothing verifies the detail is still missing"
        )


def test_anchors_are_present_in_the_documents(registry, kb):
    """Every declared conflicting value must be evidenced by literal text."""
    for family in registry.families:
        for fact in family.conflicting_facts:
            for doc_id, anchor in fact.anchors.items():
                assert anchor in kb.by_id(doc_id).body, (
                    f"{family.family_id}: {doc_id} does not contain {anchor!r}"
                )


def test_unregistered_superseded_document_is_rejected(registry, kb, monkeypatch):
    """Simulate the failure this module exists to prevent."""
    trimmed = type(registry)(
        {
            "families": [],
            "confidence_policy": registry.confidence_policy,
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


def test_fingerprints_are_full_length_sha256(kb, registry, config):
    """Truncated hashes are for terminals, not for reproducibility records."""
    assert len(kb.fingerprint()) == 64
    assert len(registry.fingerprint()) == 64
    assert len(config.fingerprint()) == 64
    assert len(kb.short_fingerprint()) == 12


def test_manifest_records_every_document(kb):
    manifest = kb.manifest()
    assert manifest["document_count"] == len(kb)
    assert len(manifest["corpus_sha256"]) == 64
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
