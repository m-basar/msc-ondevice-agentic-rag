"""Reparsing must vary the rules and nothing else.

A reparse that quietly changed the verifier's finding would let a validation
amendment look like a model improvement, which is the one thing this tool must
never do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reparse_run import reparse  # noqa: E402


def record(raw, draft, chunks=("HR-13#001", "HR-03#001")):
    return {
        "question_id": "Q1", "group_id": "G", "family_id": "CONF-01",
        "category": "conflict", "draft_answer": draft, "answer": draft,
        "answer_revised": False, "revision_rejected": False,
        "citations": [], "hallucinated_citations": [],
        "has_valid_citation_ids": False,
        "retrieval": {"results": [{"chunk_id": c} for c in chunks]},
        "verification": {"raw": raw, "relationship": "no_relationship",
                         "conflict_detected": False, "parse_failed": False},
    }


def test_the_finding_is_never_altered_by_a_reparse():
    """The validation governs what is done with a finding, not the finding."""
    raw = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-13#001"]}],
        "relationship": "supersession",
        "conflicting_chunks": ["HR-13#001", "HR-03#001"],
    })
    out = reparse(record(raw, "The rate is 55 pence [HR-13#001]."))

    assert out["verification"]["relationship"] == "supersession"
    assert out["verification"]["conflict_detected"] is True


def test_a_bare_identifier_revision_is_withheld_on_reparse():
    """The pilot 05 case, which is the reason this tool exists."""
    raw = json.dumps({
        "claims": [{"claim": "28 days", "verdict": "CONTRADICTED",
                    "contradicting": ["HR-03#001"]}],
        "relationship": "mutually_exclusive",
        "conflicting_chunks": ["HR-13#001", "HR-03#001"],
        "final_answer": "Incorrect, as outlined in HR-13#001 and HR-03#001.",
    })
    draft = "You can return an item within 28 days [HR-13#001]."
    out = reparse(record(raw, draft))

    assert out["answer"] == draft
    assert out["answer_revised"] is False
    assert out["revision_rejected"] is True
    # And the detection still stands.
    assert out["verification"]["conflict_detected"] is True


def test_a_missing_claim_audit_withholds_the_revision():
    raw = json.dumps({
        "relationship": "mutually_exclusive",
        "conflicting_chunks": ["HR-13#001", "HR-03#001"],
        "final_answer": "Two documents disagree [HR-13#001] and [HR-03#001].",
    })
    draft = "The rate is 55 pence [HR-13#001]."
    out = reparse(record(raw, draft))

    assert out["answer"] == draft
    assert out["verification"]["claim_audit_complete"] is False


def test_citation_metrics_are_recomputed_from_the_served_answer():
    """Not carried over from the record, which described the old served text."""
    raw = json.dumps({
        "claims": [{"claim": "x", "verdict": "SUPPORTED",
                    "supporting": ["HR-13#001"]}],
        "relationship": "no_relationship",
    })
    out = reparse(record(raw, "The rate is 55 pence [HR-13#001]."))

    assert out["citations"] == ["HR-13#001"]
    assert out["has_valid_citation_ids"] is True
    assert out["hallucinated_citations"] == []
