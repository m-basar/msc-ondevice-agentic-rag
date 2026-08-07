"""Tests for knowledge base loading and corpus validation.

Two kinds of test here. The first group uses temporary files to check that
malformed documents are rejected: these are fast and independent of the real
corpus. The second group loads the actual knowledge base, which guards
against a document being edited into an invalid state later in the project.
"""

from __future__ import annotations

from datetime import date

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.kb.loader import (
    Document,
    KnowledgeBaseError,
    load_document,
    load_knowledge_base,
    parse_front_matter,
)

VALID_DOC = """---
id: TEST-01
title: Test Document
category: TEST
version: 1.0
effective_date: 2026-01-01
status: current
---

# Test Document

The body of the document, which contains eleven words in total here.
"""


def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- front matter parsing ---------------------------------------------------


def test_parses_valid_front_matter(tmp_path):
    doc = load_document(write(tmp_path, "TEST-01.md", VALID_DOC))
    assert doc.doc_id == "TEST-01"
    assert doc.title == "Test Document"
    assert doc.effective_date == date(2026, 1, 1)
    assert doc.is_current
    assert doc.body.startswith("# Test Document")
    assert "---" not in doc.body


def test_missing_front_matter_is_rejected(tmp_path):
    path = write(tmp_path, "bad.md", "# No front matter\n\nJust a body.\n")
    with pytest.raises(KnowledgeBaseError, match="missing front matter"):
        load_document(path)


def test_malformed_front_matter_line_is_rejected(tmp_path):
    text = VALID_DOC.replace("category: TEST", "category TEST")
    path = write(tmp_path, "bad.md", text)
    with pytest.raises(KnowledgeBaseError, match="not 'key: value'"):
        load_document(path)


def test_missing_required_field_is_rejected(tmp_path):
    text = VALID_DOC.replace("version: 1.0\n", "")
    path = write(tmp_path, "bad.md", text)
    with pytest.raises(KnowledgeBaseError, match="missing front matter fields"):
        load_document(path)


def test_non_iso_date_is_rejected(tmp_path):
    text = VALID_DOC.replace("2026-01-01", "01/01/2026")
    path = write(tmp_path, "bad.md", text)
    with pytest.raises(KnowledgeBaseError, match="ISO format"):
        load_document(path)


def test_unknown_status_is_rejected(tmp_path):
    text = VALID_DOC.replace("status: current", "status: maybe")
    path = write(tmp_path, "bad.md", text)
    with pytest.raises(KnowledgeBaseError, match="must be one of"):
        load_document(path)


def test_empty_body_is_rejected(tmp_path):
    text = VALID_DOC.split("# Test Document")[0]
    path = write(tmp_path, "bad.md", text)
    with pytest.raises(KnowledgeBaseError, match="body is empty"):
        load_document(path)


def test_comments_and_blank_lines_in_front_matter_are_ignored():
    text = "---\n# a comment\n\nid: X\ntitle: Y\n---\nbody\n"
    metadata, body = parse_front_matter(text, "memory")
    assert metadata == {"id": "X", "title": "Y"}
    assert body == "body"


# --- corpus-level validation ------------------------------------------------


def test_duplicate_ids_are_rejected(tmp_path):
    write(tmp_path, "a.md", VALID_DOC)
    write(tmp_path, "b.md", VALID_DOC)
    with pytest.raises(KnowledgeBaseError, match="Duplicate document id"):
        load_knowledge_base(tmp_path)


def test_superseded_without_pointer_is_rejected(tmp_path):
    text = VALID_DOC.replace("status: current", "status: superseded")
    write(tmp_path, "a.md", text)
    with pytest.raises(KnowledgeBaseError, match="superseded_by is not set"):
        load_knowledge_base(tmp_path)


def test_superseded_by_unknown_document_is_rejected(tmp_path):
    text = VALID_DOC.replace(
        "status: current", "status: superseded\nsuperseded_by: NOPE-99"
    )
    write(tmp_path, "a.md", text)
    with pytest.raises(KnowledgeBaseError, match="unknown document"):
        load_knowledge_base(tmp_path)


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="No Markdown documents"):
        load_knowledge_base(tmp_path)


def test_missing_directory_is_rejected(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="directory not found"):
        load_knowledge_base(tmp_path / "nowhere")


# --- the real corpus --------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    return load_knowledge_base(load_config().path("paths.kb_docs"))


def test_real_corpus_loads(corpus):
    assert len(corpus) >= 30, "corpus has shrunk unexpectedly"


def test_real_corpus_has_a_superseded_document(corpus):
    """The conflicting-evidence test cases depend on this pair existing."""
    superseded = corpus.superseded()
    assert superseded, "no superseded document; conflict test cases cannot run"
    for doc in superseded:
        replacement = corpus.by_id(doc.superseded_by)
        assert replacement.is_current
        assert replacement.effective_date > doc.effective_date


def test_every_document_has_substantive_content(corpus):
    for doc in corpus:
        assert doc.word_count >= 100, f"{doc.doc_id} is too short to retrieve usefully"


def test_document_ids_match_filenames(corpus):
    for doc in corpus:
        assert doc.path.name.startswith(doc.doc_id), (
            f"{doc.path.name} does not start with its id {doc.doc_id}"
        )


def test_summary_reports_expected_shape(corpus):
    summary = corpus.summary()
    assert summary["document_count"] == len(corpus)
    assert summary["current_count"] + summary["superseded_count"] == len(corpus)
    assert summary["total_words"] > 0
    assert set(summary["categories"]) >= {"HR", "OPS", "IT", "FIN"}


def test_documents_are_immutable(corpus):
    doc = next(iter(corpus))
    with pytest.raises(Exception):
        doc.title = "changed"  # type: ignore[misc]
