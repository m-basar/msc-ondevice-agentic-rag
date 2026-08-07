"""Tests for structure-aware chunking.

Three properties matter more than the rest, and each has a specific failure it
prevents:

1. **Tables survive intact.** A split table produces a fragment that retrieval
   still matches but the generator cannot interpret.
2. **Provenance survives.** Without doc_id, status and effective_date on every
   chunk, the verification layer cannot distinguish a superseded policy from
   its replacement, and the planted conflicts become undetectable.
3. **Content survives.** Chunking must lose nothing. A dropped sentence is an
   answer the system can never give, and it would look like a model failure.
"""

from __future__ import annotations

import re

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.ingest.chunker import (
    Chunk,
    ChunkingError,
    chunk_corpus,
    chunk_document,
    split_blocks,
    split_sentences,
    summarise_chunks,
)
from sme_assistant.kb.loader import load_knowledge_base


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def kb(config):
    return load_knowledge_base(config.path("paths.kb_docs"))


@pytest.fixture(scope="module")
def chunks(kb, config):
    return chunk_corpus(
        kb,
        config.require("chunking.max_words"),
        config.require("chunking.overlap_sentences"),
        config.require("chunking.min_words"),
    )


# --- block parsing ----------------------------------------------------------


def test_headings_become_their_own_blocks():
    blocks = split_blocks("# Title\n\nSome prose.\n\n## Section\n\nMore prose.\n")
    kinds = [b.kind for b in blocks]
    assert kinds == ["heading", "paragraph", "heading", "paragraph"]
    assert blocks[0].level == 1
    assert blocks[2].level == 2
    assert blocks[2].text == "Section"


def test_consecutive_table_rows_form_one_block():
    body = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    blocks = split_blocks(body)
    assert len(blocks) == 1
    assert blocks[0].kind == "table"
    assert blocks[0].atomic
    assert blocks[0].text.count("\n") == 3


def test_consecutive_list_items_form_one_block():
    blocks = split_blocks("- one\n- two\n- three\n")
    assert len(blocks) == 1
    assert blocks[0].kind == "list"
    assert blocks[0].atomic


def test_paragraph_is_not_atomic():
    blocks = split_blocks("Just some prose here.\n")
    assert blocks[0].kind == "paragraph"
    assert not blocks[0].atomic


def test_sentence_splitting_respects_abbreviations():
    text = "Report it within 24 hours, e.g. by phone. Then confirm in writing."
    assert split_sentences(text) == [
        "Report it within 24 hours, e.g. by phone.",
        "Then confirm in writing.",
    ]


# --- the property that protects tables --------------------------------------


def test_tables_are_never_split(chunks, kb):
    """Every table row from a document must sit in exactly one chunk.

    Counting rows per document across chunks catches the failure directly: if a
    table were split, its rows would be distributed over two chunks and at least
    one would be missing its header.
    """
    for doc in kb:
        doc_rows = [ln for ln in doc.body.splitlines() if ln.strip().startswith("|")]
        if not doc_rows:
            continue
        doc_chunks = [c for c in chunks if c.doc_id == doc.doc_id]
        table_chunks = [c for c in doc_chunks if c.contains_table]
        chunk_rows = [
            ln for c in table_chunks for ln in c.text.splitlines() if ln.strip().startswith("|")
        ]
        assert len(chunk_rows) >= len(doc_rows), (
            f"{doc.doc_id}: {len(doc_rows)} table rows in the document but only "
            f"{len(chunk_rows)} across chunks. A table has been split or lost."
        )


def test_a_table_chunk_retains_its_header_row(chunks):
    """A data row without its header row is uninterpretable."""
    for chunk in chunks:
        if not chunk.contains_table:
            continue
        rows = [ln for ln in chunk.text.splitlines() if ln.strip().startswith("|")]
        if len(rows) < 2:
            continue
        separators = [r for r in rows if re.match(r"^\s*\|[\s:|-]+\|\s*$", r)]
        assert separators, (
            f"{chunk.chunk_id} contains table rows but no header separator, "
            "so the header was left in a different chunk"
        )


def test_oversized_table_gets_its_own_chunk(kb, config):
    """REG-02's retention schedule exceeds max_words and must stay whole."""
    doc = kb.by_id("REG-02")
    produced = chunk_document(doc, config.require("chunking.max_words"))
    table_chunks = [c for c in produced if c.contains_table]
    assert len(table_chunks) == 1
    assert table_chunks[0].word_count > config.require("chunking.max_words"), (
        "the oversized table should be allowed to exceed max_words rather than be cut"
    )


# --- the property that protects the contribution ----------------------------


def test_every_chunk_carries_full_provenance(chunks):
    for chunk in chunks:
        assert chunk.doc_id
        assert chunk.doc_title
        assert chunk.category
        assert chunk.version
        assert chunk.status in ("current", "superseded")
        assert chunk.effective_date is not None


def test_superseded_chunks_carry_the_replacement_pointer(chunks, kb):
    """Without this the verifier cannot resolve a version conflict."""
    superseded_ids = {d.doc_id for d in kb.superseded()}
    assert superseded_ids, "corpus has no superseded documents to check"
    for chunk in chunks:
        if chunk.doc_id in superseded_ids:
            assert chunk.status == "superseded"
            assert chunk.superseded_by, (
                f"{chunk.chunk_id} comes from a superseded document but does not "
                "name its replacement"
            )


def test_conflicting_chunks_are_distinguishable(chunks):
    """The mileage conflict must be resolvable from chunk metadata alone."""
    old = [c for c in chunks if c.doc_id == "HR-03" and "40 pence per mile" in c.text]
    new = [c for c in chunks if c.doc_id == "HR-13" and "55 pence per mile" in c.text]
    assert old and new, "the planted mileage conflict did not survive chunking"
    assert old[0].status == "superseded"
    assert new[0].status == "current"
    assert old[0].superseded_by == "HR-13"
    assert new[0].effective_date > old[0].effective_date


def test_chunk_ids_are_unique_and_ordered(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for chunk in chunks:
        assert chunk.chunk_id.startswith(f"{chunk.doc_id}#")
        assert int(chunk.chunk_id.split("#")[1]) == chunk.ordinal


def test_chunking_is_deterministic(kb, config):
    a = chunk_corpus(kb, config.require("chunking.max_words"))
    b = chunk_corpus(kb, config.require("chunking.max_words"))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


# --- the property that protects recall --------------------------------------


def test_no_content_is_lost(chunks, kb):
    """Every non-heading line of every document must appear in some chunk.

    A dropped line is an answer the system can never produce, and it would
    present as a model failure rather than a pipeline bug.
    """
    for doc in kb:
        combined = "\n".join(c.text for c in chunks if c.doc_id == doc.doc_id)
        for line in doc.body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert stripped in combined, (
                f"{doc.doc_id}: line lost during chunking -> {stripped[:70]!r}"
            )


def test_every_chunk_has_a_heading_path(chunks):
    """The contextual header is worthless if the path is empty."""
    missing = [c.chunk_id for c in chunks if not c.heading_path]
    assert not missing, f"chunks with no section context: {missing[:5]}"


def test_embedding_text_prepends_context(chunks):
    for chunk in chunks[:20]:
        text = chunk.embedding_text
        assert text.startswith(chunk.doc_title)
        assert chunk.heading_path[0] in text.split("\n")[0]
        assert chunk.text in text


def test_status_is_not_embedded(chunks):
    """Status is metadata for reasoning, not a token for matching."""
    for chunk in chunks:
        if chunk.status != "superseded":
            continue
        first_line = chunk.embedding_text.split("\n")[0]
        assert "superseded" not in first_line.lower()


# --- size behaviour ---------------------------------------------------------


def test_most_chunks_respect_max_words(chunks, config):
    limit = config.require("chunking.max_words")
    oversized = [c for c in chunks if c.word_count > limit]
    for chunk in oversized:
        assert chunk.contains_table, (
            f"{chunk.chunk_id} exceeds max_words at {chunk.word_count} without "
            "containing a table, so it should have been split"
        )
    assert len(oversized) / len(chunks) < 0.1


def test_summary_reports_what_chapter_three_needs(chunks):
    summary = summarise_chunks(chunks)
    for key in ("chunk_count", "mean_words", "median_words", "max_words",
                "current_chunks", "superseded_chunks"):
        assert key in summary
    assert summary["current_chunks"] + summary["superseded_chunks"] == summary["chunk_count"]


def test_empty_document_body_is_rejected(kb, config):
    from dataclasses import replace

    doc = replace(next(iter(kb)), body="")
    with pytest.raises(ChunkingError, match="no blocks"):
        chunk_document(doc, config.require("chunking.max_words"))


def test_invalid_max_words_is_rejected(kb):
    with pytest.raises(ChunkingError, match="max_words must be positive"):
        chunk_document(next(iter(kb)), 0)
