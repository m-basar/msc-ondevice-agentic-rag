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


def test_sections_are_real_headings_of_their_own_document(chunks, kb):
    """The test the old one should have been.

    Asserting only that a section list is non-empty is satisfied by inventing
    one, which is exactly what the previous implementation did: it accumulated
    every heading a chunk passed through, so a chunk containing only Principle
    content was labelled Principle > Schedule. Correctness means every named
    section is a genuine heading of that document.
    """
    for chunk in chunks:
        headings = {
            line.lstrip("#").strip()
            for line in kb.by_id(chunk.doc_id).body.splitlines()
            if line.startswith("##")
        }
        for section in chunk.sections:
            assert section in headings, (
                f"{chunk.chunk_id} claims section {section!r}, which is not a "
                f"heading in {chunk.doc_id}"
            )


def test_a_chunk_only_names_sections_whose_content_it_contains(chunks, kb):
    """REG-02 is the case that exposed the original bug.

    Its four sections are Principle, Schedule, Disposal and Review. A chunk
    holding only the Principle paragraph must not claim Schedule.
    """
    doc = kb.by_id("REG-02")
    body = doc.body
    for chunk in [c for c in chunks if c.doc_id == "REG-02"]:
        for section in chunk.sections:
            start = body.index(f"## {section}")
            following = [
                body.index(f"## {s}") for s in ("Principle", "Schedule", "Disposal", "Review")
                if f"## {s}" in body and body.index(f"## {s}") > start
            ]
            end = min(following) if following else len(body)
            section_body = body[start:end]
            words = [w for w in section_body.split() if len(w) > 4][:40]
            assert any(w in chunk.text for w in words), (
                f"{chunk.chunk_id} names section {section!r} but contains none "
                "of that section's text"
            )


def test_sections_are_siblings_not_a_hierarchy(chunks):
    """The corpus uses only H1 and H2, so no nesting exists to express.

    Rendering two adjacent sections with " > " implied one was inside the
    other. The separator must say "and", not "within".
    """
    for chunk in chunks:
        if len(chunk.sections) > 1:
            header = chunk.embedding_text.split("\n")[0]
            assert "; ".join(chunk.sections) in header
            assert f"{chunk.sections[0]} > {chunk.sections[1]}" not in header


def test_overlap_never_crosses_a_section_boundary(chunks):
    """Carried text belongs to the section it came from."""
    by_id = {c.chunk_id: c for c in chunks}
    for chunk in chunks:
        source_id = chunk.overlap_source_chunk_id
        if not source_id:
            continue
        source = by_id[source_id]
        assert set(source.sections) & set(chunk.sections), (
            f"{chunk.chunk_id} {chunk.sections} carries text from "
            f"{source_id} {source.sections}, a different section"
        )


def test_almost_no_chunk_is_too_small_to_retrieve(chunks, config):
    """One section per chunk is correct but produces unusable fragments.

    Splitting strictly at every section boundary gave 45% of chunks under 40
    words, including 10-word chunks. Adjacent short sections may therefore
    share a chunk, provided both are named.
    """
    minimum = config.require("chunking.min_words")
    tiny = [c.chunk_id for c in chunks if c.word_count < minimum]
    assert len(tiny) / len(chunks) < 0.05, (
        f"{len(tiny)} of {len(chunks)} chunks are below {minimum} words"
    )


def test_embedding_text_prepends_context(chunks):
    for chunk in chunks[:20]:
        text = chunk.embedding_text
        assert text.startswith(chunk.doc_title)
        assert chunk.sections[0] in text.split("\n")[0]
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


# --- citation attribution ---------------------------------------------------


def test_carried_text_is_never_silently_misattributed(chunks):
    """The failure this guards against is a wrong citation, not a wrong answer.

    Overlap carries the last sentence of one chunk into the next. Where the two
    chunks sit in different sections, the carried sentence would otherwise be
    presented under a heading path it does not belong to, and the generator
    would cite the wrong section for it. Every such chunk must declare the
    true source both in metadata and inline in the text the model sees.
    """
    for chunk in chunks:
        if not chunk.overlap_source:
            continue
        assert chunk.overlap_source != chunk.sections, (
            f"{chunk.chunk_id} records an overlap source identical to its own "
            "section, which should not have been marked"
        )
        marker = f"(continues from {' > '.join(chunk.overlap_source)})"
        assert chunk.text.startswith(marker), (
            f"{chunk.chunk_id} carries text from {chunk.overlap_source} but does "
            "not say so in the text the model reads"
        )


def test_unmarked_chunks_start_with_their_own_content(chunks, kb):
    """A chunk without an overlap marker must open with text from its own section.

    This is the converse check: it catches carried text that was not marked,
    which is the silent failure the marker exists to prevent.
    """
    by_doc = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    for doc_id, doc_chunks in by_doc.items():
        for previous, current in zip(doc_chunks, doc_chunks[1:]):
            if current.overlap_source:
                continue
            if previous.sections == current.sections:
                continue
            first_sentence = split_sentences(current.text)[0] if current.text else ""
            assert first_sentence not in previous.text, (
                f"{current.chunk_id} opens with a sentence from {previous.chunk_id}, "
                f"which sits in a different section, but carries no marker"
            )


def test_overlap_can_be_disabled(kb, config):
    """Overlap is a configurable choice, so it must be switchable off cleanly."""
    without = chunk_corpus(kb, config.require("chunking.max_words"), overlap_sentences=0)
    assert all(not c.overlap_source for c in without)
    assert all("(continues from" not in c.text for c in without)


def test_overlap_has_no_effect_on_this_corpus(kb, config):
    """A documented finding, not a bug.

    Overlap exists to protect a fact that straddles a chunk boundary. Since the
    rewrite, boundaries fall at section headings rather than at arbitrary word
    counts, so a boundary is a semantic break rather than a cut through prose.
    Overlap now applies only when a single section is longer than ``max_words``
    and must be split internally.

    No section in this corpus is. The largest is REG-02's Schedule at 197
    words, and that is an atomic table which is never split. Every other
    section fits inside the 180-word limit.

    ``chunking.overlap_sentences`` therefore affects zero of 133 chunks and is
    not a meaningful experimental variable for this study. The mechanism is
    retained because a corpus with longer sections would need it, and the test
    below proves it still works.
    """
    baseline = chunk_corpus(kb, 180, 0, 40)
    with_overlap = chunk_corpus(kb, 180, 2, 40)
    assert [c.text for c in baseline] == [c.text for c in with_overlap]
    assert all(not c.overlap_source_chunk_id for c in with_overlap)


def test_overlap_still_works_when_a_section_is_long_enough_to_split(kb):
    """Guards the mechanism against silently rotting while unused."""
    from dataclasses import replace

    long_section = "\n\n".join(
        f"Sentence number {i} describes a distinct operational requirement in detail."
        for i in range(1, 61)
    )
    doc = replace(kb.by_id("HR-01"), body=f"# Title\n\n## One Long Section\n\n{long_section}\n")

    produced = chunk_document(doc, max_words=60, overlap_sentences=1, min_words=20)
    assert len(produced) > 1, "the synthetic section should have split"
    carried = [c for c in produced if c.overlap_source_chunk_id]
    assert carried, "overlap did not fire on a section that was split internally"
    for chunk in carried:
        assert chunk.text.startswith("(continues from ")
        assert chunk.sections == ("One Long Section",)


# --- the short-tail merge guard ---------------------------------------------
# This guard read `if True` for several commits. The behaviour it produced was
# correct for the current corpus, but nothing enforced it.


def _doc(tmp_path, body: str, doc_id: str = "TST-01"):
    from sme_assistant.kb.loader import load_document

    text = (
        f"---\nid: {doc_id}\ntitle: T\ncategory: TST\nversion: 1.0\n"
        f"effective_date: 2026-01-01\nstatus: current\n---\n\n{body}\n"
    )
    path = tmp_path / f"{doc_id}.md"
    path.write_text(text, encoding="utf-8")
    return load_document(path)


def test_a_short_tail_is_absorbed_into_the_preceding_chunk(tmp_path):
    """The behaviour the dead guard happened to produce, now enforced."""
    from sme_assistant.ingest.chunker import chunk_document

    body = "# T\n\n## One\n\n" + ("alpha " * 100) + "\n\n## Two\n\nshort trailing note.\n"
    chunks = chunk_document(_doc(tmp_path, body), max_words=180, min_words=40)

    assert len(chunks) == 1, "the short tail should not stand as its own chunk"
    assert "One" in chunks[0].sections and "Two" in chunks[0].sections, (
        "the merged chunk must name both sections, or its text is attributed "
        "to a section it did not come from"
    )


def test_a_short_tail_is_not_merged_when_the_result_would_exceed_max_words(tmp_path):
    """The case the dead guard would have let through.

    No document in the current corpus produces it, which is why the missing
    guard was invisible. It is here for the corpus that has not been written.
    """
    from sme_assistant.ingest.chunker import chunk_document

    body = "# T\n\n## One\n\n" + ("alpha " * 179) + "\n\n## Two\n\nshort trailing note.\n"
    chunks = chunk_document(_doc(tmp_path, body), max_words=180, min_words=40)

    assert len(chunks) == 2, (
        "merging would have pushed the chunk past max_words, so the tail must "
        "stand alone even though it is short"
    )
    assert chunks[1].sections == ("Two",)


def test_the_chunk_set_fingerprint_notices_a_section_relabelling(tmp_path):
    """Identical text under a different section name is a different chunk.

    Section names are prepended to the evidence the model sees, so a
    relabelling changes the prompt. The fingerprint hashed only identifiers and
    text, so this was invisible to it.
    """
    from dataclasses import replace

    from sme_assistant.ingest.chunker import chunk_document
    from sme_assistant.ingest.index import chunk_set_fingerprint

    body = "# T\n\n## One\n\n" + ("alpha " * 60) + "\n\n## Two\n\n" + ("beta " * 60) + "\n"
    chunks = chunk_document(_doc(tmp_path, body), max_words=180, min_words=40)
    relabelled = [replace(chunks[0], sections=("Renamed",))] + list(chunks[1:])

    assert chunk_set_fingerprint(chunks) != chunk_set_fingerprint(relabelled), (
        "a section relabelling with unchanged text left the fingerprint identical, "
        "so a stale index would have loaded without complaint"
    )
