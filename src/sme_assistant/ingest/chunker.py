"""Structure-aware chunking of knowledge base documents.

Splitting text into retrievable pieces sounds mechanical. It is not, and three
decisions here determine whether the rest of the pipeline can work at all.

**Tables must not be split.** REG-02 is a retention schedule that is almost
entirely one table. Cut it between the header row and the data rows and you get
a fragment saying "6 years" with nothing to say what it applies to. That
fragment still matches a query about retention, so retrieval succeeds and the
generator reads a number with no referent. A chunk that is retrievable but
meaningless is worse than one that is never retrieved.

**Context must travel with the text.** A chunk reading "55 pence per mile for
the first 10,000 business miles" is unanswerable alone. Fifty-five pence of
what, under which policy, in which version? Each chunk is therefore embedded
with its document title and heading path prepended. This is a cheap
deterministic form of contextual retrieval, and on chunks this short it matters
more than any similarity-function tuning.

**Provenance must survive.** This is the one that decides whether the project
has a contribution. If chunks do not carry ``doc_id``, ``status`` and
``effective_date``, the verification layer cannot tell HR-03 from HR-13, and the
conflicts planted in the corpus become undetectable. The chunker is where that
information is either preserved or thrown away.

One deliberate omission: ``status`` is **not** written into the embedded text.
It would help the model notice a superseded document, but it would also pollute
the vector with a word that has nothing to do with what the passage means.
Status stays as metadata and is surfaced to the generator and verifier
downstream, where it can be reasoned about rather than merely matched.

Prompt length is also the dominant cost on edge hardware: on the Raspberry Pi,
prompt processing runs at roughly 25 tokens per second against 4,400 on the
laptop GPU. Every token this module emits is paid for at that rate, four times
over per query at ``top_k: 4``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterator, Sequence

from ..kb.loader import Document, KnowledgeBase

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
QUOTE_RE = re.compile(r"^\s*>")

# Sentence boundaries. Deliberately conservative: a full stop, question mark or
# exclamation mark, followed by whitespace, followed by something that looks
# like a new sentence. The lookbehind excludes common abbreviations so that
# "e.g. the warehouse" is not treated as two sentences.
ABBREVIATIONS = ("e.g", "i.e", "etc", "Dr", "Mr", "Mrs", "Ms", "No", "vs", "approx")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


class ChunkingError(RuntimeError):
    """Raised when a document cannot be chunked coherently."""


@dataclass(frozen=True)
class Block:
    """One structural unit of Markdown.

    ``atomic`` blocks are never split internally. Tables and lists are atomic
    because a fragment of either is not interpretable on its own.
    """

    kind: str
    text: str
    level: int = 0

    @property
    def atomic(self) -> bool:
        return self.kind in ("table", "list", "code")

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage, carrying everything needed to cite and verify it."""

    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    version: str
    status: str
    effective_date: date
    heading_path: tuple[str, ...]
    ordinal: int
    text: str
    superseded_by: str | None = None
    supersedes: str | None = None
    contains_table: bool = False
    overlap_source: tuple[str, ...] = ()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_current(self) -> bool:
        return self.status == "current"

    @property
    def embedding_text(self) -> str:
        """What is actually embedded, and what is shown to the generator.

        Document title and heading path are prepended so the passage carries
        its own context. Status and dates are deliberately excluded: they are
        metadata for reasoning, not semantics for matching.
        """
        header = self.doc_title
        if self.heading_path:
            header += " > " + " > ".join(self.heading_path)
        return f"{header}\n\n{self.text}"

    @property
    def citation(self) -> str:
        location = " > ".join(self.heading_path) if self.heading_path else "(document body)"
        return f"{self.doc_id} {self.doc_title} (v{self.version}), {location}"

    @property
    def has_carried_text(self) -> bool:
        """Whether this chunk opens with a sentence from the previous section.

        Overlap protects facts that straddle a boundary, but it means the first
        sentence of a chunk may belong to a different section than the chunk's
        heading path describes. A chunk labelled "Accommodation" that opens
        with a sentence about meals would, unmarked, invite the generator to
        cite the wrong section. Since citation accuracy is what this project
        measures, the carried text is marked inline and its true source is
        recorded here.
        """
        return bool(self.overlap_source)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "category": self.category,
            "version": self.version,
            "status": self.status,
            "effective_date": self.effective_date.isoformat(),
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "heading_path": list(self.heading_path),
            "ordinal": self.ordinal,
            "text": self.text,
            "word_count": self.word_count,
            "contains_table": self.contains_table,
            "overlap_source": list(self.overlap_source),
        }


# --- block parsing ----------------------------------------------------------


def split_blocks(body: str) -> list[Block]:
    """Parse Markdown into structural blocks.

    Consecutive table rows become one table block, consecutive list items one
    list block. This is what makes them unsplittable later.
    """
    blocks: list[Block] = []
    buffer: list[str] = []
    buffer_kind: str | None = None

    def flush() -> None:
        nonlocal buffer, buffer_kind
        if buffer and buffer_kind:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(Block(kind=buffer_kind, text=text))
        buffer = []
        buffer_kind = None

    in_code = False
    for raw_line in body.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            if in_code:
                buffer.append(line)
                flush()
                in_code = False
            else:
                flush()
                in_code = True
                buffer_kind = "code"
                buffer.append(line)
            continue

        if in_code:
            buffer.append(line)
            continue

        if not line.strip():
            flush()
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush()
            blocks.append(Block(kind="heading", text=heading.group(2), level=len(heading.group(1))))
            continue

        if TABLE_ROW_RE.match(line):
            kind = "table"
        elif LIST_ITEM_RE.match(line):
            kind = "list"
        elif QUOTE_RE.match(line):
            kind = "quote"
        else:
            kind = "paragraph"

        # A paragraph line directly beneath a list item is a continuation of it.
        if buffer_kind == "list" and kind == "paragraph" and raw_line.startswith((" ", "\t")):
            kind = "list"

        if buffer_kind and kind != buffer_kind:
            flush()
        buffer_kind = kind
        buffer.append(line)

    flush()
    return blocks


def split_sentences(text: str) -> list[str]:
    """Split into sentences, guarding against common abbreviations."""
    protected = text
    for abbreviation in ABBREVIATIONS:
        protected = protected.replace(f"{abbreviation}.", f"{abbreviation}\x00")
    parts = SENTENCE_END_RE.split(protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


# --- chunking ---------------------------------------------------------------

def _heading_path(stack: dict[int, str], skip_level: int = 1) -> tuple[str, ...]:
    """Heading path excluding the document title, which is level 1."""
    return tuple(stack[level] for level in sorted(stack) if level > skip_level)


def _overlap_text(text: str, sentences: int) -> str:
    """Trailing sentences of a chunk, carried into the next one.

    Overlap exists so that a fact split across a boundary survives intact in at
    least one chunk. It is taken from prose only: repeating a table header into
    the next chunk would duplicate rows and distort retrieval scores.
    """
    if sentences <= 0:
        return ""
    found = split_sentences(text)
    return " ".join(found[-sentences:]) if found else ""


def chunk_document(
    doc: Document,
    max_words: int,
    overlap_sentences: int = 1,
    min_words: int = 40,
) -> list[Chunk]:
    """Split one document into chunks.

    Blocks accumulate until adding the next would exceed ``max_words``. Headings
    are preferred break points, because a chunk that begins at a section
    boundary reads as a coherent passage. An atomic block larger than
    ``max_words`` becomes its own oversized chunk rather than being cut: an
    intact large table is useful, half a table is not.
    """
    if max_words <= 0:
        raise ChunkingError("max_words must be positive")

    blocks = split_blocks(doc.body)
    if not blocks:
        raise ChunkingError(f"{doc.doc_id}: document body produced no blocks")

    chunks: list[Chunk] = []
    stack: dict[int, str] = {}
    pending: list[Block] = []
    # Every section heading the accumulating chunk has passed through, not just
    # the one it started under. A chunk that spans "Mileage" and "Meals" must
    # say so: recording only the path at the chunk's start produced chunks that
    # contained the mileage rate under an empty heading path, which is exactly
    # the missing context the contextual header exists to supply.
    pending_sections: list[str] = []
    pending_words = 0
    carry_over = ""
    carry_path: tuple[str, ...] = ()

    def emit() -> None:
        nonlocal pending, pending_words, carry_over, carry_path, pending_sections
        if not pending:
            return
        body_text = "\n\n".join(block.text for block in pending).strip()
        path = tuple(pending_sections) if pending_sections else _heading_path(stack)
        overlap_source: tuple[str, ...] = ()
        if carry_over and carry_path != path:
            # The carried sentence comes from a different section, so say so.
            # Four tokens of marker buys correct attribution.
            source = " > ".join(carry_path)
            body_text = f"(continues from {source}) {carry_over}\n\n{body_text}"
            overlap_source = carry_path
        elif carry_over:
            body_text = f"{carry_over}\n\n{body_text}"
        if not body_text:
            pending, pending_words = [], 0
            return
        ordinal = len(chunks) + 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{ordinal:03d}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                category=doc.category,
                version=doc.version,
                status=doc.status,
                effective_date=doc.effective_date,
                heading_path=path,
                ordinal=ordinal,
                text=body_text,
                superseded_by=doc.superseded_by,
                supersedes=doc.supersedes,
                contains_table=any(b.kind == "table" for b in pending),
                overlap_source=overlap_source,
            )
        )
        # Overlap comes from prose only.
        prose = "\n\n".join(b.text for b in pending if b.kind == "paragraph")
        carry_over = _overlap_text(prose, overlap_sentences)
        carry_path = path
        pending, pending_words = [], 0
        # The next chunk continues under the section the last one ended in.
        pending_sections = pending_sections[-1:] if pending_sections else []

    for block in blocks:
        if block.kind == "heading":
            stack[block.level] = block.text
            for level in [l for l in stack if l > block.level]:
                del stack[level]
            # A top-level section boundary is the cleanest place to break, but
            # only once the current chunk is substantial enough to stand alone.
            if block.level <= 2 and pending_words >= min_words:
                emit()
                pending_sections = []
            if block.level > 1 and block.text not in pending_sections:
                pending_sections.append(block.text)
            continue

        if pending_words + block.word_count > max_words and pending:
            emit()

        if block.atomic and block.word_count > max_words:
            # Oversized and unsplittable. Give it a chunk of its own rather
            # than cutting it into fragments that cannot be interpreted.
            emit()
            pending = [block]
            pending_words = block.word_count
            emit()
            continue

        pending.append(block)
        pending_words += block.word_count

    emit()

    # A short trailing chunk carries little signal and dilutes retrieval.
    if len(chunks) > 1 and chunks[-1].word_count < min_words:
        tail = chunks.pop()
        previous = chunks.pop()
        chunks.append(
            Chunk(
                chunk_id=previous.chunk_id,
                doc_id=previous.doc_id,
                doc_title=previous.doc_title,
                category=previous.category,
                version=previous.version,
                status=previous.status,
                effective_date=previous.effective_date,
                heading_path=previous.heading_path,
                ordinal=previous.ordinal,
                text=f"{previous.text}\n\n{tail.text}",
                superseded_by=previous.superseded_by,
                supersedes=previous.supersedes,
                contains_table=previous.contains_table or tail.contains_table,
                overlap_source=previous.overlap_source,
            )
        )

    return chunks


def chunk_corpus(
    kb: KnowledgeBase,
    max_words: int,
    overlap_sentences: int = 1,
    min_words: int = 40,
) -> list[Chunk]:
    """Chunk every document, in deterministic document-id order."""
    chunks: list[Chunk] = []
    for doc in sorted(kb, key=lambda d: d.doc_id):
        chunks.extend(chunk_document(doc, max_words, overlap_sentences, min_words))
    return chunks


def summarise_chunks(chunks: Sequence[Chunk]) -> dict:
    """Statistics for the methodology chapter, and for tuning against cost.

    ``mean_prompt_words_at_top_k`` is the figure that matters on edge hardware:
    it estimates how much text every query will carry into the model.
    """
    if not chunks:
        return {"chunk_count": 0}
    words = [c.word_count for c in chunks]
    per_doc: dict[str, int] = {}
    for chunk in chunks:
        per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
    oversized = [c.chunk_id for c in chunks if c.word_count > 0 and c.contains_table]
    return {
        "chunk_count": len(chunks),
        "document_count": len(per_doc),
        "chunks_per_document_mean": round(len(chunks) / len(per_doc), 2),
        "chunks_per_document_max": max(per_doc.values()),
        "total_words": sum(words),
        "mean_words": round(sum(words) / len(words), 1),
        "median_words": sorted(words)[len(words) // 2],
        "min_words": min(words),
        "max_words": max(words),
        "chunks_containing_tables": len(oversized),
        "chunks_with_carried_text": sum(1 for c in chunks if c.has_carried_text),
        "current_chunks": sum(1 for c in chunks if c.is_current),
        "superseded_chunks": sum(1 for c in chunks if not c.is_current),
    }
