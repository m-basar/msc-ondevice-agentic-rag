"""Building, saving and loading the searchable index.

The index is chunks plus their embedding vectors, written to a single JSON
file. No vector database: with 147 chunks a linear scan over in-memory floats
takes well under a millisecond, and a database would add a dependency, a
service to run on the Raspberry Pi, and a component whose behaviour would have
to be described and defended in the methodology. The cost of the simple choice
is that it stops scaling somewhere in the tens of thousands of chunks, which is
two orders of magnitude beyond this corpus and worth stating as a limitation
rather than engineering around.

Staleness is the failure this module exists to prevent. An index built from an
earlier version of the corpus produces answers citing text that no longer
exists, and nothing about the output looks wrong. So the index records the
corpus fingerprint, the embedding model, the chunking parameters and the
backend that produced it, and refuses to load against a corpus it does not
match unless the caller explicitly overrides.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..common.config import Config, load_config
from ..common.llm_client import LLMClient
from ..kb.loader import KnowledgeBase
from .chunker import Chunk, chunk_corpus

SCHEMA_VERSION = "2.0"


def chunk_set_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Hash of the chunks themselves, not of the corpus they came from.

    This is the guard the corpus fingerprint could not provide. Rewriting the
    chunker changed every chunk in the corpus while leaving the documents
    untouched, so ``corpus_sha256`` was identical before and after and a stale
    index would have loaded without complaint: retrieval would have returned
    text under identifiers that no longer referred to it, and the citations in
    every answer would have pointed at the wrong passages.

    Hashing the chunk identifiers and their text subsumes the corpus, the
    chunking parameters and the chunker's behaviour in a single value. If the
    chunks differ for any reason at all, this differs.

    Sections are hashed as well as text. They were not, originally, which left
    one relabelling invisible: a chunk whose text is unchanged but whose
    ``sections`` tuple is different is a different chunk for every purpose that
    matters. Section names are prepended to the evidence block the model sees,
    so they change the prompt; and the attribution bug that motivated the
    chunker rewrite was a section-labelling bug with identical text. A
    fingerprint that could not see it was blind to exactly the failure it was
    added for.
    """
    canonical = "\n".join(
        f"{c.chunk_id}\x00{'; '.join(c.sections)}\x00{c.text}"
        for c in sorted(chunks, key=lambda c: c.chunk_id)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def index_path_for(config: "Config", *, mock: bool = False) -> "Path":
    """Where the index for a given backend lives.

    Mock and real indexes are kept in separate files. They are not
    interchangeable: mock vectors have 256 dimensions from a hashing
    vectoriser, real ones have 768 from nomic-embed-text, and a query embedded
    by one backend against an index built by the other produces similarity
    scores that are not merely wrong but meaningless.

    They were originally written to the same path, and a mock build during
    development silently replaced a real index. The embedding-model check
    caught it, but relying on a check to catch a collision that need not exist
    is worse than making the collision impossible.
    """
    base = config.path("paths.index")
    return base.with_name(f"{base.stem}_mock{base.suffix}") if mock else base


class IndexError_(RuntimeError):
    """Raised when the index is missing, malformed, or does not match the corpus."""


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk and its vector."""

    chunk: Chunk
    vector: tuple[float, ...]

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class Index:
    """Chunks, vectors and the provenance needed to trust them."""

    def __init__(self, entries: list[IndexedChunk], metadata: dict[str, Any]) -> None:
        self.entries = tuple(entries)
        self.metadata = metadata
        self._by_id = {e.chunk_id: e for e in entries}

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[IndexedChunk]:
        return iter(self.entries)

    def by_id(self, chunk_id: str) -> IndexedChunk:
        if chunk_id not in self._by_id:
            raise IndexError_(f"No chunk {chunk_id!r} in the index")
        return self._by_id[chunk_id]

    @property
    def dimensions(self) -> int:
        return len(self.entries[0].vector) if self.entries else 0

    @property
    def corpus_sha256(self) -> str:
        return self.metadata.get("corpus_sha256", "")

    @property
    def chunk_set_sha256(self) -> str:
        return self.metadata.get("chunk_set_sha256", "")

    @property
    def embedding_model(self) -> str:
        return self.metadata.get("embedding_model", "")

    @property
    def backend(self) -> str:
        return self.metadata.get("backend", "")

    def summary(self) -> dict[str, Any]:
        return {
            "chunk_count": len(self.entries),
            "dimensions": self.dimensions,
            "embedding_model": self.embedding_model,
            "backend": self.backend,
            "corpus_sha256": self.corpus_sha256[:12] + "...",
            "chunk_set_sha256": self.chunk_set_sha256[:12] + "...",
            "built_at": self.metadata.get("built_at"),
            "build_seconds": self.metadata.get("build_seconds"),
            "current_chunks": sum(1 for e in self.entries if e.chunk.is_current),
            "superseded_chunks": sum(1 for e in self.entries if not e.chunk.is_current),
        }

    # --- persistence --------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "metadata": self.metadata,
            "chunks": [
                {**entry.chunk.to_dict(), "vector": list(entry.vector)}
                for entry in self.entries
            ],
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def validate_structure(self) -> None:
        """Checks that do not need the corpus: the file must be internally sane.

        None of these should ever fail in normal operation. They exist because
        a corrupted or hand-edited index would otherwise produce similarity
        scores rather than an error, and a wrong number is harder to notice
        than a crash.
        """
        if not self.entries:
            raise IndexError_("Index contains no chunks")

        seen: set[str] = set()
        for entry in self.entries:
            if entry.chunk_id in seen:
                raise IndexError_(f"Duplicate chunk id {entry.chunk_id!r} in the index")
            seen.add(entry.chunk_id)

        dimensions = {len(e.vector) for e in self.entries}
        if len(dimensions) > 1:
            raise IndexError_(
                f"Index contains vectors of differing dimensions {sorted(dimensions)}. "
                "Similarity between them would be meaningless."
            )

        for entry in self.entries:
            for value in entry.vector:
                if not math.isfinite(value):
                    raise IndexError_(
                        f"{entry.chunk_id} has a non-finite value in its vector. "
                        "Cosine similarity would return NaN and ranking would be arbitrary."
                    )

        declared = self.metadata.get("dimensions")
        if declared and declared != self.dimensions:
            raise IndexError_(
                f"Index metadata declares {declared} dimensions but the vectors "
                f"have {self.dimensions}"
            )

    @classmethod
    def load(cls, path: Path | str, *, kb: KnowledgeBase | None = None,
             config: Config | None = None,
             allow_stale: bool = False) -> "Index":
        """Load an index, refusing a stale one unless explicitly permitted.

        If ``kb`` is supplied the corpus fingerprint is checked. Loading an
        index built from different documents is the quiet failure this guards
        against: retrieval returns text that is no longer in the corpus, the
        generator answers from it, and every citation points at a version of a
        document that no longer exists.
        """
        source = Path(path)
        if not source.exists():
            raise IndexError_(
                f"No index at {source}. Build one with scripts/build_index.py"
            )
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IndexError_(f"{source} is not valid JSON: {exc}") from exc

        if payload.get("schema_version") != SCHEMA_VERSION:
            raise IndexError_(
                f"{source} has schema {payload.get('schema_version')!r}, "
                f"expected {SCHEMA_VERSION!r}. Rebuild the index."
            )

        metadata = payload["metadata"]
        entries = []
        for record in payload["chunks"]:
            vector = record.pop("vector")
            record["sections"] = tuple(record["sections"])
            record["overlap_source"] = tuple(record.get("overlap_source", ()))
            record["effective_date"] = date.fromisoformat(record["effective_date"])
            record.pop("word_count", None)
            entries.append(IndexedChunk(chunk=Chunk(**record), vector=tuple(vector)))

        index = cls(entries, metadata)
        index.validate_structure()

        # The declared hash is checked against the chunks the file actually
        # holds, before anything is compared to the corpus. Every other check
        # here read ``metadata["chunk_set_sha256"]`` and trusted it, so a file
        # whose metadata said one thing and whose stored chunks said another
        # would pass every guard: the index would be treated as fresh while
        # serving different text under the same identifiers. Recomputing costs
        # a hash over data already in memory.
        stored = chunk_set_fingerprint([e.chunk for e in entries])
        declared = metadata.get("chunk_set_sha256")
        if declared and declared != stored:
            raise IndexError_(
                f"The index file declares chunk set {declared[:12]}... but the chunks "
                f"it contains hash to {stored[:12]}.... The file has been edited or "
                "written by a mismatched build, and its identifiers no longer refer "
                "to the text stored beside them. Rebuild it:\n"
                "  python scripts/build_index.py"
            )

        if kb is not None and not allow_stale:
            expected = kb.fingerprint()
            if index.corpus_sha256 != expected:
                raise IndexError_(
                    f"Index was built from corpus {index.corpus_sha256[:12]}... but the "
                    f"knowledge base is now {expected[:12]}.... Rebuild the index, or "
                    "pass allow_stale=True if you know what you are doing."
                )

            # The corpus can be unchanged while the chunks are entirely
            # different, which is exactly what happened when the chunker was
            # rewritten. Re-chunking costs about fifty milliseconds and is the
            # only check that catches it.
            active = config or load_config()
            current = chunk_set_fingerprint(chunk_corpus(
                kb,
                active.require("chunking.max_words"),
                active.require("chunking.overlap_sentences"),
                active.require("chunking.min_words"),
            ))
            if index.chunk_set_sha256 and index.chunk_set_sha256 != current:
                raise IndexError_(
                    f"Index holds chunk set {index.chunk_set_sha256[:12]}... but the "
                    f"corpus now chunks to {current[:12]}.... The documents are "
                    "unchanged, so the chunker or the chunking configuration has "
                    "changed. Rebuild the index:\n"
                    "  python scripts/build_index.py"
                )
        return index


def build_index(
    kb: KnowledgeBase,
    client: LLMClient,
    config: Config | None = None,
    *,
    progress: bool = False,
) -> Index:
    """Chunk the corpus and embed every chunk.

    The text embedded is ``chunk.embedding_text``, which carries the document
    title and heading path, not the bare chunk body. That is the whole point of
    the contextual header: a passage reading "55 pence per mile" must be
    findable by someone asking about the expenses policy.
    """
    config = config or load_config()
    chunks = chunk_corpus(
        kb,
        config.require("chunking.max_words"),
        config.require("chunking.overlap_sentences"),
        config.require("chunking.min_words"),
    )

    started = time.perf_counter()
    entries: list[IndexedChunk] = []
    for position, chunk in enumerate(chunks, start=1):
        if progress and (position == 1 or position % 25 == 0 or position == len(chunks)):
            elapsed = time.perf_counter() - started
            rate = position / elapsed if elapsed else 0
            remaining = (len(chunks) - position) / rate if rate else 0
            print(f"  embedding {position}/{len(chunks)}  "
                  f"{elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining", flush=True)
        vector = client.embed(chunk.embedding_text)
        entries.append(IndexedChunk(chunk=chunk, vector=tuple(vector)))
    build_seconds = time.perf_counter() - started

    dimensions = {len(e.vector) for e in entries}
    if len(dimensions) > 1:
        raise IndexError_(
            f"Embedding model returned inconsistent dimensions: {sorted(dimensions)}"
        )

    endpoint = client.describe_endpoint()
    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_seconds": round(build_seconds, 2),
        "backend": endpoint["backend"],
        "embedding_model": getattr(client, "embedding_model", "unknown"),
        "endpoint": endpoint,
        "corpus_sha256": kb.fingerprint(),
        "chunk_set_sha256": chunk_set_fingerprint(chunks),
        "document_count": len(kb),
        "chunk_count": len(entries),
        "dimensions": dimensions.pop() if dimensions else 0,
        "chunking": {
            "max_words": config.require("chunking.max_words"),
            "overlap_sentences": config.require("chunking.overlap_sentences"),
            "min_words": config.require("chunking.min_words"),
        },
        "config_sha256": config.fingerprint(),
        "seed": config.get("project.seed"),
    }
    return Index(entries, metadata)
