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

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..common.config import Config, load_config
from ..common.llm_client import LLMClient
from ..kb.loader import KnowledgeBase
from .chunker import Chunk, chunk_corpus

SCHEMA_VERSION = "1.0"


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

    @classmethod
    def load(cls, path: Path | str, *, kb: KnowledgeBase | None = None,
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

        if kb is not None and not allow_stale:
            expected = kb.fingerprint()
            if index.corpus_sha256 != expected:
                raise IndexError_(
                    f"Index was built from corpus {index.corpus_sha256[:12]}... but the "
                    f"knowledge base is now {expected[:12]}.... Rebuild the index, or "
                    "pass allow_stale=True if you know what you are doing."
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
