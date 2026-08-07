"""Vector search over the index.

Cosine similarity computed by linear scan. With 147 chunks of 768 floats that
is roughly 113,000 multiply-adds per query, which takes under a millisecond in
pure Python and is invisible next to the 18 seconds of prompt processing that
follows it on a Raspberry Pi. Optimising it would be optimising the wrong
thing.

Two features here are not standard retrieval, and both exist to serve the
research question rather than to improve ranking.

**The status filter.** ``RetrievalMode.CURRENT_ONLY`` drops superseded chunks
before ranking. That is deliberately the cheap alternative to this project's
contribution: three lines of metadata filtering that resolve every
version-supersession conflict without any reasoning about claims. It is
implemented here so the evaluation can compare against it honestly. If the
verification layer cannot beat this arm on the conflicts a filter cannot touch,
the contribution is not justified, and the evaluation should be able to say so.

**Conflict detection at retrieval time.** ``RetrievalResult.conflicts``
reports when the retrieved set contains both a superseded document and its
replacement, or two current documents that the corpus registers as
disagreeing. Note carefully: this only reports *structural* signals available
from chunk metadata. It does not read the gold registry, and it cannot detect a
conflict between two current documents on its own. Resolving those is the
verification layer's job, and this flag exists so the analysis can separate
what metadata alone achieved from what reasoning added.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..common.config import Config, load_config
from ..common.llm_client import LLMClient
from ..ingest.chunker import Chunk
from ..ingest.index import Index, IndexedChunk


class RetrievalMode(str, Enum):
    """Which chunks are eligible for ranking.

    ``ALL`` indexes everything, including superseded documents, so that
    conflicting evidence reaches the model and the verification layer has
    something to detect. ``CURRENT_ONLY`` is the metadata-filter baseline.
    """

    ALL = "all"
    CURRENT_ONLY = "current_only"


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.chunk.doc_id,
            "score": round(self.score, 4),
            "rank": self.rank,
            "status": self.chunk.status,
            "citation": self.chunk.citation,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """What retrieval returned, and what the pipeline needs to know about it."""

    query: str
    results: tuple[ScoredChunk, ...]
    mode: RetrievalMode
    top_k: int
    min_similarity: float
    best_score: float
    below_threshold: bool
    embed_seconds: float
    search_seconds: float
    candidates_considered: int
    conflicts: tuple[dict[str, str], ...] = ()

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.chunk.doc_id for s in self.results))

    @property
    def has_superseded(self) -> bool:
        return any(not s.chunk.is_current for s in self.results)

    @property
    def should_refuse(self) -> bool:
        """Whether nothing retrieved was relevant enough to answer from.

        This is a retrieval-level refusal signal, not the final decision. A
        question about pensions, which the corpus does not cover, still returns
        the four nearest chunks: vector search always returns something. The
        threshold is what distinguishes "here are the four closest passages"
        from "the corpus contains an answer".
        """
        return self.below_threshold or not self.results

    def evidence_text(self) -> str:
        """Retrieved chunks formatted for a prompt, with citable identifiers."""
        blocks = []
        for scored in self.results:
            chunk = scored.chunk
            marker = "" if chunk.is_current else f" [SUPERSEDED, replaced by {chunk.superseded_by}]"
            blocks.append(
                f"[{chunk.chunk_id}] {chunk.doc_title}"
                + (f" > {' > '.join(chunk.heading_path)}" if chunk.heading_path else "")
                + f" (v{chunk.version}, effective {chunk.effective_date}){marker}\n{chunk.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode.value,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "best_score": round(self.best_score, 4),
            "below_threshold": self.below_threshold,
            "should_refuse": self.should_refuse,
            "has_superseded": self.has_superseded,
            "candidates_considered": self.candidates_considered,
            "embed_seconds": round(self.embed_seconds, 4),
            "search_seconds": round(self.search_seconds, 6),
            "conflicts": list(self.conflicts),
            "results": [s.to_dict() for s in self.results],
        }


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, computed without assuming normalised vectors.

    ``nomic-embed-text`` returns unnormalised vectors, so a plain dot product
    would rank by magnitude as well as direction and quietly favour longer
    chunks.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _structural_conflicts(results: Sequence[ScoredChunk]) -> tuple[dict[str, str], ...]:
    """Conflicts detectable from chunk metadata alone.

    Only supersession pairs are visible here: if both a withdrawn document and
    its replacement appear in the retrieved set, the metadata says so outright.
    Two current documents that disagree look identical to two that agree, which
    is precisely why the verification layer exists.
    """
    by_doc = {s.chunk.doc_id: s.chunk for s in results}
    found = []
    for doc_id, chunk in by_doc.items():
        if chunk.superseded_by and chunk.superseded_by in by_doc:
            found.append({
                "type": "supersession",
                "superseded": doc_id,
                "current": chunk.superseded_by,
                "detectable_from": "metadata",
            })
    return tuple(found)


class Retriever:
    """Embeds a query and ranks the index against it."""

    def __init__(self, index: Index, client: LLMClient, config: Config | None = None) -> None:
        self.index = index
        self.client = client
        self.config = config or load_config()
        self.default_top_k = self.config.require("retrieval.top_k")
        self.default_min_similarity = self.config.require("retrieval.min_similarity")

        expected = index.embedding_model
        actual = getattr(client, "embedding_model", None)
        if expected and actual and expected != actual:
            raise ValueError(
                f"Index was built with embedding model {expected!r} but the client uses "
                f"{actual!r}. Query and chunk vectors would be from different spaces "
                "and every similarity score would be meaningless."
            )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
        mode: RetrievalMode = RetrievalMode.ALL,
        category: str | None = None,
    ) -> RetrievalResult:
        k = top_k if top_k is not None else self.default_top_k
        threshold = min_similarity if min_similarity is not None else self.default_min_similarity

        embed_started = time.perf_counter()
        query_vector = self.client.embed(query)
        embed_seconds = time.perf_counter() - embed_started

        search_started = time.perf_counter()
        candidates: list[IndexedChunk] = [
            entry for entry in self.index
            if (mode is RetrievalMode.ALL or entry.chunk.is_current)
            and (category is None or entry.chunk.category == category)
        ]
        scored = sorted(
            ((cosine_similarity(query_vector, e.vector), e) for e in candidates),
            key=lambda pair: (-pair[0], pair[1].chunk_id),
        )
        search_seconds = time.perf_counter() - search_started

        results = tuple(
            ScoredChunk(chunk=entry.chunk, score=score, rank=position)
            for position, (score, entry) in enumerate(scored[:k], start=1)
        )
        best = results[0].score if results else 0.0

        return RetrievalResult(
            query=query,
            results=results,
            mode=mode,
            top_k=k,
            min_similarity=threshold,
            best_score=best,
            below_threshold=best < threshold,
            embed_seconds=embed_seconds,
            search_seconds=search_seconds,
            candidates_considered=len(candidates),
            conflicts=_structural_conflicts(results),
        )
