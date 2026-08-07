"""Grounded answer generation, and the measurement of how grounded it is.

This module produces the **baseline** arm of the ablation: retrieve, prompt,
generate, extract citations. Nothing here reasons about conflicting evidence or
document currency. That is the verification layer's job, and keeping the
boundary strict is what makes the ablation a measurement rather than a
demonstration.

What this module does add is bookkeeping that the evaluation depends on:

``citations``      which chunk identifiers the answer actually cited
``hallucinated_citations``  identifiers cited that were never retrieved
``uncited``        retrieved chunks the answer ignored
``cited_superseded``  whether the answer cited a withdrawn document

That last one is the single most useful number the baseline can produce. If the
baseline cites HR-03 when asked about mileage, it has quoted a withdrawn rate,
and the ablation has something concrete to improve on. Measuring it here rather
than in the verification layer means the baseline is scored on its own terms.

A hallucinated citation is a distinct and worse failure: the model invented an
identifier that was not in its evidence at all. Separating the two matters,
because they have different causes and different fixes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from ..common.config import Config, load_config
from ..common.llm_client import Generation, LLMClient
from ..retrieve.retriever import EvidenceFormat, RetrievalResult
from .prompts import build_baseline_prompt

# Chunk identifiers look like HR-13#001. Matching the document part separately
# lets a citation of "[HR-13]" count as a document-level reference rather than
# being discarded as malformed.
CITATION_RE = re.compile(r"\[([A-Z]{2,4}-\d{2})(?:#(\d{1,3}))?\]")

# Extended after the pilot, where the model produced "I cannot provide an
# answer about the company pension scheme as it is not present in the provided
# evidence" and this list matched none of it. Extending the list does not make
# the approach sound: it remains a diagnostic, and final refusal scoring uses a
# rubric or structured output. The regression test exists so the specific miss
# cannot recur silently.
REFUSAL_MARKERS = (
    "does not contain", "no evidence", "not covered", "cannot answer",
    "cannot provide an answer", "not present in the provided evidence",
    "not present in the evidence", "does not say", "not stated",
    "no information", "does not specify", "unable to answer",
    "not provided in the evidence", "not mentioned", "is not available in",
    "do not contain", "cannot be answered",
)


@dataclass(frozen=True)
class GroundedAnswer:
    """An answer, its citations, and how well those citations hold up."""

    question: str
    answer: str
    generation: Generation
    retrieval: RetrievalResult
    prompt: str
    citations: tuple[str, ...]
    document_citations: tuple[str, ...]
    hallucinated_citations: tuple[str, ...]
    uncited_chunks: tuple[str, ...]
    cited_superseded: tuple[str, ...]
    refusal_heuristic: bool

    @property
    def has_valid_citation_ids(self) -> bool:
        """Cited at least one identifier, and every identifier was retrieved.

        Deliberately **not** called "grounded". It was, and the name claimed
        far more than the check delivered: an answer citing IT-03#001 for a
        deadline that appears only in IT-03#002 satisfied it completely. The
        identifier was real and retrieved; the passage did not support the
        claim. Whether a citation supports what it is attached to is measured
        in ``sme_assistant.evaluation.answer_scoring``, not here.
        """
        return bool(self.citations) and not self.hallucinated_citations

    @property
    def wall_seconds(self) -> float:
        """End to end: query embedding, vector search, and generation.

        Search time was previously omitted. It is only a few milliseconds, but
        an end-to-end figure that quietly excludes a stage is the wrong number
        however small the omission.
        """
        return (
            self.retrieval.embed_seconds
            + self.retrieval.search_seconds
            + self.generation.wall_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "document_citations": list(self.document_citations),
            "hallucinated_citations": list(self.hallucinated_citations),
            "uncited_chunks": list(self.uncited_chunks),
            "cited_superseded": list(self.cited_superseded),
            "refusal_heuristic": self.refusal_heuristic,
            "refusal_heuristic_is_diagnostic_only": True,
            "has_valid_citation_ids": self.has_valid_citation_ids,
            "wall_seconds": round(self.wall_seconds, 3),
            "generation": self.generation.to_dict(),
            "retrieval": self.retrieval.to_dict(),
        }


def extract_citations(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (chunk citations, document-only citations), each deduplicated.

    Order is preserved so that "the model cited HR-13 first" is recoverable,
    which matters when analysing whether it led with the current policy or the
    withdrawn one.
    """
    chunks: list[str] = []
    documents: list[str] = []
    for doc_id, ordinal in CITATION_RE.findall(text):
        if ordinal:
            identifier = f"{doc_id}#{ordinal.zfill(3)}"
            if identifier not in chunks:
                chunks.append(identifier)
        elif doc_id not in documents:
            documents.append(doc_id)
    return tuple(chunks), tuple(documents)


def looks_like_refusal(text: str) -> bool:
    """Heuristic detection of a refusal in free text.

    Deliberately a heuristic and deliberately reported as such. Turning refusal
    into a reliable structured signal is part of what the verification layer is
    for, and the baseline's inability to express it cleanly is itself a finding.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


class Generator:
    """Baseline retrieval-augmented generation."""

    def __init__(self, client: LLMClient, config: Config | None = None) -> None:
        self.client = client
        self.config = config or load_config()

    def answer(
        self,
        question: str,
        retrieval: RetrievalResult,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        evidence_format: EvidenceFormat | None = None,
    ) -> GroundedAnswer:
        evidence = "" if retrieval.should_refuse else retrieval.evidence_text(evidence_format)
        prompt = build_baseline_prompt(question, evidence)

        generation = self.client.generate(prompt, model=model, options=options)
        text = generation.text

        chunk_citations, document_citations = extract_citations(text)

        retrieved_chunk_ids = {s.chunk_id for s in retrieval}
        retrieved_doc_ids = {s.chunk.doc_id for s in retrieval}
        superseded_chunk_ids = {s.chunk_id for s in retrieval if not s.chunk.is_current}
        superseded_doc_ids = {s.chunk.doc_id for s in retrieval if not s.chunk.is_current}

        hallucinated = tuple(
            c for c in chunk_citations if c not in retrieved_chunk_ids
        ) + tuple(
            d for d in document_citations if d not in retrieved_doc_ids
        )
        cited_superseded = tuple(
            c for c in chunk_citations if c in superseded_chunk_ids
        ) + tuple(
            d for d in document_citations if d in superseded_doc_ids
        )
        uncited = tuple(sorted(retrieved_chunk_ids - set(chunk_citations)))

        return GroundedAnswer(
            question=question,
            answer=text,
            generation=generation,
            retrieval=retrieval,
            prompt=prompt,
            citations=chunk_citations,
            document_citations=document_citations,
            hallucinated_citations=hallucinated,
            uncited_chunks=uncited,
            cited_superseded=cited_superseded,
            refusal_heuristic=looks_like_refusal(text),
        )
