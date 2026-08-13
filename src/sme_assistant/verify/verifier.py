"""The verification layer. Arm D, and the contribution.

Arm B answers from evidence and stops. This runs a second pass over the same
evidence and the answer produced from it, asking a different question: does the
evidence actually support what was said, and does the evidence agree with
itself?

The second question is the one the baseline cannot ask. A generator sees six
passages and writes fluent prose from whichever ranked highest. It has no
reason to notice that passage two says 24 hours and passage five says 1 hour,
because nothing in "answer from the evidence" requires the evidence to be
consistent.

**This module reads no evaluation data.** It sees the question, the retrieved
chunks and the ordinary document metadata a real deployment would have. It does
not see which registered group a question belongs to, what relationship was
declared for it, which document was designated the safe reading, or what the
scoring criteria expect. Behaviour is inferred, and if the inference is wrong
the result says so. A verifier that looked up the answer would produce
excellent output and measure nothing.

Two passes, and the second one answers
--------------------------------------
Verification is a separate call rather than a longer prompt on the first. That
costs a second generation, which H5 predicts and RQ4 measures. The reason is
that a single prompt asking a 3B model to answer *and* audit its own answer
gets neither: the audit is written to justify the answer already produced.
Separating them means the second pass sees the answer as text to be checked,
not as a conclusion to be defended.

**The second pass returns a revised answer, not only an audit.** An earlier
version returned the audit alone and left the draft untouched, which made Arm D
unable to improve on Arm B by construction: the scorer reads the answer, so a
verifier could detect a conflict perfectly and still serve the wrong text. Two
calls, and the second one produces the answer that is scored. The draft is kept
alongside it so the change is visible and the revision rate is measurable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..common.config import Config, load_config
from ..common.llm_client import Generation, LLMClient
from ..generate.generator import GroundedAnswer, extract_citations, looks_like_refusal
from ..retrieve.retriever import RetrievalResult
from . import schema
from .schema import Verification

VERIFIER_SYSTEM = """You are a policy auditor. You are given a question, an \
answer that was produced from the evidence below, and that evidence.

Your job is NOT to answer the question. It is to check the answer against the \
evidence, and to check the evidence against itself.

Return a single JSON object and nothing else:

{
  "claims": [
    {"claim": "<one factual assertion from the answer>",
     "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
     "supporting": ["<chunk id>"],
     "contradicting": ["<chunk id>"]}
  ],
  "relationship": "supersession" | "mutually_exclusive" | "stricter_looser" \
| "contextually_compatible" | "no_relationship" | "insufficient",
  "conflicting_chunks": ["<chunk id>"],
  "safe_action": "<what to do that satisfies every passage, or null>",
  "escalate": true | false,
  "rationale": "<one or two sentences>",
  "final_answer": "<the corrected answer for the user, with citations>"
}

Rules:
1. Use only the chunk identifiers shown in the evidence. Never invent one.
2. A claim is SUPPORTED only if a named passage states it. Sounding plausible \
is not support.
3. A claim is CONTRADICTED if a named passage says otherwise.
4. Decide the relationship between the passages that bear on the question:
   - "supersession": one passage is marked as replacing the other. The \
replacement governs, so answer from it, do not present the withdrawn figure as \
current, and set escalate to false. This one has a right answer.
   - "mutually_exclusive": both are in force and no single action satisfies \
both. Set escalate to true and safe_action to null.
   - "stricter_looser": both are in force and one is stricter, so acting on \
the stricter one satisfies both. Put that action in safe_action and set \
escalate to true.
   - "contextually_compatible": they appear to disagree but apply in different \
circumstances, so both hold. Set escalate to false.
   - "no_relationship": the passages do not bear on the same point.
   - "insufficient": you cannot tell from what you were given.
5. Do not report a conflict you cannot point at. Every conflict must name the \
chunks in conflicting_chunks, from at least two different documents.
6. Write final_answer for the user. If the answer under review is correct, \
repeat it. If it is wrong, incomplete, or cites the wrong passage, correct it. \
Where the passages disagree and cannot be reconciled, state both positions, \
name both documents, say neither supersedes the other, and say the discrepancy \
should be raised. Cite chunk identifiers in square brackets."""


def build_verification_prompt(question: str, answer: str, evidence: str) -> str:
    if not evidence.strip():
        return (
            f"{VERIFIER_SYSTEM}\n\nQUESTION: {question}\n\nANSWER UNDER REVIEW:\n"
            f"{answer}\n\nEVIDENCE:\nNo evidence was retrieved.\n\n"
            "With no evidence, every claim is INSUFFICIENT_EVIDENCE and the "
            "relationship is insufficient.\n\nJSON:"
        )
    return (
        f"{VERIFIER_SYSTEM}\n\nQUESTION: {question}\n\nANSWER UNDER REVIEW:\n"
        f"{answer}\n\nEVIDENCE:\n{evidence}\n\nJSON:"
    )


@dataclass(frozen=True)
class VerifiedAnswer:
    """A grounded answer plus the audit of it."""

    answer: GroundedAnswer
    verification: Verification
    generation: Generation
    prompt: str

    @property
    def final_answer(self) -> str:
        """What Arm D serves, and what the blinded scorer reads.

        Falls back to the draft if the verifier returned nothing usable, so a
        parse failure degrades to Arm B behaviour rather than to silence.
        """
        return self.verification.final_answer or self.answer.answer

    @property
    def wall_seconds(self) -> float:
        """End to end, including the verification pass.

        H5 predicts this lands between 1.5x and 2.5x the baseline on the Pi.
        Reporting the answer's time alone would hide the cost of the thing
        being evaluated.
        """
        return self.answer.wall_seconds + self.generation.wall_seconds

    @property
    def verification_seconds(self) -> float:
        return self.generation.wall_seconds

    def _citation_metrics(self) -> dict[str, Any]:
        """Citation bookkeeping for the answer actually served.

        These were inherited from the draft. ``to_dict`` began with the
        generator's dictionary and overrode only the answer text, so
        ``citations``, ``hallucinated_citations``, ``cited_superseded`` and the
        rest described a string that had been thrown away. A revision that
        fixed a miscitation would have been recorded as still carrying it.

        The same extraction the generator uses is re-run here, against the
        final text, so the numbers describe what the reader receives.
        """
        retrieval = self.answer.retrieval
        chunks, documents = extract_citations(self.final_answer)

        retrieved_chunks = {s.chunk_id for s in retrieval}
        retrieved_docs = {s.chunk.doc_id for s in retrieval}
        superseded_chunks = {s.chunk_id for s in retrieval if not s.chunk.is_current}
        superseded_docs = {s.chunk.doc_id for s in retrieval if not s.chunk.is_current}

        hallucinated = tuple(c for c in chunks if c not in retrieved_chunks) + tuple(
            d for d in documents if d not in retrieved_docs
        )
        return {
            "citations": list(chunks),
            "document_citations": list(documents),
            "hallucinated_citations": list(hallucinated),
            "uncited_chunks": [c for c in sorted(retrieved_chunks) if c not in chunks],
            "cited_superseded": (
                [c for c in chunks if c in superseded_chunks]
                + [d for d in documents if d in superseded_docs]
            ),
            "refusal_heuristic": looks_like_refusal(self.final_answer),
            "has_valid_citation_ids": bool(chunks) and not hallucinated,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.answer.to_dict()
        # Keep the draft's figures under draft_* so the comparison survives,
        # then replace the headline ones with the served answer's.
        draft_metrics = {
            f"draft_{key}": payload[key]
            for key in ("citations", "document_citations", "hallucinated_citations",
                        "uncited_chunks", "cited_superseded", "refusal_heuristic",
                        "has_valid_citation_ids")
            if key in payload
        }
        payload.update(draft_metrics)
        payload.update(self._citation_metrics())
        payload.update({
            # The scored answer is the revised one. The draft is kept beside it
            # so a reader can see what verification changed, and so the revision
            # rate is measurable rather than assumed.
            "answer": self.final_answer,
            "draft_answer": self.answer.answer,
            "answer_revised": self.verification.revised,
            "revision_rejected": self.verification.revision_rejected,
            "verification": self.verification.to_dict(),
            "verification_generation": self.generation.to_dict(),
            "verification_prompt": self.prompt,
            "verification_seconds": round(self.verification_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
            "arm_has_verification": True,
        })
        return payload


class Verifier:
    """Second pass over an answer and the evidence it came from."""

    def __init__(self, client: LLMClient, config: Config | None = None) -> None:
        self.client = client
        self.config = config or load_config()

    @property
    def _policy(self) -> dict[str, Any]:
        # Runtime policy from config.json. A separate policy on the evaluation
        # side describes what should happen; using that here would let the
        # system consult the answer key about its own certainty.
        return dict(self.config.get("verification.confidence") or {})

    def verify(
        self,
        answer: GroundedAnswer,
        *,
        model: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> VerifiedAnswer:
        retrieval: RetrievalResult = answer.retrieval
        evidence = "" if retrieval.should_refuse else retrieval.evidence_text()
        prompt = build_verification_prompt(answer.question, answer.answer, evidence)

        chosen = model or self.config.get("llm.verification_model")
        merged = dict(self.config.require("verification"))
        merged.pop("confidence", None)
        if options:
            merged.update(options)

        started = time.perf_counter()
        generation = self.client.generate(prompt, model=chosen, options=merged)
        elapsed = time.perf_counter() - started

        verification = schema.parse(
            generation.text,
            [s.chunk_id for s in retrieval],
            self._policy,
            draft=answer.answer,
        )
        _ = elapsed  # generation.wall_seconds is authoritative; kept for clarity
        return VerifiedAnswer(
            answer=answer, verification=verification,
            generation=generation, prompt=prompt,
        )
