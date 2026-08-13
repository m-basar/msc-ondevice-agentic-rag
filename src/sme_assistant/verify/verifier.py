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

Two passes, not one
-------------------
Verification is a separate call rather than a longer prompt on the first. That
costs a second generation, which H5 predicts and RQ4 measures. The reason is
that a single prompt asking a 3B model to answer *and* audit its own answer
gets neither: the audit is written to justify the answer already produced.
Separating them means the second pass sees the answer as text to be checked,
not as a conclusion to be defended.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..common.config import Config, load_config
from ..common.llm_client import Generation, LLMClient
from ..generate.generator import GroundedAnswer
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
  "rationale": "<one or two sentences>"
}

Rules:
1. Use only the chunk identifiers shown in the evidence. Never invent one.
2. A claim is SUPPORTED only if a named passage states it. Sounding plausible \
is not support.
3. A claim is CONTRADICTED if a named passage says otherwise.
4. Decide the relationship between the passages that bear on the question:
   - "supersession": one passage is marked as replacing the other.
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
chunks in conflicting_chunks."""


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

    def to_dict(self) -> dict[str, Any]:
        payload = self.answer.to_dict()
        payload.update({
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
        )
        _ = elapsed  # generation.wall_seconds is authoritative; kept for clarity
        return VerifiedAnswer(
            answer=answer, verification=verification,
            generation=generation, prompt=prompt,
        )
