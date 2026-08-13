"""The verifier's structured output, and the guards that keep it honest.

This module defines what Arm D returns and, just as importantly, what it is
allowed to have seen. **No part of this package may read evaluation data.** Not
the conflict registry, not the family an answer belongs to, not the declared
relationship type, not which document was designated the safe reading, not the
scoring criteria.

The reason is narrow. If the verifier looked up a declared relationship and
then behaved accordingly, the experiment would measure a dictionary lookup, and
the conflict-handling result would restate the registry rather than find
anything about the system. It would also be undetectable in the output: the
answers would look excellent. The leakage tests enforce this in the import
graph, and the verifier tests enforce it again in behaviour.

What the verifier may see:

===========================  ===========================================
question                     the user's question, as asked
retrieved chunks             text, identifier, and the ordinary document
                             metadata a real deployment would have:
                             title, status, effective date, supersedes
===========================  ===========================================

What it must work out for itself: whether the evidence disagrees, what kind of
disagreement it is, which passages support or contradict each claim, and
whether a safe course exists.

The relationship vocabulary deliberately mirrors the registry's types, because
those are the distinctions that matter in the domain. Mirroring the vocabulary
is not leakage; reading the answer is. The verifier is told the names of the
categories and must assign one, exactly as a human reviewer would be briefed on
what to look for and then asked to look.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# Verdicts on a single claim.
SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
VALID_VERDICTS = frozenset({SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE})

# The relationship the verifier infers between the retrieved passages. These
# names match the registry's types because they are the real distinctions in
# the domain, not because the registry is consulted.
SUPERSESSION = "supersession"
MUTUALLY_EXCLUSIVE = "mutually_exclusive"
STRICTER_LOOSER = "stricter_looser"
CONTEXTUALLY_COMPATIBLE = "contextually_compatible"
NO_RELATIONSHIP = "no_relationship"
INSUFFICIENT = "insufficient"

VALID_RELATIONSHIPS = frozenset({
    SUPERSESSION, MUTUALLY_EXCLUSIVE, STRICTER_LOOSER,
    CONTEXTUALLY_COMPATIBLE, NO_RELATIONSHIP, INSUFFICIENT,
})

# Relationships that constitute a detected conflict. ``contextually_compatible``
# is deliberately not one: inferring that two passages apply in different
# circumstances is the verifier declining to flag, and counting it as a
# detection would make the false-conflict rate unmeasurable.
CONFLICTING_RELATIONSHIPS = frozenset({
    SUPERSESSION, MUTUALLY_EXCLUSIVE, STRICTER_LOOSER,
})

CHUNK_ID_RE = re.compile(r"\b[A-Z]{2,4}-\d{2}#\d{1,3}\b")


class VerificationError(RuntimeError):
    """Raised when the verifier's output cannot be interpreted at all."""


@dataclass(frozen=True)
class ClaimVerdict:
    """One checkable claim, judged against named passages."""

    claim: str
    verdict: str
    supporting: tuple[str, ...] = ()
    contradicting: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "supporting": list(self.supporting),
            "contradicting": list(self.contradicting),
        }


@dataclass(frozen=True)
class Verification:
    """What the verifier concluded, and what it had to discard."""

    verdicts: tuple[ClaimVerdict, ...]
    relationship: str
    conflicting_chunks: tuple[str, ...] = ()
    safe_action: str | None = None
    escalate: bool = False
    rationale: str = ""
    confidence: str = "low"
    # Identifiers the verifier returned that were never retrieved. Recorded
    # rather than silently dropped: a verifier citing evidence it was not given
    # is the same failure as a generator doing it, and it must be visible.
    invented_ids: tuple[str, ...] = ()
    parse_failed: bool = False
    raw: str = ""

    @property
    def conflict_detected(self) -> bool:
        return self.relationship in CONFLICTING_RELATIONSHIPS

    @property
    def any_contradiction(self) -> bool:
        return any(v.verdict == CONTRADICTED for v in self.verdicts)

    @property
    def all_insufficient(self) -> bool:
        return bool(self.verdicts) and all(
            v.verdict == INSUFFICIENT_EVIDENCE for v in self.verdicts
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdicts": [v.to_dict() for v in self.verdicts],
            "relationship": self.relationship,
            "conflict_detected": self.conflict_detected,
            "conflicting_chunks": list(self.conflicting_chunks),
            "safe_action": self.safe_action,
            "escalate": self.escalate,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "invented_ids": list(self.invented_ids),
            "parse_failed": self.parse_failed,
        }


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Small models wrap JSON in prose, fences, or an apology. Being strict here
    would turn a formatting slip into a missing data point, and a verifier
    scored on how often it emits clean JSON is measuring the wrong thing.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        if start == -1:
            raise VerificationError("no JSON object in the response")
        depth, end = 0, None
        for i, char in enumerate(text[start:], start):
            depth += char == "{"
            depth -= char == "}"
            if depth == 0:
                end = i + 1
                break
        if end is None:
            raise VerificationError("unterminated JSON object")
        candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"malformed JSON: {exc}") from exc


def split_ids(value: Any, retrieved: set[str], invented: set[str]) -> tuple[str, ...]:
    """Keep the identifiers that were retrieved; record the rest as invented."""
    if value is None:
        return ()
    if isinstance(value, str):
        found = CHUNK_ID_RE.findall(value)
    else:
        found = [str(v).strip() for v in value]
    kept: list[str] = []
    for identifier in found:
        match = CHUNK_ID_RE.search(identifier)
        normalised = match.group(0) if match else identifier
        if normalised in retrieved:
            if normalised not in kept:
                kept.append(normalised)
        else:
            invented.add(normalised)
    return tuple(kept)


def parse(text: str, retrieved: Iterable[str], policy: Mapping[str, Any]) -> Verification:
    """Interpret a verifier response, discarding evidence it was never given.

    Every identifier is checked against the retrieved set. A verifier that
    names a passage it was not shown has invented it, exactly as a generator
    citing an unretrieved chunk has, and the count is reported rather than
    quietly repaired.
    """
    available = set(retrieved)
    invented: set[str] = set()

    try:
        payload = extract_json(text)
    except VerificationError:
        return Verification(
            verdicts=(), relationship=INSUFFICIENT, parse_failed=True, raw=text,
            confidence=str(policy.get("on_parse_failure", "low")),
        )

    verdicts: list[ClaimVerdict] = []
    for entry in payload.get("claims") or []:
        if not isinstance(entry, dict):
            continue
        verdict = str(entry.get("verdict", "")).strip().upper()
        if verdict not in VALID_VERDICTS:
            verdict = INSUFFICIENT_EVIDENCE
        verdicts.append(ClaimVerdict(
            claim=str(entry.get("claim", "")).strip(),
            verdict=verdict,
            supporting=split_ids(entry.get("supporting"), available, invented),
            contradicting=split_ids(entry.get("contradicting"), available, invented),
        ))

    relationship = str(payload.get("relationship", "")).strip().lower()
    if relationship not in VALID_RELATIONSHIPS:
        relationship = INSUFFICIENT

    safe_action = payload.get("safe_action")
    safe_action = str(safe_action).strip() if safe_action else None
    if safe_action and safe_action.lower() in {"none", "null", "n/a", ""}:
        safe_action = None

    result = Verification(
        verdicts=tuple(verdicts),
        relationship=relationship,
        conflicting_chunks=split_ids(payload.get("conflicting_chunks"), available, invented),
        safe_action=safe_action,
        escalate=bool(payload.get("escalate", False)),
        rationale=str(payload.get("rationale", "")).strip(),
        invented_ids=tuple(sorted(invented)),
        raw=text,
    )
    return replace_confidence(result, policy)


def replace_confidence(result: Verification, policy: Mapping[str, Any]) -> Verification:
    """Apply the runtime confidence policy.

    The policy lives in ``config.json`` under ``verification``, which is
    runtime configuration. A second, separate policy exists on the evaluation
    side describing what *should* happen; that one is used only when scoring.
    Reading it here would let the verifier consult the answer key about its own
    certainty.
    """
    import dataclasses

    if result.parse_failed:
        return result

    if result.relationship in CONFLICTING_RELATIONSHIPS:
        confidence = str(policy.get("on_conflict", "low"))
    elif result.all_insufficient or result.relationship == INSUFFICIENT:
        confidence = str(policy.get("on_insufficient", "low"))
    elif any(v.verdict == CONTRADICTED for v in result.verdicts):
        confidence = str(policy.get("on_contradiction", "low"))
    elif result.invented_ids:
        confidence = str(policy.get("on_invented_evidence", "low"))
    else:
        confidence = str(policy.get("on_supported", "medium"))
    return dataclasses.replace(result, confidence=confidence)
