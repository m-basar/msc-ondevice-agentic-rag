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
# Any bracketed citation, with or without a chunk ordinal. Arm D must cite
# exact passages, so "[HR-13]" is not a valid citation for it even though the
# document exists: the reader cannot check a claim against a whole document.
BRACKET_CITE_RE = re.compile(r"\[([A-Z]{2,4}-\d{2})(#\d{1,3})?\]")

# A relationship that is resolvable from document metadata alone. The current
# document governs, so there is a right answer and no standing escalation.
# Collapsing this with the unresolvable kinds would treat "I found the
# superseded version and used the current one" as a failure state.
RESOLVED_RELATIONSHIPS = frozenset({SUPERSESSION})
UNRESOLVED_RELATIONSHIPS = CONFLICTING_RELATIONSHIPS - RESOLVED_RELATIONSHIPS


def as_bool(value: Any) -> bool:
    """Parse a boolean the way a model actually emits one.

    ``bool("false")`` is ``True``, so a verifier writing ``"escalate": "false"``
    would have escalated everything. JSON booleans, the strings both ways, and
    0/1 are all in the wild.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


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
    # The answer after verification. Arm D is scored on this, not on the draft.
    # Without it the verifier could detect a conflict perfectly and still serve
    # the unchanged wrong answer, so no amount of verification quality could
    # reach the conflict-handling metric.
    final_answer: str = ""
    revised: bool = False
    # A revision the verifier offered but which was not served, because the
    # finding it rested on failed structural validation. Downgrading the
    # relationship while accepting the prose written from it would let an
    # unsupported conflict reach the user through the back door.
    revision_rejected: bool = False
    revision_rejected_reason: str = ""
    # Every check the response failed, accumulated rather than short-circuited.
    # The first version recorded only relationship failures, so a claim verdict
    # could be downgraded for want of evidence while the revision written from
    # it was served unchanged.
    validation_failures: tuple[str, ...] = ()
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
    def is_resolved(self) -> bool:
        """A disagreement with a right answer, resolvable from metadata."""
        return self.relationship in RESOLVED_RELATIONSHIPS

    @property
    def is_unresolved(self) -> bool:
        """A disagreement with no basis in metadata for choosing."""
        return self.relationship in UNRESOLVED_RELATIONSHIPS

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
            "is_resolved": self.is_resolved,
            "is_unresolved": self.is_unresolved,
            "final_answer": self.final_answer,
            "revised": self.revised,
            "revision_rejected": self.revision_rejected,
            "revision_rejected_reason": self.revision_rejected_reason,
            "validation_failures": list(self.validation_failures),
            "invented_ids": list(self.invented_ids),
            "parse_failed": self.parse_failed,
            "raw": self.raw,
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


def parse(
    text: str,
    retrieved: Iterable[str],
    policy: Mapping[str, Any],
    draft: str = "",
) -> Verification:
    """Interpret a verifier response, and fail closed on unevidenced findings.

    Discarding an invented identifier is not enough on its own. An earlier
    version stripped the identifier and left the verdict standing, so a claim
    could be SUPPORTED with nothing supporting it and a conflict could be
    declared with one chunk. Every finding must now survive on evidence that
    was actually retrieved, or it is downgraded:

    * ``SUPPORTED`` without a valid supporting chunk becomes
      ``INSUFFICIENT_EVIDENCE``
    * ``CONTRADICTED`` without a valid contradicting chunk becomes
      ``INSUFFICIENT_EVIDENCE``
    * a relationship between documents needs at least two valid chunks from
      **distinct documents**, or the relationship becomes ``insufficient``

    The last rule is why the count is by document rather than by chunk. Two
    chunks from the same document are one side of a disagreement, not two.
    """
    available = set(retrieved)
    invented: set[str] = set()

    try:
        payload = extract_json(text)
    except VerificationError:
        return Verification(
            verdicts=(), relationship=INSUFFICIENT, parse_failed=True, raw=text,
            confidence=str(policy.get("on_parse_failure", "low")),
            final_answer=draft, revised=False,
        )

    failures: list[str] = []
    verdicts: list[ClaimVerdict] = []
    for entry in payload.get("claims") or []:
        if not isinstance(entry, dict):
            continue
        raw_verdict = str(entry.get("verdict", "")).strip()
        verdict = raw_verdict.upper()
        if verdict not in VALID_VERDICTS:
            # Normalising an unrecognised enum is a repair, and a repair is a
            # failure. A model that misspells the verdict has not told us what
            # it concluded, so a revision written alongside it rests on nothing
            # we can read.
            failures.append(
                f"verdict {raw_verdict[:40]!r} is not one of "
                f"{sorted(VALID_VERDICTS)}"
            )
            verdict = INSUFFICIENT_EVIDENCE
        supporting = split_ids(entry.get("supporting"), available, invented)
        contradicting = split_ids(entry.get("contradicting"), available, invented)

        # Fail closed. A verdict is a claim about evidence, so it cannot
        # outlive the evidence it named.
        claim_text = str(entry.get("claim", "")).strip()
        if verdict == SUPPORTED and not supporting:
            failures.append(
                f"claim {claim_text[:60]!r} was SUPPORTED with no valid supporting passage"
            )
            verdict = INSUFFICIENT_EVIDENCE
        elif verdict == CONTRADICTED and not contradicting:
            failures.append(
                f"claim {claim_text[:60]!r} was CONTRADICTED with no valid "
                "contradicting passage"
            )
            verdict = INSUFFICIENT_EVIDENCE

        verdicts.append(ClaimVerdict(
            claim=str(entry.get("claim", "")).strip(),
            verdict=verdict, supporting=supporting, contradicting=contradicting,
        ))

    raw_relationship = str(payload.get("relationship", "")).strip()
    relationship = raw_relationship.lower()
    if relationship not in VALID_RELATIONSHIPS:
        failures.append(
            "relationship "
            + (f"{raw_relationship[:40]!r}" if raw_relationship else "was missing and")
            + f" is not one of {sorted(VALID_RELATIONSHIPS)}"
        )
        relationship = INSUFFICIENT

    conflicting = split_ids(payload.get("conflicting_chunks"), available, invented)
    if relationship in CONFLICTING_RELATIONSHIPS:
        documents = {c.split("#")[0] for c in conflicting}
        if len(documents) < 2:
            failures.append(
                f"claimed {relationship} but named evidence from "
                f"{len(documents)} document(s); a disagreement needs two sides"
            )
            relationship = INSUFFICIENT
            conflicting = ()

    safe_action = payload.get("safe_action")
    safe_action = str(safe_action).strip() if safe_action else None
    if safe_action and safe_action.lower() in {"none", "null", "n/a", ""}:
        safe_action = None

    final = str(payload.get("final_answer") or "").strip()

    # Did the verification record any reason to change the answer? A layer that
    # finds nothing must be a no-op. The first version had no such rule, and in
    # pilot 02 six answers were rewritten on a verdict of no_relationship with
    # every claim SUPPORTED. One replaced a correct, cited answer about mileage
    # rates with the sentence "The answer under review does not address a
    # conflict between passages", which is commentary on the review served to
    # the user as the answer.
    #
    # The methodological cost is larger than the two bad answers. B and D share
    # a draft precisely so that any difference between them is attributable to
    # verification. Cosmetic rewrites of answers the verifier had no complaint
    # about put differences into that contrast which verification did not cause.
    # The draft citing evidence that was never retrieved is a defect the
    # verifier can see and repair, and repairing it is squarely the layer's
    # job. Without this the rule would have blocked exactly the correction it
    # exists to make: a supported claim attached to the wrong passage.
    draft_miscites = bool(
        {i for i in CHUNK_ID_RE.findall(draft) if i not in available}
        or {doc for doc, ordinal in BRACKET_CITE_RE.findall(draft) if not ordinal}
    )
    warrants_revision = (
        relationship in CONFLICTING_RELATIONSHIPS
        or any(v.verdict in {CONTRADICTED, INSUFFICIENT_EVIDENCE} for v in verdicts)
        or draft_miscites
    )
    # Withdrawing a claim the evidence does not support legitimately produces an
    # answer with nothing left to cite. Rewriting a supported answer into an
    # uncited one does not.
    is_abstention = bool(verdicts) and all(
        v.verdict == INSUFFICIENT_EVIDENCE for v in verdicts
    )

    # A revision written from a finding that did not survive validation is not
    # served. The verifier may have described a conflict in prose that the
    # structural check has just withdrawn, and accepting the prose while
    # rejecting the finding would put the unsupported claim in front of the
    # user anyway.
    if final:
        invented_in_answer = sorted({
            i for i in CHUNK_ID_RE.findall(final) if i not in available
        })
        if invented_in_answer:
            invented.update(invented_in_answer)
            failures.append(
                f"the revised answer cites evidence that was never retrieved: "
                f"{invented_in_answer}"
            )
        # A document-only citation cannot be checked against a passage, which
        # is the whole point of the exercise. Caught here rather than left to
        # the citation metrics, which run after the answer has been served.
        document_only = sorted({
            f"[{doc}]" for doc, ordinal in BRACKET_CITE_RE.findall(final) if not ordinal
        })
        if document_only:
            failures.append(
                f"the revised answer cites documents rather than passages: "
                f"{document_only}"
            )
        # Checking that cited identifiers resolve says nothing about a revision
        # that cites none: the check passes vacuously. Two of pilot 02's served
        # revisions stripped every citation from a cited draft and passed.
        if (not is_abstention
                and CHUNK_ID_RE.search(draft) and not CHUNK_ID_RE.search(final)):
            failures.append(
                "the revised answer cites no passages while the draft it "
                "replaces cited evidence"
            )

    # Kept out of ``failures`` because it is a policy about what may be served,
    # not a defect in what the model produced. Conflating the two would make
    # validation_failures unreadable as a measure of malformed output.
    unwarranted = bool(final) and not warrants_revision
    rejected_reasons = list(failures)
    if unwarranted:
        rejected_reasons.append(
            "the verification found no conflict and no unsupported claim, so "
            "there was nothing for a revision to fix"
        )

    if failures or unwarranted:
        final_answer, revised = draft, False
    else:
        final_answer = final or draft
        revised = bool(final) and final != draft.strip()

    result = Verification(
        verdicts=tuple(verdicts),
        relationship=relationship,
        conflicting_chunks=conflicting,
        safe_action=safe_action,
        escalate=as_bool(payload.get("escalate")),
        rationale=str(payload.get("rationale", "")).strip(),
        invented_ids=tuple(sorted(invented)),
        final_answer=final_answer,
        revised=revised,
        revision_rejected=bool(rejected_reasons and final),
        revision_rejected_reason="; ".join(rejected_reasons),
        validation_failures=tuple(failures),
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

    if result.invented_ids:
        return dataclasses.replace(
            result, confidence=str(policy.get("on_invented_evidence", "low"))
        )
    if result.validation_failures:
        return dataclasses.replace(
            result, confidence=str(policy.get("on_invented_evidence", "low"))
        )
    if result.is_resolved:
        # A supersession the system detected and resolved from metadata has a
        # right answer. Capping it at low would punish the system for noticing
        # the withdrawn document rather than for mishandling it.
        confidence = str(policy.get("on_resolved_conflict", "medium"))
    elif result.is_unresolved:
        confidence = str(policy.get("on_unresolved_conflict", "low"))
    elif result.all_insufficient or result.relationship == INSUFFICIENT:
        confidence = str(policy.get("on_insufficient", "low"))
    elif any(v.verdict == CONTRADICTED for v in result.verdicts):
        confidence = str(policy.get("on_contradiction", "low"))
    else:
        confidence = str(policy.get("on_supported", "medium"))
    return dataclasses.replace(result, confidence=confidence)
