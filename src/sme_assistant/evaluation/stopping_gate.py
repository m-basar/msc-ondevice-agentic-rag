"""The development stopping rule, computed rather than eyeballed.

Pilot 02 produced zero detections across eighteen genuine-conflict questions.
The question that follows is not "is that bad" but "what happens next", and
that question has to be answerable without the person reading the numbers also
being the person who wants a particular answer.

So the rule is written here, committed before the run that tests it, and
executed. Reading a table and forming an impression is how a null result gets
talked into a positive one over several attempts.

What is counted
---------------
Everything here is countable without judgement. Conflict handling quality and
answer correctness are manual and blinded, and no proxy for them appears in
this module. What the gate measures is narrower and harder to argue with:

``detected``      the verifier called the family a conflict at all
``classified``    it named the relationship the registry declares
``false positive`` it called a compatible pair a conflict

Detection and classification are separate columns on purpose. Pilot diagnostics
found qwen2.5:3b flagging three of four families while correctly classifying
one, and a single "accuracy" figure would have hidden that. A verifier that
detects but misclassifies is a different finding from one that does neither,
and it calls for a different response.

Family level, not question level
--------------------------------
Each family carries three paraphrases of the same underlying question. A family
counts as detected when the verifier flags a majority of them, because one hit
in three is closer to sampling noise than to capability. The paraphrases exist
to test robustness to wording, so a rule that accepts one of three would defeat
the reason they are there.

The registry is consulted here. That is allowed: this is evaluation code, run
after the fact, and knowing which family a question belongs to is the entire
basis of scoring. The verifier itself never imports this module, and the
package boundary is enforced by a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..verify.schema import asserts_uncited_quantity
from .aggregate import aggregate

# The registry's vocabulary and the verifier's vocabulary are separate by
# design: the verifier infers a relationship without ever seeing a declared
# type. They have to be mapped to be compared, and the mapping lives here in
# the scoring code rather than anywhere the verifier can reach.
DECLARED_TO_INFERRED = {
    "version_supersession": "supersession",
    "mutually_exclusive": "mutually_exclusive",
    "stricter_looser": "stricter_looser",
    "compatible": "contextually_compatible",
}

# Majority of the paraphrases in a family.
MAJORITY = 2


@dataclass(frozen=True)
class FamilyOutcome:
    """One conflict family's behaviour across its paraphrases."""

    family_id: str
    declared: str
    is_conflict: bool
    questions: int
    detected: int
    classified: int

    @property
    def detected_by_majority(self) -> bool:
        return self.detected >= min(MAJORITY, self.questions)

    @property
    def classified_by_majority(self) -> bool:
        return self.classified >= min(MAJORITY, self.questions)

    @property
    def falsely_detected(self) -> bool:
        """A compatible family flagged as a conflict is a false positive."""
        return not self.is_conflict and self.detected_by_majority


@dataclass(frozen=True)
class GateResult:
    """The five quantities the decision rests on, and the decision itself."""

    arm: str
    families: tuple[FamilyOutcome, ...]
    invalid_revisions_served: int
    revisions_rejected: int
    citation_completeness_delta: float | None
    parse_failures: int
    answers: int

    # --- the five quantities ------------------------------------------------

    @property
    def genuine(self) -> tuple[FamilyOutcome, ...]:
        return tuple(f for f in self.families if f.is_conflict)

    @property
    def controls(self) -> tuple[FamilyOutcome, ...]:
        return tuple(f for f in self.families if not f.is_conflict)

    @property
    def genuine_detected(self) -> int:
        return sum(f.detected_by_majority for f in self.genuine)

    @property
    def genuine_classified(self) -> int:
        return sum(f.classified_by_majority for f in self.genuine)

    @property
    def controls_falsely_detected(self) -> int:
        return sum(f.falsely_detected for f in self.controls)

    # --- the decision --------------------------------------------------------

    def decision(self) -> tuple[str, str]:
        """Return ``(decision, reason)`` from the pre-committed rule.

        The thresholds are fractions of the families available rather than raw
        counts, so the same rule reads correctly on the development split's six
        genuine families and on any other split without being rewritten to suit
        whatever number came out.
        """
        genuine = len(self.genuine)
        controls = len(self.controls)

        if self.invalid_revisions_served:
            return ("DEFECT", (
                f"{self.invalid_revisions_served} revised answers were served "
                "with citations that do not resolve. The rejection guard did "
                "not hold, so the detection figures describe a broken pipeline "
                "and mean nothing until it is fixed."
            ))

        if controls and self.controls_falsely_detected == controls:
            return ("STOP", (
                f"every compatible control ({controls}/{controls}) was flagged "
                "as a conflict. A verifier that flags everything cannot be "
                "credited with the conflicts it flags, so detection on the "
                "genuine families is uninformative."
            ))

        if genuine and self.genuine_detected <= genuine // 6:
            return ("STOP", (
                f"{self.genuine_detected}/{genuine} genuine families detected "
                "by majority. This is the null result. Freeze the verifier and "
                "report it, rather than revising the prompt again."
            ))

        if genuine and self.genuine_detected >= (2 * genuine) // 3:
            if self.genuine_classified >= genuine // 2:
                return ("PROCEED", (
                    f"{self.genuine_detected}/{genuine} detected and "
                    f"{self.genuine_classified}/{genuine} correctly classified, "
                    f"with {self.controls_falsely_detected}/{controls} false "
                    "positives on the controls."
                ))
            return ("REVISE", (
                f"{self.genuine_detected}/{genuine} detected but only "
                f"{self.genuine_classified}/{genuine} correctly classified. "
                "Detection works and classification does not, which is a "
                "prompt-shaped problem rather than a capability ceiling."
            ))

        return ("REVISE", (
            f"{self.genuine_detected}/{genuine} genuine families detected. "
            "Above the null floor and below the proceed threshold: one further "
            "prompt revision is permitted, and it is the last one."
        ))


def _get(record: Mapping[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _invalid_revision_served(record: Mapping[str, Any]) -> bool:
    """A revision that reached the user while failing the grounding standard.

    Deliberately reads the **answer that was served**, not the decision that
    served it. The verifier's own rules are enforced in ``verify.schema``; if
    this re-derived them it would agree with them by construction and catch
    nothing. Reading the output instead means a fault anywhere in the serving
    path shows up here, which is how both defects so far were found.

    The standard is the one the system rests on: an answer that states a figure
    says which passage it came from. A revision citing evidence that was never
    retrieved fails it too.

    This is a correctness check on the pipeline rather than a measurement of
    the model, so it is reported separately and vetoes the rest of the gate.
    """
    if not record.get("answer_revised") or record.get("revision_rejected"):
        return False  # not revised, or the guard held and the draft was served
    if record.get("hallucinated_citations"):
        return True
    return asserts_uncited_quantity(str(record.get("answer") or ""))


def evaluate_gate(
    records: Sequence[Mapping[str, Any]],
    registry: Any,
    *,
    arm: str = "D",
    baseline: Sequence[Mapping[str, Any]] | None = None,
) -> GateResult:
    """Compute the gate for one arm's records.

    ``baseline`` is Arm B. The citation-completeness contrast is only
    meaningful against the arm D reused drafts from, and it is reported at
    group level because that is the unit the protocol infers over.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        family_id = record.get("family_id")
        if family_id and "verification" in record:
            grouped.setdefault(family_id, []).append(record)

    families: list[FamilyOutcome] = []
    for family_id, rows in sorted(grouped.items()):
        try:
            family = registry.by_id(family_id)
        except Exception:
            continue
        expected = DECLARED_TO_INFERRED.get(family.conflict_type)
        detected = sum(bool(r["verification"]["conflict_detected"]) for r in rows)
        classified = sum(
            r["verification"].get("relationship") == expected for r in rows
        )
        families.append(FamilyOutcome(
            family_id=family_id,
            declared=family.conflict_type,
            is_conflict=bool(family.is_conflict),
            questions=len(rows),
            detected=detected,
            classified=classified,
        ))

    delta = None
    if baseline is not None:
        here = aggregate(records, "scoring.citation_completeness").group_level
        there = aggregate(baseline, "scoring.citation_completeness").group_level
        if here is not None and there is not None:
            delta = here - there

    return GateResult(
        arm=arm,
        families=tuple(families),
        invalid_revisions_served=sum(_invalid_revision_served(r) for r in records),
        revisions_rejected=sum(bool(r.get("revision_rejected")) for r in records),
        citation_completeness_delta=delta,
        parse_failures=sum(
            bool(_get(r, "verification.parse_failed")) for r in records
        ),
        answers=sum("verification" in r for r in records),
    )


def format_gate(result: GateResult) -> Iterable[str]:
    """The gate as printable lines. Counts first, decision last."""
    genuine, controls = len(result.genuine), len(result.controls)
    decision, reason = result.decision()

    yield f"Stopping gate, arm {result.arm}, over {result.answers} verified answers"
    yield f"  families detected by majority ({MAJORITY} of 3 paraphrases):"
    for family in result.families:
        kind = "conflict " if family.is_conflict else "CONTROL  "
        mark = "detected" if family.detected_by_majority else "        "
        if family.falsely_detected:
            mark = "FALSE +ve"
        yield (f"    {family.family_id:<9} {kind}{family.declared:<22}"
               f"detected {family.detected}/{family.questions}  "
               f"classified {family.classified}/{family.questions}  {mark}")
    yield ""
    yield f"  genuine families detected      {result.genuine_detected:>3} / {genuine}"
    yield f"  genuine families classified    {result.genuine_classified:>3} / {genuine}"
    yield f"  controls falsely detected      {result.controls_falsely_detected:>3} / {controls}"
    yield f"  invalid revisions served       {result.invalid_revisions_served:>3}"
    yield f"    (revisions rejected by guard {result.revisions_rejected:>3})"
    delta = result.citation_completeness_delta
    yield ("  citation completeness D-B      "
           + ("    n/a" if delta is None else f"{delta:+7.3f}") + "   (group level)")
    yield f"  parse failures                 {result.parse_failures:>3} / {result.answers}"
    yield ""
    yield f"  DECISION: {decision}"
    for line in _wrap(reason, 70):
        yield f"    {line}"


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
