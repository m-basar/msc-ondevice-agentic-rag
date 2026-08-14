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

from ..verify.schema import ABSTENTION_TEXT, cites_a_passage
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

# The gate as originally declared, before pilots 02 and 03 were seen. Named
# constants so the numbers can be read against the pre-registration rather
# than reverse-engineered from branches.
# Declared against the DEVELOPMENT split's six genuine families. An absolute
# count, not a proportion: the docstring below once claimed these were
# fractions of the families available and they never were, so on a twelve-
# family split 5 would have meant 42% where it was declared to mean 83%.
# ``format_gate(decision=False)`` is how other splits are reported instead.
DETECTION_REQUIRED = 5              # of 6 genuine development families
CONTROL_FALSE_POSITIVES_ALLOWED = 0  # of 2 compatible controls
PARSE_FAILURES_ALLOWED = 2           # of the answers in the run
CITATION_COMPLETENESS_TOLERANCE = 0.05  # against Arm B, at group level

# Prompt revisions left in the development budget. Revision 3 was spent on the
# claims-audit defect before pilot 06 and was declared the last one. With none
# remaining, REVISE is not an available verdict: a gate that keeps offering
# another revision after the budget is gone invites one to be taken.
PROMPT_REVISIONS_REMAINING = 0


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
    structurally_invalid_revisions_served: int
    revisions_rejected: int
    citation_completeness_delta: float | None
    parse_failures: int
    answers: int
    claim_audits_complete: int = 0
    # Abstention is counted structurally and split by whether the question had
    # an answer. "25 refusals" is not a finding; "21 answerable questions
    # refused and 4 unanswerable ones correctly refused" is.
    abstentions_served: int = 0
    answerable_falsely_refused: int = 0
    unanswerable_correctly_refused: int = 0
    answerable_questions: int = 0
    unanswerable_questions: int = 0
    claim_making_answers: int = 0
    validly_cited_answers: int = 0
    # A false refusal is only the verifier's error when the evidence was
    # actually in front of it. Where the expected passage was never retrieved,
    # the verifier was right that what it held did not support an answer.
    false_refusals_all_evidence: int = 0
    false_refusals_partial_evidence: int = 0
    false_refusals_no_evidence: int = 0
    false_refusals_no_expectation: int = 0

    # --- coverage, reported beside every conditional figure ------------------

    @property
    def conditional_citation_validity(self) -> float | None:
        """Of the answers that made a claim, how many cited validly."""
        if not self.claim_making_answers:
            return None
        return self.validly_cited_answers / self.claim_making_answers

    @property
    def claim_making_coverage(self) -> float | None:
        """Of the questions asked, how many got an answer at all."""
        return self.claim_making_answers / self.answers if self.answers else None

    @property
    def grounded_answer_coverage(self) -> float | None:
        """Of the questions asked, how many got a validly cited answer.

        The figure that cannot be improved by refusing. A verifier abstaining
        on everything it would have got wrong drives conditional validity
        towards 1.00 while this falls, so the two are always reported together:
        quoting the conditional figure alone rewards selective abstention.
        """
        return self.validly_cited_answers / self.answers if self.answers else None

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
        """Return ``(decision, reason)`` from the **originally declared** gate.

        A weaker version of this ran for one commit: it permitted 4 of 6
        detections, tolerated one control false positive, and ignored parse
        failures and citation completeness entirely. Those numbers were written
        after pilots 02 and 03, appeared in no document, and were looser than
        what had been declared. Thresholds relaxed after seeing the data are
        not a pre-registration, whatever they are called, so the declared ones
        are restored:

        =========================  ==================================
        detection                  >= 5 of 6 genuine families
        control false positives    0 of 2
        parse failures             <= 2 of the answers
        citation completeness      within 0.05 of Arm B
        invalid revisions served   0
        =========================  ==================================

        Every condition must hold to PROCEED. Anything genuinely below the null
        floor STOPs; everything between is a REVISE.

        **Scope.** These are absolute counts declared for the development
        split's six genuine families. They are not proportions and this rule is
        not scale-invariant, so it is meaningful on that split alone. An
        earlier docstring claimed they were fractions of the families
        available, which is what made applying the rule to the twelve-family
        test split look safe: 5 satisfied a bar declared to mean 83% at 42%,
        and the development run that failed at 4/6 was held to a stricter
        standard than the test run that passed. Other splits are reported with
        ``format_gate(decision=False)`` and no verdict.
        """
        genuine, controls = len(self.genuine), len(self.controls)
        answers = self.answers or 1

        if self.structurally_invalid_revisions_served:
            return ("DEFECT", (
                f"{self.structurally_invalid_revisions_served} revised answers "
                "were served without a citation that resolves and without "
                "being the abstention template.\n\n"
                "SCOPE: this compromises the served-answer results - citation "
                "metrics, revision counts, answer quality. The inference "
                "results above are unaffected: detection, classification and "
                "the control false-positive rate are read from the verifier's "
                "own output before anything is served, so they stand.\n\n"
                "An earlier wording said the detection figures 'describe a "
                "broken pipeline and mean nothing'. That discarded valid "
                "evidence, which is not a form of caution."
            ))

        if controls and self.controls_falsely_detected == controls:
            return ("STOP", (
                f"every compatible control ({controls}/{controls}) was flagged "
                "as a conflict. A verifier that flags everything cannot be "
                "credited with the conflicts it flags."
            ))

        if genuine and self.genuine_detected == 0:
            # Deliberately narrow. "The null result" claimed more than the run
            # supports: it reads as a finding about verification as an
            # approach, when what was tested is one model, one prompt revision
            # and one evidence window. Qwen has not been run, and neither has
            # the isolated-pair condition.
            return ("STOP", (
                "Stop further prompt tuning for llama3.2:3b at k=6; run the "
                "precommitted model/window diagnostic. "
                f"0/{genuine} genuine families detected by majority"
                + (f", and {self.answerable_falsely_refused} of "
                   f"{self.answerable_questions} answerable questions were "
                   "refused outright." if self.answerable_falsely_refused else ".")
                + " This is not yet a null result for the verification "
                "approach or for any other model."
            ))

        # The declared gate, every condition required.
        unmet = []
        if genuine and self.genuine_detected < DETECTION_REQUIRED:
            unmet.append(f"detection {self.genuine_detected}/{genuine}, "
                         f"needs {DETECTION_REQUIRED}")
        if self.controls_falsely_detected > CONTROL_FALSE_POSITIVES_ALLOWED:
            unmet.append(f"control false positives "
                         f"{self.controls_falsely_detected}/{controls}, needs "
                         f"{CONTROL_FALSE_POSITIVES_ALLOWED}")
        if self.parse_failures > PARSE_FAILURES_ALLOWED:
            unmet.append(f"parse failures {self.parse_failures}/{answers}, "
                         f"needs at most {PARSE_FAILURES_ALLOWED}")
        delta = self.citation_completeness_delta
        if delta is not None and delta < -CITATION_COMPLETENESS_TOLERANCE:
            unmet.append(f"citation completeness {delta:+.3f} against B, "
                         f"needs no worse than "
                         f"-{CITATION_COMPLETENESS_TOLERANCE}")

        # Every condition, met or not, including the ones that cannot be
        # evaluated. A gate that reports only the first failure invites the
        # reader to fix that one and re-run, and a condition silently skipped
        # for want of data reads as a condition passed.
        if delta is None:
            unmet.append("citation completeness against B: UNAVAILABLE, Arm B "
                         "absent from the run index")

        # False refusals are NOT a declared condition. Amendment 1.9 added them
        # to this list, which is adding a gate condition after seeing the data
        # the gate is judging - the exact move the pre-registration exists to
        # prevent, made by me. The count is reported above and stays out of the
        # verdict. The declared gate has five conditions and this is not one.

        if not unmet:
            return ("PROCEED", (
                f"{self.genuine_detected}/{genuine} detected, "
                f"{self.controls_falsely_detected}/{controls} control false "
                f"positives, {self.parse_failures}/{answers} parse failures, "
                "every declared condition met."
            ))

        summary = "the declared gate is not met: " + "; ".join(unmet) + "."
        if PROMPT_REVISIONS_REMAINING <= 0:
            return ("FAIL", (
                summary + " The final prompt revision has been spent, so no "
                "development revisions remain. Freeze this configuration and "
                "report the result."
            ))
        return ("REVISE", (
            summary + f" {PROMPT_REVISIONS_REMAINING} prompt revision(s) "
            "remain."
        ))

def _get(record: Mapping[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _structurally_invalid_revision_served(record: Mapping[str, Any]) -> bool:
    """A revision that reached the user while failing the grounding standard.

    Deliberately reads the **answer that was served**, not the decision that
    served it. The verifier's own rules are enforced in ``verify.schema``; if
    this re-derived them it would agree with them by construction and catch
    nothing. Reading the output instead means a fault anywhere in the serving
    path shows up here, which is how both defects so far were found.

    The standard is simple enough to state in one line: **every served revision
    cites a passage that resolves, unless it is exactly the abstention
    template.** No detection of assertions, no inspection of prose. Either the
    answer points at evidence or it is the system's fixed refusal.

    This is a correctness check on the pipeline rather than a measurement of
    the model, so it is reported separately and vetoes the rest of the gate.
    """
    if not record.get("answer_revised") or record.get("revision_rejected"):
        return False  # not revised, or the guard held and the draft was served
    served = str(record.get("answer") or "")
    if served.strip() == ABSTENTION_TEXT:
        return False  # the system's own words, which assert nothing
    # Everything else that was served must cite a passage that resolves. Stated
    # this way rather than as a list of things to detect: the previous version
    # hunted for figures in prose and could not have caught an assertion
    # phrased without digits.
    return bool(record.get("hallucinated_citations")) or not cites_a_passage(served)


def evaluate_gate(
    records: Sequence[Mapping[str, Any]],
    registry: Any,
    *,
    arm: str = "D",
    baseline: Sequence[Mapping[str, Any]] | None = None,
    expected_chunks: Mapping[str, Sequence[str]] | None = None,
) -> GateResult:
    """Compute the gate for one arm's records.

    ``baseline`` is Arm B. The citation-completeness contrast is only
    meaningful against the arm D reused drafts from, and it is reported at
    group level because that is the unit the protocol infers over.
    """
    def _abstained(record: Mapping[str, Any]) -> bool:
        verification = record.get("verification") or {}
        if "served_abstention" in verification:
            return bool(verification["served_abstention"])
        # Runs recorded before the flag existed.
        return str(record.get("answer") or "").strip() == ABSTENTION_TEXT

    def _valid_citations(record: Mapping[str, Any]) -> bool:
        return bool(record.get("has_valid_citation_ids"))

    def _evidence_state(record: Mapping[str, Any]) -> str:
        """How much of the expected evidence reached the verifier.

        Three states, not two. The first version used ``wanted & got`` - at
        least one expected chunk - and reported the result as "the verifier had
        the evidence". That is the retrieval evaluation's **lenient** rule, and
        describing it as complete evidence overstates what can be attributed to
        the verifier: TUNE-01-Q3 expected OPS-02#001 and CS-03#001 and received
        only the second, so it was refusing on half a disagreement.
        """
        wanted = set((expected_chunks or {}).get(record.get("question_id"), ()))
        if not wanted:
            return "no_expectation"
        got = {s.get("chunk_id") for s in
               (record.get("retrieval") or {}).get("results", [])}
        if wanted <= got:
            return "all"
        return "partial" if wanted & got else "none"

    claim_making = [r for r in records if not _abstained(r)]
    answerable = sum(r.get("category") != "unanswerable" for r in records)
    unanswerable = len(records) - answerable
    abstentions = sum(_abstained(r) for r in records)
    falsely_refused = sum(
        _abstained(r) and r.get("category") != "unanswerable" for r in records
    )
    correctly_refused = abstentions - falsely_refused
    false_refusals = [r for r in records
                      if _abstained(r) and r.get("category") != "unanswerable"]
    states = [_evidence_state(r) for r in false_refusals]

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
        structurally_invalid_revisions_served=sum(
            _structurally_invalid_revision_served(r) for r in records
        ),
        abstentions_served=abstentions,
        answerable_falsely_refused=falsely_refused,
        unanswerable_correctly_refused=correctly_refused,
        answerable_questions=answerable,
        unanswerable_questions=unanswerable,
        claim_making_answers=len(claim_making),
        validly_cited_answers=sum(_valid_citations(r) for r in claim_making),
        false_refusals_all_evidence=states.count("all"),
        false_refusals_partial_evidence=states.count("partial"),
        false_refusals_no_evidence=states.count("none"),
        false_refusals_no_expectation=states.count("no_expectation"),
        revisions_rejected=sum(bool(r.get("revision_rejected")) for r in records),
        citation_completeness_delta=delta,
        parse_failures=sum(
            bool(_get(r, "verification.parse_failed")) for r in records
        ),
        claim_audits_complete=sum(
            bool(_get(r, "verification.claim_audit_complete")) for r in records
        ),
        answers=sum("verification" in r for r in records),
    )


def format_gate(result: GateResult, *, decision: bool = True) -> Iterable[str]:
    """The gate as printable lines. Counts first, decision last.

    ``decision=False`` prints the measurements without a verdict, for splits
    the rule was not declared against. The thresholds were set for the
    development split's six genuine families; applying them to a split with a
    different number of families compares against a bar that was never
    declared for it.
    """
    genuine, controls = len(result.genuine), len(result.controls)

    label = "Stopping gate" if decision else "Verifier diagnostics"
    yield f"{label}, arm {result.arm}, over {result.answers} verified answers"
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
    yield ("  structurally invalid revisions "
           f"{result.structurally_invalid_revisions_served:>3}"
           "   (citations resolve, or the template)")
    yield f"    (revisions rejected by guard {result.revisions_rejected:>3})"
    yield ""
    yield "  Coverage. Conditional figures are never reported alone: a verifier"
    yield "  that abstains on what it would get wrong improves the first line"
    yield "  and worsens the third."

    def _pct(value):
        return "     -" if value is None else f"{value:6.3f}"

    yield (f"    conditional citation validity {_pct(result.conditional_citation_validity)}"
           f"   {result.validly_cited_answers}/{result.claim_making_answers}"
           "   given the answer made a claim")
    yield (f"    claim-making coverage         {_pct(result.claim_making_coverage)}"
           f"   {result.claim_making_answers}/{result.answers}"
           "   answered at all")
    yield (f"    grounded answer coverage      {_pct(result.grounded_answer_coverage)}"
           f"   {result.validly_cited_answers}/{result.answers}"
           "   cannot be improved by refusing")
    yield ""
    yield "  Abstention, counted structurally rather than read off the prose:"
    yield (f"    served                       {result.abstentions_served:>3}"
           f" / {result.answers}")
    yield (f"    answerable, falsely refused  "
           f"{result.answerable_falsely_refused:>3} / {result.answerable_questions}"
           + ("   <-- verifier error, not suppressed"
              if result.answerable_falsely_refused else ""))
    yield (f"    unanswerable, correctly      "
           f"{result.unanswerable_correctly_refused:>3} / "
           f"{result.unanswerable_questions}")
    yield "      of the false refusals, the expected evidence was:"
    yield (f"        all retrieved      {result.false_refusals_all_evidence:>3}"
           "   the verifier's error; it had everything")
    yield (f"        partly retrieved   {result.false_refusals_partial_evidence:>3}"
           "   mixed; it refused on half a disagreement")
    yield (f"        none retrieved     {result.false_refusals_no_evidence:>3}"
           "   retrieval's error, not the verifier's")
    yield (f"        none declared      {result.false_refusals_no_expectation:>3}"
           "   no expectation to check against")
    delta = result.citation_completeness_delta
    yield ("  citation completeness D-B      "
           + ("    n/a" if delta is None else f"{delta:+7.3f}") + "   (group level)")
    yield f"  parse failures                 {result.parse_failures:>3} / {result.answers}"
    yield (f"  claim audits complete          {result.claim_audits_complete:>3}"
           f" / {result.answers}   (what prompt revision 3 targeted)")
    yield ""
    if not decision:
        return
    verdict, reason = result.decision()
    yield f"  DECISION: {verdict}"
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
