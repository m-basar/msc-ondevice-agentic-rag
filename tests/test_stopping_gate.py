"""The stopping rule has to behave correctly on data it has not seen.

These tests exist because the gate's only purpose is to remove discretion from
a decision I have an interest in. If it were written and never exercised, the
first time it ran would be on the real results, and any disagreement between
what it printed and what I expected would be resolved by editing the gate.

So the cases are fixed here, before the run: a verifier that works, one that
does not, one that flags everything, and one whose revision guard has failed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sme_assistant.evaluation.stopping_gate import (
    DECLARED_TO_INFERRED,
    evaluate_gate,
    format_gate,
)


class FakeRegistry:
    """Just enough registry to resolve a family id to a declared type."""

    def __init__(self, families: dict[str, str]) -> None:
        self.families = families

    def by_id(self, family_id: str):
        declared = self.families[family_id]
        return SimpleNamespace(
            conflict_type=declared, is_conflict=declared != "compatible"
        )


def record(family_id, index, *, relationship, revised=False, rejected=False,
           hallucinated=(), valid=True, completeness=1.0, parse_failed=False):
    detected = relationship in {"supersession", "mutually_exclusive", "stricter_looser"}
    return {
        "question_id": f"{family_id}-Q{index}",
        "group_id": f"{family_id}-G",
        "family_id": family_id,
        "answer_revised": revised,
        "revision_rejected": rejected,
        "hallucinated_citations": list(hallucinated),
        "has_valid_citation_ids": valid,
        "scoring": {"citation_completeness": completeness},
        "verification": {
            "relationship": relationship,
            "conflict_detected": detected,
            "parse_failed": parse_failed,
        },
    }


def build(pattern: dict[str, list[str]], registry_types: dict[str, str], **kw):
    """``pattern`` maps family id to the relationship inferred per question."""
    return [
        record(family_id, i, relationship=rel, **kw)
        for family_id, rels in pattern.items()
        for i, rel in enumerate(rels, start=1)
    ]


DEV_TYPES = {
    "CONF-01": "version_supersession",
    "CONF-05": "stricter_looser",
    "TUNE-01": "stricter_looser",
    "TUNE-02": "stricter_looser",
    "TUNE-05": "mutually_exclusive",
    "TUNE-06": "stricter_looser",
    "TUNE-03": "compatible",
    "TUNE-04": "compatible",
}


def perfect() -> dict[str, list[str]]:
    return {fid: [DECLARED_TO_INFERRED[t]] * 3 for fid, t in DEV_TYPES.items()}


def silent() -> dict[str, list[str]]:
    return {fid: ["no_relationship"] * 3 for fid in DEV_TYPES}


# --- the four cases the decision has to separate ----------------------------


def test_a_working_verifier_proceeds():
    """A baseline is required, because one declared condition compares to B.

    Without Arm B the citation-completeness condition cannot be evaluated, and
    the gate refuses to call an unmeasured condition met. That is the point of
    the change: a condition silently skipped for want of data used to read as
    a condition passed.
    """
    registry = FakeRegistry(DEV_TYPES)
    result = evaluate_gate(build(perfect(), DEV_TYPES), registry,
                           baseline=build(perfect(), DEV_TYPES))

    assert (result.genuine_detected, len(result.genuine)) == (6, 6)
    assert result.genuine_classified == 6
    assert result.controls_falsely_detected == 0
    assert result.decision()[0] == "PROCEED"


def test_zero_detection_stops_tuning_without_claiming_a_null_result():
    """Zero detections must not be talked into a REVISE, nor overclaimed.

    This test previously asserted `"null result" in reason` and kept passing
    after the wording was corrected to "not yet a null result", because the
    old phrase is a substring of its own negation. It was asserting the
    opposite of what it was named for and nothing noticed.
    """
    registry = FakeRegistry(DEV_TYPES)
    result = evaluate_gate(build(silent(), DEV_TYPES), registry)

    assert result.genuine_detected == 0
    decision, reason = result.decision()
    assert decision == "STOP"
    # Stop tuning this configuration...
    assert "Stop further prompt tuning for llama3.2:3b at k=6" in reason
    assert "model/window diagnostic" in reason
    # ...without claiming a finding about the approach or any other model.
    assert "not yet a null result" in reason
    assert "This is the null result" not in reason


def test_a_verifier_that_flags_everything_cannot_be_credited():
    """Detection on the conflicts is uninformative if the controls also fire.

    This is the failure mode the compatible families exist to catch, and it is
    the one most likely to look like success in a table of detection rates.
    """
    pattern = {fid: ["stricter_looser"] * 3 for fid in DEV_TYPES}
    result = evaluate_gate(build(pattern, DEV_TYPES), FakeRegistry(DEV_TYPES))

    assert result.genuine_detected == 6
    assert result.controls_falsely_detected == 2
    decision, reason = result.decision()
    assert decision == "STOP"
    assert "flags everything" in reason


def test_a_broken_revision_guard_vetoes_everything():
    """A served revision with unresolvable citations is a defect, not a result."""
    rows = build(perfect(), DEV_TYPES)
    rows[0]["answer_revised"] = True
    rows[0]["hallucinated_citations"] = ["ZZ-99#003"]
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES))

    assert result.structurally_invalid_revisions_served == 1
    assert result.decision()[0] == "DEFECT"


# --- the distinctions the gate is there to preserve --------------------------


def test_detection_and_classification_are_scored_separately():
    """Qwen flagged three of four families and classified one correctly.

    A single accuracy figure would have hidden that, and the two findings call
    for different responses: one is a prompt problem, the other a ceiling.
    """
    pattern = {fid: ["stricter_looser"] * 3 for fid in DEV_TYPES if fid.startswith("CONF")}
    pattern.update({fid: ["no_relationship"] * 3 for fid in DEV_TYPES if fid.startswith("TUNE")})
    result = evaluate_gate(build(pattern, DEV_TYPES), FakeRegistry(DEV_TYPES))

    supersession = next(f for f in result.families if f.family_id == "CONF-01")
    assert supersession.detected == 3, "flagged as a conflict"
    assert supersession.classified == 0, "but called stricter_looser, not supersession"


def test_one_paraphrase_in_three_is_not_a_detection():
    """The paraphrases test robustness to wording, so a majority is required."""
    pattern = dict(silent())
    pattern["CONF-01"] = ["supersession", "no_relationship", "no_relationship"]
    result = evaluate_gate(build(pattern, DEV_TYPES), FakeRegistry(DEV_TYPES))

    family = next(f for f in result.families if f.family_id == "CONF-01")
    assert family.detected == 1
    assert not family.detected_by_majority
    assert result.genuine_detected == 0


def test_contextually_compatible_is_not_a_detection():
    """Declining to flag is not flagging, or the false-positive rate is unmeasurable."""
    pattern = {fid: ["contextually_compatible"] * 3 for fid in DEV_TYPES}
    result = evaluate_gate(build(pattern, DEV_TYPES), FakeRegistry(DEV_TYPES))

    assert result.genuine_detected == 0
    assert result.controls_falsely_detected == 0
    assert result.genuine_classified == 0
    # The controls were correctly classified even though nothing was detected.
    controls = [f for f in result.families if not f.is_conflict]
    assert all(f.classified == 3 for f in controls)


def test_the_gate_reads_the_served_answer_not_the_decision():
    """Why the gate is not simply a copy of the verifier's own rules.

    If it re-derived the decision it would agree with the decision by
    construction and catch nothing. It reads what reached the user instead,
    which is how both defects so far were found. This is pilot 03's answer.
    """
    rows = build(perfect(), DEV_TYPES)
    rows[0].update(
        answer_revised=True,
        revision_rejected=False,
        hallucinated_citations=[],
        has_valid_citation_ids=False,
        answer=("The validity period is not explicitly stated in the provided "
                "evidence. However, it can be inferred that the 14-day validity "
                "period mentioned in OPS-03#002 may not apply."),
    )
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES))

    assert result.structurally_invalid_revisions_served == 1
    assert result.decision()[0] == "DEFECT"


def test_only_the_exact_abstention_template_may_be_served_uncited():
    """The gate's whole rule, in one sentence.

    Every served revision cites a passage that resolves, unless it is exactly
    the template. No detection of assertions, no reading of prose. Model-written
    prose that merely resembles an abstention does not qualify: that is how "it
    can be inferred that the 14-day validity period" reached a user.
    """
    from sme_assistant.verify.schema import ABSTENTION_TEXT

    rows = build(perfect(), DEV_TYPES)
    rows[0].update(answer_revised=True, revision_rejected=False,
                   has_valid_citation_ids=False, answer=ABSTENTION_TEXT)
    assert evaluate_gate(
        rows, FakeRegistry(DEV_TYPES)
    ).structurally_invalid_revisions_served == 0

    rows[0]["answer"] = "The evidence does not state a validity period."
    assert evaluate_gate(
        rows, FakeRegistry(DEV_TYPES)
    ).structurally_invalid_revisions_served == 1

    rows[0]["answer"] = "It remains valid for two weeks [OPS-03#002]."
    assert evaluate_gate(
        rows, FakeRegistry(DEV_TYPES)
    ).structurally_invalid_revisions_served == 0, (
        "a cited answer passes the gate; whether the verifier should have "
        "served it is decided in schema.parse, not here"
    )


def test_a_rejected_revision_is_the_guard_working_not_a_defect():
    rows = build(perfect(), DEV_TYPES)
    rows[0].update(answer_revised=True, revision_rejected=True,
                   hallucinated_citations=["ZZ-99#003"])
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES),
                           baseline=build(perfect(), DEV_TYPES))

    assert result.structurally_invalid_revisions_served == 0
    assert result.revisions_rejected == 1
    assert result.decision()[0] == "PROCEED"


def test_the_citation_contrast_is_group_level_and_needs_a_baseline():
    registry = FakeRegistry(DEV_TYPES)
    treated = build(perfect(), DEV_TYPES, completeness=0.9)
    baseline = build(perfect(), DEV_TYPES, completeness=0.6)

    assert evaluate_gate(treated, registry).citation_completeness_delta is None
    delta = evaluate_gate(treated, registry, baseline=baseline).citation_completeness_delta
    assert delta == pytest.approx(0.3)


def test_unverified_arms_contribute_no_families():
    """Arms A to C have no verification, so they have nothing to gate."""
    rows = [{k: v for k, v in r.items() if k != "verification"}
            for r in build(perfect(), DEV_TYPES)]
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES))

    assert result.families == ()
    assert result.answers == 0


def test_the_printed_report_states_the_counts_and_the_decision():
    result = evaluate_gate(build(silent(), DEV_TYPES), FakeRegistry(DEV_TYPES))
    text = "\n".join(format_gate(result))

    assert "genuine families detected        0 / 6" in text
    assert "controls falsely detected        0 / 2" in text
    assert "structurally invalid revisions " in text
    assert "DECISION: STOP" in text
    # The wording claims one model, one prompt, one window - not the approach.
    assert "llama3.2:3b at k=6" in text
    assert "not yet a null result" in text
    assert "Abstention, counted structurally" in text


def test_the_verifier_cannot_import_the_gate():
    """The gate reads declared types. The verifier must never reach it.

    Same boundary as the rest of the evaluation package: this is checked
    structurally rather than trusted, because oracle leakage is invisible in
    results that look plausible.
    """
    import ast
    from pathlib import Path

    verify = Path(__file__).resolve().parents[1] / "src" / "sme_assistant" / "verify"
    for source in verify.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            assert not any("evaluation" in n for n in names), (
                f"{source.name} imports evaluation code: {names}"
            )


# --- abstention is counted, split, and never read off the prose ---------------


def test_abstention_is_split_by_whether_the_question_had_an_answer():
    """Pilot 04 served 25 abstentions: 21 answerable, 4 unanswerable.

    "25 refusals" is not a finding. Refusing 21 answerable questions is a
    verifier error and refusing 4 unanswerable ones is correct behaviour, and
    a single count reports neither.
    """
    from sme_assistant.verify.schema import ABSTENTION_TEXT

    rows = build(silent(), DEV_TYPES)
    for row in rows[:5]:
        row["category"] = "conflict"
        row["verification"]["served_abstention"] = True
        row["answer"] = ABSTENTION_TEXT
    for row in rows[5:7]:
        row["category"] = "unanswerable"
        row["verification"]["served_abstention"] = True
        row["answer"] = ABSTENTION_TEXT
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES))

    assert result.abstentions_served == 7
    assert result.answerable_falsely_refused == 5
    assert result.unanswerable_correctly_refused == 2


def test_a_false_refusal_is_reported_not_suppressed():
    """The verifier judging a supported claim insufficient is a real failure.

    The template stops it asserting something ungrounded. It does not make the
    misjudgement disappear, and the gate must not let it.
    """
    from sme_assistant.verify.schema import ABSTENTION_TEXT

    rows = build(perfect(), DEV_TYPES)
    for row in rows[:4]:
        row["category"] = "conflict"
        row["verification"]["served_abstention"] = True
        row["answer"] = ABSTENTION_TEXT
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES))

    assert result.structurally_invalid_revisions_served == 0, "structurally fine"
    assert result.answerable_falsely_refused == 4
    # Reported, and NOT a gate condition. Amendment 1.9 put false refusals into
    # the unmet list, which added a condition to the declared gate after seeing
    # the data it judges. The declared gate has five conditions and this is not
    # one of them.
    _, reason = result.decision()
    assert "answerable questions refused" not in reason


def test_every_unmet_condition_is_reported_not_just_the_first():
    """A gate reporting one failure invites fixing that one and re-running."""
    rows = build(silent(), DEV_TYPES)
    rows[0]["verification"]["parse_failed"] = True
    rows[1]["verification"]["parse_failed"] = True
    rows[2]["verification"]["parse_failed"] = True
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES))
    assert result.parse_failures == 3

    # Detection is zero here, so STOP fires first; the point is that when the
    # declared gate is evaluated it lists everything, including what it could
    # not evaluate.
    partial = build(perfect(), DEV_TYPES)
    partial[0]["verification"]["parse_failed"] = True
    partial[1]["verification"]["parse_failed"] = True
    partial[2]["verification"]["parse_failed"] = True
    _, reason = evaluate_gate(partial, FakeRegistry(DEV_TYPES)).decision()
    assert "parse failures" in reason
    assert "UNAVAILABLE" in reason, "a condition that could not be checked is not a pass"


def test_an_unmeasured_condition_is_not_a_passed_condition():
    """No Arm B means citation completeness cannot be checked.

    The gate says so and withholds PROCEED, rather than passing a condition it
    never evaluated. Pilot 04 printed "n/a" against this line and returned a
    decision as though every condition had been tested.
    """
    result = evaluate_gate(build(perfect(), DEV_TYPES), FakeRegistry(DEV_TYPES))

    assert result.citation_completeness_delta is None
    decision, reason = result.decision()
    assert decision != "PROCEED"
    assert "UNAVAILABLE" in reason


# --- evidence attribution is three-way, not lenient ---------------------------


def refusal(qid, retrieved):
    from sme_assistant.verify.schema import ABSTENTION_TEXT
    return {
        "question_id": qid, "group_id": "G", "family_id": None,
        "category": "conflict", "answer": ABSTENTION_TEXT,
        "answer_revised": True, "revision_rejected": False,
        "hallucinated_citations": [], "has_valid_citation_ids": False,
        "retrieval": {"results": [{"chunk_id": c} for c in retrieved]},
        "verification": {"relationship": "no_relationship",
                         "conflict_detected": False, "parse_failed": False,
                         "served_abstention": True},
    }


def test_partial_retrieval_is_not_counted_as_having_the_evidence():
    """The lenient rule overstated what the verifier can be blamed for.

    TUNE-01-Q3 expected OPS-02#001 and CS-03#001 and received only the second.
    Refusing on half a disagreement is not the same failure as refusing with
    both sides in hand, and `wanted & got` called them identical.
    """
    expected = {
        "Q-all": ("A#001", "B#001"),
        "Q-partial": ("A#001", "B#001"),
        "Q-none": ("A#001", "B#001"),
        "Q-undeclared": (),
    }
    rows = [
        refusal("Q-all", ["A#001", "B#001"]),
        refusal("Q-partial", ["A#001"]),
        refusal("Q-none", ["Z#009"]),
        refusal("Q-undeclared", ["A#001"]),
    ]
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES), expected_chunks=expected)

    assert result.answerable_falsely_refused == 4
    assert result.false_refusals_all_evidence == 1
    assert result.false_refusals_partial_evidence == 1
    assert result.false_refusals_no_evidence == 1
    assert result.false_refusals_no_expectation == 1


def test_an_undeclared_expectation_is_not_evidence_of_anything():
    """"All required evidence present" is vacuously true with none declared.

    Folding those into the attributable count inflates it, which is what an
    18 that silently included them would have done.
    """
    rows = [refusal("Q1", ["A#001"])]
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES), expected_chunks={"Q1": ()})

    assert result.false_refusals_all_evidence == 0
    assert result.false_refusals_no_expectation == 1


# --- the revision budget is spent, so REVISE is no longer offered ------------


def test_no_revisions_remain_so_the_verdict_is_fail():
    """Revision 3 was the last one and was spent before pilot 06.

    A gate that keeps saying "one prompt revision remains" after the budget is
    gone invites one to be taken. Pilot 06 read REVISE on exactly that stale
    wording.
    """
    from sme_assistant.evaluation.stopping_gate import PROMPT_REVISIONS_REMAINING

    assert PROMPT_REVISIONS_REMAINING == 0

    pattern = dict(silent())
    for fid in ("CONF-01", "TUNE-01", "TUNE-02", "TUNE-05"):
        pattern[fid] = [DECLARED_TO_INFERRED[DEV_TYPES[fid]]] * 3
    result = evaluate_gate(build(pattern, DEV_TYPES), FakeRegistry(DEV_TYPES),
                           baseline=build(pattern, DEV_TYPES))

    assert result.genuine_detected == 4, "pilot 06's figure"
    decision, reason = result.decision()
    assert decision == "FAIL"
    assert "detection 4/6, needs 5" in reason
    assert "no development revisions remain" in reason
    assert "revision remains" not in reason


def test_claim_audit_completeness_is_reported():
    """Prompt revision 3 targeted it, so the summary has to show it."""
    rows = build(perfect(), DEV_TYPES)
    for row in rows[:7]:
        row["verification"]["claim_audit_complete"] = True
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES))

    assert result.claim_audits_complete == 7
    assert "claim audits complete" in "\n".join(format_gate(result))


# --- the gate is a development instrument ------------------------------------


def test_no_verdict_is_printed_when_the_rule_was_not_declared_for_the_split():
    """The test split gets measurements and no decision.

    DETECTION_REQUIRED is an absolute 5, declared for the development split's
    six genuine families. On the twelve-family test split it was satisfied at
    5/12 - 42% - by a bar that was declared to mean 83%, and the development
    run that failed at 4/6 was held to a stricter standard than the test run
    that passed.

    Rescaling it after seeing the test results would be inventing a test
    threshold to judge data already in hand. So no verdict is printed instead.
    """
    result = evaluate_gate(build(perfect(), DEV_TYPES), FakeRegistry(DEV_TYPES),
                           baseline=build(perfect(), DEV_TYPES))

    with_decision = "\n".join(format_gate(result, decision=True))
    without = "\n".join(format_gate(result, decision=False))

    assert "DECISION:" in with_decision
    assert "DECISION:" not in without
    # The measurements survive either way; only the verdict is withheld.
    for line in ("genuine families detected", "controls falsely detected",
                 "claim audits complete", "grounded answer coverage"):
        assert line in without


def test_the_threshold_is_documented_as_absolute_not_proportional():
    """The docstring claimed fractions and the code always used counts.

    That claim is what made applying the rule to another split look safe.
    """
    from sme_assistant.evaluation import stopping_gate

    source = stopping_gate.__doc__ or ""
    decision_doc = stopping_gate.GateResult.decision.__doc__ or ""
    assert "absolute counts declared for the development" in decision_doc
    assert "not scale-invariant" in decision_doc
    assert "fractions of the families available" not in decision_doc + source
