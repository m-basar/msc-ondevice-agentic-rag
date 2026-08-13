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
    registry = FakeRegistry(DEV_TYPES)
    result = evaluate_gate(build(perfect(), DEV_TYPES), registry)

    assert (result.genuine_detected, len(result.genuine)) == (6, 6)
    assert result.genuine_classified == 6
    assert result.controls_falsely_detected == 0
    assert result.decision()[0] == "PROCEED"


def test_pilot_02_is_the_null_result():
    """Zero detections is the case that must not be talked into a REVISE."""
    registry = FakeRegistry(DEV_TYPES)
    result = evaluate_gate(build(silent(), DEV_TYPES), registry)

    assert result.genuine_detected == 0
    decision, reason = result.decision()
    assert decision == "STOP"
    assert "null result" in reason


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

    assert result.invalid_revisions_served == 1
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


def test_a_rejected_revision_is_the_guard_working_not_a_defect():
    rows = build(perfect(), DEV_TYPES)
    rows[0].update(answer_revised=True, revision_rejected=True,
                   hallucinated_citations=["ZZ-99#003"])
    result = evaluate_gate(rows, FakeRegistry(DEV_TYPES))

    assert result.invalid_revisions_served == 0
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
    assert "invalid revisions served         0" in text
    assert "DECISION: STOP" in text


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
