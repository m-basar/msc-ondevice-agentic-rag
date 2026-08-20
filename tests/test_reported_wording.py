"""Guards on the wording of reported claims.

Numbers are checked everywhere else in this suite. These tests check the
sentences the numbers are wrapped in, because that is where this project has
actually gone wrong: a correct figure described as establishing more than it
does. Every phrase pinned below was written after a specific overstatement was
found and corrected, and the test exists so the overstatement cannot come back
silently.

The assertions run over the generated analysis output rather than over the
source of the script that writes it, because the output is what the
dissertation quotes. The chapter checks at the end are unusual in a code
repository and are deliberate: the dissertation lives in this repository, its
claims are derived from these files, and a prose regression is as damaging as a
numerical one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis" / "hypotheses.json"
CHAPTERS = ROOT / "docs" / "dissertation"


@pytest.fixture(scope="module")
def report() -> dict:
    if not ANALYSIS.exists():
        pytest.skip("analysis has not been generated")
    return json.loads(ANALYSIS.read_text(encoding="utf-8"))


def text_of(node) -> str:
    """Every string anywhere under a node, lowercased and joined."""
    if isinstance(node, str):
        return node.lower()
    if isinstance(node, dict):
        return " ".join(text_of(v) for v in node.values())
    if isinstance(node, list):
        return " ".join(text_of(v) for v in node)
    return ""


# --- what the design cannot establish ----------------------------------------


def test_h2_says_the_design_cannot_rule_out_a_small_effect(report):
    """Eight families cannot detect a small effect, but "could not have
    detected" asserts a property of the design that was never characterised.
    "Cannot rule out" is the claim the evidence supports."""
    reading = report["hypotheses"]["H2"]["sensitivity_reading"].lower()
    assert "cannot rule out a small true effect" in reading
    assert "could not have detected" not in reading
    assert "not evidence for the null" in reading


def test_h2c_reports_observed_events_rather_than_a_property_of_the_verifier(report):
    """Zero events on three control families is an observation about this
    sample. It is not a finding that the verification layer does not
    over-detect, and the reported wording must not read as one."""
    h2c = report["hypotheses"]["H2c"]
    reading = h2c["reading"].lower()
    assert "no false conflicts were observed" in reading
    assert "arm d" in reading and "arm b" in reading
    assert "not a general claim" in reading
    assert "the directional prediction is falsified" not in reading


def test_h2c_states_the_denominator_cannot_rule_out_over_detection(report):
    limitation = report["hypotheses"]["H2c"]["power_limitation"].lower()
    assert "cannot rule out moderate over-detection" in limitation
    assert "could not have detected" not in limitation
    assert "floor, not a measurement" in limitation


def test_no_hypothesis_claims_an_inferential_test_was_credible_or_run(report):
    """No test was pre-registered and none was computed. Saying that none "is
    credible" invites the reply that a credible one exists; the accurate
    statement is that the decision rests on the prespecified criteria."""
    everything = text_of(report)
    for banned in ("no significance test is credible",
                   "not statistically significant",
                   "p-value", "p =", "confidence interval of"):
        assert banned not in everything, banned


def test_no_arm_is_described_as_equivalent_to_another(report):
    """No interval is computed anywhere in this study, so equivalence was never
    tested. Amendment 1.16.2 withdrew the claim.

    The check bans the assertion, not the word: the decision text has to remain
    free to say that this is *not* an equivalence test, which is the whole point
    of the correction.
    """
    decisions = text_of(report["hypotheses"]["H1"]["within_margin_leg"]["decisions"])
    for claim in ("are equivalent", "is equivalent", "shown to be equivalent",
                  "arms are equal", "demonstrates equivalence"):
        assert claim not in decisions, claim
    assert "not a statistical equivalence test" in decisions
    assert "not the same as having been shown to be equal" in decisions


def test_the_descriptive_metrics_still_carry_no_verdict(report):
    """Amendment 1.25.3. Answer correctness and superseded citation rate are
    descriptive; a threshold attached later would be a test invented after the
    data."""
    primary = report["primary_metrics"]

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)

    found = set(keys(primary))
    for banned in ("verdict", "threshold", "decision", "decisions"):
        assert banned not in found, banned


# --- the same guards over the chapters that quote them -----------------------


def flatten(text: str) -> str:
    """Lowercase with runs of whitespace collapsed.

    Prose wraps, so a phrase this suite pins can be split across a line break
    and still be present. Matching the raw text would make these guards depend
    on where a line happens to end, which is not what they are for.
    """
    return " ".join(text.lower().split())


@pytest.fixture(scope="module")
def chapters() -> dict[str, str]:
    if not CHAPTERS.exists():
        pytest.skip("dissertation chapters are not present")
    found = {p.name: flatten(p.read_text(encoding="utf-8"))
             for p in CHAPTERS.glob("chapter_*.md")}
    if not found:
        pytest.skip("dissertation chapters are not present")
    return found


@pytest.mark.parametrize("phrase", [
    "no significance test is credible",
    "could not have detected",
    "calibrated confidence",
    "so the ratio holds",
    "indicates the cost is structural",
])
def test_withdrawn_phrases_do_not_reappear_in_any_chapter(chapters, phrase):
    """Each of these was written, found to overstate the evidence, and
    withdrawn. The parametrisation names them individually so a failure says
    which one came back."""
    offenders = [name for name, text in chapters.items() if phrase in text]
    assert not offenders, f"{phrase!r} reappeared in {offenders}"


def test_the_results_chapter_states_the_inferential_position_precisely(chapters):
    results = chapters.get("chapter_4_results.md")
    if results is None:
        pytest.skip("results chapter is not present")
    assert "no inferential test was pre-registered or computed" in results


def test_the_results_chapter_keeps_citation_support_as_an_upper_bound(chapters):
    """The automatic measure is a lower bound on citation *error*. Stating it
    the other way round reverses what the gap between validity and support can
    be said to show."""
    results = chapters.get("chapter_4_results.md")
    if results is None:
        pytest.skip("results chapter is not present")
    assert "upper bound" in results
    assert "lower bound on citation **error**" in results
