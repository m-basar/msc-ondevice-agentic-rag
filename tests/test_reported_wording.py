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
import re
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


# --- amendment 1.30: what the chapters and appendices may not say -----------


@pytest.fixture(scope="module")
def appendices() -> dict[str, str]:
    if not CHAPTERS.exists():
        pytest.skip("dissertation appendices are not present")
    found = {p.name: flatten(p.read_text(encoding="utf-8"))
             for p in CHAPTERS.glob("appendix_*.md")}
    if not found:
        pytest.skip("dissertation appendices are not present")
    return found


@pytest.mark.parametrize("phrase", [
    "all registered families",
    "**45** | **20** | **6**",
    "| **total** | **33**",
])
def test_no_pooled_headline_survives_in_the_chapters(chapters, appendices, phrase):
    """Amendment 1.30.5. Rule 7 of 1.29.3 forbade a pooled headline and rule 2
    of the same amendment named one as the principal denominator. Rule 7
    governs, and the total row it forbade is gone from the chapter, the
    appendix and the analysis."""
    everywhere = {**chapters, **appendices}
    offenders = [name for name, text in everywhere.items()
                 if flatten(phrase) in text]
    assert not offenders, f"{phrase!r} reappeared in {offenders}"


def test_the_exploratory_section_says_why_it_offers_no_total(chapters):
    results = chapters.get("chapter_4_results.md")
    if results is None:
        pytest.skip("results chapter is not present")
    assert "no row of totals is offered" in results
    assert "separate hypotheses with separate decision rules" in results


def test_the_results_chapter_runs_in_numerical_order(chapters):
    """4.13 was written before 4.12 and sat between 4.11 and it. A reader who
    finds sections out of order reasonably wonders what else was not checked."""
    path = CHAPTERS / "chapter_4_results.md"
    if not path.exists():
        pytest.skip("results chapter is not present")
    text = path.read_text(encoding="utf-8")
    headings = re.findall(r"^## 4\.(\d+) ", text, flags=re.MULTILINE)
    numbers = [int(n) for n in headings]
    assert numbers == sorted(numbers), f"sections out of order: {numbers}"
    tables = [int(n) for n in re.findall(r"^\*\*Table 4\.(\d+)\*\*", text,
                                         flags=re.MULTILINE)]
    assert tables == sorted(tables), f"tables out of order: {tables}"


def test_no_chapter_quotes_a_test_count(chapters):
    """Amendment 1.30.8. The count rises with every correction, so a number in
    prose is stale by the next commit and invites a reader to check a figure
    that measures nothing in particular."""
    offenders = [name for name, text in chapters.items()
                 if re.search(r"\b\d{2,4} automated tests\b", text)]
    assert not offenders, f"a test count is quoted in {offenders}"


def test_the_amendment_appendix_footer_describes_every_phase(appendices):
    """It once said every amendment after Phase A governed scoring, analysis or
    the hardware experiment. That stopped being true when Phase E added a
    demonstrator and Phase F added exploratory analysis."""
    footer = appendices.get("appendix_amendments.md")
    if footer is None:
        pytest.skip("the amendment appendix is not present")
    for phase in ("phase b", "phase c", "phase e", "phase f", "phase g",
                  "phase h"):
        assert phase in footer, phase
    assert "demonstrator built after all evidence was frozen" in footer


def test_the_provenance_paragraph_names_the_diagnostic_source(chapters):
    results = chapters.get("chapter_4_results.md")
    if results is None:
        pytest.skip("results chapter is not present")
    assert "verifier_relationship_diagnostic" in results
    assert "load_diagnostic_source()" in results
    assert "20260814_055018_d_test" in results


def amendment_counts() -> tuple[int, int]:
    """What `docs/PREREGISTRATION.md` actually holds, counted the same way
    `scripts/make_amendment_table.py` counts it."""
    source = ROOT / "docs" / "PREREGISTRATION.md"
    if not source.exists():
        pytest.skip("the pre-registration is not present")
    text = source.read_text(encoding="utf-8")
    return (len(re.findall(r"^# Amendment 1\.\d+ ", text, flags=re.MULTILINE)),
            len(re.findall(r"^## 1\.\d+\.\d+ ", text, flags=re.MULTILINE)))


def test_the_methodology_chapter_states_the_amendment_count_the_document_holds(
        chapters):
    """Amendment 1.34. Section 3.8.4 said twenty-nine amendments across 184
    numbered entries while the document held 34 across 217. It had drifted five
    amendments behind and nothing noticed, in the one chapter whose subject is
    that the study's discretion was constrained in writing.

    A test count is banned from the chapters outright under 1.30.8, because it
    measures nothing a reader can act on. An amendment count is a different
    thing: it is a provenance claim about a table in the submitted document, and
    a reader can check it in one glance. So it stays, pinned here to the
    document rather than to whatever was true when the sentence was typed.
    """
    amendments, entries = amendment_counts()
    methodology = chapters.get("chapter_3_methodology.md")
    if methodology is None:
        pytest.skip("the methodology chapter is not present")
    stated = re.search(r"all (\d+) amendments across (\d+) numbered entries",
                       methodology)
    assert stated, "section 3.8.4 no longer states the amendment count"
    assert (int(stated.group(1)), int(stated.group(2))) == (amendments, entries), (
        f"section 3.8.4 states {stated.group(1)} amendments across "
        f"{stated.group(2)} entries; the document holds {amendments} across "
        f"{entries}")


def test_no_chapter_states_an_amendment_count_that_is_not_the_current_one(
        chapters):
    """The reconciliation note carried "the 25 amendments" long after there were
    29, in a sentence arguing that the pre-registration is the project's
    strongest methodological feature. It now names them without counting them,
    which is what a dated historical note should do."""
    amendments, _ = amendment_counts()
    offenders = [(name, int(n)) for name, text in chapters.items()
                 for n in re.findall(r"\b(\d+) amendments\b", text)
                 if int(n) != amendments]
    assert not offenders, f"stale amendment counts: {offenders}"


def test_every_section_reference_in_the_dissertation_resolves(chapters, appendices):
    """A cross-reference to a section that does not exist is a broken pointer in
    the submitted document, and the reader who follows it is the examiner.

    Added while writing Chapter 5, which cites twelve sections of Chapter 4.
    This checks that a referenced number is a heading somewhere in the
    dissertation. It cannot check that the heading is the right one: Appendix C
    pointed at section 4.12 for the throttling figures, which is a real section
    reporting the verdict table, while 4.10 is where throttling is reported.
    That was corrected by reading, not by this test, and the limit is recorded
    here rather than left for someone to discover.
    """
    directory = CHAPTERS
    if not directory.exists():
        pytest.skip("dissertation documents are not present")
    documents = {p.name: p.read_text(encoding="utf-8")
                 for p in directory.glob("*.md")}
    if not documents:
        pytest.skip("dissertation documents are not present")
    headings = set()
    for text in documents.values():
        headings.update(re.findall(r"^#{2,4}\s+(\d+\.\d+(?:\.\d+)?)\s",
                                   text, flags=re.MULTILINE))
    assert headings, "no numbered headings were found at all"
    dangling = []
    for name, text in documents.items():
        for match in re.finditer(r"\b[Ss]ections?\s+(\d+\.\d+(?:\.\d+)?)", text):
            if match.group(1) not in headings:
                line = text[:match.start()].count("\n") + 1
                dangling.append(f"{name}:{line} -> section {match.group(1)}")
    assert not dangling, f"references to sections that do not exist: {dangling}"
