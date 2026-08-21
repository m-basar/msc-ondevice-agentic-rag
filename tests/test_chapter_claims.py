"""The analysis and closing chapters' numbers, held to the files they came from.

Chapters 5 to 7 are authored prose, not generated documents, so they cannot
be regenerated and compared. What it can be held to is its arithmetic: every figure
it states is read here from `hypotheses.json`, from the Raspberry Pi performance
report or from the frozen Arm D run, formatted the way the chapter states it,
and looked for in the text.

This is the same discipline amendment 1.35 applied to the amendment count in
section 3.8.4, which had drifted five amendments behind the document it
described because nothing checked it. A chapter that quotes forty figures from
four sources will drift the same way unless something reads both.

The test fails if a figure changes in the source, if a figure is mistyped in the
chapter, or if a sentence carrying one is deleted. It cannot tell whether a
figure is being used to support a claim it does not support; that is what
supervision is for.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DISSERTATION = ROOT / "docs" / "dissertation"
CHAPTER = DISSERTATION / "chapter_5_analysis.md"
CLOSING = ("chapter_6_discussion.md", "chapter_7_conclusion.md")
ANALYSIS = ROOT / "results" / "analysis"
FROZEN_D = ROOT / "results" / "runs" / "20260814_055018_D_test" / "answers.jsonl"


def load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} is not present")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chapter() -> str:
    if not CHAPTER.exists():
        pytest.skip("Chapter 5 is not present")
    return CHAPTER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def claims() -> list[tuple[str, str]]:
    """Every figure Chapter 5 states, read from its source and formatted as the
    chapter formats it. Adding a figure to the chapter without adding it here
    leaves it unchecked, which is the defect this file exists to prevent."""
    analysis = load(ANALYSIS / "hypotheses.json")
    performance = load(ANALYSIS / "performance_latest_test_performance_pi5_cpu.json")
    joined = load(ANALYSIS / "joined.jsonl")
    frozen = load(FROZEN_D)

    h = analysis["hypotheses"]
    out: list[tuple[str, str]] = []

    def add(label, value):
        out.append((label, str(value)))

    # RQ1: citation metrics, section 4.8
    primary = h["H3"]["primary"]["by_arm"]
    add("citation validity, lowest arm",
        f"{primary['D']['citation_validity']['group_level']:.4f}")
    add("citation validity, highest arm",
        f"{primary['A']['citation_validity']['group_level']:.4f}")
    add("citation support, lowest arm",
        f"{primary['C']['citation_support']['group_level']:.4f}")
    add("citation support, highest arm",
        f"{primary['B']['citation_support']['group_level']:.4f}")
    eligible = h["H3"]["sensitivity_common_eligibility"]["by_arm"]
    add("common-eligibility validity, lowest",
        f"{eligible['D']['citation_validity']:.4f}")
    add("common-eligibility validity, highest",
        f"{eligible['C']['citation_validity']:.4f}")

    # RQ1: answer correctness, section 4.11
    correctness = analysis["primary_metrics"]["answer_correctness"]["group_level"]
    for arm in "ABCD":
        add(f"answer correctness {arm}", f"{correctness[arm]:.4f}")

    # RQ2: the confirmatory contrast and its sensitivity, sections 4.5 and 4.6
    d_vs_b = h["H2"]["contrasts"]["D_vs_B"]
    add("H2 D against B", f"{d_vs_b['paired_mean_difference']:.4f}")
    fold = h["H2"]["leave_one_family_out"]["D_vs_B"]
    add("H2 leave-one-out minimum", f"{fold['min']:.4f}")
    add("H2 leave-one-out maximum", f"{fold['max']:.4f}")
    add("H1 D against B, magnitude",
        f"{abs(h['H1']['within_margin_leg']['contrasts']['D_vs_B']['paired_mean_difference']):.4f}")
    add("H1 Arm C family level", f"{h['H1']['levels']['family_level']['C']:.4f}")
    h1_fold = h["H1"]["leave_one_family_out"]["B_vs_A"]
    add("H1 leave-one-out minimum", f"{h1_fold['min']:.4f}")
    add("H1 leave-one-out maximum", f"{h1_fold['max']:.4f}")

    # RQ2: superseded citations, section 4.5
    superseded = analysis["primary_metrics"]["superseded_citation_rate"]["by_arm"]
    for arm in "ABCD":
        add(f"superseded citations {arm}",
            f"{superseded[arm]['hits']} of {superseded[arm]['questions']}")

    # RQ4: latency and device state, section 4.10
    add("H5 ratio", f"{h['H5']['ratio']:.2f}")
    arms = performance["arms"]
    add("Arm B mean latency", f"{arms['B']['wall_seconds']['mean']:.2f}")
    add("Arm D mean latency", f"{arms['D']['wall_seconds']['mean']:.2f}")
    add("verification mean", f"{arms['D']['verification_seconds']['mean']:.2f}")
    add("draft decode rate",
        f"{arms['D']['draft_stage']['eval_tokens_per_second']['mean']:.3f}")
    add("verifier decode rate",
        f"{arms['D']['verifier_stage']['eval_tokens_per_second']['mean']:.3f}")
    add("draft temperature", f"{arms['D']['draft_stage']['cpu_temp_c']['mean']:.1f}")
    add("verifier temperature",
        f"{arms['D']['verifier_stage']['cpu_temp_c']['mean']:.1f}")
    add("maximum temperature", arms["D"]["draft_stage"]["cpu_temp_c"]["max"])
    add("draft questions throttled",
        f"{arms['D']['draft_stage']['throttled_true']} of 68")
    host = arms["B"]["environment_start"]["host"]
    add("total memory", host["memory_total_gb"])
    add("memory at its lowest",
        arms["D"]["environment_start"]["host"]["memory_available_gb"])

    # Measurement quality, section 4.3
    add("Cohen's kappa",
        f"{analysis['measurement_quality']['abstention_agreement']['cohens_kappa']:.4f}")

    # Section 5.6: what the verifier did, from the frozen run itself
    def field(record, name):
        return bool((record.get("verification") or {}).get(name))

    add("conflicts detected", sum(field(r, "conflict_detected") for r in frozen))
    add("structured abstentions served",
        sum(field(r, "served_abstention") for r in frozen))
    add("drafts revised", sum(field(r, "revised") for r in frozen))

    # Section 5.4: the reviewer's abstention flag, per arm
    abstained = collections.Counter(r["arm"] for r in joined if r["abstained"])
    for arm in "ABCD":
        add(f"abstentions judged, arm {arm}", abstained[arm])

    # Section 5.3: the diagnostic restricted to questions where both sides were
    # retrieved, which is the claim the retrieval-ceiling argument rests on
    pair = (analysis["verifier_relationship_diagnostic"]["by_hypothesis_group"]
            ["H2_live_disagreement"]["restricted_to_pair_present"])
    add("H2 pair-present questions", pair["questions"])
    add("H2 pair-present detected", pair["detected"])
    return out


def test_every_figure_chapter_5_states_is_the_figure_its_source_holds(chapter, claims):
    missing = [f"{label} ({value})" for label, value in claims
               if value not in chapter]
    assert not missing, ("Chapter 5 does not state these figures as its sources "
                         f"hold them: {missing}")


def test_the_claim_list_is_not_empty(claims):
    """A check whose input list silently became empty would pass forever."""
    assert len(claims) >= 40, len(claims)


def test_chapter_5_marks_its_post_hoc_section_as_post_hoc(chapter):
    """Section 5.6 reads fields recorded during the frozen run and sets them
    against the rubric. The decision to read them was taken after the verdicts
    were known, and the chapter has to say so wherever it is read, not once in a
    footnote. Amendments 1.29 and 1.30.4 record why."""
    for phrase in ("post-hoc",
                   "No\nthreshold is applied, no verdict is drawn, and no hypothesis is revisited",
                   "not as evidence"):
        assert phrase in chapter, phrase


def test_chapter_5_does_not_restate_the_h5_verdict_as_its_own(chapter):
    """H5 is scored in the performance report and read from it. A chapter that
    recomputes or restates the verdict creates a second place for it to drift,
    which is the defect amendment 1.32.5 records."""
    assert "H5 was\nnot supported" in chapter or "H5 was not supported" in chapter
    assert "section 4.10" in chapter, (
        "the latency figures must be attributed to the section that reports them")


def test_chapter_5_does_not_compute_a_refusal_rate(chapter):
    """Section 4.11 declines to compute a rate of refusal over answerable
    questions, because that denominator was not pre-registered. Chapter 5 uses
    counts and says why, and a percentage here would quietly reintroduce the
    metric that was refused."""
    assert "carry\nno threshold and are not a refusal rate" in chapter
    for banned in ("refusal rate of", "% of answerable", "refusal rate was"):
        assert banned not in chapter, banned


# --- Chapters 6 and 7 ---------------------------------------------------------
#
# Amendment 1.37. The discussion and conclusion quote thirty figures between
# them, and a figure quoted in a conclusion is the one a reader is most likely
# to carry away. They are checked against the concatenation of the two chapters
# rather than against each chapter separately, because which of the two states a
# given figure is an editorial decision and pinning it here would make the test
# an obstacle to editing rather than a guard on the arithmetic.


@pytest.fixture(scope="module")
def closing_chapters() -> str:
    present = [DISSERTATION / name for name in CLOSING]
    if not all(p.exists() for p in present):
        pytest.skip("the closing chapters are not present")
    return "\n".join(p.read_text(encoding="utf-8") for p in present)


@pytest.fixture(scope="module")
def closing_claims() -> list[tuple[str, str]]:
    analysis = load(ANALYSIS / "hypotheses.json")
    performance = load(ANALYSIS / "performance_latest_test_performance_pi5_cpu.json")
    frozen = load(FROZEN_D)
    h = analysis["hypotheses"]
    out: list[tuple[str, str]] = []

    def add(label, value):
        out.append((label, str(value)))

    primary = h["H3"]["primary"]["by_arm"]
    eligible = h["H3"]["sensitivity_common_eligibility"]["by_arm"]
    add("validity, lowest", f"{primary['D']['citation_validity']['group_level']:.4f}")
    add("validity, highest", f"{primary['A']['citation_validity']['group_level']:.4f}")
    add("support, lowest", f"{primary['C']['citation_support']['group_level']:.4f}")
    add("support, highest", f"{primary['B']['citation_support']['group_level']:.4f}")
    add("common eligibility, lowest", f"{eligible['D']['citation_validity']:.4f}")
    add("common eligibility, highest", f"{eligible['C']['citation_validity']:.4f}")
    add("Arm C on supersession", f"{h['H1']['levels']['family_level']['C']:.4f}")
    add("H5 ratio", f"{h['H5']['ratio']:.2f}")

    arms = performance["arms"]
    add("Arm B mean latency", f"{arms['B']['wall_seconds']['mean']:.2f}")
    add("Arm D mean latency", f"{arms['D']['wall_seconds']['mean']:.2f}")
    add("verification mean", f"{arms['D']['verification_seconds']['mean']:.2f}")
    add("draft temperature, rounded",
        f"{arms['D']['draft_stage']['cpu_temp_c']['mean']:.0f}")
    add("verifier temperature, rounded",
        f"{arms['D']['verifier_stage']['cpu_temp_c']['mean']:.0f}")

    quality = analysis["measurement_quality"]
    add("Cohen's kappa", f"{quality['abstention_agreement']['cohens_kappa']:.4f}")
    add("items in the agreement table", quality["abstention_agreement"]["n"])
    add("duplicate groups", quality["rubric_score_agreement"]["groups"])

    h1_fold = h["H1"]["leave_one_family_out"]["B_vs_A"]
    add("H1 leave-one-out minimum", f"{h1_fold['min']:.4f}")
    add("H1 leave-one-out maximum", f"{h1_fold['max']:.4f}")
    add("H2 leave-one-out range",
        f"{h['H2']['leave_one_family_out']['D_vs_B']['range']:.4f}")
    levels = h["H2"]["levels"]["by_question"]
    add("H2 lowest arm", f"{min(levels.values()):.2f}")
    add("H2 highest arm", f"{max(levels.values()):.4f}")

    superseded = analysis["primary_metrics"]["superseded_citation_rate"]["by_arm"]
    for arm in "ACD":
        add(f"superseded citations {arm}",
            f"{superseded[arm]['hits']} of {superseded[arm]['questions']}")

    def field(record, name):
        return bool((record.get("verification") or {}).get(name))

    add("drafts revised", sum(field(r, "revised") for r in frozen))
    add("structured abstentions", sum(field(r, "served_abstention") for r in frozen))
    add("conflicts detected", sum(field(r, "conflict_detected") for r in frozen))

    pair = (analysis["verifier_relationship_diagnostic"]["by_hypothesis_group"]
            ["H2_live_disagreement"])
    add("live-disagreement questions", pair["questions"])
    add("both sides retrieved", pair["restricted_to_pair_present"]["questions"])
    add("detected where both sides were retrieved",
        pair["restricted_to_pair_present"]["detected"])
    return out


def test_every_figure_the_closing_chapters_state_is_the_one_its_source_holds(
        closing_chapters, closing_claims):
    missing = [f"{label} ({value})" for label, value in closing_claims
               if value not in closing_chapters]
    assert not missing, ("Chapters 6 and 7 do not state these figures as their "
                         f"sources hold them: {missing}")


def test_the_closing_claim_list_is_not_empty(closing_claims):
    assert len(closing_claims) >= 25, len(closing_claims)


def test_no_closing_chapter_claims_the_confidence_mechanism_is_calibrated(
        closing_chapters):
    """Section 1.4 declares that a categorical confidence mechanism is
    implemented and its calibration is not evaluated. The conclusion is where
    that boundary is most likely to be lost, because a conclusion summarises and
    a summary drops qualifiers. Every sentence in these two chapters that
    mentions calibration must either deny the claim or attribute it to the
    literature."""
    import re
    for sentence in re.findall(r"[^.]*calibrat[^.]*\.", closing_chapters):
        flat = " ".join(sentence.split()).lower()
        denies = any(phrase in flat for phrase in
                     ("not measured", "no claim is made", "not evaluated",
                      "miscalibrated", "measuring whether"))
        assert denies, f"a calibration claim is made without denial: {flat[:160]}"


def test_the_conclusion_reports_the_criterion_it_was_measured_against(
        closing_chapters):
    """Section 1.4 fixed what would count as success: verification improving on
    metadata at a cost still permitting interactive use. A conclusion that
    reports the result without restating the criterion invites the reader to
    supply a softer one."""
    flat = " ".join(closing_chapters.split())
    assert "beyond what metadata alone achieves" in flat
    assert "failed both halves" in flat


def test_every_citation_in_the_closing_chapters_has_a_reference_entry(
        closing_chapters):
    """Harvard citations are checked against the master reference list.

    That list lives beside the repository rather than inside it, at
    `../references.md`, which is where the project keeps it. When it is not
    there this test skips, and the skip is honest rather than a gap in
    coverage: there is nothing to check against. Copying `references.md` into
    `docs/dissertation/` would make this check run on every machine, and that is
    a decision about where the file belongs rather than one this test should
    force.
    """
    import re

    references = ROOT.parent / "references.md"
    if not references.exists():
        pytest.skip("the master reference list is not beside the repository")
    text = references.read_text(encoding="utf-8")
    known = set()
    for authors, year in re.findall(r"^\d+\.\s+([^(]+)\((\d{4})\)", text,
                                    flags=re.MULTILINE):
        surname = re.split(r",| and |\bet al\b", authors.strip())[0].strip()
        known.add((surname, year))
    # The two Regulations have no author-year form and are matched by number.
    regulations = set(re.findall(r"Regulation \(EU\) (\d{4}/\d+)", text))
    assert known and regulations, "the reference list did not parse"

    # Narrative citations ("Chen et al. (2024)") put only the year in
    # brackets, so the scan below has to catch the surname before it.
    flat = " ".join(closing_chapters.split())
    flat = re.sub(r"([A-Z][A-Za-z'-]+(?: et al\.)?) \((\d{4})\)",
                  r"(\1, \2)", flat)
    unknown = []
    for group in re.findall(r"\(([^()]*?\b(?:19|20)\d{2}[a-z]?)\)", flat):
        for part in group.split(";"):
            match = re.match(r"^(.*?),?\s*((?:19|20)\d{2})[a-z]?$", part.strip())
            if not match:
                continue
            surname = re.split(r",| and |\bet al\b", match.group(1))[0].strip()
            if surname and (surname, match.group(2)) not in known:
                unknown.append(f"{surname} ({match.group(2)})")
    for number in re.findall(r"Regulation \(EU\) (\d{4}/\d+)", flat):
        if number not in regulations:
            unknown.append(f"Regulation (EU) {number}")
    assert not unknown, f"cited but not in references.md: {sorted(set(unknown))}"
