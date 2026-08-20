"""Post-unsealing analysis: joining the manual judgements to the arms.

This is the first module in the project permitted to know which arm produced
which answer, and it runs only after both scoring passes are complete and the
key has been opened. Everything before it was built to make that ordering
enforceable rather than promised.

Four things it does that a hand-written script would get wrong.

**It aggregates at the unit the protocol names.** Three paraphrases of one
conflict family are three questions and one observation. Section 5 makes the
family the unit of analysis, so every contrast is a mean within family followed
by a mean across families, and the sample size reported is the number of
families. Question-level figures are produced alongside for readability and are
never the basis of a claim.

**It pairs within family before averaging.** The difference between two arms is
computed family by family and then averaged, not as a difference of two overall
means. With eight families of unequal size those are not the same number, and
only the first answers "does D beat B on the same material".

**It applies the declared decision rule rather than inventing one.** A contrast
is *supported* only when the paired mean difference exceeds 0.25 **and** the
direction holds in the required number of families; meeting one but not the
other is *suggestive*; meeting neither is *not supported*, including when the
point estimate favours the contribution. The thresholds come from section 5 as
corrected by amendment 1.5.3, and are passed in explicitly so a reader can see
which rule produced which verdict.

**It reports the confounded contrast as confounded.** The arms are a tree rooted
at B, not a ladder. B versus D isolates verification. C versus D changes both
the retrieval mode and the verification layer, so it is computed and reported as
the practical comparison a practitioner would make, and is never used to
attribute an effect to verification.

Nothing here re-scores anything. The judgement logs are read-only inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .manual_scoring import load_abstention, load_judgements, load_sheet
from .question_set import QuestionSet, load_question_set
from .run_writer import read_run

SCHEMA_VERSION = "1.0"

#: Expected behaviours grouped into the sets the hypotheses are stated over.
SUPERSESSION = ("cite_current_only",)
LIVE_DISAGREEMENT = ("surface_both_and_escalate", "prefer_stricter_and_escalate")
COMPATIBLE_CONTROL = ("answer_without_flagging_conflict",)
GAPS = ("abstain",)
CORRECTNESS = ("answer_directly", "answer_and_flag_gap")

#: Section 5, as corrected by amendment 1.5.3.
EFFECT_THRESHOLD = 0.25
DIRECTION_REQUIRED = {"H1": (3, 4), "H2": (6, 8)}


class AnalysisError(RuntimeError):
    """Raised when the inputs cannot be joined into a defensible dataset."""


@dataclass(frozen=True)
class Joined:
    """One answer, with its manual judgement and its automatic metrics."""

    item: int
    arm: str
    question_id: str
    group_id: str
    category: str
    expected_behaviour: str
    score: int
    asserts_conflict: bool
    abstained: bool
    uncertain: bool
    arm_identified: bool | None
    # From the run record rather than the reviewer.
    has_valid_citation_ids: bool | None = None
    citation_support: float | None = None
    citation_completeness: float | None = None
    cited_superseded: int = 0
    wall_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


#: The four frozen quality runs, by directory name. Amendment 1.16.
#:
#: Naming them is the difference between a rule and a hope. The first version
#: of this function accepted any untagged ``*_test`` directory whose manifest
#: said ``split == "test"``, and a performance run satisfies both: it is a run
#: on the test split, and nothing stops it being written without a tag. So the
#: enforcement amendment 1.15.3 claimed did not exist. A closed list cannot be
#: satisfied by a run created later, whatever it is called.
FROZEN_QUALITY_RUNS: tuple[str, ...] = (
    "20260814_054606_A_test",
    "20260814_054754_B_test",
    "20260814_054908_C_test",
    "20260814_055018_D_test",
)


def quality_run_directories(root: Path | str) -> list[Path]:
    """The four frozen quality runs, by name, and nothing else.

    Amendment 1.15 declares the laptop quality run the sole evidential source
    for H1 to H4. This is what makes that true rather than intended.

    Three checks, and the first is the one that matters. The directory must be
    on the closed list; its manifest must still say ``split == "test"``, so a
    renamed directory cannot impersonate one; and it must not be marked
    ``purpose: performance``, which catches the case where a frozen run is
    somehow re-executed in performance mode.

    Refusing to find one of the four is an error rather than a silent shortfall.
    An analysis quietly computed over three arms would still produce numbers.
    """
    found: list[Path] = []
    for name in FROZEN_QUALITY_RUNS:
        directory = Path(root) / name
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.exists():
            raise AnalysisError(
                f"Frozen quality run {name} is missing from {root}. The reported "
                "results are defined over exactly these four runs, so an analysis "
                "without one of them is not the pre-registered analysis."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") != "test":
            raise AnalysisError(
                f"{name} is on the frozen quality list but its manifest says "
                f"split={manifest.get('split')!r}. The directory has been "
                "replaced."
            )
        if manifest.get("purpose") == "performance":
            raise AnalysisError(
                f"{name} is marked purpose=performance and cannot be used for "
                "answer quality. See amendment 1.15."
            )
        found.append(directory)
    return found


def load_key(path: Path | str) -> dict[str, str]:
    """The opaque code for each arm. Opening this is the unblinding step."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping = payload.get("mapping")
    if not mapping:
        raise AnalysisError(f"{path} contains no mapping")
    return dict(mapping)


def join(
    *,
    sheet: Path | str,
    key: Path | str,
    judgements: Path | str,
    abstention: Path | str | None,
    question_set: QuestionSet,
    runs: Sequence[Path | str] = (),
) -> tuple[Joined, ...]:
    """Join blinded judgements to arms, and to the automatic metrics.

    ``abstention`` supplies the re-pass values, which amendment 1.14.4 rule 1
    makes the reported value of that field. Passing ``None`` falls back to the
    first pass and is provided only so the superseded figures remain
    reproducible.
    """
    items = load_sheet(sheet)
    first = load_judgements(judgements)
    second = load_abstention(abstention) if abstention else {}
    code_to_arm = {code: arm for arm, code in load_key(key).items()}

    missing = [i.item for i in items if i.item not in first]
    if missing:
        raise AnalysisError(
            f"{len(missing)} items have no judgement; analysis requires a "
            "complete pass. Run `score_answers.py status`."
        )

    # Keyed on (arm, question). A second run of the same arm would overwrite
    # the first silently, and the analysis would report metrics from a run
    # nobody intended to include while the manual scores still came from the
    # frozen one. Refusing is the only safe behaviour: there is no way to tell
    # from here which of the two was meant.
    automatic: dict[tuple[str, str], dict[str, Any]] = {}
    for directory in runs:
        manifest, answers = read_run(directory)
        arm = manifest["arm"]["arm"]
        for record in answers:
            slot = (arm, record["question_id"])
            if slot in automatic:
                raise AnalysisError(
                    f"Two runs supply arm {arm} question {record['question_id']}. "
                    f"{Path(directory).name} collides with an earlier run. The "
                    "automatic metrics would take whichever was read last while "
                    "the manual scores stayed with the frozen run."
                )
            automatic[slot] = record

    joined: list[Joined] = []
    for item in items:
        arm = code_to_arm.get(item.system)
        if arm is None:
            raise AnalysisError(f"No arm for opaque code {item.system!r}")
        question = question_set.by_id(item.question_id)
        judgement = first[item.item]
        record = automatic.get((arm, item.question_id), {})
        scoring = record.get("scoring") or {}
        joined.append(
            Joined(
                item=item.item,
                arm=arm,
                question_id=item.question_id,
                group_id=question.group_id,
                category=question.category,
                expected_behaviour=question.expected_behaviour,
                score=judgement.score,
                asserts_conflict=judgement.asserts_conflict,
                abstained=(
                    second[item.item].abstained
                    if item.item in second
                    else judgement.abstained
                ),
                uncertain=judgement.uncertain,
                arm_identified=judgement.arm_identified,
                has_valid_citation_ids=record.get("has_valid_citation_ids"),
                citation_support=scoring.get("citation_support"),
                citation_completeness=scoring.get("citation_completeness"),
                cited_superseded=len(record.get("cited_superseded") or ()),
                wall_seconds=record.get("wall_seconds"),
            )
        )
    return tuple(joined)


# --- aggregation -------------------------------------------------------------


def select(rows: Iterable[Joined], behaviours: Sequence[str]) -> tuple[Joined, ...]:
    return tuple(r for r in rows if r.expected_behaviour in behaviours)


def family_table(
    rows: Iterable[Joined], *, field_name: str = "score"
) -> dict[str, dict[str, float]]:
    """Per-family, per-arm means. The family is the unit of independence."""
    rows = list(rows)
    families = sorted({r.group_id for r in rows})
    table: dict[str, dict[str, float]] = {}
    for family in families:
        table[family] = {}
        for arm in sorted({r.arm for r in rows}):
            values = [
                float(getattr(r, field_name))
                for r in rows
                if r.group_id == family and r.arm == arm
            ]
            if values:
                table[family][arm] = mean(values)
    return table


def question_table(
    rows: Iterable[Joined], *, field_name: str = "score"
) -> dict[str, float]:
    """Plain per-arm means over questions. Reported for readability only."""
    rows = list(rows)
    return {
        arm: mean([float(getattr(r, field_name)) for r in rows if r.arm == arm])
        for arm in sorted({r.arm for r in rows})
    }


@dataclass(frozen=True)
class Contrast:
    """One arm against another, paired within family."""

    treatment: str
    baseline: str
    families: tuple[str, ...]
    per_family: dict[str, float]
    treatment_mean: float
    baseline_mean: float
    paired_mean_difference: float
    better: int
    tied: int
    worse: int
    confounded: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {k: getattr(self, k) for k in self.__dataclass_fields__}
        payload["families"] = list(self.families)
        return payload


#: Contrasts that change more than one variable at once, from section 2.
CONFOUNDED = {("D", "C"), ("C", "D")}


def contrast(
    table: Mapping[str, Mapping[str, float]], treatment: str, baseline: str
) -> Contrast:
    """Difference within family, then averaged. Not a difference of means.

    With families of unequal size those two quantities differ, and only this one
    answers whether the treatment beat the baseline on the same material.
    """
    families = tuple(
        f for f in sorted(table) if treatment in table[f] and baseline in table[f]
    )
    if not families:
        raise AnalysisError(f"No families carry both {treatment} and {baseline}")
    per_family = {f: table[f][treatment] - table[f][baseline] for f in families}
    diffs = list(per_family.values())
    confounded = (treatment, baseline) in CONFOUNDED
    return Contrast(
        treatment=treatment,
        baseline=baseline,
        families=families,
        per_family=per_family,
        treatment_mean=mean(table[f][treatment] for f in families),
        baseline_mean=mean(table[f][baseline] for f in families),
        paired_mean_difference=mean(diffs),
        better=sum(1 for d in diffs if d > 0),
        tied=sum(1 for d in diffs if d == 0),
        worse=sum(1 for d in diffs if d < 0),
        confounded=confounded,
        note=(
            "Changes retrieval mode and verification together; reported as the "
            "practical comparison against the metadata filter, not as an "
            "ablation isolating verification."
            if confounded
            else ""
        ),
    )


def decide(
    result: Contrast,
    *,
    threshold: float = EFFECT_THRESHOLD,
    direction_required: int,
) -> dict[str, Any]:
    """The section 5 rule, applied rather than reinvented.

    Both criteria, or it is not supported. A point estimate that favours the
    contribution while failing the direction count is *suggestive* and is
    reported in those words, which is the whole reason the rule has three
    outcomes instead of two.
    """
    effect = result.paired_mean_difference > threshold
    direction = result.better >= direction_required
    if effect and direction:
        verdict = "supported"
    elif effect or direction:
        verdict = "suggestive"
    else:
        verdict = "not supported"
    return {
        "verdict": verdict,
        "paired_mean_difference": result.paired_mean_difference,
        "effect_threshold": threshold,
        "effect_criterion_met": effect,
        "direction": f"{result.better}/{len(result.families)}",
        "direction_required": f"{direction_required}/{len(result.families)}",
        "direction_criterion_met": direction,
        "confounded": result.confounded,
    }


def decide_within_margin(
    result: Contrast, *, threshold: float = EFFECT_THRESHOLD
) -> dict[str, Any]:
    """Is the difference inside the pre-specified margin, or outside it?

    **This is operational, not statistical.** It is not an equivalence test:
    no confidence interval is computed, no TOST is performed, and "within the
    margin" does not mean the arms have been shown to be the same. It means the
    observed paired difference does not exceed the 0.25 the pre-registration
    fixed in advance as the size worth reporting. The earlier wording said
    "equivalent", which claims something the procedure never established.

    H1 has two legs. "A < B" is a superiority claim and ``decide`` handles it.
    "B ~ C ~ D" is a claim that three arms do not differ meaningfully, and
    passing it through ``decide`` produces a misleading line: a large difference
    running *against* the treatment comes back as "direction 0/4, not
    supported", which reads as though no difference was found when the opposite
    is true.

    So an equivalence leg is decided on magnitude, and the reading names which
    arm is higher and in how many families. A difference that exceeds the
    threshold refutes equivalence regardless of which direction it runs in.

    Confounding is reported alongside and does not soften the observation.
    C to D changes retrieval mode and verification together, so the *cause*
    cannot be attributed to either alone. The difference itself was still
    observed, and saying otherwise would use a limit on attribution to make an
    inconvenient result disappear.
    """
    difference = result.paired_mean_difference
    magnitude = abs(difference)
    if difference > 0:
        higher, consistent = result.treatment, result.better
    elif difference < 0:
        higher, consistent = result.baseline, result.worse
    else:
        higher, consistent = None, 0

    within = magnitude <= threshold
    reading = (
        f"Difference within the pre-specified {threshold} margin on the "
        "three-point scale. The arms are not distinguished by this measurement, "
        "which is not the same as having been shown to be equal."
        if within
        else (
            f"The difference is outside the pre-specified {threshold} margin, in "
            f"{higher}'s favour, with {higher} higher in {consistent} of "
            f"{len(result.families)} families and {result.tied} tied."
            + (
                " This contrast changes two variables at once, so the difference "
                "cannot be attributed to either alone. It was still observed, and "
                "the confound limits attribution rather than erasing it."
                if result.confounded
                else ""
            )
        )
    )
    return {
        "verdict": "within margin" if within else "outside margin",
        "basis": (
            "Operational comparison against the pre-specified 0.25 margin. Not "
            "a statistical equivalence test; no interval is computed and no "
            "claim of equality is made."
        ),
        "paired_mean_difference": difference,
        "magnitude": magnitude,
        "threshold": threshold,
        "higher_arm": higher,
        "higher_in_families": f"{consistent}/{len(result.families)}",
        "tied_families": result.tied,
        "confounded": result.confounded,
        "reading": reading,
    }


def leave_one_family_out(
    table: Mapping[str, Mapping[str, float]], treatment: str, baseline: str
) -> dict[str, Any]:
    """How far the figure moves when any single family is removed.

    Reported as a sensitivity analysis, not as validation. With eight families
    the relevant question is whether one of them is carrying the result, and
    this answers it directly. Section 5 is explicit that this is not
    cross-validation: nothing is trained on the remaining folds.
    """
    full = contrast(table, treatment, baseline)
    folds = {}
    for held in full.families:
        reduced = {f: v for f, v in table.items() if f != held}
        folds[held] = contrast(reduced, treatment, baseline).paired_mean_difference
    values = list(folds.values())
    return {
        "full": full.paired_mean_difference,
        "folds": folds,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "most_influential_family": max(
            folds, key=lambda f: abs(folds[f] - full.paired_mean_difference)
        ),
    }


# --- rate metrics ------------------------------------------------------------


def citation_metrics(runs: Sequence[Path | str]) -> dict[str, dict[str, Any]]:
    """Citation figures on the convention that already existed in this project.

    ``scripts/summarise_arms.py`` established it before unsealing: restrict to
    answers that make a claim, since an abstention cites nothing by design and
    counting it measures refusal while calling the result citation validity;
    then report each metric at question and group level through
    ``aggregate``, with the group level as the unit for inference.

    An earlier version of this module invented a different denominator, taking
    the answers where ``citation_support`` happens to be defined and applying it
    to validity as well. That is a narrower set, it moved every validity figure,
    and it was chosen *after* the key was opened. The verdict on H3 is the same
    either way, which is exactly why the change had to be undone rather than
    kept: a denominator picked post-hoc is not made acceptable by not changing
    the answer.

    The common-eligibility variant is retained by ``common_eligibility_variant``
    and reported as a labelled sensitivity analysis, not as the figure.
    """
    from .aggregate import aggregate
    from ..verify.schema import ABSTENTION_TEXT

    def makes_a_claim(record: Mapping[str, Any]) -> bool:
        verification = record.get("verification") or {}
        if "served_abstention" in verification:
            return not verification["served_abstention"]
        return str(record.get("answer") or "").strip() != ABSTENTION_TEXT

    out: dict[str, dict[str, Any]] = {}
    for directory in runs:
        manifest, answers = read_run(directory)
        arm = manifest["arm"]["arm"]
        claiming = [r for r in answers if makes_a_claim(r)]
        block: dict[str, Any] = {
            "answers": len(answers),
            "claim_making": len(claiming),
            "abstentions_excluded": len(answers) - len(claiming),
        }
        for key, label in (
            ("has_valid_citation_ids", "citation_validity"),
            ("scoring.citation_support", "citation_support"),
            ("scoring.citation_completeness", "citation_completeness"),
        ):
            result = aggregate(claiming, key)
            block[label] = {
                "question_level": result.question_level,
                "group_level": result.group_level,
                "groups": result.group_count,
            }
        block["support_below_validity_group_level"] = (
            block["citation_support"]["group_level"]
            < block["citation_validity"]["group_level"]
        )
        out[arm] = block
    return out


def common_eligibility_variant(rows: Iterable[Joined]) -> dict[str, dict[str, Any]]:
    """Validity and support over the answers where both are defined.

    A sensitivity analysis, and labelled as one. It answers a narrower question
    than ``citation_metrics``: among answers whose citations could be checked
    for content, how often were the identifiers also real? Reported so the
    choice of denominator is visible rather than assumed, and never as the
    headline figure.
    """
    rows = [r for r in rows if r.citation_support is not None]
    out: dict[str, dict[str, Any]] = {}
    for arm in sorted({r.arm for r in rows}):
        subset = [r for r in rows if r.arm == arm]
        validity = sum(1 for r in subset if r.has_valid_citation_ids) / len(subset)
        support = sum(r.citation_support for r in subset) / len(subset)
        out[arm] = {
            "citation_validity": validity,
            "citation_support": support,
            "support_below_validity": support < validity,
            "eligible_n": len(subset),
        }
    return out


def cohens_kappa(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """Chance-corrected agreement for one binary field over two passes.

    Amendment 1.26. Raw agreement overstates reliability when one outcome
    dominates: on this field 191 of 272 items are agreed negatives, so a rater
    who marked nothing at all would score 0.702 for doing no work. Kappa removes
    the agreement expected from the marginals.

    The 2x2 table is returned with the coefficient so a reader can recompute it
    by hand. No threshold is attached and no verdict is drawn from it.
    """
    pairs = list(pairs)
    if not pairs:
        raise AnalysisError("no paired judgements to compare")
    n = len(pairs)
    both_yes = sum(1 for first, second in pairs if first and second)
    first_only = sum(1 for first, second in pairs if first and not second)
    second_only = sum(1 for first, second in pairs if not first and second)
    both_no = sum(1 for first, second in pairs if not first and not second)
    observed = (both_yes + both_no) / n
    expected = (
        (both_yes + first_only) * (both_yes + second_only)
        + (second_only + both_no) * (first_only + both_no)
    ) / (n * n)
    # Perfect agreement on a field with no variation leaves kappa undefined
    # rather than 1.0, and saying so is better than dividing by zero.
    kappa = None if expected == 1.0 else (observed - expected) / (1 - expected)
    return {
        "n": n,
        "table": {
            "both_marked": both_yes,
            "first_pass_only": first_only,
            "second_pass_only": second_only,
            "neither_marked": both_no,
        },
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohens_kappa": kappa,
        "note": (
            "Chance-corrected agreement between the two abstention passes on "
            "the same 272 items, one reviewer. No threshold is attached and no "
            "verdict is drawn from it. Amendment 1.26."
        ),
    }


def primary_metrics(rows: Iterable[Joined]) -> dict[str, Any]:
    """The two section 4 primary metrics that carry no hypothesis.

    Amendment 1.25. Answer correctness and superseded citation rate were both
    preregistered, both scored, and both frozen into ``joined.jsonl`` before the
    key was opened. Neither was ever aggregated: ``CORRECTNESS`` was imported by
    the analysis script and never referred to again, and ``cited_superseded``
    was written per item and summed nowhere. That is a defect in this layer, not
    a decision about what to report, and leaving it standing would be selective
    reporting against the study's own protocol.

    **Neither metric takes a verdict.** Section 3 states no prediction over
    either, so the section 5 decision rule does not apply. Attaching a threshold
    now would be inventing a test after the data, which is the freedom the
    pre-registration exists to remove. Superseded citation rate is the
    measurement that motivates H1; H1 remains the hypothesis and is decided
    elsewhere.

    Both aggregations reuse the helpers already tested for H1 to H4, so no
    calculation is chosen here either.
    """
    rows = list(rows)
    correctness = select(rows, CORRECTNESS)
    if not correctness:
        raise AnalysisError(
            "no answer-correctness questions found; expected the questions "
            "whose expected_behaviour is answer_directly or answer_and_flag_gap"
        )
    superseded = select(rows, SUPERSESSION)
    if not superseded:
        raise AnalysisError(
            "no supersession questions found; expected the questions whose "
            "expected_behaviour is cite_current_only"
        )
    return {
        "basis": (
            "Section 4 primary metrics that carry no hypothesis. Descriptive: "
            "no verdict, no threshold, no direction criterion. The rule was "
            "fixed in amendment 1.25.3 before either figure was computed."
        ),
        "answer_correctness": {
            "definition": (
                "Manual, blinded, three-point rubric against required_claims "
                "and forbidden_claims. 2 correct, 1 partial, 0 wrong."
            ),
            "denominator": f"expected_behaviour in {CORRECTNESS!r}",
            "unit": "group, per section 5; question level reported alongside",
            "by_question": question_table(correctness),
            "by_group": family_table(correctness),
            # The mean over group means, which is the unit section 5 names.
            # Computed here rather than in the figure layer: a number derived
            # at plot time is a number with no test behind it.
            "group_level": {
                arm: mean([
                    row[arm] for row in family_table(correctness).values()
                    if arm in row
                ])
                for arm in sorted({r.arm for r in correctness})
            },
            "groups": len({r.group_id for r in correctness}),
            "questions_per_arm": len(correctness) // len({r.arm for r in correctness}),
            "reliability": (
                "Same rubric, reviewer and session as the conflict metric: 58 "
                "of 58 duplicate groups agreed, amendment 1.14.1, under the "
                "partial blinding of amendment 1.13."
            ),
        },
        "superseded_citation_rate": {
            "definition": (
                "Fraction of answers citing at least one withdrawn document as "
                "authority. Counted per answer, not per citation, because "
                "section 4 states the metric over answers."
            ),
            "denominator": f"expected_behaviour in {SUPERSESSION!r}",
            "unit": (
                "question-level rate with the family count stated; the "
                "family-level figure is families with any false citation, "
                "matching the correction amendment 1.16.2 made to H2c"
            ),
            "by_arm": rate_by_arm(superseded, "cited_superseded"),
            "families": len({r.group_id for r in superseded}),
            "relation_to_H1": (
                "This is the measurement that motivates H1. H1 remains the "
                "hypothesis and is decided under the section 5 rule; nothing "
                "here revisits it."
            ),
        },
    }


def rate_by_arm(
    rows: Iterable[Joined], field_name: str
) -> dict[str, dict[str, Any]]:
    """A per-arm count and rate over questions, with the family count stated.

    Rates are reported at question level with the number of independent groups
    alongside, so a reader is never shown 10/10 without being told it rests on
    five gap topics.
    """
    rows = list(rows)
    out: dict[str, dict[str, Any]] = {}
    for arm in sorted({r.arm for r in rows}):
        subset = [r for r in rows if r.arm == arm]
        hits = sum(1 for r in subset if getattr(r, field_name))
        out[arm] = {
            "hits": hits,
            "questions": len(subset),
            "rate": hits / len(subset) if subset else None,
            "groups": len({r.group_id for r in subset}),
            # Both are needed and they answer different questions. For
            # abstention, "did the arm abstain on every question in this gap
            # topic" is the family-level success. For false conflicts, "did the
            # arm assert a conflict anywhere in this control family" is the
            # family-level failure, and reporting only the all-hit count would
            # show 0/3 for an arm that flagged a false conflict on a third of
            # its questions.
            "groups_all_hit": sum(
                1
                for g in {r.group_id for r in subset}
                if all(getattr(r, field_name) for r in subset if r.group_id == g)
            ),
            "groups_any_hit": sum(
                1
                for g in {r.group_id for r in subset}
                if any(getattr(r, field_name) for r in subset if r.group_id == g)
            ),
        }
    return out
