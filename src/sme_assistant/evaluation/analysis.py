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


def quality_run_directories(root: Path | str) -> list[Path]:
    """The frozen quality runs, and nothing else.

    Amendment 1.15 declares the laptop quality run the sole evidential source
    for H1 to H4, and the later hardware executions performance-only. That
    boundary is enforced here rather than remembered: ``RunWriter`` names a
    directory ``{stamp}_{arm}_{split}{label}``, so an untagged quality run ends
    in ``_test`` and any tagged run does not. A performance run therefore cannot
    be picked up by the analysis even if someone points it at the whole results
    tree.

    The check is on the manifest as well as the name, because a directory can
    be renamed and a manifest cannot be renamed by accident.
    """
    found: list[Path] = []
    for directory in sorted(Path(root).glob("*_test")):
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") == "test":
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

    automatic: dict[tuple[str, str], dict[str, Any]] = {}
    for directory in runs:
        manifest, answers = read_run(directory)
        arm = manifest["arm"]["arm"]
        for record in answers:
            automatic[(arm, record["question_id"])] = record

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


def decide_equivalence(
    result: Contrast, *, threshold: float = EFFECT_THRESHOLD
) -> dict[str, Any]:
    """The verdict for an *equivalence* claim, which is not a superiority one.

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

    equivalent = magnitude <= threshold
    reading = (
        f"Difference within {threshold} on the three-point scale; the arms are "
        "not distinguished by this measurement."
        if equivalent
        else (
            f"The difference exceeds {threshold} in {higher}'s favour, with "
            f"{higher} higher in {consistent} of {len(result.families)} families "
            f"and {result.tied} tied. Equivalence is refuted."
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
        "verdict": "equivalent" if equivalent else "not equivalent",
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
