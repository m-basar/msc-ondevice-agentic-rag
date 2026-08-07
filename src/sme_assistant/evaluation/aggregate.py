"""Aggregating results at both levels, because one level alone misleads.

A run produces one record per question. Reporting the mean over those records
treats three paraphrases of CONF-01 as three independent observations. They are
not: they retrieve the same chunks from the same document pair and almost
always succeed or fail together. The consequence is not a small bias. Standard
error falls with the square root of the sample size, so counting 48 correlated
questions as 48 independent ones understates the interval by roughly the square
root of the average group size, and a difference between arms can cross into
significance purely by the arithmetic of double counting.

The correct unit here is the group: the conflict family, or the gap topic, or
the standalone question. Every figure in this module is therefore produced
twice.

``question_level``  the plain mean over questions. Familiar, easy to read,
                    and the right denominator when the question is "how often
                    does the system get an answer right".
``group_level``     the mean of per-group means. The correct unit for
                    inference, and the number that carries the interval.

Both are reported. Neither is suppressed. Question-level numbers are useful and
intuitive; the objection is only to using them as a sample size. Where the two
diverge, the divergence is itself informative: it means performance is uneven
across families, and a group with more paraphrases was pulling the average.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

Record = Mapping[str, Any]


class AggregationError(RuntimeError):
    """Raised when records cannot be aggregated soundly."""


@dataclass(frozen=True)
class GroupResult:
    """One group's contribution, and how many questions produced it."""

    group_id: str
    mean: float
    n: int
    values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "mean": round(self.mean, 4),
            "questions": self.n,
        }


@dataclass(frozen=True)
class Aggregate:
    """A metric reported at both levels, with inference over groups."""

    metric: str
    groups: tuple[GroupResult, ...]
    question_values: tuple[float, ...]
    skipped: int = 0

    # --- question level ------------------------------------------------------

    @property
    def question_level(self) -> float | None:
        if not self.question_values:
            return None
        return round(sum(self.question_values) / len(self.question_values), 4)

    @property
    def question_count(self) -> int:
        return len(self.question_values)

    # --- group level ---------------------------------------------------------

    @property
    def group_level(self) -> float | None:
        """Macro average: mean within groups, then mean across groups."""
        if not self.groups:
            return None
        return round(sum(g.mean for g in self.groups) / len(self.groups), 4)

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def group_standard_error(self) -> float | None:
        """Standard error over group means, which is the defensible interval.

        Computed from group means rather than question values. A standard error
        over questions would treat paraphrases as independent replicates and
        report an interval narrower than the data supports.
        """
        if len(self.groups) < 2:
            return None
        means = [g.mean for g in self.groups]
        mean = sum(means) / len(means)
        variance = sum((m - mean) ** 2 for m in means) / (len(means) - 1)
        return round(math.sqrt(variance / len(means)), 4)

    def confidence_interval(self, z: float = 1.96) -> tuple[float, float] | None:
        """Normal-approximation interval at group level.

        With nine families the normal approximation is optimistic; a bootstrap
        or a t interval is preferable for the write-up. This is reported as an
        indication of spread, not as a hypothesis test.
        """
        centre = self.group_level
        error = self.group_standard_error
        if centre is None or error is None:
            return None
        return (round(centre - z * error, 4), round(centre + z * error, 4))

    @property
    def disagreement(self) -> float | None:
        """How far the two levels diverge.

        Non-zero means groups have unequal question counts and performance
        varies across them, so the question-level figure is weighted by how
        many paraphrases each family happened to receive.
        """
        if self.question_level is None or self.group_level is None:
            return None
        return round(abs(self.question_level - self.group_level), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "question_level": self.question_level,
            "question_count": self.question_count,
            "group_level": self.group_level,
            "group_count": self.group_count,
            "group_standard_error": self.group_standard_error,
            "group_confidence_interval": self.confidence_interval(),
            "level_disagreement": self.disagreement,
            "skipped_records": self.skipped,
            "per_group": [g.to_dict() for g in self.groups],
            "inference_unit": "group",
            "note": (
                "question_level is the plain mean over questions and is reported "
                "for readability. Inference uses group_level, because paraphrases "
                "within a conflict family are not independent observations. Cite "
                "group_count, not question_count, as the sample size."
            ),
        }


def _group_of(record: Record) -> str:
    for key in ("group_id", "family_id", "question_id"):
        value = record.get(key)
        if value:
            return str(value)
    raise AggregationError(
        "A record carries no group_id, family_id or question_id, so its unit of "
        "independence is unknown and it cannot be aggregated."
    )


def aggregate(
    records: Iterable[Record],
    metric: str,
    *,
    extractor: Callable[[Record], Any] | None = None,
) -> Aggregate:
    """Aggregate one metric at both levels.

    Records whose metric is ``None`` are skipped and counted, not coerced to
    zero. Citation support is undefined when an answer cited nothing, and
    silently reading that as a zero would make an abstaining system look
    dishonest rather than cautious.
    """
    get = extractor or (lambda record: _dotted(record, metric))

    values: list[float] = []
    by_group: dict[str, list[float]] = defaultdict(list)
    order: list[str] = []
    skipped = 0

    for record in records:
        group = _group_of(record)
        if group not in by_group:
            order.append(group)
            by_group[group] = []
        raw = get(record)
        if raw is None:
            skipped += 1
            continue
        value = float(raw)
        values.append(value)
        by_group[group].append(value)

    groups = tuple(
        GroupResult(
            group_id=group,
            mean=sum(by_group[group]) / len(by_group[group]),
            n=len(by_group[group]),
            values=tuple(by_group[group]),
        )
        for group in order
        if by_group[group]
    )
    return Aggregate(metric, groups, tuple(values), skipped)


def _dotted(record: Record, key: str) -> Any:
    current: Any = record
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool):
        return 1.0 if current else 0.0
    return current


def aggregate_many(
    records: Sequence[Record], metrics: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Aggregate several metrics over the same records."""
    return {metric: aggregate(records, metric).to_dict() for metric in metrics}


def by_family_type(
    records: Sequence[Record], metric: str, registry: Any
) -> dict[str, dict[str, Any]]:
    """Split results by conflict type, which is the headline comparison.

    Supersession conflicts are resolvable by a metadata filter; current_current
    conflicts are not. Pooling them hides the distinction the contribution
    rests on, and a system that only ever fixed supersession would look like a
    general improvement.
    """
    buckets: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        family_id = record.get("family_id")
        if not family_id:
            buckets["not_a_conflict"].append(record)
            continue
        try:
            family = registry.by_id(str(family_id))
        except Exception:  # unknown family: surfaced rather than silently binned
            buckets["unknown_family"].append(record)
            continue
        buckets[family.conflict_type].append(record)

    return {name: aggregate(items, metric).to_dict() for name, items in sorted(buckets.items())}


def compare_arms(
    runs: Mapping[str, Sequence[Record]], metric: str
) -> dict[str, Any]:
    """Compare arms at both levels, paired by group where possible.

    Arms answer the same questions, so the comparison is paired: the difference
    is computed per group and then averaged, rather than comparing two
    independent means. Pairing removes between-family variance, which at nine
    families is the dominant source of noise.
    """
    aggregates = {arm: aggregate(records, metric) for arm, records in runs.items()}
    arms = list(aggregates)

    paired: dict[str, Any] = {}
    if len(arms) >= 2:
        baseline = arms[0]
        base_means = {g.group_id: g.mean for g in aggregates[baseline].groups}
        for arm in arms[1:]:
            arm_means = {g.group_id: g.mean for g in aggregates[arm].groups}
            shared = sorted(set(base_means) & set(arm_means))
            differences = [arm_means[g] - base_means[g] for g in shared]
            if not differences:
                continue
            mean_difference = sum(differences) / len(differences)
            if len(differences) > 1:
                variance = sum((d - mean_difference) ** 2 for d in differences) / (
                    len(differences) - 1
                )
                error = math.sqrt(variance / len(differences))
            else:
                error = None
            paired[f"{arm}_minus_{baseline}"] = {
                "mean_difference": round(mean_difference, 4),
                "standard_error": round(error, 4) if error is not None else None,
                "paired_groups": len(shared),
                "improved": sum(1 for d in differences if d > 0),
                "unchanged": sum(1 for d in differences if d == 0),
                "worsened": sum(1 for d in differences if d < 0),
            }

    return {
        "metric": metric,
        "arms": {arm: value.to_dict() for arm, value in aggregates.items()},
        "paired_differences": paired,
        "note": (
            "Differences are paired by group and averaged over groups. The "
            "comparison unit is the family, not the question."
        ),
    }
