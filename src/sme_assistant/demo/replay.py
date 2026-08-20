"""Read the frozen quality runs for side-by-side replay.

Read-only by construction: this module opens files and returns dataclasses. It
has no write path at all, which is a stronger guarantee than a rule saying it
must not write.

The run directories are taken from ``FROZEN_QUALITY_RUNS`` rather than
discovered, for the same reason the analysis does it: a list that cannot be
satisfied by a directory created later cannot accidentally admit one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

ARMS = ("A", "B", "C", "D")


class ReplayUnavailable(RuntimeError):
    """The frozen records are missing or unreadable.

    Raised rather than returning an empty library. A replay view showing no
    questions looks like a system with nothing to say, when what has actually
    happened is that it cannot find its data.
    """


@dataclass(frozen=True)
class ArmAnswer:
    """One arm's answer to one question, as recorded during the frozen run."""

    arm: str
    question_id: str
    question: str
    category: str
    family_id: str | None
    answer: str
    draft_answer: str | None
    citations: tuple[str, ...]
    cited_superseded: tuple[str, ...]
    hallucinated_citations: tuple[str, ...]
    has_valid_citation_ids: bool | None
    wall_seconds: float | None
    verification_seconds: float | None
    generation: Mapping[str, Any]
    verification_generation: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    verification: Mapping[str, Any] | None
    scoring: Mapping[str, Any] | None

    @property
    def has_verification(self) -> bool:
        return bool(self.verification)

    @property
    def confidence(self) -> str | None:
        """The rule-based categorical level, where the arm produces one.

        Arms A to C have no verifier and therefore no confidence level. That is
        a property of the design, not a missing value, and the interface says
        so rather than showing a blank.
        """
        if not self.verification:
            return None
        return self.verification.get("confidence")

    @property
    def revised(self) -> bool:
        return bool(self.verification and self.verification.get("revised"))

    @property
    def abstained(self) -> bool:
        return bool(self.verification and self.verification.get("served_abstention"))


@dataclass(frozen=True)
class ReplayQuestion:
    """One test question and every arm's answer to it."""

    question_id: str
    question: str
    category: str
    family_id: str | None
    by_arm: dict[str, ArmAnswer] = field(default_factory=dict)

    @property
    def any_cited_superseded(self) -> tuple[str, ...]:
        return tuple(sorted({a for ans in self.by_arm.values()
                             for a in ans.cited_superseded}))


@dataclass(frozen=True)
class ReplayLibrary:
    """Every frozen question, with provenance for what produced it."""

    questions: tuple[ReplayQuestion, ...]
    provenance: Mapping[str, Any]

    def by_id(self, question_id: str) -> ReplayQuestion | None:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        return None

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({q.category for q in self.questions}))


def _tuple_of(record: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = record.get(key) or ()
    return tuple(str(v) for v in value)


def _arm_answer(record: Mapping[str, Any]) -> ArmAnswer:
    return ArmAnswer(
        arm=record["arm"],
        question_id=record["question_id"],
        question=record["question"],
        category=record.get("category", ""),
        family_id=record.get("family_id"),
        answer=record.get("answer") or "",
        draft_answer=record.get("draft_answer"),
        citations=_tuple_of(record, "citations"),
        cited_superseded=_tuple_of(record, "cited_superseded"),
        hallucinated_citations=_tuple_of(record, "hallucinated_citations"),
        has_valid_citation_ids=record.get("has_valid_citation_ids"),
        wall_seconds=record.get("wall_seconds"),
        verification_seconds=record.get("verification_seconds"),
        generation=record.get("generation") or {},
        verification_generation=record.get("verification_generation") or {},
        retrieval=record.get("retrieval") or {},
        verification=record.get("verification"),
        scoring=record.get("scoring"),
    )


def load_replay_library(runs_root: Path | str,
                        run_names: Iterable[str]) -> ReplayLibrary:
    """Join the frozen runs into one question-major view.

    Every named run must be present and must declare the arm its name implies.
    A partial library would silently drop an arm from the comparison, and a
    three-arm comparison presented as four is exactly the sort of quiet error
    this project has spent twenty-seven amendments learning to refuse.
    """
    root = Path(runs_root)
    order: list[str] = []
    merged: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {"runs": {}}

    for name in run_names:
        directory = root / name
        answers_path = directory / "answers.jsonl"
        manifest_path = directory / "manifest.json"
        if not answers_path.exists() or not manifest_path.exists():
            raise ReplayUnavailable(
                f"frozen run {name} is missing from {root}. Replay needs all "
                "four quality runs; showing fewer would present a partial "
                "comparison as a complete one."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arm = manifest["arm"]["arm"]
        if manifest.get("purpose") == "performance":
            raise ReplayUnavailable(
                f"{name} is a performance run and carries no answers to compare"
            )
        provenance["runs"][arm] = {
            "directory": name,
            "started_at": manifest.get("started_at"),
            "description": manifest["arm"].get("description"),
            "retrieval_mode": manifest["arm"].get("retrieval_mode"),
            "evidence_format": manifest["arm"].get("evidence_format"),
            "verification": manifest["arm"].get("verification"),
            "generation_model": manifest["arm"].get("generation_model"),
            "verification_model": manifest["arm"].get("verification_model"),
            "corpus_sha256": manifest["provenance"]["corpus_sha256"],
            "chunk_set_sha256": manifest["provenance"]["chunk_set_sha256"],
            "config_sha256": manifest["provenance"]["config_sha256"],
        }
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = record["question_id"]
            if question_id not in merged:
                order.append(question_id)
                merged[question_id] = {
                    "question": record["question"],
                    "category": record.get("category", ""),
                    "family_id": record.get("family_id"),
                    "by_arm": {},
                }
            merged[question_id]["by_arm"][arm] = _arm_answer(record)

    if not merged:
        raise ReplayUnavailable(f"no answers found under {root}")

    # Fail closed. Everything below was previously computed and displayed
    # rather than enforced, which is the same defect amendment 1.16.1 records:
    # a property that is reported but not checked is not a guarantee. A replay
    # that quietly drops an arm on some questions, or joins runs built over
    # different corpora, would show a comparison that was never made.
    expected_arms = set(provenance["runs"])
    if expected_arms != set(ARMS):
        raise ReplayUnavailable(
            f"expected arms {sorted(ARMS)}, the named runs declare "
            f"{sorted(expected_arms)}"
        )
    incomplete = sorted(qid for qid, entry in merged.items()
                        if set(entry["by_arm"]) != expected_arms)
    if incomplete:
        missing = {qid: sorted(expected_arms - set(merged[qid]["by_arm"]))
                   for qid in incomplete[:5]}
        raise ReplayUnavailable(
            f"{len(incomplete)} question(s) are not answered by every arm, so a "
            f"side-by-side comparison would be missing a column: {missing}"
        )
    for field in ("corpus_sha256", "chunk_set_sha256", "config_sha256"):
        values = {info[field] for info in provenance["runs"].values()}
        if len(values) != 1:
            raise ReplayUnavailable(
                f"the four runs disagree on {field}: {sorted(values)}. They were "
                "not executed over the same material and must not be compared."
            )

    provenance["corpus_consistent"] = True
    provenance["question_count"] = len(merged)
    provenance["arms"] = sorted(provenance["runs"])

    questions = tuple(
        ReplayQuestion(
            question_id=question_id,
            question=merged[question_id]["question"],
            category=merged[question_id]["category"],
            family_id=merged[question_id]["family_id"],
            by_arm=merged[question_id]["by_arm"],
        )
        for question_id in order
    )
    return ReplayLibrary(questions=questions, provenance=provenance)
