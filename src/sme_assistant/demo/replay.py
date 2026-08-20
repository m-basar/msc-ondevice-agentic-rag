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


#: The provenance fields that must agree across the four runs before they may
#: be shown side by side. Every one of them was already recorded in each
#: manifest and displayed on the page; none of them was checked. Amendment
#: 1.16.1 records what a displayed-but-unenforced property is worth.
AGREEING_PROVENANCE: tuple[str, ...] = (
    "corpus_sha256",
    "chunk_set_sha256",
    "config_sha256",
    "question_set_sha256",
    "registry_sha256",
    "index_file_sha256",
)

#: Purposes that disqualify a directory from replay. A performance run carries
#: timings and no scored answers; a demonstration directory carries whatever
#: the dashboard was pointed at. Neither is an arm of the experiment.
REJECTED_PURPOSES: frozenset[str] = frozenset({"performance", "demonstration"})


def load_replay_library(runs_root: Path | str,
                        run_names: Iterable[str],
                        expected_questions: int | None = None) -> ReplayLibrary:
    """Join the frozen runs into one question-major view, or refuse.

    Every check below was previously computed and displayed rather than
    enforced. That is the defect amendment 1.16.1 records in another place: a
    property that is reported but not checked is not a guarantee, and it gets
    quoted as though it were one. A replay that quietly drops an arm on some
    questions, joins runs built over different corpora, or shows a
    demonstration directory in an experimental column would present a
    comparison that was never made.

    ``expected_questions`` is the count the caller believes the test split
    holds, read from the question set rather than written here. Passing
    ``None`` skips only that one check; nothing else is optional.
    """
    root = Path(runs_root)
    order: list[str] = []
    merged: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {"runs": {}}
    seen_arms: dict[str, str] = {}

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

        # The split first, because a renamed directory passes every other
        # check. A run on the development split answers different questions
        # and would populate the grid with material the study never reported.
        if manifest.get("split") != "test":
            raise ReplayUnavailable(
                f"{name} declares split={manifest.get('split')!r}. Replay shows "
                "the reported test-split comparison and nothing else."
            )
        purpose = manifest.get("purpose")
        if purpose in REJECTED_PURPOSES:
            raise ReplayUnavailable(
                f"{name} is a {purpose} run and is not an arm of the "
                "experiment; it carries no comparable scored answers"
            )
        # A directory renamed to sit on the closed list still carries the
        # identifier it was written with. Names are the boundary here, so a
        # name that disagrees with its own manifest is refused rather than
        # trusted.
        declared_id = manifest.get("run_id")
        if declared_id is not None and declared_id != name:
            raise ReplayUnavailable(
                f"{name} contains a run whose manifest calls it "
                f"{declared_id!r}. The directory has been renamed, and replay "
                "identifies runs by name."
            )
        arm = manifest["arm"]["arm"]
        if arm in seen_arms:
            raise ReplayUnavailable(
                f"{name} and {seen_arms[arm]} both declare arm {arm}. Two "
                "columns of one arm presented as two arms is not a comparison."
            )
        seen_arms[arm] = name

        provenance_block = manifest.get("provenance") or {}
        entry = {
            "directory": name,
            "started_at": manifest.get("started_at"),
            "description": manifest["arm"].get("description"),
            "retrieval_mode": manifest["arm"].get("retrieval_mode"),
            "evidence_format": manifest["arm"].get("evidence_format"),
            "verification": manifest["arm"].get("verification"),
            "generation_model": manifest["arm"].get("generation_model"),
            "verification_model": manifest["arm"].get("verification_model"),
        }
        for hash_field in AGREEING_PROVENANCE:
            if hash_field not in provenance_block:
                raise ReplayUnavailable(
                    f"{name} records no {hash_field}. Replay checks that the "
                    "four runs were executed over the same material, and it "
                    "cannot check a field that is absent."
                )
            entry[hash_field] = provenance_block[hash_field]
        # How many test-split questions the runner believed existed when it
        # started. Taken from the manifest rather than from the gold question
        # set, which no part of the demonstrator may open, and independent of
        # the answers file: a truncated answers.jsonl does not shrink it.
        summary = ((provenance_block.get("question_set_metadata") or {})
                   .get("summary") or {})
        declared = (summary.get("by_split") or {}).get("test")
        if not isinstance(declared, int):
            raise ReplayUnavailable(
                f"{name} records no test-split question count, so replay "
                "cannot tell a complete run from a truncated one"
            )
        entry["declared_test_questions"] = declared
        provenance["runs"][arm] = entry

        seen_ids: set[str] = set()
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = record["question_id"]
            # The record carries its own arm. A file copied between run
            # directories keeps it, and the manifest would not notice.
            if record.get("arm") != arm:
                raise ReplayUnavailable(
                    f"{name} declares arm {arm} but question {question_id} was "
                    f"answered by arm {record.get('arm')!r}. The answers file "
                    "does not belong to this run."
                )
            if question_id in seen_ids:
                raise ReplayUnavailable(
                    f"{name} answers {question_id} more than once. Which of the "
                    "two the grid would show depends on read order, which is "
                    "not a property a comparison may rest on."
                )
            seen_ids.add(question_id)
            if question_id not in merged:
                order.append(question_id)
                merged[question_id] = {
                    "question": record["question"],
                    "category": record.get("category", ""),
                    "family_id": record.get("family_id"),
                    "by_arm": {},
                }
            else:
                # The arms must have been asked the same question. A grid whose
                # four columns answer four different prompts is the worst
                # available failure, because it looks entirely normal.
                first = merged[question_id]
                for label, value in (("question", record["question"]),
                                     ("category", record.get("category", "")),
                                     ("family_id", record.get("family_id"))):
                    if first[label] != value:
                        raise ReplayUnavailable(
                            f"{question_id} has {label} {first[label]!r} in one "
                            f"arm and {value!r} in {name}. The arms were not "
                            "asked the same question."
                        )
            merged[question_id]["by_arm"][arm] = _arm_answer(record)

    if not merged:
        raise ReplayUnavailable(f"no answers found under {root}")

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
    declared_counts = {info["declared_test_questions"]
                       for info in provenance["runs"].values()}
    if len(declared_counts) != 1:
        raise ReplayUnavailable(
            f"the four runs disagree on the size of the test split: "
            f"{sorted(declared_counts)}"
        )
    declared_count = declared_counts.pop()
    if len(merged) != declared_count:
        raise ReplayUnavailable(
            f"the runs carry {len(merged)} questions, their own manifests "
            f"declare {declared_count} on the test split. A subset shown "
            "without saying so is a different experiment."
        )
    if expected_questions is not None and declared_count != expected_questions:
        raise ReplayUnavailable(
            f"the runs declare {declared_count} test-split questions, the "
            f"caller expects {expected_questions}"
        )
    for hash_field in AGREEING_PROVENANCE:
        values = {info[hash_field] for info in provenance["runs"].values()}
        if len(values) != 1:
            raise ReplayUnavailable(
                f"the four runs disagree on {hash_field}: {sorted(values)}. They "
                "were not executed over the same material and must not be "
                "compared."
            )

    provenance["corpus_consistent"] = True
    provenance["question_count"] = len(merged)
    provenance["arms"] = sorted(provenance["runs"])
    provenance["checked"] = list(AGREEING_PROVENANCE)
    provenance["declared_test_questions"] = declared_count

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
