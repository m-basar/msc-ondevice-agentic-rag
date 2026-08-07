"""The question set, and the grouping that keeps its statistics honest.

Three paraphrases of "what is the mileage rate?" are three questions but one
observation. They interrogate the same document pair, retrieve the same chunks
and succeed or fail together. Counting them as three independent successes
inflates the sample size by a factor of three and shrinks every confidence
interval accordingly, which is how a study reports significance it has not
earned.

Two rules follow, and this module exists to enforce both rather than merely
describe them in a protocol file.

**Grouping.** Every question declares a ``group_id``: the unit of independence
it belongs to. For a conflict question that is the conflict family. For an
unanswerable question it is the gap topic, because "does the company offer a
pension?" and "what is the auto-enrolment rate?" both probe one absence.
Ordinary factual questions are their own group. Aggregation is then a mean
within groups followed by a mean across them, and the sample size reported is
the number of groups.

**Split integrity.** A group must lie entirely within one split. If one
paraphrase of CONF-01 sits in development and another in test, the test set has
been contaminated: the system was tuned against the same document pair, the
same conflicting figures and the same retrieval behaviour it is later scored
on. The leak is invisible in the numbers because the question strings differ.
``validate_question_set`` refuses to load such a set.

The same reasoning drives the cross-validation protocol. Folds are held out by
family, not by question, so no fold is ever tested on a family it saw.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

SCHEMA_VERSION = "1.0"

VALID_SPLITS = frozenset({"dev", "test"})
VALID_CATEGORIES = frozenset(
    {"conflict", "unanswerable", "partial", "factual", "synthesis"}
)
VALID_ANSWERABILITY = frozenset({"answerable", "unanswerable", "partial"})
VALID_RISK = frozenset({"low", "medium", "high"})

# What a correct system should do, as distinct from what it should say. Scored
# separately from answer content because a system can state the right figure
# while failing to disclose that another document says otherwise.
VALID_BEHAVIOURS = frozenset(
    {
        "cite_current_only",      # supersession: answer from the live document
        "surface_both_and_qualify",  # current_current: no basis to choose
        "abstain",                # nothing in the corpus supports an answer
        "answer_and_flag_gap",    # partial: say what is known, name what is not
        "answer_directly",        # ordinary factual question
    }
)


class QuestionSetError(RuntimeError):
    """Raised when a question set is malformed or its splits are unsound."""


@dataclass(frozen=True)
class Question:
    """One question, with everything needed to score an answer to it."""

    question_id: str
    text: str
    category: str
    group_id: str
    split: str
    answerability: str
    expected_behaviour: str
    risk_level: str = "low"
    family_id: str | None = None
    gap_topic: str | None = None
    paraphrase_of: str | None = None
    gold_answer: str = ""
    gold_facts: tuple[str, ...] = ()
    expected_documents: tuple[str, ...] = ()
    expected_chunks: tuple[str, ...] = ()
    must_not_cite: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_paraphrase(self) -> bool:
        return self.paraphrase_of is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "category": self.category,
            "group_id": self.group_id,
            "split": self.split,
            "answerability": self.answerability,
            "expected_behaviour": self.expected_behaviour,
            "risk_level": self.risk_level,
            "family_id": self.family_id,
            "gap_topic": self.gap_topic,
            "paraphrase_of": self.paraphrase_of,
            "gold_answer": self.gold_answer,
            "gold_facts": list(self.gold_facts),
            "expected_documents": list(self.expected_documents),
            "expected_chunks": list(self.expected_chunks),
            "must_not_cite": list(self.must_not_cite),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class QuestionSet:
    """A collection of questions, indexed by group and split."""

    questions: tuple[Question, ...]
    source: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    def by_id(self, question_id: str) -> Question:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        raise QuestionSetError(f"No question {question_id!r}")

    def split(self, name: str) -> QuestionSet:
        return QuestionSet(
            tuple(q for q in self.questions if q.split == name), self.source, self.metadata
        )

    def of_category(self, category: str) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.category == category)

    def of_group(self, group_id: str) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.group_id == group_id)

    @property
    def groups(self) -> tuple[str, ...]:
        """Group identifiers in first-appearance order."""
        seen: list[str] = []
        for question in self.questions:
            if question.group_id not in seen:
                seen.append(question.group_id)
        return tuple(seen)

    @property
    def family_groups(self) -> tuple[str, ...]:
        """Groups that correspond to a conflict family, for leave-one-out folds."""
        seen: list[str] = []
        for question in self.questions:
            if question.family_id and question.group_id not in seen:
                seen.append(question.group_id)
        return tuple(seen)

    def group_split_map(self) -> dict[str, set[str]]:
        """Which splits each group appears in. Sound sets have one apiece."""
        mapping: dict[str, set[str]] = defaultdict(set)
        for question in self.questions:
            mapping[question.group_id].add(question.split)
        return dict(mapping)

    def leave_one_family_out(self) -> Iterator[tuple[str, tuple[Question, ...], tuple[Question, ...]]]:
        """Yield (held_out_family, held_out_questions, remaining_questions).

        Folds are defined by family, never by question. A fold that held out
        one paraphrase while training on its siblings would be testing on a
        family it had already seen, which is the leak this module exists to
        prevent expressed as a cross-validation bug instead of a split bug.
        """
        for group in self.family_groups:
            held = tuple(q for q in self.questions if q.group_id == group)
            rest = tuple(q for q in self.questions if q.group_id != group)
            yield group, held, rest

    def summary(self) -> dict[str, Any]:
        by_split = Counter(q.split for q in self.questions)
        by_category = Counter(q.category for q in self.questions)
        sizes = Counter(q.group_id for q in self.questions)
        return {
            "question_count": len(self.questions),
            "group_count": len(self.groups),
            "family_group_count": len(self.family_groups),
            "by_split": dict(sorted(by_split.items())),
            "by_category": dict(sorted(by_category.items())),
            "by_answerability": dict(
                sorted(Counter(q.answerability for q in self.questions).items())
            ),
            "by_risk": dict(sorted(Counter(q.risk_level for q in self.questions).items())),
            "groups_per_split": {
                name: len({q.group_id for q in self.questions if q.split == name})
                for name in sorted(by_split)
            },
            "largest_group": max(sizes.values()) if sizes else 0,
            "effective_sample_size_note": (
                f"{len(self.questions)} questions but {len(self.groups)} independent "
                "groups. Report inference over groups."
            ),
        }


# --- validation --------------------------------------------------------------


def validate_question_set(
    question_set: QuestionSet,
    *,
    registry: Any = None,
    kb: Any = None,
) -> None:
    """Refuse a question set whose splits or grouping would corrupt a result.

    ``registry`` and ``kb`` are optional so the structural checks can run
    without loading the corpus, but both should be supplied in tests. Passing
    neither leaves family identifiers and cited documents unverified.
    """
    _validate_fields(question_set)
    _validate_split_integrity(question_set)
    _validate_paraphrases(question_set)
    if registry is not None:
        _validate_against_registry(question_set, registry)
    if kb is not None:
        _validate_against_corpus(question_set, kb)


def _validate_fields(question_set: QuestionSet) -> None:
    seen: set[str] = set()
    for question in question_set:
        where = question.question_id
        if where in seen:
            raise QuestionSetError(f"Duplicate question id {where!r}")
        seen.add(where)
        if not question.text.strip():
            raise QuestionSetError(f"{where}: empty question text")
        if not question.group_id:
            raise QuestionSetError(
                f"{where}: no group_id. Every question must declare its unit of "
                "independence, or it cannot be aggregated correctly."
            )
        for value, allowed, name in (
            (question.split, VALID_SPLITS, "split"),
            (question.category, VALID_CATEGORIES, "category"),
            (question.answerability, VALID_ANSWERABILITY, "answerability"),
            (question.expected_behaviour, VALID_BEHAVIOURS, "expected_behaviour"),
            (question.risk_level, VALID_RISK, "risk_level"),
        ):
            if value not in allowed:
                raise QuestionSetError(
                    f"{where}: {name} {value!r} must be one of {sorted(allowed)}"
                )
        if question.category == "conflict" and not question.family_id:
            raise QuestionSetError(f"{where}: a conflict question must name its family")
        if question.family_id and question.group_id != question.family_id:
            raise QuestionSetError(
                f"{where}: group_id {question.group_id!r} does not match family_id "
                f"{question.family_id!r}. Conflict questions group by family."
            )
        if question.answerability == "unanswerable" and question.expected_chunks:
            raise QuestionSetError(
                f"{where}: declared unanswerable but names expected chunks"
            )
        if question.answerability == "answerable" and not question.gold_answer.strip():
            raise QuestionSetError(f"{where}: answerable but has no gold answer")


def _validate_split_integrity(question_set: QuestionSet) -> None:
    """The check that stops paraphrase leakage across the split boundary."""
    straddling = {
        group: sorted(splits)
        for group, splits in question_set.group_split_map().items()
        if len(splits) > 1
    }
    if straddling:
        detail = "; ".join(f"{g} in {s}" for g, s in sorted(straddling.items()))
        raise QuestionSetError(
            "These groups appear in more than one split: "
            f"{detail}. Paraphrases of one conflict family test the same document "
            "pair, so tuning on one and scoring on another leaks the test set. "
            "Move every question in a group to the same split."
        )


def _validate_paraphrases(question_set: QuestionSet) -> None:
    ids = {q.question_id: q for q in question_set}
    for question in question_set:
        if not question.paraphrase_of:
            continue
        parent = ids.get(question.paraphrase_of)
        if parent is None:
            raise QuestionSetError(
                f"{question.question_id}: paraphrase_of {question.paraphrase_of!r} "
                "is not a question in this set"
            )
        if parent.group_id != question.group_id:
            raise QuestionSetError(
                f"{question.question_id}: paraphrases {parent.question_id} but is in "
                f"group {question.group_id!r} while its parent is in "
                f"{parent.group_id!r}. A paraphrase is by definition the same "
                "observation as what it paraphrases."
            )


def _validate_against_registry(question_set: QuestionSet, registry: Any) -> None:
    known = {family.family_id for family in registry.families}
    topics = {gap.topic for gap in registry.fully_absent}
    topics |= {gap.topic for gap in registry.partially_present}

    for question in question_set:
        if question.family_id and question.family_id not in known:
            raise QuestionSetError(
                f"{question.question_id}: family {question.family_id!r} is not in "
                "the conflict registry"
            )
        if question.gap_topic and question.gap_topic not in topics:
            raise QuestionSetError(
                f"{question.question_id}: gap topic {question.gap_topic!r} is not a "
                "declared gap, so nothing guarantees it is still absent"
            )
        if question.family_id:
            family = registry.by_id(question.family_id)
            expected = (
                "cite_current_only"
                if family.is_filter_resolvable
                else "surface_both_and_qualify"
            )
            if question.expected_behaviour != expected:
                raise QuestionSetError(
                    f"{question.question_id}: {family.family_id} is "
                    f"{family.conflict_type}, so expected_behaviour should be "
                    f"{expected!r}, not {question.expected_behaviour!r}"
                )

    covered = {q.family_id for q in question_set if q.family_id}
    missing = sorted(known - covered)
    if missing:
        raise QuestionSetError(
            f"No question exercises {missing}. A registered family with no question "
            "contributes nothing and inflates the apparent scope of the evaluation."
        )


def _validate_against_corpus(question_set: QuestionSet, kb: Any) -> None:
    known_docs = {doc.doc_id for doc in kb}
    for question in question_set:
        for doc_id in tuple(question.expected_documents) + tuple(question.must_not_cite):
            if doc_id not in known_docs:
                raise QuestionSetError(
                    f"{question.question_id}: references unknown document {doc_id!r}"
                )
        for chunk_id in question.expected_chunks:
            doc_id = chunk_id.split("#")[0]
            if doc_id not in known_docs:
                raise QuestionSetError(
                    f"{question.question_id}: expected chunk {chunk_id!r} belongs to "
                    "no known document"
                )


# --- loading and writing -----------------------------------------------------


def _question_from_dict(payload: dict[str, Any]) -> Question:
    def tuple_of(key: str) -> tuple[str, ...]:
        return tuple(payload.get(key) or ())

    try:
        return Question(
            question_id=payload["question_id"],
            text=payload["text"],
            category=payload["category"],
            group_id=payload.get("group_id") or payload.get("family_id") or payload["question_id"],
            split=payload["split"],
            answerability=payload["answerability"],
            expected_behaviour=payload["expected_behaviour"],
            risk_level=payload.get("risk_level", "low"),
            family_id=payload.get("family_id"),
            gap_topic=payload.get("gap_topic"),
            paraphrase_of=payload.get("paraphrase_of"),
            gold_answer=payload.get("gold_answer", ""),
            gold_facts=tuple_of("gold_facts"),
            expected_documents=tuple_of("expected_documents"),
            expected_chunks=tuple_of("expected_chunks"),
            must_not_cite=tuple_of("must_not_cite"),
            notes=payload.get("notes", ""),
        )
    except KeyError as exc:
        raise QuestionSetError(
            f"Question {payload.get('question_id', '?')!r} is missing {exc}"
        ) from exc


def load_question_set(path: Path | str) -> QuestionSet:
    source = Path(path)
    if not source.exists():
        raise QuestionSetError(f"Question set not found: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QuestionSetError(f"{source} is not valid JSON: {exc}") from exc
    if "questions" not in data:
        raise QuestionSetError(f"{source}: missing 'questions'")

    questions = tuple(_question_from_dict(entry) for entry in data["questions"])
    metadata = {k: v for k, v in data.items() if k != "questions"}
    question_set = QuestionSet(questions, source, metadata)
    _validate_fields(question_set)
    _validate_split_integrity(question_set)
    _validate_paraphrases(question_set)
    return question_set


def write_question_set(
    question_set: QuestionSet, path: Path | str, **metadata: Any
) -> Path:
    """Write a question set, refusing to write one whose splits are unsound."""
    _validate_fields(question_set)
    _validate_split_integrity(question_set)
    _validate_paraphrases(question_set)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "_warning": (
            "Gold data. Not reachable from config.json and not to be read by any "
            "inference component."
        ),
        **metadata,
        "summary": question_set.summary(),
        "questions": [q.to_dict() for q in question_set],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def assign_splits(
    question_set: QuestionSet, *, test_groups: Sequence[str]
) -> QuestionSet:
    """Reassign splits by group, so a family can never be divided.

    Splits are chosen at the level of the group and applied to every question
    within it. Assigning per question is the mistake this function makes
    unavailable.
    """
    wanted = set(test_groups)
    unknown = wanted - set(question_set.groups)
    if unknown:
        raise QuestionSetError(f"Unknown groups requested for test split: {sorted(unknown)}")

    from dataclasses import replace

    return QuestionSet(
        tuple(
            replace(q, split="test" if q.group_id in wanted else "dev")
            for q in question_set
        ),
        question_set.source,
        question_set.metadata,
    )
