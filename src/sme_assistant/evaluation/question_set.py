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
        "cite_current_only",              # supersession: answer from the live document
        "surface_both_and_escalate",      # mutually_exclusive: no basis to choose
        "prefer_stricter_and_escalate",   # stricter_looser: name the safe reading
        "answer_without_flagging_conflict",  # compatible: negative control
        "abstain",                        # nothing in the corpus supports an answer
        "answer_and_flag_gap",            # partial: say what is known, name what is not
        "answer_directly",                # ordinary factual question
    }
)


# The manual three-point scale, defined once per expected behaviour rather
# than repeated on every question. A reviewer scores against these, blind to
# which arm produced the answer. Written into the question set file so the
# rubric that was in force cannot drift away from the scores it produced.
SCORING_RUBRICS: dict[str, dict[str, str]] = {
    "cite_current_only": {
        "2": "States the current figure and attributes it to the current document. Does not present the withdrawn figure as being in force.",
        "1": "States the current figure but attributes it to the withdrawn document, or presents both figures without saying which governs.",
        "0": "States the withdrawn figure as the answer.",
    },
    "surface_both_and_escalate": {
        "2": "States both positions, names both documents, says neither supersedes the other, AND escalates, by directing the reader to have the discrepancy resolved. All four.",
        "1": "States both positions and names both documents, but selects one without qualification, or does not escalate.",
        "0": "States one position as settled policy, with no indication that the corpus disagrees with itself.",
    },
    "prefer_stricter_and_escalate": {
        "2": "Names the stricter figure as the safe course, states the other figure and its document, and escalates the discrepancy.",
        "1": "Gives the stricter figure without mentioning the other document, or gives both without identifying which is safe to follow.",
        "0": "Gives the looser figure as the answer. Following it would breach the stricter document, which is why this is scored as a failure rather than a partial success.",
    },
    "answer_without_flagging_conflict": {
        "2": "Answers correctly, covering both requirements where both apply, and does NOT assert that the documents contradict each other.",
        "1": "Answers correctly but describes the documents as conflicting or inconsistent, or covers only one of two requirements that both apply.",
        "0": "Declines to answer on the grounds of a conflict, or escalates. The documents agree; treating them as contradictory is a false positive, which is the failure this family exists to detect.",
    },
    "abstain": {
        "2": "Declines to answer and says the evidence does not cover the topic.",
        "1": "Declines but hedges in a way that implies an answer, or answers a different question instead.",
        "0": "Answers, including any specific figure. The corpus contains none, so any figure is invented.",
    },
    "answer_and_flag_gap": {
        "2": "Reports what the corpus does say and names the detail that is missing.",
        "1": "Does one of the two: reports the partial evidence without flagging the gap, or flags the gap without reporting the evidence.",
        "0": "Supplies the missing detail, which the corpus does not contain.",
    },
    "answer_directly": {
        "2": "Every required claim is present and no forbidden claim is made.",
        "1": "Some required claims are present, none forbidden. Incomplete rather than wrong.",
        "0": "A required claim is contradicted, or a forbidden claim is made.",
    },
}

# Content correctness is scored above. Citation validity, support and
# completeness are scored separately and automatically in
# ``sme_assistant.evaluation.answer_scoring``.
#
# They were entangled: the ``answer_directly`` criterion previously awarded 1
# to "correct, but a citation is missing or points at the wrong passage", which
# meant a metric named answer correctness could not report correctness. A right
# answer with a bad citation and a half-right answer with a good one scored the
# same, and neither number meant what it said.
SCORING_SEPARATION_NOTE = (
    "These criteria score content only. Citation validity, support and "
    "completeness are measured separately and automatically, and are reported "
    "alongside rather than folded into this scale."
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
    # Per question, not per family. Three questions in one family may share a
    # document pair and still target different claims, and a family-level fact
    # list marks a concise correct answer wrong while passing an incomplete one.
    required_claims: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    acceptable_variants: tuple[str, ...] = ()
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
            "required_claims": list(self.required_claims),
            "forbidden_claims": list(self.forbidden_claims),
            "acceptable_variants": list(self.acceptable_variants),
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
        if question.answerability != "unanswerable" and not question.required_claims:
            raise QuestionSetError(
                f"{where}: no required_claims, so no rubric decides whether an "
                "answer is right. A gold answer alone leaves the scorer to judge "
                "by resemblance."
            )
        if question.answerability == "unanswerable" and question.required_claims:
            raise QuestionSetError(
                f"{where}: unanswerable but declares required claims"
            )


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
    tuning = {family.family_id for family in getattr(registry, "tuning_families", ())}
    topics = {gap.topic for gap in registry.fully_absent}
    topics |= {gap.topic for gap in registry.partially_present}

    for question in question_set:
        if question.family_id and question.family_id not in (known | tuning):
            raise QuestionSetError(
                f"{question.question_id}: family {question.family_id!r} is not in "
                "the conflict registry"
            )
        # Tuning families exist so the pipeline can be developed against real
        # conflicts. That only works if they are the *only* conflicts anyone
        # develops against, which means the boundary has to run both ways: a
        # reported family may not be inspected during tuning, and a tuning
        # family may not contribute to a result.
        if question.family_id in known and question.split != "test":
            raise QuestionSetError(
                f"{question.question_id}: {question.family_id} is a reported family "
                f"and cannot sit in the {question.split!r} split. Tuning against a "
                "family that is later scored contaminates the result."
            )
        if question.family_id in tuning and question.split != "dev":
            raise QuestionSetError(
                f"{question.question_id}: {question.family_id} is a tuning family "
                f"and cannot sit in the {question.split!r} split. Tuning families "
                "were inspected during development and must never be reported."
            )
        if question.gap_topic and question.gap_topic not in topics:
            raise QuestionSetError(
                f"{question.question_id}: gap topic {question.gap_topic!r} is not a "
                "declared gap, so nothing guarantees it is still absent"
            )
        if question.family_id:
            family = registry.by_id(question.family_id)
            # Derived from the conflict type, so a question cannot declare a
            # behaviour the classification does not justify. CONF-07 and CONF-09
            # were classified as conflicts and carried conflict behaviour for
            # three days; deriving it means a reclassification propagates rather
            # than leaving the questions asserting the old rule.
            expected = family.expected_behaviour
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
            required_claims=tuple_of("required_claims"),
            forbidden_claims=tuple_of("forbidden_claims"),
            acceptable_variants=tuple_of("acceptable_variants"),
            expected_documents=tuple_of("expected_documents"),
            expected_chunks=tuple_of("expected_chunks"),
            must_not_cite=tuple_of("must_not_cite"),
            notes=payload.get("notes", ""),
        )
    except KeyError as exc:
        raise QuestionSetError(
            f"Question {payload.get('question_id', '?')!r} is missing {exc}"
        ) from exc


def check_provenance(
    question_set: QuestionSet,
    *,
    corpus_sha256: str | None = None,
    chunk_set_sha256: str | None = None,
    registry_sha256: str | None = None,
) -> None:
    """Compare the hashes recorded in the file against the current state.

    Recording a hash and checking it are different things, and only the second
    prevents anything. A question set names specific chunk identifiers as the
    evidence for each answer. If the chunker changes, those identifiers keep
    resolving but point at different text, and every scored result is quietly
    wrong. The corpus hash cannot see that, which is why the chunk-set hash is
    checked here and is the one that matters most.

    Missing recorded hashes are reported rather than skipped. A question set
    written before this check existed cannot be trusted to be current, and
    silently accepting it would defeat the purpose.
    """
    expectations = {
        "corpus_sha256": corpus_sha256,
        "chunk_set_sha256": chunk_set_sha256,
        "registry_sha256": registry_sha256,
    }
    problems: list[str] = []
    for name, actual in expectations.items():
        if actual is None:
            continue
        recorded = question_set.metadata.get(name)
        if not recorded:
            problems.append(f"{name} is not recorded in the question set")
        elif recorded != actual:
            problems.append(
                f"{name} recorded {recorded[:12]} but the current value is {actual[:12]}"
            )
    if problems:
        raise QuestionSetError(
            "The question set does not match the current state: "
            + "; ".join(problems)
            + ". Rebuild it with scripts/build_question_set.py, and treat any "
            "result produced against the old one as invalid."
        )


def load_question_set(
    path: Path | str,
    *,
    corpus_sha256: str | None = None,
    chunk_set_sha256: str | None = None,
    registry_sha256: str | None = None,
) -> QuestionSet:
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
    check_provenance(
        question_set,
        corpus_sha256=corpus_sha256,
        chunk_set_sha256=chunk_set_sha256,
        registry_sha256=registry_sha256,
    )
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
        "scoring_rubrics": SCORING_RUBRICS,
        "scoring_separation": SCORING_SEPARATION_NOTE,
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
