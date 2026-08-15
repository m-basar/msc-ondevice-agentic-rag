"""Blinded manual scoring: the interactive half of a procedure that was
already half written.

``run_writer.write_review_sheet`` produces the blinded sheet - answers pooled
across arms, shuffled, arm labels replaced by opaque codes, key written to a
separate file. What did not exist was anything to score it *with*, so the
primary metric of the study had no instrument. This module is that instrument.

Three properties matter more than the interface.

**The judgement survives the session.** Every judgement is appended and fsynced
before the next item is shown. A pass over 272 answers is not one sitting, and
a scorer that loses an evening's work to a closed terminal will be abandoned
half way through, which is a worse failure than an ugly prompt. Resuming reads
the log back and continues at the first unscored item.

**The opaque code is never displayed.** The sheet carries ``system_A`` and so
on because the scores have to be joined back to the arms eventually. Putting
that code on screen would hand the reviewer a stable label to accumulate
against: score forty items, notice that one code escalates far more often than
the others, and the blinding is gone for the remaining two hundred. This is the
same defeat that amendment 1.1.10 closed when it removed the evidence block,
and it is closed here the same way, by not emitting the thing. Judgement
records do not store the code either; they key on the item number, and the join
happens once, afterwards, from the sheet.

**Both flags are asked on every item.** ``asserts_conflict`` exists because the
three-point scale cannot express H2c: under ``answer_without_flagging_conflict``
a score of 1 covers both "described the documents as conflicting" and "covered
only one of two requirements", and only the first is a false conflict.
``abstained`` exists because appropriate abstention is currently measured by a
heuristic that the pre-registration itself labels diagnostic only. Asking both
everywhere, rather than only where a hypothesis needs them, keeps the prompt
identical across items so the question does not disclose which family type the
reviewer is looking at - and it makes over-detection outside the negative
controls visible instead of unmeasured.

**The blinding is partial, and the tool says so rather than assuming it.**
Amendment 1.13 records that thirteen items in the built sheet are the verbatim
``ABSTENTION_TEXT`` template, which only the verified arm can emit, and that
opaque codes are stable, so recognising the template once exposes all 68 items
carrying that code. The sheet cannot be repaired: item numbers are positions in
a shuffled file that judgements already key on, and redacting the template would
leave nothing to score on items whose whole question is whether declining was
right. Two things follow here. ``arm_signature_audit`` refuses to let a *new*
sheet be built with the defect, which fixes the class rather than the instance.
And the ``i`` flag records, per item, that the reviewer believes they can
identify the arm, so the unblinding rate is measured instead of asserted.

That flag deliberately does not ask *which* arm. Prompting for a guess on every
item invites speculation and would inflate the quantity it is meant to observe;
blinding indices are conventionally collected after assessment for the same
reason. The resulting rate is a self-report and bounds nothing in either
direction, and amendment 1.13.4 says so.

No model is called from this module and nothing here reads a run directory, the
conflict registry or the key. The inputs are the blinded sheet and the
reviewer.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import textwrap
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..verify.schema import ABSTENTION_TEXT

SCHEMA_VERSION = "1.1"

VALID_SCORES = (0, 1, 2)

#: Flag letters accepted after a score, and what each records.
FLAGS: dict[str, str] = {
    "c": "asserts_conflict",
    "a": "abstained",
    "u": "uncertain",
    "i": "arm_identified",
}

# Fixed text the system serves verbatim. ABSTENTION_TEXT is written by the
# source rather than by the model, under amendment 1.7.6, so its presence in an
# answer is a byte-exact signature of the verified arm. Auditing for served
# templates rather than for a list of previously known give-aways is the
# difference between catching this class of leak and catching the last one.
SERVED_TEMPLATES: dict[str, str] = {"ABSTENTION_TEXT": ABSTENTION_TEXT}

#: ``n`` requests a note. It is not a recorded field of its own.
NOTE_FLAG = "n"

# Fields that would tell a reviewer which system produced an answer, directly
# or by inference. write_review_sheet does not emit any of them; this list is
# what makes that a checked property of the file being scored rather than a
# property of the function that happened to write it. A sheet built by some
# future variant, or edited by hand, is refused here.
FORBIDDEN_SHEET_FIELDS = frozenset({
    "arm",
    "model",
    "generation_model",
    "verification_model",
    "generation",
    "verification",
    "prompt",
    "prompt_sha256",
    "evidence",
    "evidence_sha256",
    "retrieval",
    "citations",
    "document_citations",
    "hallucinated_citations",
    "uncited_chunks",
    "cited_superseded",
    "has_valid_citation_ids",
    "refusal_heuristic",
    "wall_seconds",
    "scoring",
    "confidence",
    "extra",
    "split",
})

REQUIRED_SHEET_FIELDS = frozenset({"item", "question_id", "question", "answer"})


class ScoringError(RuntimeError):
    """Raised when a sheet, a judgement log or a session is unusable."""


class InputError(ValueError):
    """Raised on malformed reviewer input. Recoverable: reprompt."""


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- the sheet ---------------------------------------------------------------


@dataclass(frozen=True)
class ReviewItem:
    """One blinded answer, with everything needed to judge it and nothing else."""

    item: int
    question_id: str
    question: str
    answer: str
    group_id: str = ""
    system: str = ""  # opaque code. Held for the join, never rendered.
    required_claims: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    acceptable_variants: tuple[str, ...] = ()
    scoring_criteria: dict[str, str] = field(default_factory=dict)

    @property
    def answer_sha256(self) -> str:
        return sha256_of_text(self.answer)

    @property
    def duplicate_key(self) -> tuple[str, str]:
        """What counts as "the same answer" for the consistency check.

        Keyed on the question as well as the text. Arm D replays Arm B's
        drafts, so a verification pass that changes nothing leaves two
        byte-identical answers in the pool, and those two must not receive
        different scores. But the system-written abstention template is also
        byte-identical across many *different* questions, and those legitimately
        differ, because they are judged against different rubrics. Keying on the
        pair separates the two cases.
        """
        return (self.question_id, self.answer_sha256)


def load_sheet(path: Path | str) -> tuple[ReviewItem, ...]:
    """Read the blinded sheet, refusing one that is not blind.

    The forbidden-field check is not defensive programming for its own sake.
    The sheet is the only thing standing between the reviewer and the arm
    labels, and it is a plain text file in the results tree that anyone can
    regenerate with different arguments. Checking it on every load costs
    nothing and converts "the blinding was correct when it was written" into
    "the blinding is correct now".
    """
    source = Path(path)
    if not source.exists():
        raise ScoringError(
            f"Review sheet not found: {source}. Build it first with "
            "`python scripts/score_answers.py build`."
        )

    items: list[ReviewItem] = []
    seen: set[int] = set()
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoringError(f"{source} line {number} is not valid JSON: {exc}") from exc

        leaked = sorted(FORBIDDEN_SHEET_FIELDS & set(record))
        if leaked:
            raise ScoringError(
                f"{source} line {number} carries {leaked}, which identifies the "
                "system that produced the answer. This sheet is not blind and "
                "must not be scored. Rebuild it with "
                "run_writer.write_review_sheet."
            )
        missing = sorted(REQUIRED_SHEET_FIELDS - set(record))
        if missing:
            raise ScoringError(f"{source} line {number} is missing {missing}")
        if record["item"] in seen:
            raise ScoringError(
                f"{source} line {number}: duplicate item number {record['item']}. "
                "Item numbers key the judgement log, so duplicates would silently "
                "overwrite each other's scores."
            )
        seen.add(record["item"])

        items.append(
            ReviewItem(
                item=int(record["item"]),
                question_id=str(record["question_id"]),
                question=str(record["question"]),
                answer=str(record["answer"]),
                group_id=str(record.get("group_id", "")),
                system=str(record.get("system", "")),
                required_claims=tuple(record.get("required_claims") or ()),
                forbidden_claims=tuple(record.get("forbidden_claims") or ()),
                acceptable_variants=tuple(record.get("acceptable_variants") or ()),
                scoring_criteria=dict(record.get("scoring_criteria") or {}),
            )
        )

    if not items:
        raise ScoringError(f"{source} contains no items")
    return tuple(items)


# --- judgements --------------------------------------------------------------


@dataclass(frozen=True)
class Judgement:
    """One reviewer decision about one item.

    ``revision`` counts re-scores of the same item. Nothing is ever deleted
    from the log, so a changed judgement leaves both the original and the
    replacement on disk. A reviewer who revisits item 37 after learning how the
    rubric behaves in practice should be able to say so, and the record should
    show that they did rather than quietly presenting the second score as the
    first.
    """

    item: int
    question_id: str
    score: int
    asserts_conflict: bool = False
    abstained: bool = False
    uncertain: bool = False
    # Tri-state, and the third state is load-bearing. ``False`` means the
    # reviewer was asked and said no. ``None`` means the item was scored before
    # the flag existed and the question was never put. Collapsing the two would
    # move those items into the denominator of the unblinding rate as evidence
    # of blinding that was never gathered.
    arm_identified: bool | None = None
    note: str = ""
    revision: int = 0
    recorded_at: str = ""

    def __post_init__(self) -> None:
        if self.score not in VALID_SCORES:
            raise ScoringError(
                f"Score {self.score!r} for item {self.item} is not one of "
                f"{VALID_SCORES}. The rubric is a three-point scale."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "item": self.item,
            "question_id": self.question_id,
            "score": self.score,
            "asserts_conflict": self.asserts_conflict,
            "abstained": self.abstained,
            "uncertain": self.uncertain,
            "arm_identified": self.arm_identified,
            "note": self.note,
            "revision": self.revision,
            "recorded_at": self.recorded_at,
        }


def _append_jsonl(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Append one record and force it to disk before returning.

    The flush and fsync are the whole point. Buffered writes mean a terminal
    closed at item 180 loses however much of the log the interpreter had not
    got round to writing, and the reviewer cannot tell which items those were.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_judgement(path: Path | str, judgement: Judgement) -> Judgement:
    """Append one judgement, durably."""
    stamped = judgement if judgement.recorded_at else replace(judgement, recorded_at=_now())
    _append_jsonl(path, stamped.to_dict())
    return stamped


def load_judgements(path: Path | str) -> dict[int, Judgement]:
    """Replay the log, last write wins.

    Reading the whole log rather than the last state of each item is what makes
    re-scoring safe: the file is append-only, so the current judgement is
    whatever was written most recently, and the earlier ones remain available
    to anyone auditing how a score changed.
    """
    source = Path(path)
    if not source.exists():
        return {}

    current: dict[int, Judgement] = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoringError(
                f"{source} line {number} is not valid JSON: {exc}. The judgement "
                "log is append-only; a corrupt line usually means a write was "
                "interrupted, and the line can be deleted by hand."
            ) from exc
        current[int(record["item"])] = Judgement(
            item=int(record["item"]),
            question_id=str(record.get("question_id", "")),
            score=int(record["score"]),
            asserts_conflict=bool(record.get("asserts_conflict", False)),
            abstained=bool(record.get("abstained", False)),
            uncertain=bool(record.get("uncertain", False)),
            # Absent means the flag did not exist when this line was written.
            # Reading it as False would fabricate a negative answer.
            arm_identified=(
                None
                if record.get("arm_identified") is None
                else bool(record["arm_identified"])
            ),
            note=str(record.get("note", "")),
            revision=int(record.get("revision", 0)),
            recorded_at=str(record.get("recorded_at", "")),
        )
    return current


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- session integrity -------------------------------------------------------


def open_session(
    session_path: Path | str, sheet_path: Path | str, *, item_count: int
) -> dict[str, Any]:
    """Bind a judgement log to the exact sheet it was started against.

    Item numbers are the key of the whole scheme, and they are positions in a
    shuffled file. Rebuild the sheet from a different set of runs, or with a
    different seed, and item 37 is a different answer while the log still says
    it was scored. Recording the sheet's hash at the start and refusing a
    mismatch makes that failure loud instead of invisible.
    """
    session = Path(session_path)
    fingerprint = sha256_of_file(sheet_path)

    if session.exists():
        existing = json.loads(session.read_text(encoding="utf-8"))
        if existing.get("sheet_sha256") != fingerprint:
            raise ScoringError(
                "The review sheet has changed since scoring began.\n"
                f"  recorded: {existing.get('sheet_sha256')}\n"
                f"  current:  {fingerprint}\n"
                "Item numbers index a shuffled file, so judgements already "
                "recorded no longer refer to the answers now at those "
                "positions. Restore the original sheet, or start a new "
                "judgement log against this one."
            )
        return existing

    session.parent.mkdir(parents=True, exist_ok=True)
    created = {
        "schema_version": SCHEMA_VERSION,
        "sheet": str(sheet_path),
        "sheet_sha256": fingerprint,
        "item_count": item_count,
        "started_at": _now(),
    }
    session.write_text(json.dumps(created, indent=2), encoding="utf-8")
    return created


# --- input grammar -----------------------------------------------------------


@dataclass(frozen=True)
class ParsedScore:
    score: int
    asserts_conflict: bool = False
    abstained: bool = False
    uncertain: bool = False
    arm_identified: bool = False
    wants_note: bool = False


def parse_score_input(text: str) -> ParsedScore:
    """Parse ``2``, ``1c``, ``0 a u``, ``2cun`` and reject everything else.

    Deliberately strict. A silently swallowed stray character in a 272-item
    pass is a wrong score in the primary metric, and the cost of being told to
    type it again is two seconds.
    """
    cleaned = "".join(text.split()).lower()
    if not cleaned:
        raise InputError("Nothing entered.")

    head, tail = cleaned[0], cleaned[1:]
    if head not in "012":
        raise InputError(f"{text.strip()!r} does not start with a score of 0, 1 or 2.")

    seen: set[str] = set()
    for letter in tail:
        if letter not in FLAGS and letter != NOTE_FLAG:
            raise InputError(
                f"{letter!r} is not a flag. Use c, a, u, i or n."
            )
        if letter in seen:
            raise InputError(f"{letter!r} given twice.")
        seen.add(letter)

    return ParsedScore(
        score=int(head),
        asserts_conflict="c" in seen,
        abstained="a" in seen,
        uncertain="u" in seen,
        arm_identified="i" in seen,
        # ``i`` deliberately does not force a note. Friction on an honesty flag
        # suppresses the flag, and an undercounted unblinding rate is worse
        # than an unexplained one. ``n`` is there if the reviewer wants to say
        # why.
        wants_note=NOTE_FLAG in seen or "u" in seen,
    )


# --- rendering ---------------------------------------------------------------

_WIDTH = 78
_RULE = "-" * _WIDTH


def _wrap(text: str, indent: str = "  ") -> str:
    paragraphs = (text or "").splitlines() or [""]
    out: list[str] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            out.append("")
            continue
        out.extend(
            textwrap.wrap(
                paragraph,
                width=_WIDTH,
                initial_indent=indent,
                subsequent_indent=indent,
            )
            or [indent]
        )
    return "\n".join(out)


def _bullets(label: str, values: Sequence[str]) -> list[str]:
    if not values:
        return []
    block = [f"{label}"]
    for value in values:
        block.extend(
            textwrap.wrap(
                value, width=_WIDTH, initial_indent="  - ", subsequent_indent="    "
            )
            or ["  - "]
        )
    return block + [""]


def render_item(
    item: ReviewItem,
    *,
    position: int,
    total: int,
    scored: int,
    existing: Judgement | None = None,
) -> str:
    """The screen the reviewer sees.

    ``item.system`` is not referenced anywhere in this function, and
    ``tests/test_manual_scoring.py`` asserts that no opaque code reaches the
    output. That assertion is the blinding, expressed as a test rather than as
    a comment.
    """
    lines = [
        "",
        _RULE,
        f" Item {position} of {total}".ljust(28)
        + f"scored {scored}".ljust(16)
        + f"remaining {total - scored}",
        _RULE,
        "",
        f"QUESTION  [{item.question_id}]",
        _wrap(item.question),
        "",
        "ANSWER",
        _wrap(item.answer),
        "",
    ]

    if item.scoring_criteria:
        lines.append("RUBRIC  what a correct answer does")
        for score in ("2", "1", "0"):
            criterion = item.scoring_criteria.get(score)
            if criterion:
                lines.extend(
                    textwrap.wrap(
                        criterion,
                        width=_WIDTH,
                        initial_indent=f"  {score}  ",
                        subsequent_indent="     ",
                    )
                )
        lines.append("")

    lines.extend(_bullets("REQUIRED CLAIMS", item.required_claims))
    lines.extend(_bullets("FORBIDDEN CLAIMS", item.forbidden_claims))
    lines.extend(_bullets("ACCEPTABLE VARIANTS", item.acceptable_variants))

    if existing is not None:
        flags = describe_flags(existing) or "no flags"
        lines.append(f"ALREADY SCORED  {existing.score}, {flags}. Re-scoring revises it.")
        if existing.note:
            lines.append(_wrap(f"note: {existing.note}"))
        lines.append("")

    return "\n".join(lines)


def describe_flags(judgement: Judgement) -> str:
    names = []
    if judgement.asserts_conflict:
        names.append("asserts a conflict")
    if judgement.abstained:
        names.append("declined to answer")
    if judgement.uncertain:
        names.append("uncertain")
    if judgement.arm_identified:
        names.append("arm identifiable")
    return ", ".join(names)


HELP = """
  0 1 2     the rubric score for this answer

  add any of these letters straight after the score:
    c       the answer asserts the documents conflict, or declines on
            those grounds
    a       the answer declines to answer rather than answering
    u       you are not confident in this judgement (prompts for a note)
    i       you believe you can tell which system produced this answer,
            for any reason. Judge it anyway; the flag records that the
            blinding did not hold here, so the rate is measured rather
            than assumed. Do not write down which system you think it is.
    n       add a note

  examples:  2      1c      0a      2cu      1 c n      2i

  commands:
    s       skip this item, come back to it at the end
    b       back one item
    g N     go to item N
    p       redisplay this item
    ?       this help
    q       save and quit
"""


# --- progress and consistency ------------------------------------------------


def progress(
    items: Sequence[ReviewItem], judgements: Mapping[int, Judgement]
) -> dict[str, Any]:
    numbers = [i.item for i in items]
    scored = [n for n in numbers if n in judgements]
    asked = [n for n in scored if judgements[n].arm_identified is not None]
    identified = [n for n in asked if judgements[n].arm_identified]
    return {
        "total": len(numbers),
        "scored": len(scored),
        "remaining": len(numbers) - len(scored),
        "uncertain": sum(1 for n in scored if judgements[n].uncertain),
        "revised": sum(1 for n in scored if judgements[n].revision > 0),
        "with_notes": sum(1 for n in scored if judgements[n].note),
        "arm_identified": len(identified),
        "identification_asked": len(asked),
        "identification_not_asked": len(scored) - len(asked),
        # Reported over the items where the question was actually put. Dividing
        # by everything scored would treat "not asked" as "said no".
        "unblinding_rate": (len(identified) / len(asked)) if asked else None,
        "complete": len(scored) == len(numbers),
    }


def arm_signature_audit(items: Iterable[ReviewItem]) -> dict[str, Any]:
    """Which items carry text the system serves verbatim, and how far it reaches.

    The build-time blinding check was a list of give-aways already known: arm
    labels, model names, opaque codes, evidence status markers. It could not
    catch ``ABSTENTION_TEXT``, because that constant was introduced by amendment
    1.7.6 long after the list was written. Checking for served templates instead
    tests the property rather than enumerating its past violations.

    The second figure is the one that matters. A template appearing on thirteen
    items is not a thirteen-item leak: opaque codes are stable across the sheet,
    so recognising it once identifies every item sharing that code. The audit
    reports the exposure, not the trigger count.
    """
    pooled = list(items)
    by_code: dict[str, int] = {}
    for item in pooled:
        by_code[item.system] = by_code.get(item.system, 0) + 1

    findings: list[dict[str, Any]] = []
    exposed: set[str] = set()
    for name, template in SERVED_TEMPLATES.items():
        matches = [i for i in pooled if i.answer.strip() == template.strip()]
        if not matches:
            continue
        codes = {i.system for i in matches}
        exposed |= codes
        findings.append({
            "template": name,
            "items": sorted(i.item for i in matches),
            "questions": len({i.question_id for i in matches}),
            "codes_carrying_it": len(codes),
            "items_exposed": sum(by_code.get(code, 0) for code in codes),
        })

    return {
        "total_items": len(pooled),
        "findings": findings,
        "items_exposed": sum(by_code.get(code, 0) for code in exposed),
        "blind": not findings,
    }


def next_unscored(
    items: Sequence[ReviewItem],
    judgements: Mapping[int, Judgement],
    *,
    after: int = 0,
) -> int | None:
    """Position (1-based) of the next item without a judgement, or None."""
    for position, item in enumerate(items, start=1):
        if position > after and item.item not in judgements:
            return position
    for position, item in enumerate(items, start=1):
        if item.item not in judgements:
            return position
    return None


# --- the abstention re-pass --------------------------------------------------
#
# The first pass collected ``abstained`` alongside a three-point rubric score.
# The scores proved perfectly consistent across 58 blind repeats. The flag did
# not: six duplicate groups disagreed, and in every one the items marked came
# earlier in the sequence than the items not marked, which is drift rather than
# noise. Half the sheet is singletons where the same drift leaves no trace, so
# the observable disagreement rate is an estimate of an error rate that is
# mostly invisible.
#
# The fix is a second pass that asks one question and nothing else. Reading an
# answer for "did this decline?" is a different and much cheaper task than
# scoring it against a rubric, and the flag stops competing for attention with
# the judgement it was collected beside.

ABSTENTION_ORDER_SEED = 4114


@dataclass(frozen=True)
class AbstentionJudgement:
    """One answer to the single question the re-pass asks."""

    item: int
    question_id: str
    abstained: bool
    note: str = ""
    revision: int = 0
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "pass": "abstention",
            "item": self.item,
            "question_id": self.question_id,
            "abstained": self.abstained,
            "note": self.note,
            "revision": self.revision,
            "recorded_at": self.recorded_at,
        }


def append_abstention(
    path: Path | str, judgement: AbstentionJudgement
) -> AbstentionJudgement:
    stamped = judgement if judgement.recorded_at else replace(judgement, recorded_at=_now())
    _append_jsonl(path, stamped.to_dict())
    return stamped


def load_abstention(path: Path | str) -> dict[int, AbstentionJudgement]:
    source = Path(path)
    if not source.exists():
        return {}
    current: dict[int, AbstentionJudgement] = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoringError(f"{source} line {number} is not valid JSON: {exc}") from exc
        current[int(record["item"])] = AbstentionJudgement(
            item=int(record["item"]),
            question_id=str(record.get("question_id", "")),
            abstained=bool(record["abstained"]),
            note=str(record.get("note", "")),
            revision=int(record.get("revision", 0)),
            recorded_at=str(record.get("recorded_at", "")),
        )
    return current


def abstention_order(
    items: Sequence[ReviewItem], *, seed: int = ABSTENTION_ORDER_SEED
) -> tuple[ReviewItem, ...]:
    """Present the re-pass in a different order from the first pass.

    This is the point of the exercise. The first pass drifted with position, so
    a second pass in the same order would drift at the same places, the two
    would agree wrongly, and the agreement rate would report reliability the
    instrument does not have. Re-ordering decorrelates any new drift from the
    old, which is what makes the comparison worth making.

    Item numbers are untouched, so the join back to the first pass, to the
    sheet and to the key is unaffected.
    """
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled)


def render_abstention_item(
    item: ReviewItem, *, position: int, total: int, decided: int
) -> str:
    """One question, no rubric, no score, and no sight of the first pass.

    The first pass's answer is deliberately not displayed. Showing it would
    anchor the second and produce agreement that measures suggestion rather
    than reliability.
    """
    return "\n".join([
        "",
        _RULE,
        f" Abstention pass  {position} of {total}".ljust(40) + f"decided {decided}",
        _RULE,
        "",
        f"QUESTION  [{item.question_id}]",
        _wrap(item.question),
        "",
        "ANSWER",
        _wrap(item.answer),
        "",
        "  Did this answer decline to answer, rather than answering?",
        "  y = declined      n = answered      (y? or n? to add a note)",
        "",
    ])


ABSTENTION_HELP = """
  y       the answer declines: it says the evidence does not support an
          answer, or otherwise refuses rather than committing to one
  n       the answer answers, however well or badly

  y? n?   as above, and prompt for a note

  commands:
    s     skip, come back to it
    b     back one item
    p     redisplay
    ?     this help
    q     save and quit

  Judge only whether it declined. Not whether declining was correct: that is
  the rubric score, and it is already recorded.
"""


def parse_abstention_input(text: str) -> tuple[bool, bool]:
    """Return (abstained, wants_note)."""
    cleaned = "".join(text.split()).lower()
    if not cleaned:
        raise InputError("Nothing entered.")
    wants_note = cleaned.endswith("?")
    head = cleaned[:-1] if wants_note else cleaned
    if head not in {"y", "n", "yes", "no"}:
        raise InputError(f"{text.strip()!r} is not y or n.")
    return head.startswith("y"), wants_note


def abstention_agreement(
    first: Mapping[int, Judgement], second: Mapping[int, AbstentionJudgement]
) -> dict[str, Any]:
    """How far the two passes agree on the one field they share.

    This replaces an estimate with a measurement. The first pass could only be
    checked where the sheet happened to repeat an answer, which was 147 of 272
    items; the re-pass covers every item, so the disagreement rate is observed
    rather than extrapolated from the half that was visible.

    Which pass is reported is fixed in advance by amendment 1.14 and is not
    decided here or from these numbers.
    """
    shared = sorted(set(first) & set(second))
    disagreements = [
        {
            "item": n,
            "question_id": second[n].question_id,
            "first_pass": first[n].abstained,
            "second_pass": second[n].abstained,
        }
        for n in shared
        if first[n].abstained != second[n].abstained
    ]
    only_second = sum(1 for n in shared if second[n].abstained and not first[n].abstained)
    return {
        "compared": len(shared),
        "first_pass_only": sorted(set(first) - set(second)),
        "second_pass_only": sorted(set(second) - set(first)),
        "agreed": len(shared) - len(disagreements),
        "disagreed": len(disagreements),
        "agreement_rate": (
            (len(shared) - len(disagreements)) / len(shared) if shared else None
        ),
        # Direction matters. The first pass drifted towards under-marking, so a
        # second pass finding abstentions the first missed is the expected
        # shape, and the reverse would need explaining.
        "missed_by_first_pass": only_second,
        "missed_by_second_pass": len(disagreements) - only_second,
        "disagreements": disagreements,
    }


def positional_drift_report(
    items: Sequence[ReviewItem],
    first: Mapping[int, Judgement],
    second: Mapping[int, AbstentionJudgement],
    *,
    order_seed: int = ABSTENTION_ORDER_SEED,
) -> dict[str, Any]:
    """Locate each disagreement in *both* passes' orders, and check each pass
    against itself on the repeated answers.

    This is what the re-ordering was for, and it is the only way to tell the two
    competing explanations apart. If a set of disagreements clusters at the end
    of one pass's sequence and scatters in the other's, the cause is that pass's
    position, which is consistent with fatigue or with criterion drift over
    the sequence. If it scatters in both, position is not the
    cause and the disagreement is a real difference of criterion.

    Written as a function rather than left as the shell command that first
    produced the figures, so the numbers quoted in amendment 1.14.7 can be
    regenerated rather than trusted.
    """
    order_one = {item.item: n for n, item in enumerate(items, start=1)}
    order_two = {
        item.item: n
        for n, item in enumerate(abstention_order(items, seed=order_seed), start=1)
    }
    shared = sorted(set(first) & set(second))

    def locate(numbers: Sequence[int]) -> dict[str, Any]:
        if not numbers:
            return {"n": 0}
        in_one = sorted(order_one[n] for n in numbers)
        in_two = sorted(order_two[n] for n in numbers)
        total = len(items)
        return {
            "n": len(numbers),
            "items": sorted(numbers),
            "positions_in_first_pass": in_one,
            "positions_in_second_pass": in_two,
            # A contiguous tail in one order and a spread in the other is the
            # signature of a position effect in that pass, not proof of its
            # mechanism.
            "first_pass_span": [in_one[0], in_one[-1]],
            "second_pass_span": [in_two[0], in_two[-1]],
            "confined_to_first_pass_tail": in_one[0] > total * 0.75,
            "confined_to_second_pass_tail": in_two[0] > total * 0.75,
        }

    groups: dict[tuple[str, str], list[int]] = {}
    for item in items:
        groups.setdefault(item.duplicate_key, []).append(item.item)
    dupes = {k: sorted(v) for k, v in groups.items() if len(v) > 1}

    def internal(source: Mapping[int, Any]) -> dict[str, Any]:
        divergent = [
            nums for nums in dupes.values()
            if len({source[n].abstained for n in nums if n in source}) > 1
        ]
        return {
            "groups": len(dupes),
            "consistent": len(dupes) - len(divergent),
            "divergent": len(divergent),
        }

    return {
        "compared": len(shared),
        "second_pass_only": locate([n for n in shared
                                    if second[n].abstained and not first[n].abstained]),
        "first_pass_only": locate([n for n in shared
                                   if first[n].abstained and not second[n].abstained]),
        "internal_consistency": {
            "first_pass": internal(first),
            "second_pass": internal(second),
        },
    }


def consistency_report(
    items: Iterable[ReviewItem],
    judgements: Mapping[int, Judgement],
    abstention: Mapping[int, AbstentionJudgement] | None = None,
) -> dict[str, Any]:
    """Where identical text on the same question received different judgements.

    This is the price of scoring every item independently. Arm D replays Arm
    B's drafts, so whenever verification changes nothing the pool holds two
    byte-identical answers, and independent scoring can put a 1 on one and a 2
    on the other. That difference is noise, it lands squarely on the B versus D
    contrast, and left unexamined it is indistinguishable from an effect.

    Reporting it rather than preventing it keeps the pre-registered procedure
    intact: every item was judged on its own, and the disagreements are then
    visible, countable and reconcilable on the record instead of silently
    averaged into the result. Nothing here reveals which arm produced anything.

    ``abstention`` supplies the re-pass values. Amendment 1.14.4 rule 1 makes
    the second pass the reported value of ``abstained``, so a report that used
    the first pass's would count divergences the re-pass has already settled
    and overstate the remaining inconsistency several times over. Passing them
    applies a rule that was fixed before unblinding; it is not a judgement made
    after it. Omitting the argument still reports the first pass against
    itself, which is the 52 of 58 figure cited in 1.14.6.
    """
    groups: dict[tuple[str, str], list[ReviewItem]] = {}
    for item in items:
        groups.setdefault(item.duplicate_key, []).append(item)

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}

    divergent: list[dict[str, Any]] = []
    consistent = 0
    incomplete = 0
    for (question_id, digest), members in sorted(
        duplicate_groups.items(), key=lambda kv: min(m.item for m in kv[1])
    ):
        decided = [judgements[m.item] for m in members if m.item in judgements]
        if len(decided) < len(members):
            incomplete += 1
            continue
        def abstained_for(j: Judgement) -> bool:
            if abstention is not None and j.item in abstention:
                return abstention[j.item].abstained
            return j.abstained

        signatures = {
            (j.score, j.asserts_conflict, abstained_for(j)) for j in decided
        }
        if len(signatures) == 1:
            consistent += 1
            continue
        divergent.append({
            "question_id": question_id,
            "answer_sha256": digest,
            "items": sorted(m.item for m in members),
            "judgements": [
                {
                    "item": j.item,
                    "score": j.score,
                    "asserts_conflict": j.asserts_conflict,
                    "abstained": abstained_for(j),
                }
                for j in sorted(decided, key=lambda j: j.item)
            ],
        })

    return {
        "duplicate_groups": len(duplicate_groups),
        "fully_scored_groups": consistent + len(divergent),
        "not_yet_fully_scored": incomplete,
        "consistent": consistent,
        "divergent": divergent,
        "note": (
            "Identical answer text on the same question that received different "
            "scores or flags. Each entry is a judgement to reconcile before the "
            "scores are joined to the arms."
        ),
    }
