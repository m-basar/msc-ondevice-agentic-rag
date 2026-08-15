"""Tests for the blinded manual scoring instrument.

The primary metric of this study is a human judgement, which makes the tool
that collects it part of the method rather than a convenience. Three things
have to hold, and each is asserted here rather than described in a docstring:

- the reviewer is never shown anything that identifies the arm, including the
  opaque code, which is a stable per-arm label and therefore a pattern to
  accumulate against
- a judgement survives the terminal closing
- the key is not reachable during a scoring pass

The last one is tested by recording every file the scoring path opens and
asserting the key is not among them. A comment saying "we do not read the key"
is worth nothing; the pre-registration is explicit that boundaries are enforced
in code rather than by intention, and this is that boundary.
"""

from __future__ import annotations

import builtins
import json
import pathlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from score_answers import main  # noqa: E402

from sme_assistant.common.config import load_config
from sme_assistant.common.llm_client import MockClient
from sme_assistant.evaluation.manual_scoring import (
    AbstentionJudgement,
    InputError,
    Judgement,
    ScoringError,
    abstention_agreement,
    abstention_order,
    append_abstention,
    append_judgement,
    arm_signature_audit,
    consistency_report,
    load_abstention,
    load_judgements,
    load_sheet,
    next_unscored,
    open_session,
    parse_abstention_input,
    parse_score_input,
    positional_drift_report,
    progress,
    render_abstention_item,
    render_item,
)
from sme_assistant.evaluation.question_set import load_question_set
from sme_assistant.evaluation.config import load_evaluation_config
from sme_assistant.evaluation.run_writer import (
    ArmDefinition,
    RunWriter,
    write_review_sheet,
)
from sme_assistant.generate.generator import Generator
from sme_assistant.ingest.index import build_index
from sme_assistant.kb.loader import load_knowledge_base
from sme_assistant.retrieve.retriever import EvidenceFormat, Retriever
from sme_assistant.verify.schema import ABSTENTION_TEXT


# --- fixtures ---------------------------------------------------------------


def sheet_line(item: int, question_id: str, answer: str, system: str = "system_A", **extra):
    return {
        "item": item,
        "question_id": question_id,
        "group_id": question_id,
        "question": f"question for {question_id}?",
        "system": system,
        "answer": answer,
        "required_claims": ["the rate is 45p per mile"],
        "forbidden_claims": ["the rate is 40p per mile"],
        "acceptable_variants": [],
        "scoring_criteria": {"2": "correct", "1": "partial", "0": "wrong"},
        **extra,
    }


def write_sheet(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def sheet(tmp_path):
    return write_sheet(
        tmp_path / "review.jsonl",
        [
            sheet_line(1, "TEST-001", "the rate is 45p", "system_C"),
            sheet_line(2, "TEST-001", "the rate is 45p", "system_A"),
            sheet_line(3, "TEST-002", "no evidence covers this", "system_D"),
            sheet_line(4, "TEST-003", "the rate is 40p", "system_B"),
        ],
    )


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def kb(config):
    return load_knowledge_base(config.path("paths.kb_docs"))


@pytest.fixture(scope="module")
def index(kb, config):
    return build_index(kb, MockClient(config), config)


# --- the sheet must be blind ------------------------------------------------


def test_a_sheet_carrying_the_arm_is_refused(tmp_path):
    """The sheet is a text file anyone can regenerate with different arguments.

    Checking it on load turns "the blinding was correct when written" into
    "the blinding is correct now".
    """
    path = write_sheet(tmp_path / "leaky.jsonl", [sheet_line(1, "TEST-001", "x", arm="D")])
    with pytest.raises(ScoringError, match="not blind"):
        load_sheet(path)


@pytest.mark.parametrize(
    "leak", ["arm", "prompt", "evidence", "wall_seconds", "generation", "scoring"]
)
def test_every_arm_identifying_field_is_refused(tmp_path, leak):
    path = write_sheet(tmp_path / "leaky.jsonl", [sheet_line(1, "TEST-001", "x", **{leak: 1})])
    with pytest.raises(ScoringError, match="not blind"):
        load_sheet(path)


def test_duplicate_item_numbers_are_refused(tmp_path):
    """Item numbers key the judgement log, so duplicates overwrite silently."""
    path = write_sheet(
        tmp_path / "dupes.jsonl",
        [sheet_line(1, "TEST-001", "a"), sheet_line(1, "TEST-002", "b")],
    )
    with pytest.raises(ScoringError, match="duplicate item number"):
        load_sheet(path)


def test_the_opaque_code_never_reaches_the_screen(sheet):
    """system_C is a stable label for one arm across the whole pass.

    Show it and a reviewer who notices that one code escalates more than the
    others has deanonymised every remaining item. This is the defeat amendment
    1.1.10 closed for the evidence block, closed the same way: by not emitting
    the thing.
    """
    items = load_sheet(sheet)
    for position, item in enumerate(items, start=1):
        rendered = render_item(item, position=position, total=len(items), scored=0)
        assert item.system not in rendered
        assert "system_" not in rendered
        assert item.answer in rendered
        assert item.question_id in rendered


def test_the_judgement_record_does_not_store_the_code_either(tmp_path, sheet):
    items = load_sheet(sheet)
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=items[0].item, question_id="TEST-001", score=2))
    written = json.loads(log.read_text(encoding="utf-8").strip())
    assert "system" not in written
    assert not any("system_" in str(v) for v in written.values())


# --- durability and resumption ----------------------------------------------


def test_a_judgement_is_on_disk_before_the_next_item_is_shown(tmp_path):
    """A pass over 272 answers is not one sitting."""
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=7, question_id="TEST-001", score=1))
    assert log.exists()
    assert json.loads(log.read_text(encoding="utf-8").strip())["score"] == 1


def test_scoring_resumes_where_an_interrupted_pass_stopped(sheet, tmp_path):
    items = load_sheet(sheet)
    log = tmp_path / "judgements.jsonl"
    for item in items[:2]:
        append_judgement(log, Judgement(item=item.item, question_id=item.question_id, score=2))

    reloaded = load_judgements(log)
    assert len(reloaded) == 2
    assert next_unscored(items, reloaded) == 3
    assert progress(items, reloaded)["remaining"] == 2
    assert progress(items, reloaded)["complete"] is False


def test_a_rescored_item_wins_but_the_original_is_not_erased(tmp_path):
    """A reviewer who revisits an item should be able to, and the record
    should show that they did rather than presenting the second score as
    the first."""
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=3, question_id="TEST-002", score=0))
    append_judgement(log, Judgement(item=3, question_id="TEST-002", score=2, revision=1))

    current = load_judgements(log)
    assert current[3].score == 2
    assert current[3].revision == 1
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_a_corrupt_final_line_is_named_rather_than_skipped(tmp_path):
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=1, question_id="TEST-001", score=2))
    with log.open("a", encoding="utf-8") as handle:
        handle.write('{"item": 2, "sco')
    with pytest.raises(ScoringError, match="append-only"):
        load_judgements(log)


def test_a_score_outside_the_three_point_scale_is_refused():
    with pytest.raises(ScoringError, match="three-point"):
        Judgement(item=1, question_id="TEST-001", score=3)


# --- the sheet a log was started against ------------------------------------


def test_a_rebuilt_sheet_invalidates_the_log(tmp_path, sheet):
    """Item numbers are positions in a shuffled file.

    Rebuild it from different runs and item 37 is a different answer while the
    log still claims it was scored.
    """
    session = tmp_path / "session.json"
    open_session(session, sheet, item_count=4)

    write_sheet(sheet, [sheet_line(1, "TEST-009", "something else")])
    with pytest.raises(ScoringError, match="has changed since scoring began"):
        open_session(session, sheet, item_count=1)


# --- input grammar ----------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2", (2, False, False, False)),
        ("0", (0, False, False, False)),
        ("1c", (1, True, False, False)),
        ("0a", (0, False, True, False)),
        ("2cu", (2, True, False, True)),
        (" 1 c a ", (1, True, True, False)),
        ("2N", (2, False, False, False)),
    ],
)
def test_input_grammar(text, expected):
    parsed = parse_score_input(text)
    assert (
        parsed.score,
        parsed.asserts_conflict,
        parsed.abstained,
        parsed.uncertain,
    ) == expected


def test_uncertain_always_prompts_for_a_note():
    assert parse_score_input("1u").wants_note is True
    assert parse_score_input("1n").wants_note is True
    assert parse_score_input("1").wants_note is False


@pytest.mark.parametrize("text", ["", "3", "x", "2z", "2cc", "c2", "22"])
def test_malformed_input_is_rejected_rather_than_guessed(text):
    """A swallowed stray character in a 272-item pass is a wrong score in the
    primary metric."""
    with pytest.raises(InputError):
        parse_score_input(text)


# --- the blinding is partial, and measured rather than assumed --------------


def test_the_served_abstention_template_is_caught_as_an_arm_signature(tmp_path):
    """The leak amendment 1.13 records.

    Only the verified arm serves ABSTENTION_TEXT, so its presence is a
    byte-exact arm signature. The build-time check that missed it looked for
    give-aways already known; this one looks for text the system serves
    verbatim, which is the property rather than a list of its past violations.
    """
    path = write_sheet(
        tmp_path / "leaky.jsonl",
        [
            sheet_line(1, "TEST-001", ABSTENTION_TEXT, "system_B"),
            sheet_line(2, "TEST-002", "a real answer", "system_B"),
            sheet_line(3, "TEST-003", "another answer", "system_A"),
        ],
    )
    audit = arm_signature_audit(load_sheet(path))
    assert audit["blind"] is False
    assert audit["findings"][0]["template"] == "ABSTENTION_TEXT"
    assert audit["findings"][0]["items"] == [1]


def test_the_audit_reports_the_exposure_not_the_trigger_count(tmp_path):
    """One template on one item unblinds every item sharing its code.

    Reporting 1 here rather than 2 would describe the leak as an order of
    magnitude smaller than it is, which is the mistake the original audit made
    by counting nothing at all.
    """
    path = write_sheet(
        tmp_path / "leaky.jsonl",
        [
            sheet_line(1, "TEST-001", ABSTENTION_TEXT, "system_B"),
            sheet_line(2, "TEST-002", "a real answer", "system_B"),
            sheet_line(3, "TEST-003", "another answer", "system_A"),
        ],
    )
    audit = arm_signature_audit(load_sheet(path))
    assert audit["findings"][0]["items_exposed"] == 2
    assert audit["items_exposed"] == 2


def test_a_clean_sheet_passes_the_audit(sheet):
    assert arm_signature_audit(load_sheet(sheet))["blind"] is True


def test_build_refuses_to_write_a_sheet_carrying_a_served_template(
    tmp_path, monkeypatch, capsys
):
    """The class fix. The existing sheet cannot be repaired; the next one can
    be prevented."""
    import score_answers

    target = tmp_path / "new_sheet.jsonl"

    def fake_write(runs, output, *, seed=42, question_set=None):
        write_sheet(Path(output), [sheet_line(1, "TEST-001", ABSTENTION_TEXT)])
        return {"D": "system_A"}

    monkeypatch.setattr(score_answers, "write_review_sheet", fake_write)
    monkeypatch.setattr(score_answers, "discover_test_runs", lambda root: [tmp_path])
    monkeypatch.setattr(score_answers, "read_run", lambda d: ({}, []))
    monkeypatch.setattr(score_answers, "load_question_set", lambda p: None)
    monkeypatch.setattr(
        score_answers,
        "load_evaluation_config",
        lambda: SimpleNamespace(path=lambda name: tmp_path / "question_set.json"),
    )

    code = main(["--sheet", str(target),
                 "--judgements", str(tmp_path / "j.jsonl"), "build"])
    assert code == 1
    assert "not blind" in capsys.readouterr().err


def test_the_identification_flag_is_recorded(tmp_path, sheet, monkeypatch):
    log = tmp_path / "judgements.jsonl"
    scripted_input(monkeypatch, ["2", "1i", "0", "2"])
    main(["--sheet", str(sheet), "--judgements", str(log),
          "score", "--session", str(tmp_path / "session.json")])

    judgements = load_judgements(log)
    assert judgements[2].arm_identified is True
    assert judgements[1].arm_identified is False


def test_a_line_written_before_the_flag_existed_is_not_asked_rather_than_no(tmp_path):
    """Some items were scored before the flag existed.

    Reading a missing field as False would fabricate answers to a question that
    was never put, and move those items into the denominator of the unblinding
    rate as evidence of blinding that was never gathered.
    """
    log = tmp_path / "judgements.jsonl"
    log.write_text(
        json.dumps({"item": 1, "question_id": "TEST-001", "score": 2}) + "\n",
        encoding="utf-8",
    )
    assert load_judgements(log)[1].arm_identified is None


def test_the_unblinding_rate_excludes_items_that_were_never_asked(sheet, tmp_path):
    items = load_sheet(sheet)
    log = tmp_path / "judgements.jsonl"
    log.write_text(
        json.dumps({"item": 1, "question_id": "TEST-001", "score": 2}) + "\n",
        encoding="utf-8",
    )
    append_judgement(log, Judgement(item=2, question_id="TEST-001", score=2,
                                    arm_identified=True))
    append_judgement(log, Judgement(item=3, question_id="TEST-002", score=2,
                                    arm_identified=False))

    state = progress(items, load_judgements(log))
    assert state["identification_not_asked"] == 1
    assert state["identification_asked"] == 2
    assert state["arm_identified"] == 1
    assert state["unblinding_rate"] == pytest.approx(0.5)


def test_the_identification_flag_does_not_force_a_note(monkeypatch):
    """Friction on an honesty flag suppresses the flag."""
    assert parse_score_input("2i").wants_note is False
    assert parse_score_input("2i").arm_identified is True


def test_identification_is_not_treated_as_a_scoring_disagreement(sheet, tmp_path):
    """It is a fact about the reviewer, not about the answer.

    Two identical answers where one was recognised and one was not is not a
    scoring inconsistency, and reporting it as one would bury the real ones.
    """
    items = load_sheet(sheet)
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=1, question_id="TEST-001", score=2,
                                    arm_identified=True))
    append_judgement(log, Judgement(item=2, question_id="TEST-001", score=2,
                                    arm_identified=False))
    assert consistency_report(items, load_judgements(log))["divergent"] == []


# --- the abstention re-pass -------------------------------------------------


def test_the_repass_uses_a_different_order(sheet):
    """A second pass in the same order drifts at the same places.

    The two would then agree wrongly and the agreement rate would report
    reliability the instrument does not have.
    """
    items = load_sheet(sheet)
    reordered = abstention_order(items, seed=7)
    assert [i.item for i in reordered] != [i.item for i in items]
    assert sorted(i.item for i in reordered) == sorted(i.item for i in items)


def test_the_repass_order_is_reproducible(sheet):
    items = load_sheet(sheet)
    assert abstention_order(items, seed=7) == abstention_order(items, seed=7)


def test_the_repass_screen_shows_no_rubric_and_no_first_pass_answer(sheet):
    """Showing the first pass would anchor the second and measure suggestion
    rather than reliability."""
    item = load_sheet(sheet)[0]
    rendered = render_abstention_item(item, position=1, total=4, decided=0)
    assert item.answer in rendered
    assert "RUBRIC" not in rendered
    assert "system_" not in rendered
    for criterion in item.scoring_criteria.values():
        assert criterion not in rendered


@pytest.mark.parametrize(
    "text,expected", [("y", (True, False)), ("n", (False, False)),
                      ("Y", (True, False)), ("yes", (True, False)),
                      ("no", (False, False)), ("y?", (True, True)),
                      ("n?", (False, True))],
)
def test_abstention_input_grammar(text, expected):
    assert parse_abstention_input(text) == expected


@pytest.mark.parametrize("text", ["", "2", "maybe", "yn", "?y"])
def test_malformed_abstention_input_is_rejected(text):
    with pytest.raises(InputError):
        parse_abstention_input(text)


def test_the_repass_is_recorded_to_its_own_log(tmp_path, sheet, monkeypatch):
    """The first pass is not edited, so both remain independently auditable."""
    judgements = tmp_path / "judgements.jsonl"
    repass = tmp_path / "abstention.jsonl"
    scripted_input(monkeypatch, ["2", "1", "0a", "2"])
    main(["--sheet", str(sheet), "--judgements", str(judgements),
          "score", "--session", str(tmp_path / "session.json")])
    before = judgements.read_text(encoding="utf-8")

    scripted_input(monkeypatch, ["y", "n", "y", "n"])
    main(["--sheet", str(sheet), "--judgements", str(judgements),
          "--abstention-log", str(repass),
          "abstention", "--session", str(tmp_path / "session.json")])

    assert judgements.read_text(encoding="utf-8") == before, "the first pass was edited"
    assert len(load_abstention(repass)) == 4


def test_the_repass_resumes_after_an_interruption(tmp_path, sheet, monkeypatch):
    repass = tmp_path / "abstention.jsonl"
    scripted_input(monkeypatch, ["y", "n", "q"])
    main(["--sheet", str(sheet), "--judgements", str(tmp_path / "j.jsonl"),
          "--abstention-log", str(repass),
          "abstention", "--session", str(tmp_path / "session.json")])
    assert len(load_abstention(repass)) == 2

    scripted_input(monkeypatch, ["y", "y"])
    main(["--sheet", str(sheet), "--judgements", str(tmp_path / "j.jsonl"),
          "--abstention-log", str(repass),
          "abstention", "--session", str(tmp_path / "session.json")])
    assert len(load_abstention(repass)) == 4


def test_agreement_measures_every_item_not_only_the_repeated_ones(tmp_path, sheet):
    """The first pass could only be checked where the sheet repeated an answer.

    The re-pass covers everything, so the disagreement rate is observed rather
    than extrapolated from the half that was visible.
    """
    judgements = tmp_path / "judgements.jsonl"
    repass = tmp_path / "abstention.jsonl"
    for number, abstained in ((1, True), (2, False), (3, False), (4, False)):
        append_judgement(judgements, Judgement(item=number, question_id="Q",
                                               score=2, abstained=abstained))
    for number, abstained in ((1, True), (2, False), (3, True), (4, False)):
        append_abstention(repass, AbstentionJudgement(item=number,
                                                      question_id="Q",
                                                      abstained=abstained))

    report = abstention_agreement(load_judgements(judgements), load_abstention(repass))
    assert report["compared"] == 4
    assert report["disagreed"] == 1
    assert report["agreement_rate"] == pytest.approx(0.75)
    assert report["missed_by_first_pass"] == 1
    assert report["missed_by_second_pass"] == 0


def test_unseal_refuses_until_the_repass_is_done_too(tmp_path, sheet, monkeypatch, capsys):
    """Unsealing between the passes would let the arm mapping reach the second
    one."""
    key = sheet.with_name(sheet.stem + "_key.json")
    key.write_text(json.dumps({"mapping": {"A": "system_C"}}), encoding="utf-8")
    judgements = tmp_path / "judgements.jsonl"
    repass = tmp_path / "abstention.jsonl"

    scripted_input(monkeypatch, ["2", "2", "0", "2"])
    main(["--sheet", str(sheet), "--judgements", str(judgements),
          "score", "--session", str(tmp_path / "session.json")])
    capsys.readouterr()

    code = main(["--sheet", str(sheet), "--judgements", str(judgements),
                 "--abstention-log", str(repass),
                 "unseal", "--i-have-finished-scoring"])
    assert code == 1
    captured = capsys.readouterr()
    assert "abstention re-pass" in captured.err
    assert "system_C" not in captured.out

    scripted_input(monkeypatch, ["y", "n", "y", "n"])
    main(["--sheet", str(sheet), "--judgements", str(judgements),
          "--abstention-log", str(repass),
          "abstention", "--session", str(tmp_path / "session.json")])
    capsys.readouterr()

    assert main(["--sheet", str(sheet), "--judgements", str(judgements),
                 "--abstention-log", str(repass),
                 "unseal", "--i-have-finished-scoring"]) == 0
    assert "system_C" in capsys.readouterr().out


def test_drift_report_tells_fatigue_apart_from_a_difference_of_criterion(tmp_path):
    """The whole reason the re-pass runs in a different order.

    Disagreements confined to the tail of one pass and scattered in the other's
    are that pass running out of attention. Scattered in both, they are a real
    difference of criterion, and no amount of care would have removed them.
    """
    records = [sheet_line(n, f"TEST-{n:03d}", f"answer {n}") for n in range(1, 21)]
    path = write_sheet(tmp_path / "review.jsonl", records)
    items = load_sheet(path)

    first, second = {}, {}
    for item in items:
        # The first pass stops marking after item 15: fatigue by position.
        first[item.item] = Judgement(item=item.item, question_id=item.question_id,
                                     score=2, abstained=item.item <= 15)
        second[item.item] = AbstentionJudgement(item=item.item,
                                                question_id=item.question_id,
                                                abstained=True)

    report = positional_drift_report(items, first, second, order_seed=11)
    block = report["second_pass_only"]
    assert block["n"] == 5
    assert block["first_pass_span"] == [16, 20]
    assert block["confined_to_first_pass_tail"] is True
    assert block["confined_to_second_pass_tail"] is False


def test_drift_report_counts_each_pass_against_itself(sheet, tmp_path):
    items = load_sheet(sheet)
    first, second = {}, {}
    for item in items:
        first[item.item] = Judgement(item=item.item, question_id=item.question_id,
                                     score=2, abstained=item.item == 1)
        second[item.item] = AbstentionJudgement(item=item.item,
                                                question_id=item.question_id,
                                                abstained=True)

    internal = positional_drift_report(items, first, second)["internal_consistency"]
    # Items 1 and 2 are the same answer to the same question; the first pass
    # marked one and not the other, the second marked both.
    assert internal["first_pass"]["divergent"] == 1
    assert internal["second_pass"]["divergent"] == 0


# --- consistency ------------------------------------------------------------


def test_identical_answers_scored_differently_are_reported(sheet, tmp_path):
    """Arm D replays Arm B's drafts, so a verification pass that changes
    nothing leaves two byte-identical answers in the pool. Independent scoring
    can put a 1 on one and a 2 on the other, and that noise lands on the B
    versus D contrast."""
    items = load_sheet(sheet)
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=1, question_id="TEST-001", score=2))
    append_judgement(log, Judgement(item=2, question_id="TEST-001", score=1))
    append_judgement(log, Judgement(item=3, question_id="TEST-002", score=2))
    append_judgement(log, Judgement(item=4, question_id="TEST-003", score=0))

    report = consistency_report(items, load_judgements(log))
    assert report["duplicate_groups"] == 1
    assert len(report["divergent"]) == 1
    assert report["divergent"][0]["items"] == [1, 2]
    assert report["consistent"] == 0


def test_the_repass_settles_divergences_the_first_pass_left(sheet, tmp_path):
    """Amendment 1.14.4 rule 1 makes the second pass the reported value.

    A report built on the first pass counts divergences the re-pass has already
    settled, which overstates the remaining inconsistency. Omitting the
    argument still reports the first pass against itself, which is the figure
    the amendment cites.
    """
    items = load_sheet(sheet)
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=1, question_id="TEST-001", score=2,
                                    abstained=True))
    append_judgement(log, Judgement(item=2, question_id="TEST-001", score=2,
                                    abstained=False))
    first = load_judgements(log)
    assert len(consistency_report(items, first)["divergent"]) == 1

    repass = tmp_path / "abstention.jsonl"
    for number in (1, 2):
        append_abstention(repass, AbstentionJudgement(item=number,
                                                      question_id="TEST-001",
                                                      abstained=True))
    settled = consistency_report(items, first, load_abstention(repass))
    assert settled["divergent"] == []
    assert settled["consistent"] == 1


def test_agreeing_duplicates_are_not_reported(sheet, tmp_path):
    items = load_sheet(sheet)
    log = tmp_path / "judgements.jsonl"
    for number in (1, 2):
        append_judgement(log, Judgement(item=number, question_id="TEST-001", score=2))
    report = consistency_report(items, load_judgements(log))
    assert report["divergent"] == []
    assert report["consistent"] == 1


def test_a_differing_flag_counts_as_a_divergence(sheet, tmp_path):
    items = load_sheet(sheet)
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=1, question_id="TEST-001", score=2))
    append_judgement(
        log, Judgement(item=2, question_id="TEST-001", score=2, asserts_conflict=True)
    )
    report = consistency_report(items, load_judgements(log))
    assert len(report["divergent"]) == 1


def test_identical_text_on_different_questions_is_not_a_duplicate(tmp_path):
    """The system-written abstention template is byte-identical across many
    different gaps. Those are judged against different rubrics and may
    legitimately differ, so keying on text alone would manufacture
    disagreements."""
    path = write_sheet(
        tmp_path / "review.jsonl",
        [
            sheet_line(1, "TEST-010", "The evidence does not cover this."),
            sheet_line(2, "TEST-011", "The evidence does not cover this."),
        ],
    )
    items = load_sheet(path)
    log = tmp_path / "judgements.jsonl"
    append_judgement(log, Judgement(item=1, question_id="TEST-010", score=2))
    append_judgement(log, Judgement(item=2, question_id="TEST-011", score=0))
    report = consistency_report(items, load_judgements(log))
    assert report["duplicate_groups"] == 0
    assert report["divergent"] == []


# --- the command line, end to end -------------------------------------------


@pytest.fixture
def watched_opens(monkeypatch):
    """Record every path the code under test opens."""
    opened: list[str] = []

    real_open = builtins.open
    real_path_open = pathlib.Path.open
    real_read_text = pathlib.Path.read_text
    real_read_bytes = pathlib.Path.read_bytes

    def note(target):
        opened.append(str(target))

    def traced_open(file, *args, **kwargs):
        note(file)
        return real_open(file, *args, **kwargs)

    def traced_path_open(self, *args, **kwargs):
        note(self)
        return real_path_open(self, *args, **kwargs)

    def traced_read_text(self, *args, **kwargs):
        note(self)
        return real_read_text(self, *args, **kwargs)

    def traced_read_bytes(self, *args, **kwargs):
        note(self)
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", traced_open)
    monkeypatch.setattr(pathlib.Path, "open", traced_path_open)
    monkeypatch.setattr(pathlib.Path, "read_text", traced_read_text)
    monkeypatch.setattr(pathlib.Path, "read_bytes", traced_read_bytes)
    return opened


def scripted_input(monkeypatch, answers):
    supplied = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(supplied)
        except StopIteration:  # pragma: no cover - a test that ran out of input
            raise EOFError

    monkeypatch.setattr(builtins, "input", fake_input)


def test_a_full_pass_never_opens_the_key(tmp_path, sheet, monkeypatch, watched_opens):
    key = sheet.with_name(sheet.stem + "_key.json")
    key.write_text(
        json.dumps({"warning": "Do not open", "seed": 42,
                    "mapping": {"A": "system_C", "B": "system_A",
                                "C": "system_D", "D": "system_B"}}),
        encoding="utf-8",
    )
    log = tmp_path / "judgements.jsonl"
    scripted_input(monkeypatch, ["2", "1", "0a", "2"])

    # Creating the key above is itself a file operation, and the tracer sees
    # it. Only what the scoring pass touches is the claim being tested.
    watched_opens.clear()

    assert main([
        "--sheet", str(sheet), "--judgements", str(log),
        "score", "--session", str(tmp_path / "session.json"),
    ]) == 0

    assert not any("_key.json" in path for path in watched_opens), (
        "the scoring pass touched the key file"
    )
    assert progress(load_sheet(sheet), load_judgements(log))["complete"] is True


def test_the_flags_are_recorded_through_the_command_line(tmp_path, sheet, monkeypatch):
    log = tmp_path / "judgements.jsonl"
    scripted_input(monkeypatch, ["2", "1c", "0a", "2u", "borderline"])
    main([
        "--sheet", str(sheet), "--judgements", str(log),
        "score", "--session", str(tmp_path / "session.json"),
    ])

    judgements = load_judgements(log)
    assert judgements[2].asserts_conflict is True
    assert judgements[3].abstained is True
    assert judgements[4].uncertain is True
    assert judgements[4].note == "borderline"
    assert judgements[1].asserts_conflict is False


def test_quitting_keeps_everything_scored_so_far(tmp_path, sheet, monkeypatch):
    log = tmp_path / "judgements.jsonl"
    scripted_input(monkeypatch, ["2", "1", "q"])
    main([
        "--sheet", str(sheet), "--judgements", str(log),
        "score", "--session", str(tmp_path / "session.json"),
    ])
    assert len(load_judgements(log)) == 2

    scripted_input(monkeypatch, ["0", "2"])
    main([
        "--sheet", str(sheet), "--judgements", str(log),
        "score", "--session", str(tmp_path / "session.json"),
    ])
    assert progress(load_sheet(sheet), load_judgements(log))["complete"] is True


def test_unseal_refuses_while_items_are_unscored(tmp_path, sheet, monkeypatch, capsys):
    key = sheet.with_name(sheet.stem + "_key.json")
    key.write_text(json.dumps({"mapping": {"A": "system_C"}}), encoding="utf-8")
    log = tmp_path / "judgements.jsonl"
    scripted_input(monkeypatch, ["2", "q"])
    main(["--sheet", str(sheet), "--judgements", str(log),
          "score", "--session", str(tmp_path / "session.json")])

    code = main(["--sheet", str(sheet), "--judgements", str(log),
                 "unseal", "--i-have-finished-scoring"])
    assert code == 1
    assert "Refusing to unseal" in capsys.readouterr().err
    assert "system_C" not in capsys.readouterr().out


def test_unseal_refuses_without_the_awkward_flag(tmp_path, sheet, monkeypatch, capsys):
    key = sheet.with_name(sheet.stem + "_key.json")
    key.write_text(json.dumps({"mapping": {"A": "system_C"}}), encoding="utf-8")
    log = tmp_path / "judgements.jsonl"
    scripted_input(monkeypatch, ["2", "1", "0", "2"])
    main(["--sheet", str(sheet), "--judgements", str(log),
          "score", "--session", str(tmp_path / "session.json")])
    capsys.readouterr()

    assert main(["--sheet", str(sheet), "--judgements", str(log), "unseal"]) == 1
    captured = capsys.readouterr()
    assert "system_C" not in captured.out


def test_unseal_prints_the_mapping_once_the_pass_is_complete(
    tmp_path, sheet, monkeypatch, capsys
):
    key = sheet.with_name(sheet.stem + "_key.json")
    key.write_text(json.dumps({"mapping": {"A": "system_C", "B": "system_A"}}),
                   encoding="utf-8")
    log = tmp_path / "judgements.jsonl"
    repass = tmp_path / "abstention.jsonl"
    scripted_input(monkeypatch, ["2", "2", "0", "2"])
    main(["--sheet", str(sheet), "--judgements", str(log),
          "score", "--session", str(tmp_path / "session.json")])
    scripted_input(monkeypatch, ["y", "n", "y", "n"])
    main(["--sheet", str(sheet), "--judgements", str(log),
          "--abstention-log", str(repass),
          "abstention", "--session", str(tmp_path / "session.json")])
    capsys.readouterr()

    assert main(["--sheet", str(sheet), "--judgements", str(log),
                 "--abstention-log", str(repass),
                 "unseal", "--i-have-finished-scoring"]) == 0
    assert "system_C" in capsys.readouterr().out


def test_build_refuses_to_renumber_a_sheet_that_has_been_scored(
    tmp_path, sheet, monkeypatch, capsys
):
    log = tmp_path / "judgements.jsonl"
    scripted_input(monkeypatch, ["2", "q"])
    main(["--sheet", str(sheet), "--judgements", str(log),
          "score", "--session", str(tmp_path / "session.json")])

    assert main(["--sheet", str(sheet), "--judgements", str(log), "build"]) == 1
    assert "Refusing to rebuild" in capsys.readouterr().err


# --- integration with the pre-registered blinding function ------------------


def test_a_real_review_sheet_loads_and_carries_a_rubric(tmp_path, kb, index, config):
    """The sheet the pre-registration names must be scoreable by this tool.

    write_review_sheet is the blinding procedure of record. If its output did
    not load here, the tool would be scoring something else.
    """
    client = MockClient(config)
    retriever = Retriever(index, client, config)
    generator = Generator(client, config)
    question_set = load_question_set(load_evaluation_config().path("question_set"))
    questions = [q for q in question_set if q.split == "dev"][:3]

    runs = []
    for name, fmt in (("A", EvidenceFormat.PLAIN), ("D", EvidenceFormat.WITH_STATUS)):
        writer = RunWriter(
            ArmDefinition(
                arm=name, description="test", retrieval_mode="all",
                evidence_format=fmt.value, verification=False,
                generation_model="mock-generate",
            ),
            split="dev", config=config, kb=kb, index=index, root=tmp_path / name,
        )
        for question in questions:
            retrieval = retriever.retrieve(question.text, min_similarity=0.0)
            answer = generator.answer(question.text, retrieval, evidence_format=fmt)
            writer.record(
                question_id=question.question_id,
                question=question.text,
                answer=answer,
                group_id=question.group_id,
                category=question.category,
            )
        runs.append(writer.finish())

    sheet = tmp_path / "review.jsonl"
    write_review_sheet(runs, sheet, question_set=question_set)

    items = load_sheet(sheet)
    assert len(items) == 2 * len(questions)
    assert all(item.scoring_criteria for item in items), "every item needs its rubric"
    assert all(set(item.scoring_criteria) == {"0", "1", "2"} for item in items)

    rendered = "\n".join(
        render_item(item, position=n, total=len(items), scored=0)
        for n, item in enumerate(items, start=1)
    )
    for leak in ("system_", "Arm ", "llama", "qwen", "SUPERSEDED"):
        assert leak not in rendered, f"the scoring screen leaks {leak!r}"
