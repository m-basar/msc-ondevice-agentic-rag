"""The dashboard, and the boundary it must not cross.

Pre-registration amendment 1.27 declares that the demonstrator contributes no
evidence to any hypothesis and never writes into a frozen run directory.
Amendment 1.16.1 records what happened the last time this document declared a
boundary and did not enforce one, so most of what follows is enforcement rather
than feature testing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from .sealing import seal

from sme_assistant.demo import load_replay_library
from sme_assistant.demo.replay import ReplayUnavailable
from sme_assistant.demo.render import arm_card, live_page, replay_page
from sme_assistant.demo.server import build_server
from sme_assistant.evaluation.analysis import (
    FROZEN_QUALITY_RUNS,
    quality_run_directories,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def frozen_available() -> bool:
    return all((RUNS / name / "answers.jsonl").exists()
               for name in FROZEN_QUALITY_RUNS)


needs_frozen = pytest.mark.skipif(
    not frozen_available(), reason="frozen quality runs are not present")


@pytest.fixture(scope="module")
def library():
    if not frozen_available():
        pytest.skip("frozen quality runs are not present")
    return load_replay_library(RUNS, FROZEN_QUALITY_RUNS)


# --- the boundary ------------------------------------------------------------


def test_a_demonstration_run_cannot_enter_the_quality_analysis(tmp_path):
    """The guarantee must not rest on the dashboard writing elsewhere.

    Amendment 1.27.3 rule 3. Even a demonstration directory placed directly
    into results/runs must be refused, because the closed list is the boundary
    and a naming convention is not.
    """
    for name in FROZEN_QUALITY_RUNS:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"split": "test", "arm": {"arm": name.split("_")[1]}}),
            encoding="utf-8")
    intruder = tmp_path / "20260820_120000_D_test_demo"
    intruder.mkdir()
    (intruder / "manifest.json").write_text(
        json.dumps({"split": "test", "purpose": "demonstration",
                    "arm": {"arm": "D"}}), encoding="utf-8")

    admitted = {p.name for p in quality_run_directories(tmp_path)}
    assert intruder.name not in admitted
    assert admitted == set(FROZEN_QUALITY_RUNS)


def test_the_demo_package_does_not_import_the_experiment_runner():
    """The dependency runs one way.

    A demonstrator that imports the runner puts itself on the experiment's
    path; a runner that imports the demonstrator would be worse still. Neither
    happens, and this asserts it over the import graph rather than by reading.
    """
    banned = {"run_arms", "analyse_results", "analyse_performance",
              "answer_scoring", "manual_scoring"}
    for module in (ROOT / "src" / "sme_assistant" / "demo").glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            for name in names:
                tail = name.rsplit(".", 1)[-1]
                assert tail not in banned, f"{module.name} imports {name}"


def test_the_demo_package_has_no_write_path():
    """Replay is read-only by construction rather than by rule.

    A module with no call that opens a file for writing cannot write to a
    frozen run whatever its caller does.
    """
    forbidden_attrs = {"write_text", "write_bytes", "mkdir", "unlink", "rmdir"}
    for module in (ROOT / "src" / "sme_assistant" / "demo").glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs, (
                    f"{module.name} calls {node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "open":
                    args = [a for a in node.args[1:]]
                    for arg in args:
                        if isinstance(arg, ast.Constant) and "w" in str(arg.value):
                            pytest.fail(f"{module.name} opens a file for writing")


@needs_frozen
def test_a_replay_session_leaves_every_frozen_file_untouched(library):
    """Amendment 1.27.3 rule 4, measured rather than assumed."""
    def digest() -> dict[str, str]:
        out = {}
        for name in FROZEN_QUALITY_RUNS:
            for filename in ("answers.jsonl", "manifest.json", "summary.json"):
                path = RUNS / name / filename
                if path.exists():
                    out[f"{name}/{filename}"] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
        return out

    before = digest()
    reloaded = load_replay_library(RUNS, FROZEN_QUALITY_RUNS)
    for question in reloaded.questions[:5]:
        replay_page(reloaded, question, [q.question_id for q in reloaded.questions])
    assert digest() == before


# --- replay reads what it should --------------------------------------------


@needs_frozen
def test_replay_joins_all_four_arms_over_the_test_split(library):
    assert library.provenance["arms"] == ["A", "B", "C", "D"]
    assert len(library.questions) == 68
    assert all(set(q.by_arm) == {"A", "B", "C", "D"} for q in library.questions)
    assert library.provenance["corpus_consistent"] is True


def test_a_missing_run_raises_rather_than_returning_a_partial_library(tmp_path):
    """Three arms presented as four is the failure this refuses.

    An empty or short library renders as a system with less to say, when what
    has happened is that it cannot find its data.
    """
    with pytest.raises(ReplayUnavailable, match="missing"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_a_performance_run(tmp_path):
    """A performance run is on the test split and carries an arm, so nothing
    but the purpose field distinguishes it."""
    _four_stubs(tmp_path)
    directory = tmp_path / FROZEN_QUALITY_RUNS[3]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["purpose"] = "performance"
    (directory / "manifest.json").write_text(json.dumps(manifest),
                                             encoding="utf-8")
    seal(directory)
    with pytest.raises(ReplayUnavailable, match="performance"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_a_run_that_is_not_on_the_closed_list(tmp_path):
    """Amendment 1.31.2. A directory that is not one of the four frozen runs is
    refused by name before anything inside it is read."""
    directory = tmp_path / "perf"
    directory.mkdir()
    (directory / "answers.jsonl").write_text("", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps({"split": "test", "purpose": "performance",
                    "arm": {"arm": "D"}}),
        encoding="utf-8")
    with pytest.raises(ReplayUnavailable, match="not one of the four"):
        load_replay_library(tmp_path, ["perf"])


@needs_frozen
def test_the_supersession_case_shows_the_arms_diverging(library):
    """The demonstration case, pinned so a change to the reader is visible.

    Arm A cites the withdrawn sick-pay policy on this question and the other
    three do not. If that stops being true, either the reader is broken or the
    frozen data has moved, and both need to be noticed.
    """
    question = library.by_id("CONF-02-Q1")
    assert question is not None
    assert question.by_arm["A"].cited_superseded
    assert not question.by_arm["B"].cited_superseded
    assert not question.by_arm["C"].cited_superseded


@needs_frozen
def test_arms_without_a_verifier_report_no_confidence(library):
    """Absent by design, not missing. The interface has to be able to tell."""
    question = library.questions[0]
    for arm in ("A", "B", "C"):
        assert question.by_arm[arm].confidence is None
        assert question.by_arm[arm].has_verification is False
    assert question.by_arm["D"].has_verification is True


# --- rendering ---------------------------------------------------------------


@needs_frozen
def test_the_replay_page_names_its_mode_and_its_provenance(library):
    question = library.by_id("CONF-02-Q1")
    html = replay_page(library, question, [q.question_id for q in library.questions])
    assert "Frozen experimental replay" in html
    assert "Live demonstration" not in html
    assert "FROZEN_QUALITY_RUNS" in html
    assert library.provenance["runs"]["A"]["corpus_sha256"][:12] in html


def test_the_live_page_names_its_mode_and_never_claims_evidence():
    html = live_page(None, None, None, {"ready": False, "detail": "no models"})
    assert "Live demonstration, not part of the reported evaluation" in html
    assert "Frozen experimental replay" not in html
    assert "Only Arm D runs live" in html


def test_a_typed_question_is_escaped_before_it_reaches_the_page():
    """The question comes from a URL. It is rendered, so it is escaped."""
    html = live_page("<script>alert(1)</script>", None, None,
                     {"ready": True, "generation": "g", "verification": "v"})
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


@needs_frozen
def test_confidence_is_labelled_as_rule_based_wherever_it_appears(library):
    """Chapter 3 says the mechanism is a declared mapping, not a calibrated
    score. The interface must not quietly imply otherwise."""
    question = library.by_id("CONF-02-Q1")
    html = arm_card(question.by_arm["D"])
    assert "rule-based" in html
    assert "calibrated" not in html.lower()


# --- the server --------------------------------------------------------------


@needs_frozen
def test_replay_serves_on_a_machine_with_no_models():
    """The failure mode this is built against: a demonstration on a laptop
    where Ollama is not running, or a marker opening the repository cold."""
    server = build_server(ROOT, host="127.0.0.1", port=8791)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:8791/replay?q=CONF-02-Q1", timeout=20) as response:
            assert response.status == 200
            body = response.read().decode("utf-8")
        assert "Frozen experimental replay" in body
        assert "Arm A" in body and "Arm D" in body
    finally:
        server.shutdown()
        server.server_close()


# --- amendments 1.28 and 1.30: replay fails closed -------------------------
#
# Amendment 1.28 claimed replay "now fails closed" when only three of these
# properties were checked. The rest were computed, displayed on the page and
# never compared. What follows is one test per refusal, because a guarantee
# with no failing case behind it is the thing 1.16.1 records.

STUB_HASHES = {
    "corpus_sha256": "c0",
    "chunk_set_sha256": "k0",
    "config_sha256": "g0",
    "question_set_sha256": "q0",
    "registry_sha256": "r0",
    "index_file_sha256": "i0",
}


def _stub_run(directory: Path, arm: str, question_ids, *, run_id=None,
              split="test", purpose=None, hashes=None, declared=None,
              record_arm=None, records=None, extra=None, seal_it=True):
    """A stub run that passes every check, so that a test can remove one.

    Written valid by default and broken one property at a time. A helper that
    produced something already refused for another reason could not show which
    refusal a test was exercising, which is how the first version of this file
    came to assert a corpus-mismatch message it never reached.
    """
    directory.mkdir(parents=True, exist_ok=True)
    provenance = dict(hashes or STUB_HASHES)
    provenance["question_set_metadata"] = {
        "summary": {"by_split": {"test": len(question_ids)
                                 if declared is None else declared}}}
    (directory / "manifest.json").write_text(json.dumps({
        "run_id": directory.name if run_id is None else run_id,
        "split": split,
        **({"purpose": purpose} if purpose is not None else {}),
        "arm": {"arm": arm, "description": f"stub {arm}"},
        "provenance": provenance,
    }), encoding="utf-8")
    lines = []
    for entry in (records if records is not None else question_ids):
        record = ({"question_id": entry, "question": entry,
                   "arm": arm if record_arm is None else record_arm,
                   "category": "conflict", "family_id": None, "answer": "a"}
                  if isinstance(entry, str) else dict(entry))
        record.update(extra or {})
        lines.append(json.dumps(record))
    (directory / "answers.jsonl").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    # Amendment 1.31.2: replay authenticates content before it checks anything
    # else, so a stub has to be sealed or every test below would exercise the
    # digest rather than the property it names. conftest restores the table.
    if seal_it:
        seal(directory)


def _four_stubs(tmp_path: Path, **overrides) -> None:
    """Four valid stub runs, one per arm, one question each."""
    for arm, name in zip("ABCD", FROZEN_QUALITY_RUNS):
        _stub_run(tmp_path / name, arm, ["Q1"], **overrides.get(arm, {}))


def test_the_stub_runs_are_accepted_when_nothing_is_wrong_with_them(tmp_path):
    """The control. Every refusal below is a departure from this."""
    _four_stubs(tmp_path)
    library = load_replay_library(tmp_path, FROZEN_QUALITY_RUNS,
                                  expected_questions=1)
    assert [q.question_id for q in library.questions] == ["Q1"]


def test_replay_refuses_a_question_an_arm_did_not_answer(tmp_path):
    """A missing column is a comparison that was never made.

    Previously the library would join whatever it found and render three cards
    where there should be four, which reads as a system with nothing to say
    rather than as an incomplete input.
    """
    for arm, name in zip("ABCD", FROZEN_QUALITY_RUNS):
        ids = ["Q1", "Q2"] if arm != "C" else ["Q1"]
        _stub_run(tmp_path / name, arm, ids, declared=2)
    with pytest.raises(ReplayUnavailable, match="not answered by every arm"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


@pytest.mark.parametrize("field", sorted(STUB_HASHES))
def test_replay_refuses_runs_that_disagree_on_any_provenance_hash(tmp_path, field):
    """Amendment 1.28 enforced three of the six. The other three, including the
    question set and the conflict registry, were displayed and not compared."""
    broken = dict(STUB_HASHES, **{field: "DIFFERENT"})
    _four_stubs(tmp_path, D={"hashes": broken})
    with pytest.raises(ReplayUnavailable, match=f"disagree on {field}"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


@pytest.mark.parametrize("field", sorted(STUB_HASHES))
def test_replay_refuses_a_run_that_records_no_provenance_hash(tmp_path, field):
    """A field that is absent cannot be compared, and treating absence as
    agreement is how four runs over three corpora would pass."""
    partial = {k: v for k, v in STUB_HASHES.items() if k != field}
    _four_stubs(tmp_path, D={"hashes": partial})
    with pytest.raises(ReplayUnavailable, match=f"records no {field}"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_a_run_on_the_development_split(tmp_path):
    """The dev split answers different questions. A grid populated from it
    would show material the study never reported, and look identical."""
    _four_stubs(tmp_path, D={"split": "dev"})
    with pytest.raises(ReplayUnavailable, match="declares split='dev'"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


@pytest.mark.parametrize("purpose", ["performance", "demonstration"])
def test_replay_refuses_a_run_by_purpose(tmp_path, purpose):
    """A demonstration directory is whatever the dashboard was pointed at, and
    amendment 1.27.3 rule 3 forbids it entering an experimental column."""
    _four_stubs(tmp_path, D={"purpose": purpose})
    with pytest.raises(ReplayUnavailable, match=purpose):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_a_renamed_run_directory(tmp_path):
    """Replay identifies runs by name, so a directory renamed onto the closed
    list must not be trusted over the identifier it was written with."""
    _four_stubs(tmp_path, D={"run_id": "20260814_055018_D_test_SOMETHING_ELSE"})
    with pytest.raises(ReplayUnavailable, match="has been renamed"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_two_runs_declaring_the_same_arm(tmp_path):
    """Two columns of one arm presented as two arms is not a comparison. The
    arm-coverage check alone does not catch it: it would report arm D missing
    and say nothing about the duplicate."""
    for arm, name in zip("ABCD", FROZEN_QUALITY_RUNS):
        _stub_run(tmp_path / name, "D" if arm == "C" else arm, ["Q1"])
    with pytest.raises(ReplayUnavailable, match="both declare arm D"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_an_answers_file_from_another_arm(tmp_path):
    """The record carries its own arm. A file copied between run directories
    keeps it, and the manifest would not notice."""
    _four_stubs(tmp_path, D={"record_arm": "B"})
    with pytest.raises(ReplayUnavailable, match="answered by arm 'B'"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_a_question_answered_twice_in_one_run(tmp_path):
    """Which of the two the grid would show depends on read order."""
    _four_stubs(tmp_path, D={"records": ["Q1", "Q1"], "declared": 1})
    with pytest.raises(ReplayUnavailable, match="more than once"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


@pytest.mark.parametrize("field,value", [("question", "a different prompt"),
                                         ("category", "factual"),
                                         ("family_id", "F-99")])
def test_replay_refuses_arms_that_were_not_asked_the_same_question(
        tmp_path, field, value):
    """Four columns answering four different prompts is the worst available
    failure, because it looks entirely normal."""
    record = {"question_id": "Q1", "question": "Q1", "arm": "D",
              "category": "conflict", "family_id": None, "answer": "a",
              field: value}
    _four_stubs(tmp_path, D={"records": [record]})
    with pytest.raises(ReplayUnavailable, match="were not asked the same"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_a_truncated_answers_file(tmp_path):
    """The manifest states the size of the split before the run begins, so a
    run that stopped early disagrees with its own record of itself."""
    for arm, name in zip("ABCD", FROZEN_QUALITY_RUNS):
        _stub_run(tmp_path / name, arm, ["Q1"], declared=68)
    with pytest.raises(ReplayUnavailable, match="declare 68 on the test split"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_runs_that_disagree_on_the_size_of_the_split(tmp_path):
    _four_stubs(tmp_path, D={"declared": 2})
    with pytest.raises(ReplayUnavailable, match="disagree on the size"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_a_run_that_states_no_size_for_the_split(tmp_path):
    """Absence is not agreement. A manifest with no count cannot show that the
    answers file is complete."""
    _four_stubs(tmp_path)
    directory = tmp_path / FROZEN_QUALITY_RUNS[3]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["provenance"].pop("question_set_metadata")
    (directory / "manifest.json").write_text(json.dumps(manifest),
                                             encoding="utf-8")
    seal(directory)
    with pytest.raises(ReplayUnavailable, match="no test-split question count"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_a_split_size_the_caller_did_not_expect(tmp_path):
    """Two independent statements of the same number. The server states 68; a
    set of runs internally consistent at some other size still fails."""
    _four_stubs(tmp_path)
    with pytest.raises(ReplayUnavailable, match="caller expects 68"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS, expected_questions=68)


@needs_frozen
def test_the_frozen_runs_satisfy_every_check_at_the_reported_size():
    """The four committed runs pass the whole of the above at 68 questions."""
    library = load_replay_library(RUNS, FROZEN_QUALITY_RUNS,
                                  expected_questions=68)
    assert len(library.questions) == 68
    assert library.provenance["declared_test_questions"] == 68
    assert sorted(library.provenance["checked"]) == sorted(STUB_HASHES)
    assert sorted(library.provenance["arms"]) == ["A", "B", "C", "D"]


@needs_frozen
def test_the_claim_audit_shows_contradicting_evidence_too(library):
    """Showing only the supporting column hides half of what the verifier
    said, and on this very question the contradicting side is the interesting
    half."""
    html = arm_card(library.by_id("CONF-02-Q1").by_arm["D"], expanded=True)
    assert "Contradicted by" in html
    assert "Supported by" in html


@needs_frozen
def test_the_claim_audit_is_labelled_as_model_output_not_a_key(library):
    """In the frozen run the verifier marked a correct claim contradicted and
    endorsed a withdrawn document's claim on the same question. An interface
    that presents that as an adjudication misleads the person reading it."""
    html = arm_card(library.by_id("CONF-02-Q1").by_arm["D"], expanded=True)
    assert "Recorded verifier output, not ground truth" in html
    assert "were not checked against the answer key" in html


# --- amendment 1.30: live mode is the frozen Arm D, not something like it ----


@needs_frozen
def test_the_live_arm_matches_the_frozen_arm_d_manifest():
    """What is demonstrated must be what was measured.

    The live constants are restated in the demo package rather than imported
    from the runner, so that the demonstrator does not sit on the experiment's
    path. That decision has a cost: the two can drift silently, and a
    demonstration of current-only retrieval labelled Arm D would be a
    misrepresentation of the result rather than a bug. This compares them.

    The models are compared through config.json, which is what both the frozen
    run and live mode actually read.
    """
    from sme_assistant.common.config import load_config
    from sme_assistant.demo import live as live_module

    manifest = json.loads(
        (RUNS / "20260814_055018_D_test" / "manifest.json").read_text(
            encoding="utf-8"))
    arm = manifest["arm"]
    assert live_module.LIVE_ARM == arm["arm"]
    assert live_module.LIVE_RETRIEVAL.value == arm["retrieval_mode"]
    assert live_module.LIVE_EVIDENCE.value == arm["evidence_format"]
    assert arm["verification"] is True

    config = load_config(ROOT / "config.json")
    assert config.require("llm.generation_model") == arm["generation_model"]
    assert config.get("llm.verification_model") == arm["verification_model"]


@needs_frozen
def test_live_mode_scores_nothing():
    """Amendment 1.27. Nothing live mode produces may be mistaken later for a
    measured result, which is enforced by there being no scorer in the import
    graph rather than by the output not being saved."""
    tree = ast.parse((ROOT / "src" / "sme_assistant" / "demo" / "live.py")
                     .read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
    assert not {n for n in names if "scor" in n.rsplit(".", 1)[-1].lower()}


# --- amendment 1.30: the POST rule, enforced at the server -------------------


@needs_frozen
def test_a_get_to_live_neither_answers_nor_reflects_the_question():
    """The defect amendment 1.28 introduced while claiming to fix it.

    1.28 recorded that live questions had moved to POST. What had changed was
    the form's method attribute. A question pasted into the address bar was
    still answered, and still passed through the browser history and every log
    between the address bar and this handler, which is the whole of what the
    rule was for.

    The private string must appear nowhere in the response, so a form that
    helpfully repopulated its input would fail here too.
    """
    private = "PRIVATE-DISCIPLINARY-CASE-42"
    calls = []

    server = build_server(ROOT, host="127.0.0.1", port=8792)
    handler = server.RequestHandlerClass
    handler.state.live_error = None
    handler.state._live = _RefusingAssistant(calls)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:8792/live?q={private}", timeout=20) as response:
            assert response.status == 200
            body = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()

    assert calls == [], f"the assistant was invoked by a GET: {calls}"
    assert private not in body
    assert "method='post'" in body


class _RefusingAssistant:
    """Records any attempt to answer. Never reached on the passing path."""

    def __init__(self, calls: list) -> None:
        self.calls = calls

    def model_status(self) -> dict:
        return {"ready": True, "generation": "g", "verification": "v",
                "base_url": "http://localhost:11434", "detail": ""}

    def frozen_arm_d_agreement(self, manifest) -> dict:
        """Amendment 1.31.3. The live page states whether this pipeline is
        configured as the frozen Arm D run was, so a stand-in has to answer
        it. Reporting agreement is not answering a question."""
        return {"matches": True, "differs": [], "fields": {}}

    def answer(self, question: str):
        self.calls.append(question)
        raise AssertionError("a GET reached the assistant")


def test_the_get_handler_passes_no_query_to_the_live_view():
    """Read over the source as well as over the wire.

    The wire test above depends on the live side reporting itself ready. This
    one holds whatever the machine running it has installed.
    """
    source = (ROOT / "src" / "sme_assistant" / "demo" / "server.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    do_get = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "do_GET")
    live_calls = [n for n in ast.walk(do_get)
                  if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "_live"]
    assert len(live_calls) == 1
    call = live_calls[0]
    assert isinstance(call.args[0], ast.Dict) and not call.args[0].keys, (
        "do_GET passes a query to the live view")
    assert any(kw.arg == "executable" and kw.value.value is False
               for kw in call.keywords), "do_GET does not disable execution"


def test_a_live_question_is_not_placed_in_the_url():
    """A question about someone's sick pay does not belong in a browser history
    or a proxy log. Replay stays on GET: its parameter is an identifier from a
    fixed list, and a shareable link to a comparison is useful."""
    html = live_page(None, None, None, {"ready": True, "generation": "g",
                                        "verification": "v"})
    assert "method='post'" in html
    assert "method='get'" not in html


@needs_frozen
def test_the_timing_panel_separates_the_draft_from_the_verifier(library):
    """A single end-to-end total hides the entire latency finding."""
    html = arm_card(library.by_id("CONF-02-Q1").by_arm["D"], expanded=True)
    for label in ("Draft, total", "Verification, total", "Verifier, prompt",
                  "Draft tokens", "Verifier tokens"):
        assert label in html, label


PI_RUN = RUNS / "20260815_040341_D_test_perf_pi5"

needs_pi = pytest.mark.skipif(
    not (PI_RUN / "answers.jsonl").exists(),
    reason="the Pi 5 performance run is not present")


@needs_pi
def test_thermal_telemetry_is_reported_by_stage_not_merged():
    """Amendment 1.30. One row called "CPU temperature" took the draft's
    reading, or the verifier's if the draft had none, and printed it as though
    it described the machine. On this record the draft ran at 84.8 degrees and
    the verifier, arriving second onto an already hot core, at 88.1. The
    difference is the thermal finding, and merging the rows deleted it.
    """
    from sme_assistant.demo.replay import _arm_answer

    record = json.loads(
        next(line for line in (PI_RUN / "answers.jsonl")
             .read_text(encoding="utf-8").splitlines() if line.strip()))
    html = arm_card(_arm_answer(record), expanded=True)
    for label in ("Draft CPU temperature", "Verifier CPU temperature",
                  "Draft throttled", "Verifier throttled"):
        assert label in html, label
    assert "<th>CPU temperature</th>" not in html
    assert "<th>Throttled</th>" not in html
    draft = record["generation"]["cpu_temp_c"]
    verifier = record["verification_generation"]["cpu_temp_c"]
    assert draft != verifier, "this record no longer distinguishes the stages"
    assert f"{draft:.1f} &deg;C" in html and f"{verifier:.1f} &deg;C" in html


@needs_frozen
def test_absent_thermal_telemetry_is_shown_as_absent_not_as_no(library):
    """The laptop runs report neither field. A row reading "no" would be a
    measurement that was never taken, which is the reporting failure this
    project spends most of its amendments on."""
    html = arm_card(library.by_id("CONF-02-Q1").by_arm["D"], expanded=True)
    assert "not reported on this host" in html
    assert "throttled</th><td class='num'>no<" not in html
