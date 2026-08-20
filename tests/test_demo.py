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
    directory = tmp_path / "perf"
    directory.mkdir()
    (directory / "answers.jsonl").write_text("", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps({"purpose": "performance", "arm": {"arm": "D"}}),
        encoding="utf-8")
    with pytest.raises(ReplayUnavailable, match="performance"):
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


# --- amendment 1.28 ----------------------------------------------------------


def _stub_run(directory: Path, arm: str, question_ids, *, corpus="c0", extra=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({
        "split": "test",
        "arm": {"arm": arm, "description": f"stub {arm}"},
        "provenance": {"corpus_sha256": corpus, "chunk_set_sha256": "k0",
                       "config_sha256": "g0"},
    }), encoding="utf-8")
    lines = []
    for qid in question_ids:
        record = {"question_id": qid, "question": qid, "arm": arm,
                  "category": "conflict", "answer": "a"}
        record.update(extra or {})
        lines.append(json.dumps(record))
    (directory / "answers.jsonl").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")


def test_replay_refuses_a_question_an_arm_did_not_answer(tmp_path):
    """A missing column is a comparison that was never made.

    Previously the library would join whatever it found and render three cards
    where there should be four, which reads as a system with nothing to say
    rather than as an incomplete input.
    """
    for arm, name in zip("ABCD", FROZEN_QUALITY_RUNS):
        ids = ["Q1", "Q2"] if arm != "C" else ["Q1"]
        _stub_run(tmp_path / name, arm, ids)
    with pytest.raises(ReplayUnavailable, match="not answered by every arm"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


def test_replay_refuses_runs_built_over_different_corpora(tmp_path):
    """Amendment 1.28. The hashes were computed and displayed but never
    enforced, which is the defect amendment 1.16.1 records in another place."""
    for arm, name in zip("ABCD", FROZEN_QUALITY_RUNS):
        _stub_run(tmp_path / name, arm, ["Q1"],
                  corpus="c0" if arm != "D" else "DIFFERENT")
    with pytest.raises(ReplayUnavailable, match="disagree on corpus_sha256"):
        load_replay_library(tmp_path, FROZEN_QUALITY_RUNS)


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
