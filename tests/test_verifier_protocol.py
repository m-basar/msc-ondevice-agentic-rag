"""The diagnostic protocol's two load-bearing mechanisms.

Neither is about model behaviour. Both are about whether the protocol can be
believed: the guard that stops the decision rules being written after the
numbers, and the confound check that decides whether the two evidence
conditions are comparable at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verifier_protocol import (  # noqa: E402
    CONDITIONS,
    MODELS,
    PROTOCOL,
    pair_is_present,
    require_committed_protocol,
)


# --- the confound check ------------------------------------------------------


def test_both_sides_of_the_dispute_must_be_present():
    anchors = ["HR-03#001", "HR-13#001"]
    assert pair_is_present(anchors, {"HR-03#001", "HR-13#001", "HR-01#002"})


def test_one_side_is_not_a_pair():
    """The case the check exists for.

    A verifier shown only the current rate has no disagreement in front of it.
    Reading its silence as a reasoning failure would attribute a retrieval
    problem to the model, which is the specific confusion this protocol is
    meant to resolve.
    """
    anchors = ["HR-03#001", "HR-13#001"]
    assert not pair_is_present(anchors, {"HR-13#001"})
    assert not pair_is_present(anchors, set())


def test_two_chunks_from_one_document_are_not_two_sides():
    """A disagreement needs two documents, not two passages."""
    anchors = ["HR-13#001", "HR-13#004"]
    assert not pair_is_present(anchors, {"HR-13#001", "HR-13#004"})


def test_anchors_absent_from_the_evidence_do_not_count():
    anchors = ["HR-03#001", "HR-13#001"]
    assert not pair_is_present(anchors, {"HR-03#001", "FIN-02#002"})


# --- the pre-commitment guard ------------------------------------------------


def test_the_guard_refuses_an_uncommitted_protocol(monkeypatch):
    """A rule written once the numbers are visible is not a committed rule.

    The guard is the only thing standing between this protocol and the
    failure mode it exists to prevent, so it is tested rather than trusted.
    """
    import verifier_protocol

    monkeypatch.setattr(verifier_protocol, "git",
                        lambda *a: " M docs/VERIFIER_PROTOCOL.md" if a[0] == "status" else "")
    with pytest.raises(SystemExit) as excinfo:
        require_committed_protocol()
    assert "runtime state is not committed" in str(excinfo.value)


def test_the_guard_refuses_uncommitted_verifier_code(monkeypatch):
    """The protocol being committed proves nothing about what will run.

    An edited prompt or parsing rule would influence all 96 calls while the
    document sat frozen in git looking authoritative. This is the case that
    checking only docs/ would have missed.
    """
    import verifier_protocol

    monkeypatch.setattr(
        verifier_protocol, "git",
        lambda *a: " M src/sme_assistant/verify/verifier.py" if a[0] == "status" else "",
    )
    with pytest.raises(SystemExit) as excinfo:
        require_committed_protocol()
    assert "verifier.py" in str(excinfo.value)


def test_the_guard_checks_prompt_config_and_gold_data():
    """Naming the paths in a comment is not the same as checking them."""
    import verifier_protocol

    checked = verifier_protocol.RUNTIME_PATHS
    assert "src" in checked, "the prompt and the parsing rules live here"
    assert "config.json" in checked, "models, top_k and min_similarity"
    assert "gold" in checked, "the declared types the classification is scored against"
    assert "docs/VERIFIER_PROTOCOL.md" in checked
    # results/ is where this run writes; refusing on it would only produce a
    # habit of passing a bypass flag.
    assert not any(p.startswith("results") for p in checked)


def test_the_guard_passes_on_a_clean_runtime_state(monkeypatch):
    import verifier_protocol

    monkeypatch.setattr(verifier_protocol, "git",
                        lambda *a: "" if a[0] == "status" else "abc123")
    provenance = require_committed_protocol()
    assert provenance["commit"] == "abc123"
    assert provenance["other_paths_dirty"] is False
    assert provenance["runtime_paths_checked"] == list(verifier_protocol.RUNTIME_PATHS)


# --- the design is what the protocol document says it is ---------------------


def test_the_design_matches_the_committed_document():
    """96 calls, two models, two conditions, phi3 excluded.

    If the script and the document disagree about the design, the document is
    not a pre-registration of anything.
    """
    text = PROTOCOL.read_text(encoding="utf-8")

    assert MODELS == ("llama3.2:3b", "qwen2.5:3b")
    assert "phi3" not in " ".join(MODELS)
    assert CONDITIONS == ("full", "oracle_pair")
    assert "96 verifier calls" in text
    for model in MODELS:
        assert model in text


def test_the_dry_run_reports_ninety_six_calls():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verifier_protocol.py"), "--dry-run"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "Calls       96" in out.stdout
    assert "Questions   24" in out.stdout
