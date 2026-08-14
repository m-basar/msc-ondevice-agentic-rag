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
    REPEATS,
    stability,
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

    An edited prompt or parsing rule would influence every call while the
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
    """288 calls, two models, two conditions, three repeats, phi3 excluded.

    If the script and the document disagree about the design, the document is
    not a pre-registration of anything.
    """
    text = PROTOCOL.read_text(encoding="utf-8")

    assert MODELS == ("llama3.2:3b", "qwen2.5:3b")
    assert "phi3" not in " ".join(MODELS)
    assert CONDITIONS == ("full", "oracle_pair")
    assert REPEATS == 3, "set by the corrected reproducibility check"
    assert "288 verifier calls" in text
    assert "complete passes in protocol order" in text
    assert "three results" in text and "not 288 independent observations" in text
    for model in MODELS:
        assert model in text


def test_the_dry_run_reports_the_committed_call_count():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verifier_protocol.py"), "--dry-run"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr
    assert "Calls       288" in out.stdout
    assert "Questions   24" in out.stdout


# --- stability is measured per prompt, never from the totals ------------------


def row(qid, repeat, **kw):
    base = dict(model="m", condition="full", question_id=qid, repeat=repeat,
                detected=False, classified=False, parse_failed=False,
                inferred_relationship="no_relationship", raw="{}")
    base.update(kw)
    return base


def test_agreement_is_one_when_every_repeat_agrees():
    rows = [row("Q1", r) for r in (1, 2, 3)]
    measures = stability(rows)
    assert measures["detected"] == 1.0
    assert measures["raw_text"] == 1.0


def test_a_prompt_that_changes_between_sweeps_is_counted():
    rows = [row("Q1", 1, raw='{"a":1}'), row("Q1", 2, raw='{"a":2}'),
            row("Q1", 3, raw='{"a":2}')]
    measures = stability(rows)
    assert measures["raw_text"] == 0.0
    assert measures["detected"] == 1.0, "the wording moved, the finding did not"


def test_cancelling_changes_do_not_look_stable():
    """The trap from pilots 02 and 03.

    Two questions moved one way and two the other, so the aggregate counts were
    identical while a tenth of the questions churned. Measuring per prompt is
    what stops that reading as reproducibility.
    """
    rows = [
        row("Q1", 1, inferred_relationship="insufficient"),
        row("Q1", 2, inferred_relationship="no_relationship"),
        row("Q2", 1, inferred_relationship="no_relationship"),
        row("Q2", 2, inferred_relationship="insufficient"),
    ]
    from collections import Counter
    totals = [Counter(r["inferred_relationship"] for r in rows if r["repeat"] == n)
              for n in (1, 2)]
    assert totals[0] == totals[1], "the totals are identical, which is the trap"
    assert stability(rows)["inferred_relationship"] == 0.0


# --- the protocol is not changeable from the command line ---------------------


def test_no_flag_can_change_the_committed_design():
    """A protocol that can be reconfigured at the prompt was not pre-registered.

    The flags would be reached for at exactly the moment the guard mattered.
    Changing the design means editing the file, which the freeze check then
    requires to be committed.
    """
    source = (ROOT / "scripts" / "verifier_protocol.py").read_text(encoding="utf-8")
    for flag in ("--models", "--repeats", "--i-accept-a-defective-pipeline"):
        assert f'add_argument("{flag}"' not in source, (
            f"{flag} lets the committed design be bypassed"
        )
    # --dry-run stays: it runs nothing and changes nothing.
    assert 'add_argument("--dry-run"' in source


def test_git_failure_is_not_read_as_a_clean_tree(monkeypatch):
    """The failure mode of the first version was "permit everything".

    A failed git status returns empty stdout, which the guard read as a clean
    working tree: the one answer that lets an unfrozen experiment run.
    """
    import subprocess as sp

    import verifier_protocol

    class Failed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(sp, "run", lambda *a, **k: Failed())
    with pytest.raises(SystemExit) as excinfo:
        verifier_protocol.git("status", "--porcelain")
    assert "failed with status 128" in str(excinfo.value)
    assert "reads as a clean tree" in str(excinfo.value)


# --- the design is validated before any call is made --------------------------


def test_a_missing_family_stops_the_run():
    """Seven families would still produce a full results file and a decision."""
    from types import SimpleNamespace

    import verifier_protocol

    questions = [SimpleNamespace(family_id=f"F{i}", question_id=f"F{i}-Q{j}")
                 for i in range(7) for j in range(3)]
    registry = SimpleNamespace(by_id=lambda f: SimpleNamespace(is_conflict=True))
    with pytest.raises(SystemExit) as excinfo:
        verifier_protocol.validate_design(questions, registry)
    assert "7 families" in str(excinfo.value)


def test_a_family_with_two_paraphrases_stops_the_run():
    from types import SimpleNamespace

    import verifier_protocol

    questions = [SimpleNamespace(family_id=f"F{i}", question_id=f"F{i}-Q{j}")
                 for i in range(8) for j in range(3)]
    questions = [q for q in questions if q.question_id != "F0-Q2"]
    registry = SimpleNamespace(
        by_id=lambda f: SimpleNamespace(is_conflict=f not in ("F6", "F7"))
    )
    with pytest.raises(SystemExit) as excinfo:
        verifier_protocol.validate_design(questions, registry)
    assert "exactly 3 paraphrases" in str(excinfo.value)


def test_the_real_question_set_matches_the_design():
    import verifier_protocol
    from sme_assistant.evaluation.config import load_evaluation_config
    from sme_assistant.evaluation.conflicts import load_conflicts
    from sme_assistant.evaluation.question_set import load_question_set

    evaluation = load_evaluation_config()
    registry = load_conflicts(evaluation.path("conflicts"))
    question_set = load_question_set(evaluation.path("question_set"))
    questions = [q for q in question_set.split("dev") if q.family_id]
    verifier_protocol.validate_design(questions, registry)  # must not raise


# --- the decision rules are executed, not printed for a reader ----------------


def block(model="llama3.2:3b", condition="full", block=1, detected=0,
          classified=0, controls_false=0, genuine=6, controls=2):
    return dict(model=model, condition=condition, block=block,
                genuine_detected=detected, genuine_classified=classified,
                genuine=genuine, controls_false=controls_false,
                controls=controls, pair_absent=0)


def test_r1_fires_on_the_null_result():
    import verifier_protocol

    rule, reason = verifier_protocol.apply_rules(
        [block(m, c, b) for m in verifier_protocol.MODELS
         for c in CONDITIONS for b in (1, 2, 3)], {})
    assert rule.startswith("R1")
    assert "null result" in reason


def test_r0_refuses_to_credit_a_model_that_flags_every_control():
    import verifier_protocol

    entries = [block("llama3.2:3b", "full", b, detected=6, controls_false=2)
               for b in (1, 2, 3)]
    entries += [block("qwen2.5:3b", c, b) for c in CONDITIONS for b in (1, 2, 3)]
    entries += [block("llama3.2:3b", "oracle_pair", b) for b in (1, 2, 3)]
    rule, _ = verifier_protocol.apply_rules(entries, {})
    assert not rule.startswith("R1"), "detection existed, so this is not the null"


def test_r2_fires_when_the_oracle_condition_rescues_detection():
    import verifier_protocol

    entries = [block(m, "full", b, detected=1) for m in verifier_protocol.MODELS
               for b in (1, 2, 3)]
    entries += [block(m, "oracle_pair", b, detected=5, classified=5)
                for m in verifier_protocol.MODELS for b in (1, 2, 3)]
    rule, reason = verifier_protocol.apply_rules(entries, {})
    assert rule.startswith("R2")
    assert "retrieval finding" in reason


def test_r4_separates_detection_from_classification():
    import verifier_protocol

    entries = [block(m, c, b, detected=5, classified=1)
               for m in verifier_protocol.MODELS for c in CONDITIONS
               for b in (1, 2, 3)]
    rule, reason = verifier_protocol.apply_rules(entries, {})
    assert rule.startswith("R4")
    assert "revision 3 is permitted" in reason
