"""Tests for the hardware boundary: the runner gate and the timing analyser.

Amendment 1.15 declared the boundary, 1.16 made the quality side enforce it, and
1.17 closes the runner and reporting side. Nothing here has been exercised
against a real Pi yet, which is precisely why it is tested now rather than
discovered mid-run on a machine that takes hours per pass.

Two failure modes drive most of these. A performance run that is
indistinguishable from the frozen quality run contaminates the reported result.
And a timing report that reads Arm D's *draft* metrics reports Arm B's numbers
under D's name, because D replays B's draft and only the verifier is new.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyse_performance  # noqa: E402
import run_arms  # noqa: E402

from sme_assistant.common.llm_client import MockClient, OllamaClient  # noqa: E402
from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.evaluation.analysis import AnalysisError  # noqa: E402


# --- the runner gate ---------------------------------------------------------


BASE = ["--performance-only", "--split", "test", "--arms", "B", "D",
        "--tag", "pi5", "--placement", "cpu",
        "--hardware-condition", "pi5_cpu"]


def run_main(monkeypatch, argv):
    """Parse arguments only. The gate must reject before anything executes."""
    monkeypatch.setattr(sys, "argv", ["run_arms.py"] + argv)
    return run_arms.main()


def expect_refusal(monkeypatch, argv, fragment):
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, argv)
    assert exit_info.value.code == 2, "argparse should refuse, not run"
    return fragment


def swap(argv, flag, value):
    out = list(argv)
    index = out.index(flag)
    out[index + 1] = value
    return out


def test_a_performance_run_must_be_on_the_test_split(monkeypatch, capsys):
    """--split defaults to dev, so this is the easy mistake to make."""
    expect_refusal(monkeypatch, swap(BASE, "--split", "dev"), "split")
    assert "must be 'test'" in capsys.readouterr().err


def test_a_performance_run_must_use_exactly_b_and_d(monkeypatch, capsys):
    """H5 is a ratio. One arm cannot produce it."""
    argv = [a for a in BASE if a != "D"]
    expect_refusal(monkeypatch, argv, "arms")
    assert "exactly B and D" in capsys.readouterr().err


def test_a_performance_run_may_not_include_another_arm(monkeypatch, capsys):
    argv = list(BASE)
    argv.insert(argv.index("D") + 1, "C")
    expect_refusal(monkeypatch, argv, "arms")
    assert "exactly B and D" in capsys.readouterr().err


def test_a_performance_run_may_not_be_mocked(monkeypatch, capsys):
    """Mock timings measure the harness."""
    expect_refusal(monkeypatch, BASE + ["--mock"], "mock")
    assert "mock" in capsys.readouterr().err


def test_the_condition_must_match_the_placement(monkeypatch, capsys):
    """pi5_cpu recorded from a GPU run is a false provenance block, not a
    slightly wrong number."""
    expect_refusal(monkeypatch, swap(BASE, "--placement", "gpu"), "placement")
    assert "requires --placement cpu" in capsys.readouterr().err


def test_laptop_gpu_requires_gpu_placement(monkeypatch, capsys):
    argv = swap(swap(BASE, "--hardware-condition", "laptop_gpu"),
                "--placement", "cpu")
    expect_refusal(monkeypatch, argv, "placement")
    assert "requires --placement gpu" in capsys.readouterr().err


def test_a_performance_run_requires_a_tag(monkeypatch, capsys):
    argv = [a for a in BASE if a not in ("--tag", "pi5")]
    expect_refusal(monkeypatch, argv, "tag")
    assert "--tag is required" in capsys.readouterr().err


def test_placement_flags_are_refused_outside_performance_mode(monkeypatch, capsys):
    argv = ["--split", "dev", "--placement", "cpu"]
    expect_refusal(monkeypatch, argv, "placement")
    assert "apply only to" in capsys.readouterr().err


# --- embeddings stay on the CPU ---------------------------------------------


def test_embeddings_are_posted_with_cpu_placement(monkeypatch):
    """The gap amendment 1.17 closes.

    The embedding call posted no options at all, so the device was whatever
    Ollama chose. A cpu-only condition whose query embedding was offloaded is
    not the condition it claims to be.
    """
    client = OllamaClient(load_config())
    sent: dict = {}

    def fake_post(endpoint, payload):
        sent["endpoint"] = endpoint
        sent["payload"] = payload
        return {"embedding": [0.1, 0.2]}

    monkeypatch.setattr(client, "_post", fake_post)
    client.embed("what is the mileage rate?")

    assert sent["endpoint"] == "/api/embeddings"
    assert sent["payload"]["options"]["num_gpu"] == 0


def test_the_client_can_evict_and_observe_placement():
    """Requested placement is a hope until something looks."""
    client = MockClient(load_config())
    assert client.unload()["mock"] is True
    observed = client.observed_placement()
    assert set(observed) == {"models_loaded", "any_on_gpu", "vram_bytes"}


# --- the timing analyser -----------------------------------------------------


def write_run(root: Path, name: str, arm: str, *, purpose="performance",
              questions=68, verifier=False, scoring=False, split="test",
              question_prefix="Q", reused=None, hashes="same"):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "split": split,
        "purpose": purpose,
        "arm": {"arm": arm},
        "provenance": {
            "corpus_sha256": hashes, "chunk_set_sha256": hashes,
            "question_set_sha256": hashes, "registry_sha256": hashes,
            "config_sha256": hashes,
        },
        "environment": {"start": True},
    }), encoding="utf-8")

    lines = []
    for n in range(questions):
        record = {
            "question_id": f"{question_prefix}-{n:03d}",
            "wall_seconds": 2.0 if arm == "B" else 5.0,
            "generation": {
                "model": "llama3.2:3b",
                "eval_tokens_per_second": 20.0,
                "prompt_tokens_per_second": 100.0,
                "load_seconds": 0.5,
                "cpu_temp_c": 60.0,
                "throttled": None,
            },
        }
        if verifier:
            record["verification_seconds"] = 3.0
            record["verification_generation"] = {
                "model": "qwen2.5:3b",
                "eval_tokens_per_second": 8.0,
                "prompt_tokens_per_second": 40.0,
                "load_seconds": 0.4,
                "cpu_temp_c": 75.0,
                "throttled": True,
            }
        if scoring:
            record["scoring"] = {"citation_support": 1.0}
        lines.append(json.dumps(record))
    (directory / "answers.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps({
        "environment_at_end": {"end": True},
        "elapsed_seconds": 120.0,
        "drafts_reused_from": reused,
    }), encoding="utf-8")
    return directory


def build_index(tmp_path, **overrides):
    b = write_run(tmp_path, "b_test_pi5", "B")
    d = write_run(tmp_path, "d_test_pi5", "D", verifier=True,
                  reused="results/runs/b_test_pi5")
    payload = {
        "B": str(b.relative_to(tmp_path)),
        "D": str(d.relative_to(tmp_path)),
        "_performance": {
            "hardware_condition": "pi5_cpu",
            "requested_placement": "cpu",
            "observed_placement": {"any_on_gpu": False, "models_loaded": ["x"]},
            "split": "test",
        },
    }
    payload["_performance"].update(overrides)
    return payload, b, d


def test_a_quality_run_is_refused(tmp_path):
    directory = write_run(tmp_path, "quality", "B", purpose="quality")
    with pytest.raises(AnalysisError, match="not a performance run"):
        analyse_performance.performance_run(directory)


def test_a_performance_run_carrying_scoring_is_refused(tmp_path):
    directory = write_run(tmp_path, "scored", "B", scoring=True)
    with pytest.raises(AnalysisError, match="answer scoring"):
        analyse_performance.performance_run(directory)


def test_arm_d_metrics_separate_the_draft_from_the_verifier(tmp_path):
    """The defect this rewrite fixes.

    D replays B's draft, so ``generation`` on a D record is B's number
    generated on another machine. Reporting it as D's decode rate reports the
    wrong model at the wrong speed on the wrong hardware.
    """
    directory = write_run(tmp_path, "d", "D", verifier=True, reused="b")
    manifest, answers, summary = analyse_performance.performance_run(directory)
    block = analyse_performance.timings(answers, manifest, summary)

    assert block["draft_stage"]["model"] == "llama3.2:3b"
    assert block["verifier_stage"]["model"] == "qwen2.5:3b"
    assert block["draft_stage"]["eval_tokens_per_second"]["mean"] == 20.0
    assert block["verifier_stage"]["eval_tokens_per_second"]["mean"] == 8.0
    assert block["has_verifier_stage"] is True


def test_unknown_throttling_is_not_counted_as_not_throttled(tmp_path):
    """Missing instrumentation must not read as a clean thermal record."""
    directory = write_run(tmp_path, "d", "D", verifier=True, reused="b")
    manifest, answers, summary = analyse_performance.performance_run(directory)
    block = analyse_performance.timings(answers, manifest, summary)

    draft = block["draft_stage"]
    assert draft["throttled_unknown"] == 68
    assert draft["throttled_false"] == 0
    assert block["verifier_stage"]["throttled_true"] == 68


def test_both_environments_are_reported(tmp_path):
    """A Pi that began at 60 degrees and finished throttled at 85 was not one
    machine throughout."""
    directory = write_run(tmp_path, "b", "B")
    manifest, answers, summary = analyse_performance.performance_run(directory)
    block = analyse_performance.timings(answers, manifest, summary)
    assert block["environment_start"] == {"start": True}
    assert block["environment_end"] == {"end": True}


# --- validation --------------------------------------------------------------


def validated(tmp_path, payload):
    meta = payload.pop("_performance")
    runs = {arm: analyse_performance.performance_run(tmp_path / rel)
            for arm, rel in payload.items()}
    return analyse_performance.validate(runs, meta)


def test_a_sound_index_validates(tmp_path):
    payload, _, _ = build_index(tmp_path)
    assert validated(tmp_path, payload)["valid"] is True


def test_a_placement_mismatch_is_caught(tmp_path):
    payload, _, _ = build_index(
        tmp_path, observed_placement={"any_on_gpu": True, "models_loaded": ["x"]}
    )
    result = validated(tmp_path, payload)
    assert result["valid"] is False
    assert any("placement was gpu" in f for f in result["findings"])


def test_an_unobserved_placement_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, observed_placement=None)
    result = validated(tmp_path, payload)
    assert result["valid"] is False
    assert any("not observed" in f for f in result["findings"])


def test_a_wrong_question_count_is_caught(tmp_path):
    write_run(tmp_path, "b_test_pi5", "B", questions=41)
    write_run(tmp_path, "d_test_pi5", "D", verifier=True, reused="b")
    payload = {"B": "b_test_pi5", "D": "d_test_pi5"}
    meta = {"requested_placement": "cpu",
            "observed_placement": {"any_on_gpu": False}}
    runs = {a: analyse_performance.performance_run(tmp_path / r)
            for a, r in payload.items()}
    result = analyse_performance.validate(runs, meta)
    assert any("expected 68" in f for f in result["findings"])


def test_mismatched_question_sets_are_caught(tmp_path):
    write_run(tmp_path, "b", "B")
    write_run(tmp_path, "d", "D", verifier=True, reused="b", question_prefix="Z")
    runs = {a: analyse_performance.performance_run(tmp_path / a.lower())
            for a in ("B", "D")}
    result = analyse_performance.validate(
        runs, {"requested_placement": "cpu",
               "observed_placement": {"any_on_gpu": False}}
    )
    assert any("different question set" in f for f in result["findings"])


def test_divergent_provenance_hashes_are_caught(tmp_path):
    write_run(tmp_path, "b", "B")
    write_run(tmp_path, "d", "D", verifier=True, reused="b", hashes="different")
    runs = {a: analyse_performance.performance_run(tmp_path / a.lower())
            for a in ("B", "D")}
    result = analyse_performance.validate(
        runs, {"requested_placement": "cpu",
               "observed_placement": {"any_on_gpu": False}}
    )
    assert any("provenance hashes" in f for f in result["findings"])


def test_arm_d_not_replaying_drafts_is_caught(tmp_path):
    """Without the replay, B versus D is not the comparison the frozen run
    made."""
    write_run(tmp_path, "b", "B")
    write_run(tmp_path, "d", "D", verifier=True, reused=None)
    runs = {a: analyse_performance.performance_run(tmp_path / a.lower())
            for a in ("B", "D")}
    result = analyse_performance.validate(
        runs, {"requested_placement": "cpu",
               "observed_placement": {"any_on_gpu": False}}
    )
    assert any("did not replay" in f for f in result["findings"])


def test_an_arm_whose_manifest_disagrees_with_the_index_is_caught(tmp_path):
    write_run(tmp_path, "b", "B")
    write_run(tmp_path, "d", "C", verifier=True, reused="b")
    runs = {a: analyse_performance.performance_run(tmp_path / a.lower())
            for a in ("B", "D")}
    result = analyse_performance.validate(
        runs, {"requested_placement": "cpu",
               "observed_placement": {"any_on_gpu": False}}
    )
    assert any("manifest says arm 'C'" in f for f in result["findings"])
