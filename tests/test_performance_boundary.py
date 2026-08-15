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
              question_prefix="Q", reused=None, hashes="same",
              condition="pi5_cpu", placement="cpu", loaded_end=None,
              loaded_start=None, drafts_match=True, retrieval_match=True,
              generation_match=True, duplicate_ids=False):
    directory = root / name
    directory.mkdir(parents=True)
    if loaded_end is None:
        loaded_end = [{"name": "llama3.2:3b", "size": 100, "size_vram": 0}]
    (directory / "manifest.json").write_text(json.dumps({
        "split": split,
        "purpose": purpose,
        "hardware_condition": condition,
        "placement": placement,
        "arm": {"arm": arm},
        "provenance": {key: hashes for key in (
            "corpus_sha256", "chunk_set_sha256", "question_set_sha256",
            "registry_sha256", "config_sha256")},
        "environment": {"ollama": {"loaded": loaded_start or []}},
    }), encoding="utf-8")

    lines = []
    for n in range(questions):
        qid = f"{question_prefix}-{0 if duplicate_ids else n:03d}"
        record = {
            "question_id": qid,
            "wall_seconds": 2.0,
            "answer": f"draft for {qid}",
            "retrieval": {"chunks": [qid]} if retrieval_match else {"chunks": [n]},
            "generation": {
                "model": "llama3.2:3b",
                "eval_tokens_per_second": 20.0,
                "prompt_tokens_per_second": 100.0,
                "load_seconds": 0.5,
                "cpu_temp_c": 60.0,
                "throttled": None,
            } if generation_match else {"model": "other"},
        }
        if verifier:
            record["draft_answer"] = (
                f"draft for {qid}" if drafts_match else "something else"
            )
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
        "environment_at_end": {"ollama": {"loaded": loaded_end}},
        "elapsed_seconds": 120.0,
        "drafts_reused_from": reused,
    }), encoding="utf-8")
    return directory


def build_index(tmp_path, *, b_kwargs=None, d_kwargs=None, **overrides):
    d_defaults = {"verifier": True, "reused": "results/runs/b_test_pi5"}
    d_defaults.update(d_kwargs or {})
    b = write_run(tmp_path, "b_test_pi5", "B", **(b_kwargs or {}))
    d = write_run(tmp_path, "d_test_pi5", "D", **d_defaults)
    payload = {
        "B": str(b.relative_to(tmp_path)),
        "D": str(d.relative_to(tmp_path)),
        "_performance": {
            "hardware_condition": "pi5_cpu",
            "requested_placement": "cpu",
            "observed_placement": {"any_on_gpu": False, "models_loaded": ["x"],
                                   "vram_bytes": {"x": 0}},
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
    started = [{"name": "llama3.2:3b", "size": 100, "size_vram": 0}]
    directory = write_run(tmp_path, "b", "B", loaded_start=started)
    manifest, answers, summary = analyse_performance.performance_run(directory)
    block = analyse_performance.timings(answers, manifest, summary)
    assert analyse_performance.loaded_models(block["environment_start"]) == started
    assert analyse_performance.loaded_models(block["environment_end"])


# --- validation: one test per rejection route --------------------------------


def validated(tmp_path, payload):
    meta = payload.pop("_performance")
    runs = {arm: analyse_performance.performance_run(tmp_path / rel)
            for arm, rel in payload.items()}
    return analyse_performance.validate(runs, meta)


def failing(result) -> set[str]:
    return set(result["checks_failed"])


def test_a_sound_index_passes_every_check(tmp_path):
    result = validated(tmp_path, build_index(tmp_path)[0])
    assert result["valid"] is True, result["findings"]
    assert result["checks_failed"] == []
    assert result["checks_run"] > 20


def test_arms_must_be_exactly_b_and_d(tmp_path):
    write_run(tmp_path, "b", "B")
    runs = {"B": analyse_performance.performance_run(tmp_path / "b")}
    result = analyse_performance.validate(runs, {"hardware_condition": "pi5_cpu",
                                                 "requested_placement": "cpu",
                                                 "observed_placement": {"any_on_gpu": False}})
    assert "arms_exactly_b_and_d" in failing(result)


def test_a_wrong_question_count_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, b_kwargs={"questions": 41})
    assert "B_question_count" in failing(validated(tmp_path, payload))


def test_duplicate_question_ids_are_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, b_kwargs={"duplicate_ids": True})
    assert "B_question_ids_unique" in failing(validated(tmp_path, payload))


def test_mismatched_question_sets_are_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"question_prefix": "Z"})
    assert "question_ids_match" in failing(validated(tmp_path, payload))


def test_a_manifest_declaring_the_wrong_arm_is_caught(tmp_path):
    b = write_run(tmp_path, "b", "C")
    d = write_run(tmp_path, "d", "D", verifier=True, reused="b")
    runs = {"B": analyse_performance.performance_run(b),
            "D": analyse_performance.performance_run(d)}
    result = analyse_performance.validate(runs, {"hardware_condition": "pi5_cpu",
                                                 "requested_placement": "cpu",
                                                 "observed_placement": {"any_on_gpu": False}})
    assert "B_manifest_declares_its_arm" in failing(result)


def test_a_condition_disagreeing_with_the_index_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"condition": "laptop_cpu"})
    assert "D_condition_matches_index" in failing(validated(tmp_path, payload))


def test_a_placement_disagreeing_with_the_index_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"placement": "gpu"})
    assert "D_placement_matches_index" in failing(validated(tmp_path, payload))


def test_empty_provenance_hashes_are_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, b_kwargs={"hashes": ""})
    failures = failing(validated(tmp_path, payload))
    assert "B_provenance_non_empty" in failures
    assert "provenance_equal" in failures


def test_divergent_provenance_hashes_are_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"hashes": "different"})
    assert "provenance_equal" in failing(validated(tmp_path, payload))


def test_a_missing_verifier_stage_is_caught(tmp_path):
    b = write_run(tmp_path, "b", "B")
    d = write_run(tmp_path, "d", "D", verifier=False, reused="b")
    runs = {"B": analyse_performance.performance_run(b),
            "D": analyse_performance.performance_run(d)}
    result = analyse_performance.validate(runs, {"hardware_condition": "pi5_cpu",
                                                 "requested_placement": "cpu",
                                                 "observed_placement": {"any_on_gpu": False}})
    assert "D_verifier_records_complete" in failing(result)


def test_a_verifier_stage_on_arm_b_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, b_kwargs={"verifier": True})
    assert "B_has_no_verifier_stage" in failing(validated(tmp_path, payload))


def test_scoring_fields_are_caught_by_the_loader(tmp_path):
    """Refused before validation even begins."""
    directory = write_run(tmp_path, "scored", "B", scoring=True)
    with pytest.raises(AnalysisError, match="answer scoring"):
        analyse_performance.performance_run(directory)


def test_an_inexact_draft_replay_is_caught(tmp_path):
    """Without byte-exact replay, B versus D is not the frozen comparison."""
    payload, _, _ = build_index(tmp_path, d_kwargs={"drafts_match": False})
    assert "draft_replay_exact" in failing(validated(tmp_path, payload))


def test_differing_generation_records_are_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"generation_match": False})
    assert "generation_records_match" in failing(validated(tmp_path, payload))


def test_differing_retrieval_records_are_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"retrieval_match": False})
    assert "retrieval_records_match" in failing(validated(tmp_path, payload))


def test_arm_d_not_replaying_drafts_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"reused": None})
    assert "arm_d_replayed_b_drafts" in failing(validated(tmp_path, payload))


# --- placement fails closed --------------------------------------------------


def test_missing_api_ps_evidence_fails_closed(tmp_path):
    """An unknown device is not a hardware condition."""
    payload, _, _ = build_index(tmp_path, observed_placement=None)
    failures = failing(validated(tmp_path, payload))
    assert "placement_observed_at_run_time" in failures


def test_missing_per_model_evidence_fails_closed(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"loaded_end": []})
    assert "per_model_evidence_present" in failing(validated(tmp_path, payload))


def test_a_cpu_condition_with_a_model_in_vram_is_caught(tmp_path):
    """Per-model, not a single any_on_gpu boolean."""
    payload, _, _ = build_index(tmp_path, d_kwargs={
        "loaded_end": [{"name": "qwen2.5:3b", "size": 100, "size_vram": 100}]
    })
    assert "cpu_placement_holds_for_every_model" in failing(validated(tmp_path, payload))


def test_a_gpu_condition_with_nothing_in_vram_is_caught(tmp_path):
    payload, _, _ = build_index(
        tmp_path,
        b_kwargs={"condition": "laptop_gpu", "placement": "gpu"},
        d_kwargs={"condition": "laptop_gpu", "placement": "gpu"},
        hardware_condition="laptop_gpu", requested_placement="gpu",
        observed_placement={"any_on_gpu": True, "models_loaded": ["x"]},
    )
    assert "gpu_placement_observed_for_at_least_one_model" in failing(
        validated(tmp_path, payload)
    )


def test_an_embedding_model_on_cpu_does_not_fail_a_gpu_run(tmp_path):
    """Embeddings are pinned to the CPU by design, so a zero-VRAM embedding
    model alongside an offloaded generator is correct, not a failure."""
    gpu = [{"name": "llama3.2:3b", "size": 100, "size_vram": 100},
           {"name": "nomic-embed-text:latest", "size": 10, "size_vram": 0}]
    payload, _, _ = build_index(
        tmp_path,
        b_kwargs={"condition": "laptop_gpu", "placement": "gpu", "loaded_end": gpu},
        d_kwargs={"condition": "laptop_gpu", "placement": "gpu", "loaded_end": gpu},
        hardware_condition="laptop_gpu", requested_placement="gpu",
        observed_placement={"any_on_gpu": True, "models_loaded": ["x"]},
    )
    result = validated(tmp_path, payload)
    assert result["valid"] is True, result["findings"]


def test_an_observed_placement_contradicting_the_request_is_caught(tmp_path):
    payload, _, _ = build_index(
        tmp_path, observed_placement={"any_on_gpu": True, "models_loaded": ["x"]}
    )
    assert "observed_placement_matches_request" in failing(validated(tmp_path, payload))


def test_a_condition_implying_a_different_placement_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, requested_placement="gpu")
    assert "condition_implies_requested_placement" in failing(validated(tmp_path, payload))


def test_an_unknown_condition_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, hardware_condition="desktop_tpu")
    assert "condition_is_known" in failing(validated(tmp_path, payload))


# --- eviction routes the embedding model correctly ---------------------------


def test_the_embedding_model_is_evicted_through_the_embedding_endpoint(monkeypatch):
    """An embedding model does not serve /api/generate, so evicting it there
    leaves it resident with its previous placement. Amendment 1.18."""
    client = OllamaClient(load_config())
    calls: list = []
    monkeypatch.setattr(client, "_post",
                        lambda endpoint, payload: calls.append((endpoint, payload)))

    client.unload("nomic-embed-text", embedding=True)
    client.unload("llama3.2:3b")

    assert calls[0][0] == "/api/embeddings"
    assert calls[0][1]["keep_alive"] == 0
    assert calls[1][0] == "/api/generate"
