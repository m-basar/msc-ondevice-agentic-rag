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
import preflight_placement  # noqa: E402
import run_arms  # noqa: E402

from sme_assistant.common.llm_client import (  # noqa: E402
    LLMError,
    MockClient,
    OllamaClient,
)
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
    assert {"models_loaded", "any_on_gpu", "vram_bytes", "complete"} <= set(observed)


# --- the timing analyser -----------------------------------------------------


def write_run(root: Path, name: str, arm: str, *, purpose="performance",
              questions=68, verifier=False, scoring=False, split="test",
              question_prefix="Q", reused=None, hashes="same",
              condition="pi5_cpu", placement="cpu", loaded_end=None,
              loaded_start=None, drafts_match=True, retrieval_match=True,
              generation_match=True, duplicate_ids=False,
              drop_wall_seconds=0, zero_wall_seconds=False,
              drop_verification_seconds=0, break_wall_identity=False,
              generation_model="llama3.2:3b", verification_model="qwen2.5:3b"):
    directory = root / name
    directory.mkdir(parents=True)
    if loaded_end is None:
        loaded_end = [{"name": "llama3.2:3b", "size": 100, "size_vram": 0},
                      {"name": "qwen2.5:3b", "size": 100, "size_vram": 0}]
    (directory / "manifest.json").write_text(json.dumps({
        "split": split,
        "purpose": purpose,
        "hardware_condition": condition,
        "placement": placement,
        "arm": {"arm": arm, "generation_model": generation_model,
                "verification_model": verification_model},
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
            "wall_seconds": (
                0.0 if zero_wall_seconds
                else (5.0 + (1.0 if break_wall_identity else 0.0)) if verifier
                else 2.0
            ),
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
            if n >= drop_verification_seconds:
                record["verification_seconds"] = 3.0
            record["verification_generation"] = {
                "model": "qwen2.5:3b",
                "eval_tokens_per_second": 8.0,
                "prompt_tokens_per_second": 40.0,
                "load_seconds": 0.4,
                "cpu_temp_c": 75.0,
                "throttled": True,
            }
        if n < drop_wall_seconds:
            record.pop("wall_seconds")
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
    block = analyse_performance.timings(answers=answers, manifest=manifest, summary=summary)

    assert block["draft_stage"]["model"] == "llama3.2:3b"
    assert block["verifier_stage"]["model"] == "qwen2.5:3b"
    assert block["draft_stage"]["eval_tokens_per_second"]["mean"] == 20.0
    assert block["verifier_stage"]["eval_tokens_per_second"]["mean"] == 8.0
    assert block["has_verifier_stage"] is True


def test_unknown_throttling_is_not_counted_as_not_throttled(tmp_path):
    """Missing instrumentation must not read as a clean thermal record."""
    directory = write_run(tmp_path, "d", "D", verifier=True, reused="b")
    manifest, answers, summary = analyse_performance.performance_run(directory)
    block = analyse_performance.timings(answers=answers, manifest=manifest, summary=summary)

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
    block = analyse_performance.timings(answers=answers, manifest=manifest, summary=summary)
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
    assert "D_cpu_placement_holds_for_every_model" in failing(validated(tmp_path, payload))


def test_a_gpu_condition_with_nothing_in_vram_is_caught(tmp_path):
    payload, _, _ = build_index(
        tmp_path,
        b_kwargs={"condition": "laptop_gpu", "placement": "gpu"},
        d_kwargs={"condition": "laptop_gpu", "placement": "gpu"},
        hardware_condition="laptop_gpu", requested_placement="gpu",
        observed_placement={"any_on_gpu": True, "models_loaded": ["x"]},
    )
    failures = failing(validated(tmp_path, payload))
    assert "B_stage_model_on_gpu" in failures
    assert "D_stage_model_on_gpu" in failures


def test_an_embedding_model_on_cpu_does_not_fail_a_gpu_run(tmp_path):
    """Embeddings are pinned to the CPU by design, so a zero-VRAM embedding
    model alongside an offloaded generator is correct, not a failure."""
    gpu = [{"name": "llama3.2:3b", "size": 100, "size_vram": 100},
           {"name": "qwen2.5:3b", "size": 100, "size_vram": 100},
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


# --- amendment 1.19: fail-open paths closed ----------------------------------


def test_gpu_placement_is_judged_on_the_model_being_timed(tmp_path):
    """A leftover Llama on the GPU does not make D a GPU run.

    D times the *verifier*. If Qwen ran on the CPU while a residual Llama held
    VRAM, an "any model on GPU" check passed it, and the ratio would compare a
    GPU draft against a CPU verification.
    """
    both_gpu = [{"name": "llama3.2:3b", "size": 100, "size_vram": 100},
                {"name": "qwen2.5:3b", "size": 100, "size_vram": 100}]
    leftover_only = [{"name": "llama3.2:3b", "size": 100, "size_vram": 100},
                     {"name": "qwen2.5:3b", "size": 100, "size_vram": 0}]
    payload, _, _ = build_index(
        tmp_path,
        b_kwargs={"condition": "laptop_gpu", "placement": "gpu",
                  "loaded_end": both_gpu},
        d_kwargs={"condition": "laptop_gpu", "placement": "gpu",
                  "loaded_end": leftover_only},
        hardware_condition="laptop_gpu", requested_placement="gpu",
        observed_placement={"any_on_gpu": True, "models_loaded": ["x"]},
    )
    failures = failing(validated(tmp_path, payload))
    assert "D_stage_model_on_gpu" in failures
    assert "B_stage_model_on_gpu" not in failures


def test_the_timed_stage_model_must_be_resident(tmp_path):
    """D's verifier missing from residency means nothing confirms where the
    timed work ran."""
    payload, _, _ = build_index(tmp_path, d_kwargs={
        "loaded_end": [{"name": "llama3.2:3b", "size": 100, "size_vram": 0}]
    })
    assert "D_stage_model_resident" in failing(validated(tmp_path, payload))


def test_a_successful_but_empty_residency_fails(tmp_path):
    """all([]) is True, so an empty model list previously passed as CPU."""
    payload, _, _ = build_index(tmp_path, d_kwargs={"loaded_end": []})
    failures = failing(validated(tmp_path, payload))
    assert "D_residency_not_empty" in failures


def test_a_malformed_size_fails_closed(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={
        "loaded_end": [{"name": "qwen2.5:3b", "size": "big", "size_vram": 0}]
    })
    assert "D_residency_fully_reported" in failing(validated(tmp_path, payload))


def test_d_wall_time_must_equal_draft_plus_verification(tmp_path):
    """If it does not, D timed something other than a replay of B."""
    payload, _, _ = build_index(tmp_path, d_kwargs={"break_wall_identity": True})
    assert "D_wall_time_is_draft_plus_verification" in failing(
        validated(tmp_path, payload)
    )


def test_the_wall_time_identity_tolerates_only_recorded_rounding(tmp_path):
    payload, _, _ = build_index(tmp_path)
    result = validated(tmp_path, payload)
    identity = result["checks"]["D_wall_time_is_draft_plus_verification"]
    assert identity["pass"] is True
    assert identity["compared"] == 68
    assert identity["tolerance_seconds"] == analyse_performance.WALL_TIME_TOLERANCE


def test_a_missing_size_vram_fails_closed(tmp_path):
    """Absent residency is not evidence of CPU placement."""
    payload, _, _ = build_index(tmp_path, d_kwargs={
        "loaded_end": [{"name": "qwen2.5:3b", "size": 100}]
    })
    assert "D_residency_fully_reported" in failing(validated(tmp_path, payload))


def test_a_non_numeric_size_vram_fails_closed(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={
        "loaded_end": [{"name": "qwen2.5:3b", "size": 100, "size_vram": "unknown"}]
    })
    assert "D_residency_fully_reported" in failing(validated(tmp_path, payload))


def test_missing_wall_seconds_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, b_kwargs={"drop_wall_seconds": 3})
    assert "B_wall_seconds_complete" in failing(validated(tmp_path, payload))


def test_non_positive_wall_seconds_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, b_kwargs={"zero_wall_seconds": True})
    assert "B_wall_seconds_complete" in failing(validated(tmp_path, payload))


def test_missing_verification_seconds_is_caught(tmp_path):
    payload, _, _ = build_index(tmp_path, d_kwargs={"drop_verification_seconds": 5})
    assert "D_verification_seconds_complete" in failing(validated(tmp_path, payload))


def test_api_ps_failure_raises_rather_than_reporting_nothing_loaded(monkeypatch):
    """Returning [] made an unreachable endpoint look like CPU placement."""
    from sme_assistant.common.llm_client import LLMError

    client = OllamaClient(load_config())

    def boom(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(LLMError, match="Residency is unknown"):
        client.residency()


def test_a_rejected_report_discloses_no_latency(tmp_path, monkeypatch, capsys):
    """A rejected run's timing is not a smaller result. It is not computed."""
    payload, _, _ = build_index(tmp_path, d_kwargs={"reused": None})
    index_path = tmp_path / "latest_test_performance_pi5_cpu.json"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(analyse_performance, "ROOT", tmp_path)
    monkeypatch.setattr(
        analyse_performance, "load_config",
        lambda: type("C", (), {"path": staticmethod(lambda _k: tmp_path)})(),
    )
    monkeypatch.setattr(analyse_performance, "timings",
                        lambda *a, **k: pytest.fail("timings must not be computed"))

    out = tmp_path / "analysis"
    code = analyse_performance.main(["--index", index_path.name, "--out", str(out)])
    assert code == 1

    written = json.loads((out / "performance_latest_test_performance_pi5_cpu_REJECTED.json")
                         .read_text(encoding="utf-8"))
    assert written["rejected"] is True
    # Structural, not substring: check *names* legitimately mention
    # wall_seconds. What must be absent is any computed timing.
    assert "arms" not in written, "rejected report carries per-arm timings"
    assert "H5" not in written, "rejected report carries an H5 verdict"

    def numeric_timing_keys(node, path=""):
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"ratio", "mean", "median", "mean_wall_seconds",
                           "eval_tokens_per_second", "prompt_tokens_per_second"}:
                    found.append(f"{path}.{key}")
                found.extend(numeric_timing_keys(value, f"{path}.{key}"))
        elif isinstance(node, list):
            for n, value in enumerate(node):
                found.extend(numeric_timing_keys(value, f"{path}[{n}]"))
        return found

    assert numeric_timing_keys(written) == [], "rejected report leaked a timing"


# --- the standalone preflight, amendment 1.20 --------------------------------


class FakeOllama:
    """A live server, simulated at the /api boundary rather than at _post.

    It drives the **real** ``wait_until_unloaded`` from ``OllamaClient``, with
    an injected clock so the polling logic is exercised deterministically and
    without sleeping. Amendment 1.22.
    """

    UNLOAD_TIMEOUT_SECONDS = OllamaClient.UNLOAD_TIMEOUT_SECONDS
    UNLOAD_POLL_INTERVAL_SECONDS = OllamaClient.UNLOAD_POLL_INTERVAL_SECONDS

    def __init__(self, *, placement="cpu", sticky=(), refuse_embedding=False,
                 unload_delay_polls=0, ps_failures=0):
        self.loaded: dict[str, int] = {}
        self.placement = placement
        self.sticky = {self._tagged(m) for m in sticky}
        self.refuse_embedding = refuse_embedding
        self.unload_delay_polls = unload_delay_polls
        self.ps_failures = ps_failures
        self.calls: list[tuple[str, str]] = []
        self.default_model = "llama3.2:3b"
        self.applied_num_gpu = None
        self.applied_keep_alive = None
        self._clock = 0.0
        self._pending: dict[str, int] = {}

    @staticmethod
    def _tagged(model):
        model = model or ""
        return model if ":" in model else f"{model}:latest"

    def unload(self, model=None, *, embedding=False):
        self.calls.append(("unload", model))
        if embedding and self.refuse_embedding:
            return {}
        name = self._tagged(model)
        if name in self.sticky:
            return {}
        if self.unload_delay_polls:
            self._pending[name] = self.unload_delay_polls
        else:
            self.loaded.pop(name, None)
        return {}

    def _load(self, model, *, embedding):
        name = self._tagged(model)
        self._pending.pop(name, None)
        vram = 0 if embedding else (
            100 if (self.applied_num_gpu != 0 and self.placement == "gpu") else 0
        )
        self.loaded[name] = vram

    def generate(self, prompt, *, model=None, options=None, keep_alive=None):
        self.calls.append(("generate", model))
        self.applied_num_gpu = (options or {}).get("num_gpu")
        self.applied_keep_alive = keep_alive
        self._load(model or self.default_model, embedding=False)

    def embed(self, text, *, model=None, options=None, keep_alive=None):
        self.calls.append(("embed", model))
        self.applied_keep_alive = keep_alive
        self._load("nomic-embed-text", embedding=True)

    def observed_placement(self):
        if self.ps_failures > 0:
            self.ps_failures -= 1
            raise LLMError("simulated /api/ps failure")
        return {
            "models_loaded": list(self.loaded),
            "any_on_gpu": any(v > 0 for v in self.loaded.values()),
            "vram_bytes": dict(self.loaded),
            "sizes": {k: 100 for k in self.loaded},
            "complete": True,
        }

    def _tick(self, interval):
        self._clock += interval
        for name in list(self._pending):
            self._pending[name] -= 1
            if self._pending[name] <= 0:
                self.loaded.pop(name, None)
                del self._pending[name]

    def wait_until_unloaded(self, models, **kwargs):
        kwargs.setdefault("now", lambda: self._clock)
        kwargs.setdefault("sleep", self._tick)
        return OllamaClient.wait_until_unloaded(self, models, **kwargs)


def run_preflight(monkeypatch, tmp_path, server, placement="cpu"):
    monkeypatch.setattr(preflight_placement, "OllamaClient", lambda config: server)
    return preflight_placement.main(
        ["--placement", placement, "--out", str(tmp_path)]
    )


def latest_report(tmp_path):
    files = sorted(tmp_path.glob("*_preflight_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8"))


def test_the_preflight_checks_all_three_models(monkeypatch, tmp_path):
    """Embedding eviction was the reason it was added, and the first version
    only ever touched Llama."""
    server = FakeOllama()
    assert run_preflight(monkeypatch, tmp_path, server) == 0

    report = latest_report(tmp_path)
    checked = [stage["model"] for stage in report["stages"]]
    assert checked == ["llama3.2:3b", "qwen2.5:3b", "nomic-embed-text"]
    assert report["result"] == "pass"


def test_the_preflight_exercises_the_embedding_endpoint(monkeypatch, tmp_path):
    server = FakeOllama()
    run_preflight(monkeypatch, tmp_path, server)
    embedding_stage = latest_report(tmp_path)["stages"][-1]
    assert embedding_stage["endpoint"] == "/api/embeddings"
    assert embedding_stage["expected_placement"] == "cpu"
    assert ("embed", None) in server.calls


def test_the_preflight_checks_the_verifier_not_only_the_generator(monkeypatch, tmp_path):
    server = FakeOllama()
    run_preflight(monkeypatch, tmp_path, server)
    models = [stage["model"] for stage in latest_report(tmp_path)["stages"]]
    assert "qwen2.5:3b" in models


def test_the_preflight_leaves_nothing_loaded(monkeypatch, tmp_path):
    """A preflight that leaves a model resident warms the next timed run."""
    server = FakeOllama()
    assert run_preflight(monkeypatch, tmp_path, server) == 0
    assert server.loaded == {}
    assert latest_report(tmp_path)["clean_exit"] is True


def test_the_preflight_fails_when_a_model_survives_eviction(monkeypatch, tmp_path):
    server = FakeOllama(sticky={"nomic-embed-text"})
    server.loaded["nomic-embed-text"] = 0
    assert run_preflight(monkeypatch, tmp_path, server) == 1
    report = latest_report(tmp_path)
    assert report["result"] == "fail"
    assert "was still resident" in report["error"]
    assert "of polling" in report["error"]


def test_the_preflight_catches_a_server_that_ignores_the_placement(monkeypatch, tmp_path):
    """Applying num_gpu is a request, not a guarantee.

    A server that disregards it must still be caught, which is why the check
    reads residency back rather than trusting that the option was honoured.
    """
    class IgnoresNumGpu(FakeOllama):
        def _load(self, model, *, embedding):
            name = model if ":" in model else f"{model}:latest"
            self.loaded[name] = 0 if embedding else 100   # offloads regardless

    assert run_preflight(monkeypatch, tmp_path, IgnoresNumGpu(),
                         placement="cpu") == 1
    assert "loaded onto gpu" in latest_report(tmp_path)["error"]


def test_the_preflight_uses_no_test_questions(monkeypatch, tmp_path):
    server = FakeOllama()
    run_preflight(monkeypatch, tmp_path, server)
    report = latest_report(tmp_path)
    assert report["uses_test_questions"] is False
    assert report["synthetic_prompt"] == "ping"


def test_nothing_calls_the_preflight_automatically():
    """It must not warm a subsequent timed run, so the runner does not invoke
    it. Amendment 1.20."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_arms.py")
    text = source.read_text(encoding="utf-8")
    assert "preflight(" not in text
    assert "preflight_placement.py" in text, "the runner should point at it"


# --- runner rejection records, exercised rather than inferred ----------------


def test_the_runner_writes_a_retained_rejection_record(tmp_path):
    """Tested directly, not only through the client exception."""
    class FakeConfig:
        @staticmethod
        def path(_key):
            return tmp_path

    record = run_arms.write_rejection(
        FakeConfig(), split="test", condition="pi5_cpu", stage="placement_D",
        reason="placement_mismatch", arm="D", requested_placement="cpu",
        observed_placement="gpu",
    )
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["reason"] == "placement_mismatch"
    assert payload["arm"] == "D"
    assert "retained unchanged" in payload["retention"]
    assert "delete" not in payload["retention"].lower().replace("deleted", "")


def test_rejection_records_do_not_overwrite_each_other(tmp_path):
    class FakeConfig:
        @staticmethod
        def path(_key):
            return tmp_path

    first = run_arms.write_rejection(FakeConfig(), split="test",
                                     condition="pi5_cpu", stage="eviction",
                                     reason="eviction_failed")
    second = run_arms.write_rejection(FakeConfig(), split="test",
                                      condition="pi5_cpu", stage="eviction",
                                      reason="eviction_failed")
    assert first != second
    assert len(list(tmp_path.glob("REJECTED_*.json"))) == 2


def test_no_rejection_record_instructs_deletion():
    """Amendment 1.19.5. Deleting the evidence of a failed condition is how a
    failed condition becomes an unrecorded one."""
    text = (Path(__file__).resolve().parents[1] / "scripts" / "run_arms.py").read_text(
        encoding="utf-8"
    )
    for phrase in ("Delete or re-tag", "delete or re-tag", "re-tag the"):
        assert phrase not in text


# --- amendment 1.21 ----------------------------------------------------------


def test_the_implicit_latest_tag_does_not_defeat_the_preflight(monkeypatch, tmp_path):
    """config.json says nomic-embed-text; /api/ps says nomic-embed-text:latest.

    An exact comparison fails in both directions, and both are dangerous: a
    failed eviction reads as success, and a correctly loaded model is rejected.
    """
    server = FakeOllama()
    assert run_preflight(monkeypatch, tmp_path, server) == 0
    report = latest_report(tmp_path)
    assert report["result"] == "pass"
    # The server reported the tagged form throughout.
    assert any(":latest" in name
               for stage in report["stages"]
               for name in stage.get("loaded_after_synthetic_call", []))


def test_a_failed_embedding_eviction_is_caught_despite_the_tag(monkeypatch, tmp_path):
    """The fail-open direction: sticky under the tagged name."""
    server = FakeOllama(sticky={"nomic-embed-text:latest"})
    server.loaded["nomic-embed-text:latest"] = 0
    assert run_preflight(monkeypatch, tmp_path, server) == 1
    assert "was still resident" in latest_report(tmp_path)["error"]


def test_canonical_model_name_normalises_the_implicit_tag():
    from sme_assistant.common.llm_client import canonical_model_name

    assert canonical_model_name("nomic-embed-text:latest") == "nomic-embed-text"
    assert canonical_model_name("nomic-embed-text") == "nomic-embed-text"
    assert canonical_model_name("llama3.2:3b") == "llama3.2:3b"
    assert canonical_model_name(None) == ""


def test_the_preflight_applies_the_placement_rather_than_only_observing_it(
    monkeypatch, tmp_path
):
    """Without num_gpu the server picks a device, and the check becomes a
    report of what Ollama chose."""
    server = FakeOllama(placement="gpu")
    run_preflight(monkeypatch, tmp_path, server, placement="cpu")
    assert server.applied_num_gpu == 0, "cpu placement was never applied"
    stage = latest_report(tmp_path)["stages"][0]
    assert stage["options_applied"]["num_gpu"] == 0


def test_gpu_placement_is_applied_as_full_offload(monkeypatch, tmp_path):
    server = FakeOllama(placement="gpu")
    run_preflight(monkeypatch, tmp_path, server, placement="gpu")
    assert server.applied_num_gpu == -1


def test_cleanup_runs_even_when_a_check_fails(monkeypatch, tmp_path):
    """1.20 promised nothing is left loaded. A raise after loading previously
    jumped past the eviction."""
    class FailsOnVerifier(FakeOllama):
        def observed_placement(self):
            observed = super().observed_placement()
            if ("generate", "qwen2.5:3b") in self.calls:
                observed["any_on_gpu"] = True
                observed["vram_bytes"] = {k: 100 for k in self.loaded}
            return observed

    server = FailsOnVerifier()
    assert run_preflight(monkeypatch, tmp_path, server, placement="cpu") == 1
    report = latest_report(tmp_path)
    assert report["result"] == "fail"
    # The failure happened part way through, and cleanup still ran.
    assert report["project_models_remaining"] == [], \
        "a failed preflight left a project model loaded"
    assert report["project_cleanup_complete"] is True


def test_exit_residency_is_recorded_on_failure(monkeypatch, tmp_path):
    server = FakeOllama(sticky={"nomic-embed-text:latest"})
    server.loaded["nomic-embed-text:latest"] = 0
    run_preflight(monkeypatch, tmp_path, server)
    report = latest_report(tmp_path)
    assert "models_loaded_at_exit" in report
    assert "project_models_remaining" in report
    assert "project_cleanup_complete" in report
    assert report["project_cleanup_complete"] is False


def test_a_stray_third_party_model_does_not_fail_the_preflight(monkeypatch, tmp_path):
    """Cleanup waits for *this project's* three models, not for the machine to
    be idle.

    A model loaded by something else is not ours to evict, and failing on it
    would make the preflight depend on what else happens to be running.
    """
    server = FakeOllama()
    original = server.observed_placement

    def with_stray():
        observed = original()
        observed["models_loaded"] = list(observed["models_loaded"]) + ["other:latest"]
        return observed

    monkeypatch.setattr(server, "observed_placement", with_stray)
    assert run_preflight(monkeypatch, tmp_path, server) == 0
    report = latest_report(tmp_path)
    assert report["project_cleanup_complete"] is True
    assert report["project_models_remaining"] == []
    # The stray is reported rather than hidden. Amendment 1.23.
    assert "other:latest" in report["models_loaded_at_exit"]


def test_one_of_our_models_remaining_is_a_dirty_exit(monkeypatch, tmp_path):
    server = FakeOllama(sticky={"qwen2.5:3b"})
    assert run_preflight(monkeypatch, tmp_path, server) == 1
    report = latest_report(tmp_path)
    assert report["project_cleanup_complete"] is False
    assert "qwen2.5:3b" in report["project_models_remaining"]
    assert "qwen2.5:3b" in " ".join(report["models_loaded_at_exit"] or [])


# --- polling, amendment 1.22 -------------------------------------------------


def test_a_delayed_disappearance_is_waited_for_not_failed(monkeypatch, tmp_path):
    """The defect the 08:01:59 diagnostic exposed: checked once, immediately."""
    server = FakeOllama(unload_delay_polls=4)
    assert run_preflight(monkeypatch, tmp_path, server) == 0
    report = latest_report(tmp_path)
    assert report["result"] == "pass"
    waited = report["cleanup_wait"]
    assert waited["cleared"] is True
    assert waited["elapsed_seconds"] > 0, "it should record how long it waited"
    assert waited["successful_observations"] >= 1


def test_permanent_residue_still_fails_after_the_timeout(monkeypatch, tmp_path):
    """A poll that never gives up would pass on Ollama's own five-minute
    expiry instead of on an effective unload."""
    server = FakeOllama(sticky={"llama3.2:3b", "qwen2.5:3b", "nomic-embed-text"})
    server.loaded.update({"llama3.2:3b": 0, "qwen2.5:3b": 0,
                          "nomic-embed-text:latest": 0})
    assert run_preflight(monkeypatch, tmp_path, server) == 1
    report = latest_report(tmp_path)
    waited = report["stages"][0]["unload_wait"] if report["stages"] else None
    assert report["result"] == "fail"
    if waited:
        assert waited["cleared"] is False
        assert waited["elapsed_seconds"] >= waited["timeout_seconds"]


def test_a_transient_api_ps_failure_is_tolerated_during_the_wait(monkeypatch, tmp_path):
    """A momentary refusal is not evidence either way, so it is recorded and
    the poll continues."""
    server = FakeOllama(ps_failures=3)
    assert run_preflight(monkeypatch, tmp_path, server) == 0
    report = latest_report(tmp_path)
    assert report["result"] == "pass"
    recorded = report["stages"][0]["unload_wait"]["transient_errors"]
    assert recorded, "the transient failure should be recorded, not hidden"
    assert all("api/ps" in message for message in recorded)


def test_a_wait_that_only_ever_errors_fails_closed():
    """No successful empty observation means no evidence, and no evidence is
    not a pass."""
    class AlwaysFails(FakeOllama):
        def observed_placement(self):
            raise LLMError("simulated /api/ps failure")

    server = AlwaysFails()
    result = server.wait_until_unloaded(["llama3.2:3b"])
    assert result["cleared"] is False
    assert result["successful_observations"] == 0
    assert result["transient_errors"]


def test_the_poll_budget_is_predeclared():
    """A timeout tuned until the check passed would not be a rule."""
    assert OllamaClient.UNLOAD_TIMEOUT_SECONDS == 30.0
    assert OllamaClient.UNLOAD_POLL_INTERVAL_SECONDS == 0.25


def test_the_dead_client_preflight_is_gone():
    """It was unused and still compared model names exactly, which is the bug
    amendment 1.21 fixed everywhere else."""
    assert not hasattr(OllamaClient, "preflight")
    source = (Path(__file__).resolve().parents[2] if False else
              Path(__file__).resolve().parents[1])
    text = (source / "src" / "sme_assistant" / "common" / "llm_client.py").read_text(
        encoding="utf-8"
    )
    assert "def preflight" not in text


def test_a_wall_time_breach_message_carries_no_magnitude(tmp_path):
    """Amendment 1.19 said rejected reports contain no timing values, and the
    message said "off by 0.4213s"."""
    payload, _, _ = build_index(tmp_path, d_kwargs={"break_wall_identity": True})
    result = validated(tmp_path, payload)
    identity = result["checks"]["D_wall_time_is_draft_plus_verification"]

    assert identity["pass"] is False
    assert identity["breaches"] == 68
    assert identity["breached_question_ids"], "identifiers should be named"
    blob = json.dumps(identity)
    assert "off by" not in blob
    for finding in result["findings"]:
        assert "off by" not in finding
    # No float in the message other than the declared tolerance.
    assert "0.002" in json.dumps(identity["tolerance_seconds"])


def test_the_analyser_matches_stage_models_canonically(tmp_path):
    """A manifest naming a model without its tag must still match /api/ps."""
    payload, _, _ = build_index(
        tmp_path,
        d_kwargs={"verification_model": "qwen2.5",
                  "loaded_end": [{"name": "qwen2.5:latest", "size": 100,
                                  "size_vram": 0}]},
    )
    assert "D_stage_model_resident" not in failing(validated(tmp_path, payload))


# --- retention is stated, not assumed. Amendment 1.23 ------------------------


def test_the_synthetic_load_states_its_retention(monkeypatch, tmp_path):
    """Without it, "gone within 30 seconds" is only conclusive on a server
    that happens to be on the five-minute default."""
    server = FakeOllama()
    assert run_preflight(monkeypatch, tmp_path, server) == 0
    assert server.applied_keep_alive == preflight_placement.PREFLIGHT_KEEP_ALIVE
    for stage in latest_report(tmp_path)["stages"]:
        assert stage["load_keep_alive"] == "10m"


def test_keep_alive_is_top_level_and_not_an_option(monkeypatch):
    """Ollama reads retention from the request body. Buried in options it would
    be ignored and the server would stay on its configured default."""
    client = OllamaClient(load_config())
    sent: dict = {}
    monkeypatch.setattr(client, "_post",
                        lambda endpoint, payload: sent.update(payload)
                        or {"response": "", "embedding": [0.1]})

    client.generate("ping", options={"num_predict": 1}, keep_alive="10m")
    assert sent["keep_alive"] == "10m"
    assert "keep_alive" not in sent["options"]

    sent.clear()
    client.embed("ping", keep_alive="10m")
    assert sent["keep_alive"] == "10m"
    assert "keep_alive" not in sent["options"]


def test_experimental_calls_never_set_retention(monkeypatch):
    """A run must be unaffected by the preflight's retention choice."""
    client = OllamaClient(load_config())
    sent: dict = {}
    monkeypatch.setattr(client, "_post",
                        lambda endpoint, payload: sent.update(payload)
                        or {"response": "", "embedding": [0.1]})

    client.generate("what is the mileage rate?")
    assert "keep_alive" not in sent

    sent.clear()
    client.embed("what is the mileage rate?")
    assert "keep_alive" not in sent


def test_polling_returns_the_complete_observed_model_list():
    class WithStray(FakeOllama):
        def observed_placement(self):
            observed = super().observed_placement()
            observed["models_loaded"] = list(observed["models_loaded"]) + ["other:latest"]
            return observed

    server = WithStray()
    result = server.wait_until_unloaded(["llama3.2:3b"])
    assert result["cleared"] is True
    assert result["remaining"] == []
    assert "other:latest" in result["models_loaded"]


# --- the success path, end to end. Amendment 1.24 ----------------------------


def test_a_valid_index_produces_a_report_end_to_end(tmp_path, monkeypatch):
    """The gap that let an argument-order bug reach a live run.

    Every earlier test of ``main`` exercised the *rejection* path, which
    returns before timings are computed. Nothing ran it through to a report, so
    ``timings(*runs[arm])`` transposing manifest and answers was invisible
    until the real index was analysed.
    """
    payload, _, _ = build_index(tmp_path)
    index_path = tmp_path / "latest_test_performance_pi5_cpu.json"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(analyse_performance, "ROOT", tmp_path)
    monkeypatch.setattr(
        analyse_performance, "load_config",
        lambda: type("C", (), {"path": staticmethod(lambda _k: tmp_path)})(),
    )

    out = tmp_path / "analysis"
    assert analyse_performance.main(
        ["--index", index_path.name, "--out", str(out)]
    ) == 0

    report = json.loads(
        (out / "performance_latest_test_performance_pi5_cpu.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["validation"]["valid"] is True
    assert set(report["arms"]) == {"B", "D"}
    # The transposition would have made these empty or wrong.
    assert report["arms"]["B"]["questions"] == 68
    assert report["arms"]["D"]["draft_stage"]["model"] == "llama3.2:3b"
    assert report["arms"]["D"]["verifier_stage"]["model"] == "qwen2.5:3b"
    assert report["arms"]["D"]["has_verifier_stage"] is True
    assert report["H5"]["verdict"] in {"supported", "not supported"}


def test_a_laptop_index_gets_no_h5_verdict_end_to_end(tmp_path, monkeypatch):
    """H5 names the Pi. A laptop ratio is descriptive under RQ4."""
    payload, _, _ = build_index(
        tmp_path,
        b_kwargs={"condition": "laptop_cpu"},
        d_kwargs={"condition": "laptop_cpu"},
        hardware_condition="laptop_cpu",
    )
    index_path = tmp_path / "latest_test_performance_laptop_cpu.json"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(analyse_performance, "ROOT", tmp_path)
    monkeypatch.setattr(
        analyse_performance, "load_config",
        lambda: type("C", (), {"path": staticmethod(lambda _k: tmp_path)})(),
    )

    out = tmp_path / "analysis"
    assert analyse_performance.main(
        ["--index", index_path.name, "--out", str(out)]
    ) == 0
    report = json.loads(
        (out / "performance_latest_test_performance_laptop_cpu.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["H5"]["verdict"] == "not applicable"
    assert "pi5_cpu" in report["H5"]["verdict_basis"]


def test_timings_cannot_be_called_positionally():
    """Keyword-only, so the transposition cannot recur."""
    import inspect

    signature = inspect.signature(analyse_performance.timings)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
               for p in signature.parameters.values())
