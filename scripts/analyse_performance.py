"""Timing for RQ4 and H5. Reads performance runs, validates them, refuses quality ones.

    python scripts/analyse_performance.py --index latest_test_performance_pi5_cpu.json

H5 predicts that Arm D costs between 1.5x and 2.5x Arm B **on the Pi 5**: enough
to show the verification pass is doing real work, not so much that it is doing
more than one extra generation.

Three things this script gets right that an obvious version does not.

**Arm D is two stages, not one.** D replays B's draft and then verifies it, so
``record["generation"]`` on a D record describes the *reused draft*, produced by
a different model in the immediately preceding arm of the same invocation. Reading only that field reports B's prefill, decode,
load, temperature and throttle state and labels them D. The verifier's own
figures are in ``verification_generation``, and the two are reported separately
as well as summed. The wall-clock total is still the right quantity for H5,
because what a user waits for is draft plus verification.

**Unknown is not false.** ``throttled`` is ``None`` on a platform that does not
expose the counter. Counting those as "not throttled" turns missing
instrumentation into a clean thermal record.

**H5 is stated over the Pi 5.** A laptop ratio is reported as a descriptive
figure under RQ4 and does not receive an H5 verdict, because the hypothesis
names the platform and prefill dominance differs.

The script validates before it reports: arm identity, the question set, the
provenance hashes, the requested against the observed placement, and whether D
actually reused B's drafts. A timing run that silently differed from the frozen
one in any of those is not comparable, and is refused rather than averaged.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.evaluation.analysis import AnalysisError  # noqa: E402
from sme_assistant.common.llm_client import canonical_model_name  # noqa: E402
from sme_assistant.evaluation.run_writer import read_run  # noqa: E402

H5_LOWER, H5_UPPER = 1.5, 2.5
H5_CONDITION = "pi5_cpu"
EXPECTED_QUESTIONS = 68
EXPECTED_ARMS = frozenset({"B", "D"})
PROVENANCE_KEYS = (
    "corpus_sha256", "chunk_set_sha256", "question_set_sha256",
    "registry_sha256", "config_sha256",
)
CONDITION_PLACEMENT = {"laptop_gpu": "gpu", "laptop_cpu": "cpu", "pi5_cpu": "cpu"}

#: Which model each arm's *timed stage* actually runs on.
#: B times a draft, so its stage model is the generator. D replays that draft
#: and times a verification, so its stage model is the verifier. Checking "some
#: model held VRAM" would pass a D run where a leftover Llama sat on the GPU
#: while Qwen, the model being timed, ran on the CPU. Amendment 1.20.
STAGE_MODEL_KEY = {"B": "generation_model", "D": "verification_model"}

#: Derived, not fitted. See amendment 1.21 for the full derivation.
#:
#: ``VerifiedAnswer.wall_seconds`` is ``self.answer.wall_seconds +
#: self.generation.wall_seconds``, computed from the **live float attributes**
#: of the reused draft and the verification, not from anything already
#: serialised. Three values are then stored at three decimals: B's own wall
#: time, D's verification_seconds, and D's total. The comparison
#: ``|D_total - (B_wall + D_verification)|`` therefore admits at most three
#: half-ULP errors at 3 dp, or 0.0015 s.
#:
#: The 4-decimal roundings on ``prompt_seconds``, ``eval_seconds``,
#: ``load_seconds`` and ``embed_seconds``, and the 6-decimal rounding on
#: ``search_seconds``, apply to separately serialised sub-fields that do not
#: enter this arithmetic. They are named here because a derivation that ignores
#: the other roundings in the same record is not a derivation a reader can
#: check.
#:
#: 0.002 leaves headroom over the 0.0015 bound rather than sitting exactly on
#: it, so the check fails on a genuine breach rather than on a boundary case.
WALL_TIME_TOLERANCE = 0.002


def performance_run(directory: Path) -> tuple[dict, list[dict], dict]:
    """Load a run, refusing anything not declared performance-only."""
    manifest, answers = read_run(directory)
    if manifest.get("purpose") != "performance":
        raise AnalysisError(
            f"{Path(directory).name} is not a performance run: its manifest says "
            f"purpose={manifest.get('purpose')!r}. This script reports timing "
            "only and must not be pointed at the frozen quality run. See "
            "amendment 1.15."
        )
    scored = [r for r in answers if r.get("scoring")]
    if scored:
        raise AnalysisError(
            f"{Path(directory).name} carries answer scoring on {len(scored)} "
            "records. A performance run must not produce quality figures; "
            "rerun it with --performance-only."
        )
    summary_path = Path(directory) / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists() else {}
    )
    return manifest, answers, summary


def loaded_models(environment: dict | None) -> list[dict]:
    """Per-model residency evidence from a captured environment block."""
    return list(((environment or {}).get("ollama") or {}).get("loaded") or [])


def placement_of(models: list[dict]) -> dict:
    """Per-model offload, derived from ``size`` and ``size_vram``.

    A single "any_on_gpu" boolean is not enough. A partially offloaded model,
    or an embedding model that should be pinned to the CPU while the generator
    is on the GPU, are different situations and only per-model figures
    distinguish them.
    """
    out: dict = {}
    for model in models:
        # Canonicalised, because /api/ps reports the implicit ":latest" tag that
        # config.json omits. Amendment 1.21.
        name = canonical_model_name(model.get("name") or model.get("model"))
        size, vram = model.get("size"), model.get("size_vram")
        fraction = (
            (vram / size) if isinstance(size, (int, float)) and size
            and isinstance(vram, (int, float)) else None
        )
        out[name] = {"size": size, "size_vram": vram, "offload_fraction": fraction}
    return out


def validate(runs: dict[str, tuple[dict, list[dict], dict]], index_meta: dict) -> dict:
    """Every check, before any timing value is read.

    Amendment 1.18. Each check is named and recorded pass or fail, so a
    rejection says which property failed rather than that something did. The
    six hardware runs were already complete when this was written, and it was
    written before any latency value was examined; the checks are therefore
    properties of a sound run rather than a description of these ones.

    Missing evidence **fails closed**. An absent ``/api/ps`` observation is not
    a run that happened to be on the right device, it is a run whose device is
    unknown, and an unknown device is not a hardware condition.
    """
    checks: dict[str, dict] = {}
    detail: dict = {}

    def check(name: str, ok: bool, message: str = "", **extra) -> None:
        checks[name] = {"pass": bool(ok), **({"message": message} if message else {}),
                        **extra}

    # --- shape of the index -------------------------------------------------
    check("arms_exactly_b_and_d", set(runs) == set(EXPECTED_ARMS),
          f"expected arms {sorted(EXPECTED_ARMS)}, got {sorted(runs)}")

    # --- per-run properties -------------------------------------------------
    condition = index_meta.get("hardware_condition")
    requested = index_meta.get("requested_placement")

    for arm, (manifest, answers, _) in sorted(runs.items()):
        declared = (manifest.get("arm") or {}).get("arm")
        check(f"{arm}_manifest_declares_its_arm", declared == arm,
              f"index says {arm}, manifest says {declared!r}")
        check(f"{arm}_purpose_is_performance",
              manifest.get("purpose") == "performance",
              f"purpose={manifest.get('purpose')!r}")
        check(f"{arm}_split_is_test", manifest.get("split") == "test",
              f"split={manifest.get('split')!r}")
        check(f"{arm}_condition_matches_index",
              manifest.get("hardware_condition") == condition,
              f"manifest {manifest.get('hardware_condition')!r} against index "
              f"{condition!r}")
        check(f"{arm}_placement_matches_index",
              manifest.get("placement") == requested,
              f"manifest {manifest.get('placement')!r} against index {requested!r}")

        ids = [r.get("question_id") for r in answers]
        check(f"{arm}_question_count", len(answers) == EXPECTED_QUESTIONS,
              f"{len(answers)} answers, expected {EXPECTED_QUESTIONS}")
        check(f"{arm}_question_ids_unique", len(set(ids)) == len(ids),
              f"{len(ids) - len(set(ids))} duplicate identifiers")

        provenance = manifest.get("provenance") or {}
        missing = [k for k in PROVENANCE_KEYS if not provenance.get(k)]
        check(f"{arm}_provenance_non_empty", not missing, f"missing {missing}")

        generation = sum(1 for r in answers if r.get("generation"))
        verifier = sum(1 for r in answers if r.get("verification_generation"))
        check(f"{arm}_generation_records_complete",
              generation == len(answers),
              f"{generation} generation blocks over {len(answers)} answers")
        if arm == "D":
            check("D_verifier_records_complete", verifier == len(answers),
                  f"{verifier} verifier blocks over {len(answers)} answers")
        else:
            check(f"{arm}_has_no_verifier_stage", verifier == 0,
                  f"{verifier} verifier blocks on an unverified arm")

        scored = [r.get("question_id") for r in answers if "scoring" in r]
        check(f"{arm}_no_scoring_fields", not scored,
              f"{len(scored)} records carry answer scoring")

    # --- cross-arm identity -------------------------------------------------
    if set(runs) == set(EXPECTED_ARMS):
        b_manifest, b_answers, _ = runs["B"]
        d_manifest, d_answers, _ = runs["D"]
        b = {r["question_id"]: r for r in b_answers}
        d = {r["question_id"]: r for r in d_answers}
        common = sorted(set(b) & set(d))

        check("question_ids_match", set(b) == set(d),
              f"{len(set(b) ^ set(d))} identifiers differ")
        check("matched_question_count", len(common) == EXPECTED_QUESTIONS,
              f"{len(common)} matched, expected {EXPECTED_QUESTIONS}")

        replayed = sum(1 for q in common if d[q].get("draft_answer") == b[q].get("answer"))
        check("draft_replay_exact", replayed == len(common),
              f"{replayed}/{len(common)} of D's drafts equal B's answers",
              matched=replayed, of=len(common))

        same_generation = sum(1 for q in common if d[q].get("generation") == b[q].get("generation"))
        check("generation_records_match", same_generation == len(common),
              f"{same_generation}/{len(common)} generation blocks identical")

        same_retrieval = sum(1 for q in common if d[q].get("retrieval") == b[q].get("retrieval"))
        check("retrieval_records_match", same_retrieval == len(common),
              f"{same_retrieval}/{len(common)} retrieval blocks identical")

        b_hashes = {k: (b_manifest.get("provenance") or {}).get(k) for k in PROVENANCE_KEYS}
        d_hashes = {k: (d_manifest.get("provenance") or {}).get(k) for k in PROVENANCE_KEYS}
        check("provenance_equal", b_hashes == d_hashes,
              "the two runs disagree on their provenance hashes")
        detail["provenance"] = {"B": b_hashes, "D": d_hashes}

    # --- placement, fail closed --------------------------------------------
    check("condition_is_known", condition in CONDITION_PLACEMENT,
          f"unknown hardware condition {condition!r}")
    check("condition_implies_requested_placement",
          CONDITION_PLACEMENT.get(condition) == requested,
          f"{condition} implies {CONDITION_PLACEMENT.get(condition)!r}, index "
          f"requested {requested!r}")

    observed = index_meta.get("observed_placement")
    check("placement_observed_at_run_time", bool(observed),
          "no /api/ps evidence in the run index; the device is unknown")

    per_model: dict = {}
    for arm, (manifest, _, summary) in sorted(runs.items()):
        per_model[arm] = {
            "at_start": placement_of(loaded_models(manifest.get("environment"))),
            "at_end": placement_of(loaded_models(summary.get("environment_at_end"))),
        }
    detail["per_model_placement"] = per_model

    end_evidence = {arm: per_model[arm]["at_end"] for arm in per_model}
    check("per_model_evidence_present",
          all(bool(v) for v in end_evidence.values()),
          "a run finished with no loaded-model evidence, so its residency "
          "cannot be confirmed")

    if requested:
        # Placement is judged per arm and, within an arm, on the model whose
        # work is being timed. Amendment 1.20.
        for arm, (manifest, _, _) in sorted(runs.items()):
            models = end_evidence.get(arm) or {}

            check(f"{arm}_residency_not_empty", bool(models),
                  "no model was resident at the end of this arm, so nothing "
                  "confirms where it ran")
            if not models:
                continue

            unreported = [
                name for name, block in models.items()
                if not isinstance(block.get("size_vram"), (int, float))
                or not isinstance(block.get("size"), (int, float))
            ]
            check(f"{arm}_residency_fully_reported", not unreported,
                  f"{unreported} reported no numeric size or size_vram, so "
                  "placement cannot be confirmed")
            if unreported:
                continue

            stage_model = canonical_model_name(
                (manifest.get("arm") or {}).get(
                    STAGE_MODEL_KEY.get(arm, "generation_model")
                )
            )
            check(f"{arm}_stage_model_resident", stage_model in models,
                  f"the timed stage model {stage_model!r} is not among the "
                  f"resident models {sorted(models)}",
                  stage_model=stage_model)

            if requested == "cpu":
                held = [f"{name} held {block['size_vram']} bytes"
                        for name, block in models.items()
                        if block["size_vram"] > 0]
                check(f"{arm}_cpu_placement_holds_for_every_model", not held,
                      "; ".join(held))
            elif stage_model in models:
                # Independently, per arm, on the model actually being timed.
                on_gpu = models[stage_model]["size_vram"] > 0
                check(f"{arm}_stage_model_on_gpu", on_gpu,
                      f"{stage_model} held no VRAM on an arm requesting gpu "
                      "placement, so the timed stage ran on the CPU",
                      size_vram=models[stage_model]["size_vram"])

    if observed:
        seen = "gpu" if observed.get("any_on_gpu") else "cpu"
        detail["placement"] = {"requested": requested, "observed": seen,
                               "vram_bytes": observed.get("vram_bytes")}
        check("observed_placement_matches_request", seen == requested,
              f"requested {requested}, observed {seen}")
    else:
        detail["placement"] = {"requested": requested, "observed": None}

    # --- timing completeness ------------------------------------------------
    # Amendment 1.19. ``_series`` silently drops non-numeric values, so a run
    # missing half its timings would average the half that survived and report
    # it as the arm's latency. A mean over an unknown denominator is not a
    # measurement.
    def positive_numbers(records, field):
        values = [r.get(field) for r in records]
        return [v for v in values if isinstance(v, (int, float))
                and not isinstance(v, bool) and v > 0]

    for arm, (_, answers, _) in sorted(runs.items()):
        wall = positive_numbers(answers, "wall_seconds")
        check(f"{arm}_wall_seconds_complete", len(wall) == EXPECTED_QUESTIONS,
              f"{len(wall)} positive numeric wall_seconds over {len(answers)} "
              f"answers, expected {EXPECTED_QUESTIONS}")
    if "D" in runs:
        verification = positive_numbers(runs["D"][1], "verification_seconds")
        check("D_verification_seconds_complete",
              len(verification) == EXPECTED_QUESTIONS,
              f"{len(verification)} positive numeric verification_seconds, "
              f"expected {EXPECTED_QUESTIONS}")

    # --- D's wall time must be B's draft plus D's verification ---------------
    # VerifiedAnswer.wall_seconds is defined as the reused draft's wall time
    # plus the verification's, so on a sound replay this identity holds for
    # every question. If it does not, D timed something other than a replay of
    # B, and the ratio would answer a different question. Amendment 1.20.
    if set(runs) == set(EXPECTED_ARMS):
        b_by_id = {r["question_id"]: r for r in runs["B"][1]}
        # Amendment 1.21. The breach list carries question identifiers only.
        # The previous version wrote "off by 0.4213s" into the message, and
        # that message is copied verbatim into the rejection file, so a
        # rejected report disclosed a timing-derived quantity in the very field
        # amendment 1.19 said would contain none.
        breached_ids: list[str] = []
        unmeasurable_ids: list[str] = []
        compared = 0
        for record in runs["D"][1]:
            baseline = b_by_id.get(record["question_id"])
            if baseline is None:
                continue
            parts = (baseline.get("wall_seconds"),
                     record.get("verification_seconds"),
                     record.get("wall_seconds"))
            if not all(isinstance(v, (int, float)) for v in parts):
                unmeasurable_ids.append(record["question_id"])
                continue
            compared += 1
            if abs(parts[2] - (parts[0] + parts[1])) > WALL_TIME_TOLERANCE:
                breached_ids.append(record["question_id"])
        check("D_wall_time_is_draft_plus_verification",
              not breached_ids and not unmeasurable_ids
              and compared == EXPECTED_QUESTIONS,
              (f"{len(breached_ids)} of {EXPECTED_QUESTIONS} questions fall "
               f"outside the rounding tolerance"
               + (f"; {len(unmeasurable_ids)} not measurable"
                  if unmeasurable_ids else "")
               + (f"; only {compared} comparable"
                  if compared != EXPECTED_QUESTIONS else "")),
              compared=compared,
              breaches=len(breached_ids),
              breached_question_ids=sorted(breached_ids),
              unmeasurable_question_ids=sorted(unmeasurable_ids),
              tolerance_seconds=WALL_TIME_TOLERANCE)

    # --- draft provenance ---------------------------------------------------
    if "D" in runs:
        summary = runs["D"][2]
        reused = summary.get("drafts_reused_from") or summary.get("drafts_replayed_from")
        detail["arm_d_replayed_drafts_from"] = reused
        check("arm_d_replayed_b_drafts", bool(reused),
              "arm D generated its own drafts, so B versus D is not the "
              "comparison the frozen run made")

    failed = sorted(name for name, result in checks.items() if not result["pass"])
    return {
        "schema_version": "1.1",
        "checks": checks,
        "checks_run": len(checks),
        "checks_failed": failed,
        "findings": [
            f"{name}: {checks[name].get('message', 'failed')}" for name in failed
        ],
        "valid": not failed,
        **detail,
    }


def _series(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def stage(records: list[dict], key: str) -> dict:
    """Metrics for one generation stage: the draft, or the verifier."""
    blocks = [r.get(key) or {} for r in records]
    blocks = [b for b in blocks if b]
    throttle_flags = [b.get("throttled") for b in blocks]
    return {
        "records": len(blocks),
        "model": next((b.get("model") for b in blocks if b.get("model")), None),
        "prompt_tokens_per_second": _series(
            b.get("prompt_tokens_per_second") for b in blocks
        ),
        "eval_tokens_per_second": _series(
            b.get("eval_tokens_per_second") for b in blocks
        ),
        "load_seconds": _series(b.get("load_seconds") for b in blocks),
        "cpu_temp_c": _series(b.get("cpu_temp_c") for b in blocks),
        # Unknown is unknown. Counting None as not-throttled turns missing
        # instrumentation into a clean thermal record.
        "throttled_true": sum(1 for t in throttle_flags if t is True),
        "throttled_false": sum(1 for t in throttle_flags if t is False),
        "throttled_unknown": sum(1 for t in throttle_flags if t is None),
    }


def timings(answers: list[dict], manifest: dict, summary: dict) -> dict:
    draft = stage(answers, "generation")
    verifier = stage(answers, "verification_generation")
    return {
        "questions": len(answers),
        # What a user waits for: draft plus verification where both ran.
        "wall_seconds": _series(r.get("wall_seconds") for r in answers),
        "verification_seconds": _series(
            r.get("verification_seconds") for r in answers
        ),
        "draft_stage": draft,
        "verifier_stage": verifier,
        "has_verifier_stage": verifier["records"] > 0,
        "environment_start": manifest.get("environment"),
        "environment_end": summary.get("environment_at_end"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", required=True,
                        help="run index written by a --performance-only run")
    parser.add_argument("--out", default=str(ROOT / "results" / "analysis"))
    args = parser.parse_args(argv)

    config = load_config()
    index_path = Path(config.path("paths.results")) / args.index
    if not index_path.exists():
        print(f"No run index at {index_path}", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index_meta = index.pop("_performance", {})
    if not index_meta:
        raise AnalysisError(
            f"{args.index} carries no _performance block, so it was not written "
            "by a --performance-only run. This script reports timing only."
        )

    runs = {arm: performance_run(ROOT / rel) for arm, rel in sorted(index.items())}

    # Amendment 1.19. Validation runs first and timings are not computed until
    # it passes. The previous version built the whole report, timings included,
    # and only then checked the verdict, so a rejected run still had its latency
    # calculated and written into the rejection file. A rejected run's timing is
    # not a smaller result; it is a number nobody is entitled to see, and the
    # only way to be sure it is not disclosed is not to compute it.
    validation = validate(runs, index_meta)
    if not validation["valid"]:
        rejection = {
            "schema_version": "1.1",
            "index": args.index,
            "hardware_condition": index_meta.get("hardware_condition"),
            "rejected": True,
            "validation": validation,
            "note": (
                "No timing was computed. This file deliberately contains no "
                "latency, no per-arm figures and no H5 field. The run "
                "directories are retained unchanged; see amendment 1.19."
            ),
        }
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"performance_{Path(args.index).stem}_REJECTED.json"
        target.write_text(json.dumps(rejection, indent=2), encoding="utf-8",
                          newline="\n")
        print("Refusing to report. Validation failed:", file=sys.stderr)
        for finding in validation["findings"]:
            print(f"  - {finding}", file=sys.stderr)
        print(f"\n  rejection record written to {target}", file=sys.stderr)
        print("  no latency was computed", file=sys.stderr)
        return 1

    report: dict = {
        "index": args.index,
        "hardware_condition": index_meta.get("hardware_condition"),
        "validation": validation,
        "arms": {arm: timings(*runs[arm]) for arm in sorted(runs)},
    }

    condition = report["hardware_condition"]
    if {"B", "D"} <= set(report["arms"]):
        b = report["arms"]["B"]["wall_seconds"]["mean"]
        d = report["arms"]["D"]["wall_seconds"]["mean"]
        ratio = d / b if b else None
        block = {
            "statement": (
                f"latency(D) is between {H5_LOWER}x and {H5_UPPER}x latency(B) "
                f"on {H5_CONDITION}"
            ),
            "condition": condition,
            "mean_wall_seconds": {"B": b, "D": d},
            "ratio": ratio,
            "reading": (
                "Wall clock is draft plus verification, which is what a user "
                "waits for. Above the range indicates the verifier is doing "
                "more work than one extra generation; below it indicates it is "
                "not verifying much. This says nothing about answer quality."
            ),
        }
        # H5 names the platform. A laptop ratio is descriptive under RQ4 and
        # does not get an H5 verdict, because prefill dominance differs.
        if condition == H5_CONDITION:
            block["verdict"] = (
                "supported" if ratio is not None and H5_LOWER <= ratio <= H5_UPPER
                else "not supported"
            )
        else:
            block["verdict"] = "not applicable"
            block["verdict_basis"] = (
                f"H5 is stated over {H5_CONDITION}; this index is {condition}. "
                "Reported as a descriptive RQ4 figure only."
            )
        report["H5"] = block
    else:
        report["H5"] = {"verdict": "pending",
                        "note": "needs both arm B and arm D in one index"}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"performance_{Path(args.index).stem}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    print(f"  condition {condition}, placement "
          f"{report['validation']['placement']['observed']}")
    for arm, block in sorted(report["arms"].items()):
        wall = block["wall_seconds"]
        line = (f"  arm {arm}: {block['questions']} questions, "
                f"mean {wall.get('mean', float('nan')):.2f}s total")
        if block["has_verifier_stage"]:
            line += f", verifier {block['verification_seconds'].get('mean', 0):.2f}s"
        print(line)
        for name in ("draft_stage", "verifier_stage"):
            s = block[name]
            if s["records"]:
                print(f"      {name:<15} {s['model']}  "
                      f"decode {s['eval_tokens_per_second'].get('mean', 0):.1f} tok/s  "
                      f"throttled {s['throttled_true']}/{s['records']} "
                      f"({s['throttled_unknown']} unknown)")
    h5 = report["H5"]
    if h5.get("ratio") is not None:
        print(f"\n  D/B wall-clock ratio = {h5['ratio']:.3f}  -> H5 {h5['verdict']}")
    print(f"\n  written to {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)
