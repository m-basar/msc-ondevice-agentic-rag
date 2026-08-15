"""Timing for RQ4 and H5. Reads performance runs, validates them, refuses quality ones.

    python scripts/analyse_performance.py --index latest_test_performance_pi5_cpu.json

H5 predicts that Arm D costs between 1.5x and 2.5x Arm B **on the Pi 5**: enough
to show the verification pass is doing real work, not so much that it is doing
more than one extra generation.

Three things this script gets right that an obvious version does not.

**Arm D is two stages, not one.** D replays B's draft and then verifies it, so
``record["generation"]`` on a D record describes the *reused draft*, generated
earlier on another machine. Reading only that field reports B's prefill, decode,
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
from sme_assistant.evaluation.run_writer import read_run  # noqa: E402

H5_LOWER, H5_UPPER = 1.5, 2.5
H5_CONDITION = "pi5_cpu"
EXPECTED_QUESTIONS = 68


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


def validate(runs: dict[str, tuple[dict, list[dict], dict]], index_meta: dict) -> dict:
    """Independent checks, before any number is computed.

    None of these is hypothetical. A run of the wrong arm, over the wrong
    questions, against a different corpus, on a device other than the one
    requested, or with D generating fresh drafts instead of replaying B's,
    would all produce a plausible ratio that means something other than H5.
    """
    findings: list[str] = []
    detail: dict = {}

    for arm, (manifest, answers, _) in sorted(runs.items()):
        declared = manifest.get("arm", {}).get("arm")
        if declared != arm:
            findings.append(
                f"index lists {arm} but the manifest says arm {declared!r}"
            )
        if manifest.get("split") != "test":
            findings.append(f"{arm}: split is {manifest.get('split')!r}, not test")
        if len(answers) != EXPECTED_QUESTIONS:
            findings.append(
                f"{arm}: {len(answers)} answers, expected {EXPECTED_QUESTIONS}"
            )

    question_sets = {
        arm: [r["question_id"] for r in answers]
        for arm, (_, answers, _) in runs.items()
    }
    if len(runs) > 1:
        reference_arm, reference = next(iter(sorted(question_sets.items())))
        for arm, ids in sorted(question_sets.items()):
            if sorted(ids) != sorted(reference):
                findings.append(
                    f"{arm} answered a different question set from {reference_arm}"
                )
    detail["question_ids_match"] = not any("question set" in f for f in findings)

    provenance_keys = ("corpus_sha256", "chunk_set_sha256", "question_set_sha256",
                       "registry_sha256", "config_sha256")
    hashes = {
        arm: {k: (manifest.get("provenance") or {}).get(k) for k in provenance_keys}
        for arm, (manifest, _, _) in runs.items()
    }
    if len({json.dumps(h, sort_keys=True) for h in hashes.values()}) > 1:
        findings.append("the runs disagree on their provenance hashes")
    detail["provenance"] = hashes

    requested = index_meta.get("requested_placement")
    observed = index_meta.get("observed_placement") or {}
    if requested and observed:
        seen = "gpu" if observed.get("any_on_gpu") else "cpu"
        detail["placement"] = {"requested": requested, "observed": seen}
        if seen != requested:
            findings.append(
                f"placement was {seen} but {requested} was requested"
            )
    else:
        detail["placement"] = {"requested": requested, "observed": None}
        findings.append("placement was not observed at run time")

    if "D" in runs:
        summary = runs["D"][2]
        reused = summary.get("drafts_reused_from") or summary.get(
            "drafts_replayed_from"
        )
        detail["arm_d_replayed_drafts_from"] = reused
        if not reused:
            findings.append(
                "arm D did not replay arm B's drafts, so B versus D is not the "
                "same comparison as the frozen run"
            )

    detail["findings"] = findings
    detail["valid"] = not findings
    return detail


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

    report: dict = {
        "index": args.index,
        "hardware_condition": index_meta.get("hardware_condition"),
        "validation": validate(runs, index_meta),
        "arms": {arm: timings(*runs[arm]) for arm in sorted(runs)},
    }

    if not report["validation"]["valid"]:
        print("Refusing to report. Validation failed:", file=sys.stderr)
        for finding in report["validation"]["findings"]:
            print(f"  - {finding}", file=sys.stderr)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"performance_{Path(args.index).stem}_REJECTED.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8", newline="\n"
        )
        return 1

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
