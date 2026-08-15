"""Timing for RQ4 and H5. Reads performance runs and refuses quality ones.

    python scripts/analyse_performance.py --index latest_test_performance_pi5_cpu.json

H5 predicts that Arm D costs between 1.5x and 2.5x Arm B on the Pi 5: enough to
show the verification pass is doing real work, not so much that it is doing more
than one extra generation.

**This script cannot report answer quality and will not read a quality run.**
Amendment 1.15 makes the frozen laptop run the sole evidential source for H1 to
H4, and amendment 1.16 makes that structural: a run is admitted here only if its
manifest says ``purpose: performance``, and the quality analyser admits only the
four runs on its closed list. Neither can be pointed at the other's data.

The metrics are wall-clock, prefill and decode rates, load time, and the thermal
and throttle state at both ends of the run. A Pi that began at 60 degrees and
finished throttled at 85 was not one machine throughout, which is why the
environment is captured twice and reported as a range rather than a value.
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


def performance_run(directory: Path) -> tuple[dict, list[dict]]:
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
    return manifest, answers


def timings(answers: list[dict]) -> dict:
    def series(pick):
        values = [pick(r) for r in answers]
        return [v for v in values if isinstance(v, (int, float))]

    wall = series(lambda r: r.get("wall_seconds"))
    generation = [r.get("generation") or {} for r in answers]
    prefill = [g.get("prompt_tokens_per_second") for g in generation]
    decode = [g.get("eval_tokens_per_second") for g in generation]
    load = [g.get("load_seconds") for g in generation]
    temps = [g.get("cpu_temp_c") for g in generation]
    throttled = [bool(g.get("throttled")) for g in generation]

    def summarise(values, label):
        values = [v for v in values if isinstance(v, (int, float))]
        if not values:
            return {"n": 0, "note": f"{label} not recorded on this platform"}
        return {
            "n": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "questions": len(answers),
        "wall_seconds": summarise(wall, "wall clock"),
        "prompt_tokens_per_second": summarise(prefill, "prefill rate"),
        "eval_tokens_per_second": summarise(decode, "decode rate"),
        "load_seconds": summarise(load, "model load"),
        "cpu_temp_c": summarise(temps, "CPU temperature"),
        "throttled_questions": sum(throttled),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index", required=True,
                        help="run index file written by a --performance-only run")
    parser.add_argument("--out", default=str(ROOT / "results" / "analysis"))
    args = parser.parse_args(argv)

    config = load_config()
    index_path = Path(config.path("paths.results")) / args.index
    if not index_path.exists():
        print(f"No run index at {index_path}", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text(encoding="utf-8"))

    report: dict = {"index": args.index, "arms": {}, "conditions": {}}
    for arm, relative in sorted(index.items()):
        manifest, answers = performance_run(ROOT / relative)
        report["arms"][arm] = timings(answers)
        report["conditions"][arm] = {
            "hardware_condition": manifest.get("hardware_condition"),
            "placement": manifest.get("placement"),
            "host": manifest.get("host"),
            "generation_options": manifest.get("provenance", {}).get(
                "generation_options"
            ),
            "environment_start": manifest.get("environment"),
        }

    conditions = {c["hardware_condition"] for c in report["conditions"].values()}
    if len(conditions) > 1:
        raise AnalysisError(
            f"This index mixes hardware conditions {sorted(conditions)}. A "
            "latency ratio across two machines is not a measurement."
        )

    if {"B", "D"} <= set(report["arms"]):
        b = report["arms"]["B"]["wall_seconds"]["mean"]
        d = report["arms"]["D"]["wall_seconds"]["mean"]
        ratio = d / b if b else None
        report["H5"] = {
            "statement": f"latency(D) is between {H5_LOWER}x and {H5_UPPER}x latency(B)",
            "condition": conditions.pop() if conditions else None,
            "mean_wall_seconds": {"B": b, "D": d},
            "ratio": ratio,
            "verdict": (
                "supported" if ratio is not None and H5_LOWER <= ratio <= H5_UPPER
                else "not supported"
            ),
            "reading": (
                "Above the range indicates the verifier is doing more work than "
                "one extra generation; below it indicates it is not verifying "
                "much. This is a timing result and says nothing about answer "
                "quality."
            ),
        }
    else:
        report["H5"] = {"verdict": "pending",
                        "note": "needs both arm B and arm D in one index"}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"performance_{Path(args.index).stem}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    for arm, block in sorted(report["arms"].items()):
        wall = block["wall_seconds"]
        print(f"  arm {arm}: {block['questions']} questions, "
              f"mean {wall.get('mean', float('nan')):.2f}s, "
              f"median {wall.get('median', float('nan')):.2f}s, "
              f"throttled on {block['throttled_questions']}")
    if report["H5"].get("ratio") is not None:
        print(f"\n  H5 ratio D/B = {report['H5']['ratio']:.3f}"
              f"  -> {report['H5']['verdict']}")
    print(f"\n  written to {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1)
