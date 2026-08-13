"""Does re-running the verifier change the numbers, or only the prose?

Those are different questions and the first version of this script could not
tell them apart. It repeated each prompt three times back to back, compared
sha256 of the raw text, and called a prompt reproducible only when all three
calls equalled the recorded run. That conflated two things and hid the most
important fact in its own output: **every first call matched the recording**.

What the first version actually established:

    call 1 == the recorded pilot 03 run        12 / 12
    calls 2 and 3 agree with each other        12 / 12
    output changes under immediate repetition   4 / 12

which is evidence *for* cross-session reproducibility of a fresh prompt, and
evidence of state carried between adjacent calls on the same prompt. It says
nothing about whether any reported metric moves, because a reordered JSON key
or a reworded rationale changes the hash and changes no result.

So this version:

* runs **complete passes** in protocol order, never adjacent repeats, so a
  repeat meets the server in the same state a real run does;
* replays the **options and model recorded in the run**, not whatever the
  config says today, because the config has changed since;
* **parses** every response and compares the outcomes that are reported, one
  at a time, rather than hashing the text and inferring.

    python scripts/check_determinism.py --repeats 3

Two comparisons come out of it, and they answer different questions:

``pass 1 vs the recorded run``   is a fresh prompt reproducible across sessions?
``pass to pass``                 is it reproducible within one session?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import build_client  # noqa: E402
from sme_assistant.evaluation.run_writer import read_run  # noqa: E402
from sme_assistant.generate.generator import extract_citations  # noqa: E402
from sme_assistant.verify import schema  # noqa: E402

# Every outcome that reaches a reported figure, compared separately. A single
# "reproducible" number would average a stable primary metric together with a
# volatile rationale and describe neither.
OUTCOMES = (
    "raw_sha256",
    "relationship",
    "conflict_detected",
    "verdicts",
    "parse_failed",
    "validation_failures",
    "invented_ids",
    "revision_served",
    "final_answer_sha256",
    "citations",
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def latest_run(config, pattern: str) -> Path:
    """The most recent run directory matching ``pattern``.

    ``paths.results`` already points at ``results/runs``; an earlier version
    globbed ``runs/*`` beneath it and matched nothing.
    """
    root = Path(config.path("paths.results"))
    runs = sorted(p for p in root.glob(pattern) if p.is_dir())
    if not runs:
        available = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise SystemExit(
            f"No run directory matching {pattern!r} in {root}.\n"
            + ("Available:\n" + "\n".join(f"  {n}" for n in available[-10:])
               if available else "  (none)")
            + "\n\nPass --run with a directory to check a specific one."
        )
    return runs[-1]


def outcomes_of(raw: str, record: dict) -> dict:
    """Parse one response into the outcomes that are actually reported."""
    available = [s["chunk_id"] for s in record["retrieval"]["results"]]
    draft = record.get("draft_answer") or record.get("answer") or ""
    result = schema.parse(raw, available, {}, draft=draft)
    served = result.final_answer or draft
    citations, _ = extract_citations(served)
    return {
        "raw_sha256": digest(raw),
        "relationship": result.relationship,
        "conflict_detected": result.conflict_detected,
        "verdicts": tuple(sorted(v.verdict for v in result.verdicts)),
        "parse_failed": result.parse_failed,
        "validation_failures": tuple(sorted(result.validation_failures)),
        "invented_ids": tuple(sorted(result.invented_ids)),
        "revision_served": bool(result.revised),
        "final_answer_sha256": digest(served),
        "citations": tuple(citations),
    }


def replay_options(record: dict, manifest: dict, config) -> tuple[str, dict]:
    """The model and options the run actually used, falling back loudly.

    Using today's configuration to re-run yesterday's prompts would test a
    different system and call the difference non-determinism.
    """
    options = record.get("verification_options")
    model = (record.get("verification_generation") or {}).get("model")
    model = model or (manifest.get("arm") or {}).get("verification_model")
    if not options:
        options = {k: v for k, v in config.require("verification").items()
                   if k != "confidence" and not k.startswith("_")}
        print("  WARNING: this run predates option recording. Falling back to "
              "the current config,\n           so an options difference cannot "
              "be ruled out as the cause of any change.\n")
    return model, dict(options)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="run directory; defaults to the latest dev D run")
    parser.add_argument("--repeats", type=int, default=3, help="complete passes")
    parser.add_argument("--questions", type=int, default=0,
                        help="limit the prompts; 0 means all of them")
    args = parser.parse_args()

    config = load_config()
    directory = Path(args.run) if args.run else latest_run(config, "*_D_dev_*")
    if not directory.is_absolute():
        directory = ROOT / directory
    manifest, records = read_run(directory)
    records = [r for r in records if r.get("verification_prompt")]
    if args.questions:
        records = records[:args.questions]
    if not records:
        raise SystemExit(f"{directory.name} has no recorded verification prompts.")

    client = build_client(config)
    model, options = replay_options(records[0], manifest, config)

    print(f"Run        {directory.name}")
    print(f"Model      {model}   (as recorded, not from the current config)")
    print(f"Options    {json.dumps(options, sort_keys=True)}")
    print(f"Prompts    {len(records)}")
    print(f"Passes     {args.repeats}, complete and in protocol order\n")

    observed: dict[str, list[dict]] = defaultdict(list)
    for repeat in range(1, args.repeats + 1):
        for position, record in enumerate(records, start=1):
            generation = client.generate(
                record["verification_prompt"], model=model, options=options
            )
            observed[record["question_id"]].append(
                outcomes_of(generation.text.strip(), record)
            )
            if position % 10 == 0 or position == len(records):
                print(f"  pass {repeat}: {position}/{len(records)}")
        print()

    recorded = {
        r["question_id"]: outcomes_of(
            (r["verification"].get("raw") or "").strip(), r
        ) for r in records
    }

    # --- the two comparisons, kept apart -------------------------------------
    print(f"{'outcome':<24}{'pass 1 vs recorded':>20}{'pass to pass':>16}")
    print("-" * 60)
    summary = {}
    for field in OUTCOMES:
        across_sessions = sum(
            observed[qid][0][field] == recorded[qid][field] for qid in recorded
        )
        within_session = sum(
            len({tuple(str(o[field])) for o in observed[qid]}) == 1
            for qid in recorded
        )
        n = len(recorded)
        summary[field] = {
            "pass1_matches_recorded": across_sessions,
            "stable_across_passes": within_session,
            "n": n,
        }
        mark = "" if within_session == n and across_sessions == n else "   <--"
        print(f"{field:<24}{f'{across_sessions}/{n}':>20}"
              f"{f'{within_session}/{n}':>16}{mark}")

    n = len(recorded)
    detection_stable = (summary["conflict_detected"]["stable_across_passes"] == n
                        and summary["conflict_detected"]["pass1_matches_recorded"] == n)
    metrics = [f for f in OUTCOMES if f != "raw_sha256"]
    outcomes_stable = all(
        summary[f]["stable_across_passes"] == n
        and summary[f]["pass1_matches_recorded"] == n for f in metrics
    )
    prose_varies = summary["raw_sha256"]["stable_across_passes"] < n

    print()
    if outcomes_stable and prose_varies:
        print("  Raw text varies; every reported outcome is stable.")
        print("  Retain the 96-call protocol and document prose-level "
              "variability as a finding.")
    elif outcomes_stable:
        print("  Reproducible on every axis measured, including the raw text.")
        print("  Retain the 96-call protocol.")
    else:
        moved = [f for f in metrics if summary[f]["stable_across_passes"] < n
                 or summary[f]["pass1_matches_recorded"] < n]
        print(f"  Reported outcomes move between runs: {', '.join(moved)}.")
        print("  Amend the protocol to three complete 96-call blocks. Apply "
              "R0 to R5 independently")
        print("  to each block and report the three block outcomes with their "
              "mean and range.")
        print("  288 calls are three blocks, not 288 independent observations.")
        if detection_stable:
            print("\n  Conflict detection itself did not move. Say that "
                  "specifically rather than")
            print("  describing the whole system as unreproducible.")

    out = Path(config.path("paths.results")).parent / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"{stamp}_determinism_passes.json"
    path.write_text(json.dumps({
        "schedule": "complete passes in protocol order, not adjacent repeats",
        "run": directory.name,
        "model": model,
        "options": options,
        "passes": args.repeats,
        "environment": manifest.get("environment"),
        "summary": summary,
        "recorded": {k: {f: str(v[f]) for f in OUTCOMES} for k, v in recorded.items()},
        "observed": {k: [{f: str(o[f]) for f in OUTCOMES} for o in v]
                     for k, v in observed.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
