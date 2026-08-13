"""Is the verifier reproducible at a fixed seed? Measure it rather than assume.

Pilot 03 replayed Arm B's pilot 02 drafts, so the 41 verification prompts were
byte-identical to the ones pilot 02 sent. The model, the model store
fingerprint, the Ollama version, the host and the seed were also identical.

**24 of the 41 produced different output.**

Everything downstream rests on that not being true. A single run of the test
split is presented as the result; if the same prompt yields different findings
on different days, a detection count is one sample from a distribution nobody
has measured, and the B versus D contrast carries noise that was never in the
error budget.

Three explanations, and this script separates them:

``sampling``   the model is not deterministic even within one session, despite
               ``seed`` and ``temperature: 0``. Response: quantify the
               variability and report every figure as a mean over repeats.
``session``    deterministic within a session but not across model loads, which
               is a known consequence of batching and reduction order in
               llama.cpp. Response: the same, plus record the load boundary.
``options``    pilot 02 posted different options. It predates option recording,
               so this cannot be ruled out from the record alone - which is
               itself the finding that the instrumentation existed to prevent.

    python scripts/check_determinism.py
    python scripts/check_determinism.py --repeats 3 --questions 12

The comparison against the recorded output is the across-run test. The repeats
within this process are the within-session test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import build_client  # noqa: E402
from sme_assistant.evaluation.run_writer import read_run  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="run directory; defaults to the latest D pilot03")
    parser.add_argument("--repeats", type=int, default=2,
                        help="calls per prompt within this session")
    parser.add_argument("--questions", type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    directory = Path(args.run) if args.run else latest_run(config, "*_D_dev_*")
    if not directory.is_absolute():
        directory = ROOT / directory
    manifest, records = read_run(directory)
    records = [r for r in records if r.get("verification_prompt")][:args.questions]
    if not records:
        raise SystemExit(f"{directory.name} has no recorded verification prompts.")

    client = build_client(config)
    options = {k: v for k, v in config.require("verification").items()
               if k != "confidence" and not k.startswith("_")}
    model = config.get("llm.verification_model")

    print(f"Run        {directory.name}")
    print(f"Model      {model}")
    print(f"Options    {json.dumps(options, sort_keys=True)}")
    print(f"Prompts    {len(records)}, {args.repeats} repeats each\n")
    print(f"{'question':<32}{'recorded':<10}" +
          "".join(f"{'call ' + str(i + 1):<10}" for i in range(args.repeats)) +
          "verdict")
    print("-" * (42 + 10 * args.repeats + 20))

    rows, within, across = [], Counter(), Counter()
    for record in records:
        prompt = record["verification_prompt"]
        recorded = (record["verification"].get("raw") or "").strip()
        outputs = [
            client.generate(prompt, model=model, options=options).text.strip()
            for _ in range(args.repeats)
        ]

        stable_within = len(set(outputs)) == 1
        matches_recorded = bool(recorded) and all(o == recorded for o in outputs)
        within["stable" if stable_within else "varies"] += 1
        across["matches" if matches_recorded else "differs"] += 1

        verdict = ("reproducible" if stable_within and matches_recorded else
                   "differs across runs" if stable_within else
                   "VARIES WITHIN SESSION")
        print(f"{record['question_id']:<32}{digest(recorded):<10}"
              + "".join(f"{digest(o):<10}" for o in outputs) + verdict)

        rows.append({
            "question_id": record["question_id"],
            "prompt_sha256": digest(prompt),
            "recorded_sha256": digest(recorded),
            "repeat_sha256": [digest(o) for o in outputs],
            "stable_within_session": stable_within,
            "matches_recorded_run": matches_recorded,
            "outputs": outputs,
            "recorded": recorded,
        })

    print(f"\nWithin this session   stable {within['stable']}/{len(rows)}")
    print(f"Against the recorded run   matches {across['matches']}/{len(rows)}")
    print()
    if within["varies"]:
        print("  The model is not deterministic even within one session, at "
              "seed and temperature 0.")
        print("  Every reported figure must be a mean over repeats, with the "
              "spread stated.")
    elif across["differs"]:
        print("  Deterministic within a session, not across them. This is the "
              "known llama.cpp behaviour")
        print("  where reduction order depends on batching and model load "
              "state. Results are")
        print("  reproducible only within a single process, which has to be "
              "said rather than implied.")
    else:
        print("  Reproducible. The pilot 02 to pilot 03 differences then come "
              "from the options,")
        print("  which pilot 02 did not record - itself the gap the "
              "instrumentation closed.")

    out = Path(config.path("paths.results")).parent / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"{stamp}_determinism.json"
    path.write_text(json.dumps({
        "run": directory.name,
        "model": model,
        "options": options,
        "repeats": args.repeats,
        "environment": manifest.get("environment"),
        "within_session": dict(within),
        "against_recorded": dict(across),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
