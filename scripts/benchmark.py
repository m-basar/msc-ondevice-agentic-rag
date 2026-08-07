"""Benchmark a machine's Ollama inference throughput.

Runs identically on Windows and on the Raspberry Pi, measures through the
same HTTP API the pipeline itself uses, and pins every generation option from
config.json. The point is that the numbers it produces are the numbers the
real pipeline will see, rather than whatever defaults the `ollama run` CLI
happens to apply.

Three measurements are taken:

  short   a 37-token prompt, which mostly measures generation speed
  long    a prompt built from real knowledge base chunks, roughly the size of
          an actual retrieval-augmented request; this is what dominates
          latency in practice and is the number the CLI cannot give you
  embed   query embedding latency, paid once per question

Results, together with the full hardware and Ollama environment, are written
to results/benchmarks/ so they can be cited in the dissertation rather than
retyped from a terminal.

    python scripts/benchmark.py
    python scripts/benchmark.py --model qwen2.5:3b --repeat 5
    python scripts/benchmark.py --all-models --tag pi5_baseline
    python scripts/benchmark.py --cpu-only --tag laptop_cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.hostinfo import (  # noqa: E402
    cpu_temperature_c,
    environment,
    gpu_offload_fraction,
    ollama_info,
    throttle_state,
)
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402

SHORT_PROMPT = "Write a 150 word summary of workplace fire safety procedures."


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_long_prompt(config) -> str:
    """A prompt the size of a real retrieval-augmented request.

    Built from actual corpus documents rather than lorem ipsum, so the token
    count and vocabulary match what the pipeline will really send.
    """
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    top_k = config.require("retrieval.top_k")
    selected = sorted(kb, key=lambda d: d.doc_id)[:top_k]
    evidence = "\n\n".join(
        f"[{doc.doc_id}] {doc.title}\n{doc.body[:900]}" for doc in selected
    )
    return (
        "Answer the question using only the evidence below. Cite the document "
        "identifier for every claim.\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        "QUESTION: What are the reporting deadlines that staff must observe?\n\nANSWER:"
    )


def time_generation(
    base_url: str, model: str, prompt: str, options: dict[str, Any], timeout: int
) -> dict[str, Any]:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    wall_start = time.perf_counter()
    response = post_json(f"{base_url}/api/generate", payload, timeout)
    wall = time.perf_counter() - wall_start

    def rate(count_key: str, duration_key: str) -> float | None:
        count = response.get(count_key)
        duration = response.get(duration_key)
        if not count or not duration:
            return None
        return round(count / (duration / 1e9), 2)

    return {
        "wall_seconds": round(wall, 3),
        "load_seconds": round((response.get("load_duration") or 0) / 1e9, 3),
        "prompt_tokens": response.get("prompt_eval_count"),
        "prompt_tokens_per_second": rate("prompt_eval_count", "prompt_eval_duration"),
        "eval_tokens": response.get("eval_count"),
        "eval_tokens_per_second": rate("eval_count", "eval_duration"),
        "total_seconds": round((response.get("total_duration") or 0) / 1e9, 3),
    }


def time_embedding(base_url: str, model: str, repeat: int, timeout: int) -> dict[str, Any]:
    prompt = "How many days of annual leave do I get?"
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        post_json(f"{base_url}/api/embeddings", {"model": model, "prompt": prompt}, timeout)
        timings.append(time.perf_counter() - start)
    return {
        "repeat": repeat,
        "mean_ms": round(statistics.mean(timings) * 1000, 1),
        "median_ms": round(statistics.median(timings) * 1000, 1),
        "min_ms": round(min(timings) * 1000, 1),
        "max_ms": round(max(timings) * 1000, 1),
    }


def summarise(runs: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = [r[key] for r in runs if r.get(key) is not None]
    if not values:
        return None
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "n": len(values),
    }


def benchmark_model(
    base_url: str,
    model: str,
    options: dict[str, Any],
    long_prompt: str,
    repeat: int,
    timeout: int,
) -> dict[str, Any]:
    print(f"\n  {model}")

    # Warm-up. The first call pays model load cost, which would otherwise
    # dominate the first timing and skew the mean.
    print("    warm-up ...", end="", flush=True)
    warm = time_generation(base_url, model, "Say OK.", {**options, "num_predict": 5}, timeout)
    print(f" loaded in {warm['load_seconds']}s")

    short_runs, long_runs = [], []
    for i in range(repeat):
        print(f"    run {i + 1}/{repeat} short ...", end="", flush=True)
        short = time_generation(base_url, model, SHORT_PROMPT, options, timeout)
        short_runs.append(short)
        print(f" {short['eval_tokens_per_second']} tok/s", end="", flush=True)

        print("  long ...", end="", flush=True)
        long = time_generation(base_url, model, long_prompt, options, timeout)
        long_runs.append(long)
        print(
            f" {long['eval_tokens_per_second']} tok/s eval,"
            f" {long['prompt_tokens_per_second']} tok/s prompt"
            f" ({long['prompt_tokens']} prompt tokens),"
            f" {long['wall_seconds']}s wall"
        )

    return {
        "model": model,
        "short": {
            "eval_tokens_per_second": summarise(short_runs, "eval_tokens_per_second"),
            "wall_seconds": summarise(short_runs, "wall_seconds"),
            "runs": short_runs,
        },
        "long": {
            "prompt_tokens": long_runs[0]["prompt_tokens"],
            "eval_tokens_per_second": summarise(long_runs, "eval_tokens_per_second"),
            "prompt_tokens_per_second": summarise(long_runs, "prompt_tokens_per_second"),
            "wall_seconds": summarise(long_runs, "wall_seconds"),
            "runs": long_runs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="single model to benchmark")
    parser.add_argument("--all-models", action="store_true", help="benchmark every candidate model")
    parser.add_argument("--repeat", type=int, default=3, help="timed runs per model (default 3)")
    parser.add_argument("--cpu-only", action="store_true", help="force CPU execution with num_gpu=0")
    parser.add_argument("--tag", default="", help="label recorded in the results filename")
    parser.add_argument("--no-save", action="store_true", help="print only, do not write a results file")
    args = parser.parse_args()

    config = load_config()
    base_url = config.require("llm.ollama_url")
    timeout = config.require("llm.timeout_seconds")

    probe = ollama_info(base_url)
    if not probe.get("reachable"):
        print(f"Ollama is not reachable at {base_url}: {probe.get('error')}", file=sys.stderr)
        return 1

    if args.all_models:
        models = list(config.require("llm.candidate_models"))
    elif args.model:
        models = [args.model]
    else:
        models = [config.require("llm.generation_model")]

    available = {m["name"] for m in probe.get("models", [])}
    available |= {name.split(":")[0] for name in available}
    missing = [m for m in models if m not in available and m.split(":")[0] not in available]
    if missing:
        print(f"Models not present on this server: {missing}", file=sys.stderr)
        print(f"Server holds: {sorted(m['name'] for m in probe.get('models', []))}", file=sys.stderr)
        return 1

    options = dict(config.require("generation"))
    if args.cpu_only:
        # num_gpu is the number of layers offloaded to the GPU. Zero forces
        # CPU execution without restarting the server or touching environment
        # variables, so condition A and condition B differ by one key.
        options["num_gpu"] = 0

    before = environment(base_url)
    host = before["host"]

    print("=" * 72)
    print(f"Host          {host['hostname']}  {host['system']} {host['machine']}")
    print(f"CPU cores     {host['cpu_count']}   RAM {host['memory_total_gb']} GB")
    if host["gpus"]:
        for gpu in host["gpus"]:
            print(f"GPU           {gpu['name']}  {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB used")
    else:
        print("GPU           none detected")
    if host["power_profile"]:
        print(f"Power profile {host['power_profile']}")
    if host["cpu_temperature_c"] is not None:
        print(f"CPU temp      {host['cpu_temperature_c']:.1f} C (before)")
    print(f"Ollama        {probe.get('version')} at {base_url}")
    print(f"Model store   fingerprint {probe.get('model_store_fingerprint')}")
    print(f"Options       {json.dumps(options)}")
    print(f"Mode          {'CPU only (num_gpu=0)' if args.cpu_only else 'default placement'}")
    print("=" * 72)

    long_prompt = build_long_prompt(config)

    results = []
    for model in models:
        results.append(benchmark_model(base_url, model, options, long_prompt, args.repeat, timeout))

    embed_model = config.require("llm.embedding_model")
    print(f"\n  {embed_model} (embedding)")
    embedding = time_embedding(base_url, embed_model, max(args.repeat * 3, 5), timeout)
    print(f"    {embedding['median_ms']} ms median over {embedding['repeat']} calls")

    after = environment(base_url)
    offload = gpu_offload_fraction(after["ollama"])

    print("\n" + "=" * 72)
    print(f"{'Model':<16} {'short eval':>11} {'long eval':>11} {'long prompt':>12} {'long wall':>10}")
    for entry in results:
        short = entry["short"]["eval_tokens_per_second"]
        long_eval = entry["long"]["eval_tokens_per_second"]
        long_prompt_rate = entry["long"]["prompt_tokens_per_second"]
        wall = entry["long"]["wall_seconds"]
        print(
            f"{entry['model']:<16} "
            f"{(short['median'] if short else 0):>8.2f}t/s "
            f"{(long_eval['median'] if long_eval else 0):>8.2f}t/s "
            f"{(long_prompt_rate['median'] if long_prompt_rate else 0):>9.2f}t/s "
            f"{(wall['median'] if wall else 0):>8.2f}s"
        )
    print("=" * 72)
    if offload is not None:
        print(f"GPU offload   {offload:.0%} of loaded model resident in VRAM")
    temp_after = cpu_temperature_c()
    if temp_after is not None:
        print(f"CPU temp      {temp_after:.1f} C (after)")
    throttle = throttle_state()
    if throttle:
        active = [k for k, v in throttle["now"].items() if v]
        historic = [k for k, v in throttle["since_boot"].items() if v]
        print(f"Throttle      now={active or 'none'}  since boot={historic or 'none'}")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "config_fingerprint": config.fingerprint(),
        "generation_options": options,
        "cpu_only": args.cpu_only,
        "environment_before": before,
        "environment_after": after,
        "gpu_offload_fraction": offload,
        "models": results,
        "embedding": {"model": embed_model, **embedding},
    }

    if args.no_save:
        return 0

    out_dir = config.path("paths.results").parent / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"{stamp}_{host['hostname']}{label}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
