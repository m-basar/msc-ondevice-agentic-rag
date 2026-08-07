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
import hashlib
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
    arm_frequency_hz,
    cpu_temperature_c,
    environment,
    git_commit,
    nominal_frequency_hz,
    ollama_info,
    throttle_state,
)
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
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


class BenchmarkTimeout(RuntimeError):
    """A single generation exceeded the timeout."""


def time_generation(
    base_url: str,
    model: str,
    prompt: str,
    options: dict[str, Any],
    timeout: int,
    cache_buster: str = "",
) -> dict[str, Any]:
    """Time one generation.

    ``cache_buster`` is prepended to the prompt, not appended. Ollama caches the
    KV state of a repeated prompt *prefix*. An earlier version of this function
    appended the marker, which left the entire 900-token prefix cached and only
    forced reprocessing of the ten-token suffix. The reported prompt rate then
    reached seventy thousand tokens per second, which is not a measurement of
    anything. Putting the marker first invalidates the whole prefix, so the work
    is done every time, exactly as it is for the real pipeline where every
    question differs.
    """
    if cache_buster:
        prompt = f"[run {cache_buster}]\n\n{prompt}"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    wall_start = time.perf_counter()
    try:
        response = post_json(f"{base_url}/api/generate", payload, timeout)
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        raise BenchmarkTimeout(
            f"{model} exceeded {timeout}s. On a CPU-only edge device a prompt of "
            f"{len(prompt.split())} words can take several minutes to process alone."
        ) from exc
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


def time_embedding(
    base_url: str, model: str, repeat: int, timeout: int, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Time query embedding, honouring device placement and separating cold start.

    Two bugs are fixed here. The first is that this function previously sent no
    options at all, so ``--cpu-only`` silently did not apply to the embedding
    model: a reported "CPU-only" embedding latency was in fact measured with
    the model resident in VRAM. The second is that the first call after a load
    pays model initialisation, which was being averaged in with warm calls.
    """
    prompt = "How many days of annual leave do I get?"
    payload_base: dict[str, Any] = {"model": model, "prompt": prompt}
    if options:
        payload_base["options"] = options

    unload_all(base_url)

    cold_start = time.perf_counter()
    post_json(f"{base_url}/api/embeddings", payload_base, timeout)
    cold_ms = (time.perf_counter() - cold_start) * 1000

    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        post_json(f"{base_url}/api/embeddings", payload_base, timeout)
        timings.append(time.perf_counter() - start)

    return {
        "options": options or {},
        "cold_start_ms": round(cold_ms, 1),
        "repeat": repeat,
        "warm_mean_ms": round(statistics.mean(timings) * 1000, 1),
        "warm_median_ms": round(statistics.median(timings) * 1000, 1),
        "warm_min_ms": round(min(timings) * 1000, 1),
        "warm_max_ms": round(max(timings) * 1000, 1),
        "vram_resident_fraction": model_offload(base_url, model),
    }


def unload_all(base_url: str, timeout: int = 30) -> list[str]:
    """Evict every loaded model so the next load starts from a known state.

    Ollama keeps a model resident for keep_alive after a request, *with the
    device placement it was originally loaded with*. Running a CPU-only
    benchmark and then a default-placement benchmark within that window means
    the second run silently re-measures the CPU-pinned copy. That produced a
    laptop figure for llama3.2:3b identical to its CPU-only figure, with 0% of
    the model resident in VRAM, while every other model correctly used the GPU.

    Sending keep_alive=0 evicts the model immediately.
    """
    evicted = []
    for entry in ollama_info(base_url).get("loaded") or []:
        name = entry.get("name")
        if not name:
            continue
        try:
            post_json(f"{base_url}/api/generate", {"model": name, "keep_alive": 0}, timeout)
            evicted.append(name)
        except (TimeoutError, urllib.error.URLError, OSError):
            pass
    if evicted:
        time.sleep(2)
    return evicted


def cool_down(target_c: float, max_wait_s: int) -> dict[str, Any]:
    """Wait for the CPU to drop below ``target_c`` before the next model.

    Without this the benchmark measures a thermal ramp rather than a set of
    models. In the first Pi run the board started at 63C and finished at 85C,
    actively frequency-capped, so the model benchmarked first ran cool and the
    model benchmarked last ran throttled. Any comparison between them confounds
    model with temperature.
    """
    start_temp = cpu_temperature_c()
    if start_temp is None:
        return {"skipped": "no temperature sensor"}
    waited = 0.0
    while cpu_temperature_c() > target_c and waited < max_wait_s:
        time.sleep(5)
        waited += 5
    end_temp = cpu_temperature_c()
    return {
        "start_c": start_temp,
        "end_c": end_temp,
        "waited_s": waited,
        "reached_target": end_temp is not None and end_temp <= target_c,
    }


def model_offload(base_url: str, model: str) -> float | None:
    """Fraction of THIS model resident in VRAM.

    Reported per model, not across everything Ollama has loaded. The embedding
    model may well be on the GPU while a generation model is deliberately
    pinned to CPU, and aggregating the two produced a nonsensical figure for
    the CPU-only condition.
    """
    info = ollama_info(base_url)
    for entry in info.get("loaded") or []:
        name = entry.get("name") or ""
        if name == model or name.split(":")[0] == model.split(":")[0]:
            total = entry.get("size") or 0
            vram = entry.get("size_vram") or 0
            return round(vram / total, 4) if total else None
    return None


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
    cool_to: float | None = None,
    cool_max_wait: int = 300,
) -> dict[str, Any]:
    print(f"\n  {model}")

    # Evict anything resident. A model left loaded from a previous run keeps
    # its original device placement, so without this the benchmark can silently
    # re-measure a CPU-pinned copy under a GPU-enabled configuration.
    evicted = unload_all(base_url)
    if evicted:
        print(f"    evicted {', '.join(evicted)}")

    # Warm-up. The first call pays model load cost, which would otherwise
    # dominate the first timing and skew the mean.
    print("    warm-up ...", end="", flush=True)
    try:
        warm = time_generation(base_url, model, "Say OK.", {**options, "num_predict": 5}, timeout)
    except BenchmarkTimeout as exc:
        print(f" TIMED OUT: {exc}")
        return {"model": model, "timed_out": True, "stage": "warm-up", "error": str(exc)}
    print(f" loaded in {warm['load_seconds']}s")

    short_runs, long_runs = [], []
    timed_out = None
    for i in range(repeat):
        stamp = f"{time.time_ns()}-{i}"
        # Cool before EVERY repetition, not only between models. Without this a
        # second repetition begins at the temperature the first one ended at,
        # which for this board is the throttle point.
        if cool_to is not None and i > 0:
            cooled = cool_down(cool_to, cool_max_wait)
            if not cooled.get("skipped") and not cooled["reached_target"]:
                print(f"    ABORTING: could not cool below {cool_to}C "
                      f"(stuck at {cooled['end_c']:.1f}C after {cooled['waited_s']:.0f}s)")
                timed_out = {"stage": "cooling", "run": i + 1, "error": "cool-down target not reached"}
                break
        temp_before = cpu_temperature_c()
        freq_before = arm_frequency_hz()
        print(f"    run {i + 1}/{repeat} short ...", end="", flush=True)
        try:
            short = time_generation(base_url, model, SHORT_PROMPT, options, timeout, stamp)
        except BenchmarkTimeout as exc:
            print(f" TIMED OUT")
            timed_out = {"stage": "short", "run": i + 1, "error": str(exc)}
            break
        short_runs.append(short)
        print(f" {short['eval_tokens_per_second']} tok/s", end="", flush=True)

        print("  long ...", end="", flush=True)
        try:
            long = time_generation(base_url, model, long_prompt, options, timeout, stamp)
        except BenchmarkTimeout as exc:
            print(f" TIMED OUT")
            timed_out = {"stage": "long", "run": i + 1, "error": str(exc)}
            break
        temp_after = cpu_temperature_c()
        freq_after = arm_frequency_hz()
        long["cpu_temp_before_c"] = temp_before
        long["cpu_temp_after_c"] = temp_after
        long["arm_frequency_before_hz"] = freq_before
        long["arm_frequency_after_hz"] = freq_after
        nominal = nominal_frequency_hz()
        long["arm_frequency_nominal_hz"] = nominal
        long["clock_ratio"] = round(freq_after / nominal, 3) if (freq_after and nominal) else None
        long["throttle"] = throttle_state()
        long_runs.append(long)
        temp_note = ""
        if temp_after is not None:
            throttled = (long["throttle"] or {}).get("now", {}).get("throttled")
            temp_note = f", {temp_after:.1f}C"
            if freq_after:
                temp_note += f" @{freq_after / 1e9:.2f}GHz"
            if throttled:
                temp_note += " THROTTLED"
        print(
            f" {long['eval_tokens_per_second']} tok/s eval,"
            f" {long['prompt_tokens_per_second']} tok/s prompt"
            f" ({long['prompt_tokens']} prompt tokens),"
            f" {long['wall_seconds']}s wall{temp_note}"
        )

    if not long_runs:
        return {"model": model, "timed_out": True, **(timed_out or {})}

    return {
        "model": model,
        "timed_out": bool(timed_out),
        "timeout_detail": timed_out,
        "gpu_offload_fraction": model_offload(base_url, model),
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
    parser.add_argument(
        "--cool-to", type=float, default=None,
        help="wait between models until CPU temperature falls below this, in Celsius. "
             "Without it the benchmark measures a thermal ramp rather than a set of models.")
    parser.add_argument(
        "--cool-max-wait", type=int, default=300,
        help="give up cooling after this many seconds (default 300)")
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="per-request timeout in seconds; overrides llm.timeout_seconds. "
             "A CPU-only Pi needs several hundred seconds for a single "
             "retrieval-sized prompt, so the default is often too low.")
    args = parser.parse_args()

    config = load_config()
    base_url = config.require("llm.ollama_url")
    timeout = args.timeout or config.require("llm.timeout_seconds")

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
    options.setdefault("seed", config.get("project.seed", 42))
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
    print(f"Timeout       {timeout}s per request")
    print(f"Mode          {'CPU only (num_gpu=0)' if args.cpu_only else 'default placement'}")
    print("=" * 72)

    long_prompt = build_long_prompt(config)

    results = []
    for index, model in enumerate(models):
        if args.cool_to is not None:
            print(f"\n  cooling to {args.cool_to}C ...", end="", flush=True)
            cooled = cool_down(args.cool_to, args.cool_max_wait)
            if "skipped" in cooled:
                print(f" {cooled['skipped']}")
            else:
                print(f" {cooled['start_c']:.1f} -> {cooled['end_c']:.1f}C "
                      f"after {cooled['waited_s']:.0f}s"
                      f"{'' if cooled['reached_target'] else ' (gave up)'}")
        results.append(benchmark_model(
            base_url, model, options, long_prompt, args.repeat, timeout,
            args.cool_to, args.cool_max_wait))

    embed_model = config.require("llm.embedding_model")
    print(f"\n  {embed_model} (embedding)")
    embed_options = {"num_gpu": 0} if args.cpu_only else None
    embedding = time_embedding(
        base_url, embed_model, max(args.repeat * 3, 5), timeout, embed_options)
    print(f"    cold start {embedding['cold_start_ms']} ms, "
          f"warm {embedding['warm_median_ms']} ms median over {embedding['repeat']} calls, "
          f"{(embedding['vram_resident_fraction'] or 0):.0%} in VRAM")

    after = environment(base_url)

    print("\n" + "=" * 72)
    print(f"{'Model':<16} {'short eval':>11} {'long eval':>11} {'long prompt':>12} {'long wall':>10}")
    for entry in results:
        if entry.get("timed_out") and "short" not in entry:
            print(f"{entry['model']:<16} {'TIMED OUT at ' + str(timeout) + 's':>45}")
            continue
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
    for entry in results:
        frac = entry.get("gpu_offload_fraction")
        if frac is not None:
            print(f"  {entry['model']:<16} {frac:.0%} resident in VRAM")
    temp_after = cpu_temperature_c()
    if temp_after is not None:
        print(f"CPU temp      {temp_after:.1f} C (after)")
    throttle = throttle_state()
    if throttle:
        active = [k for k, v in throttle["now"].items() if v]
        historic = [k for k, v in throttle["since_boot"].items() if v]
        print(f"Throttle      now={active or 'none'}  since boot={historic or 'none'}")

    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(load_evaluation_config().path("conflicts"))
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prompt_sha = hashlib.sha256(long_prompt.encode("utf-8")).hexdigest()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "provenance": {
            "config_sha256": config.fingerprint(),
            "corpus_sha256": kb.fingerprint(),
            "conflict_registry_sha256": registry.fingerprint(),
            "benchmark_script_sha256": script_sha,
            "long_prompt_sha256": prompt_sha,
            "long_prompt_tokens_reported_by_server": results[0].get("long", {}).get("prompt_tokens")
                if results and "long" in results[0] else None,
            "git": git_commit(),
            "seed": config.get("project.seed"),
            "command": " ".join(sys.argv),
            "arguments": vars(args),
        },
        "config_fingerprint": config.fingerprint(),
        "generation_options": options,
        "cpu_only": args.cpu_only,
        "environment_before": before,
        "environment_after": after,
        # Per model, never aggregated. Summing residency across everything Ollama
        # has loaded once made a CPU-only run appear 46% GPU-offloaded, because
        # the embedding model was still on the GPU.
        "gpu_offload_by_model": {
            entry["model"]: entry.get("gpu_offload_fraction")
            for entry in results
        },
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
