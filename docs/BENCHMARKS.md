# Hardware Benchmarking

Baseline inference measurements for both machines. Written for Chapter 3 (experimental setup) and Chapter 5 (analysis).

**Every figure here comes from a run classified VALID in `results/benchmarks/README.md`.** Five earlier runs are retained in that directory but are marked SUPERSEDED and must not be cited: they predate one or more of the measurement corrections described at the end of this document.

Produced by `scripts/benchmark.py`, measured through the same HTTP API the pipeline uses, with generation options pinned from `config.json`. Every run records corpus, registry, config, script and prompt SHA-256, git commit and working-tree state, seed, and the exact command.

| | Value |
|---|---|
| Runs cited | `20260807_181428_Basar_laptop_gpu_v2`, `20260807_181611_Basar_laptop_cpu_v2`, `20260807_191119_agi-pi_pi5_v2` |
| Git commit | `ca511829602f` |
| Corpus SHA-256 | `dd51cf741e59...` (identical across all three; **historical**, the corpus these runs used, since changed) |
| Seed | 42 |
| Ollama | 0.32.6 on both machines |
| Repetitions | 5 per model, prompt cache defeated on every one |

## Test conditions

| | Laptop | Raspberry Pi 5 |
|---|---|---|
| CPU | AMD Ryzen 7 7840HS, 8C/16T | Cortex-A76, 4C, 2.4 GHz nominal |
| RAM | 15.3 GB | 15.8 GB |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB | None |
| Architecture | x86_64 | aarch64 |
| Cooling | Laptop internal | Official Active Cooler, `FAN_TEMP0..3` lowered to 45/55/62/68°C |

Prompt: 903 tokens of real knowledge base evidence under the Llama tokenizer, matching an actual retrieval-augmented request. `num_ctx: 4096`, `num_predict: 400`, `temperature: 0.1`.

## Laptop, GPU

All models 100% resident in VRAM.

| Model | Prompt | Generation | Wall | Prompt tokens |
|---|---|---|---|---|
| llama3.2:1b | 12,299.77 tok/s | 118.24 tok/s | 0.89s | 903 |
| gemma:2b | 6,392.37 tok/s | 97.95 tok/s | 0.87s | 960 |
| llama3.2:3b | 4,434.80 tok/s | 76.19 tok/s | 1.26s | 903 |
| qwen2.5:3b | 4,124.34 tok/s | 78.39 tok/s | 1.62s | 952 |
| phi3:mini | 3,675.84 tok/s | 60.32 tok/s | 5.75s | 1,160 |

Variance across five repetitions is small: gemma:2b ranged 97.60 to 98.12 tok/s generation.

## Laptop, CPU only (`num_gpu: 0`)

| Model | Prompt | Generation | Wall | VRAM |
|---|---|---|---|---|
| llama3.2:3b | 183.91 tok/s | 22.50 tok/s | 7.96s | 0% |

## Raspberry Pi 5, CPU only

Cool-down to 65°C before every repetition. Temperature and ARM clock recorded per request.

| Model | Prompt | Generation | Wall | Temp range | Clock range |
|---|---|---|---|---|---|
| gemma:2b | 54.42 tok/s | 5.24 tok/s | **27.90s** | 83.2-84.2°C | 2.15-2.37 GHz |
| llama3.2:1b | 49.50 tok/s | 6.45 tok/s | 29.99s | 82.0-83.7°C | 2.20-2.37 GHz |
| phi3:mini | 26.23 tok/s | 2.86 tok/s | 154.10s | 83.2-83.7°C | 1.50-2.31 GHz |
| qwen2.5:3b | 25.29 tok/s | 4.19 tok/s | 62.14s | 83.2-84.8°C | 1.50-2.20 GHz |
| llama3.2:3b | 24.54 tok/s | 3.66 tok/s | 54.34s | 83.2-84.8°C | 1.50-2.26 GHz |

Throttle flags were clear at end of run (`now=none`).

## Embedding, `nomic-embed-text`

| Machine | Placement | Cold start | Warm median |
|---|---|---|---|
| Laptop | GPU (100% VRAM) | 1,073.7 ms | 25.1 ms |
| Laptop | CPU (0% VRAM) | 315.7 ms | 27.2 ms |
| Pi 5 | CPU | 847.9 ms | 71.6 ms |

## Findings

### 1. Software configuration dominated hardware capability

Pi prompt processing for llama3.2:3b, across three configurations of the same board:

| Configuration | Prompt | Wall | Change |
|---|---|---|---|
| Ollama 0.18.3, stock fan curve | 5.67 tok/s | 181.5s | baseline |
| Ollama 0.32.6, stock fan curve | 21.27 tok/s | 62.3s | 3.75x |
| Ollama 0.32.6, corrected fan curve | 24.54 tok/s | 54.3s | **4.33x** |

A runtime upgrade and four EEPROM values produced a 4.3x improvement with no hardware change. The stock fan curve held the fan at cooling state 2 of 4 while the CPU sat at the 85°C hard limit, capping the ARM clock at 1.70 GHz against a 2.4 GHz nominal.

Any edge benchmark that does not pin and report the inference runtime version and the thermal configuration is close to meaningless.

### 2. Prefill is penalised far more than decode on accelerator-free hardware

llama3.2:3b, laptop GPU against laptop CPU:

| | GPU | CPU | Ratio |
|---|---|---|---|
| Prompt processing | 4,434.80 tok/s | 183.91 tok/s | **24.1x** |
| Generation | 76.19 tok/s | 22.50 tok/s | 3.4x |

And laptop GPU against Pi 5:

| | Laptop GPU | Pi 5 | Ratio |
|---|---|---|---|
| Prompt processing | 4,434.80 tok/s | 24.54 tok/s | **180.7x** |
| Generation | 76.19 tok/s | 3.66 tok/s | 20.8x |

The prompt gap is roughly nine times wider than the generation gap in both comparisons. Prefill processes all tokens in parallel and is compute-bound; decode emits one token at a time and is memory-bandwidth-bound. Parallel hardware can only help the first.

The consequence for this project: **retrieval-augmented workloads are disproportionately penalised at the edge**, because they are prompt-heavy by construction. Moving RAG to an edge device is harder than moving plain chat, and this is why.

It also makes `top_k` and chunk size first-order design decisions rather than tuning details.

### 3. Parameter count does not predict edge latency

On the Pi, `gemma:2b` completes in 27.90s while `llama3.2:1b` takes 29.99s, despite having twice the parameters, because its prompt processing is faster (54.42 against 49.50 tok/s).

`phi3:mini` is the clearest case. The same evidence encodes to 903 tokens under the Llama tokenizer, 952 under Qwen, 960 under Gemma and **1,160 under Phi-3**: a 28% penalty on every call. Combined with the slowest generation rate, it takes 154s per request against 28s for gemma:2b, **5.5x slower**.

Tokenizer efficiency and prefill throughput matter more than parameter count for edge deployment.

### 4. There is no reason to put the embedding model on the GPU

Warm embedding latency differs by 2.1 ms between GPU and CPU on the laptop, while cold start is 3.4x worse on GPU. For a 274 MB model and a short query, transfer overhead cancels any compute benefit. Keeping the embedding model on CPU frees VRAM for generation at no measurable cost.

This only became visible once cold start was separated from warm.

## Threats to measurement validity encountered

Each produced plausible-looking but wrong numbers. Documented because none is obvious, and because a benchmark that looks reasonable is the hardest kind of error to catch.

| # | Error | How it presented | Detection | Mitigation |
|---|---|---|---|---|
| 1 | **Endpoint substitution.** VS Code Remote-SSH auto-forwarded the Pi's Ollama port to the laptop's localhost, so all "laptop" measurements executed on the Pi. | Laptop and Pi reported near-identical rates. Model loaded 100% CPU despite a working GPU. | The Pi's fan audibly spinning up during a laptop command. | Auto port forwarding disabled. `model_store_fingerprint` recorded per run: a hash of the responding server's model digests, so substitution is visible in the results file. |
| 2 | **Runtime version mismatch** between machines, 0.18.3 against 0.32.6. | Different prompt-cache behaviour, invalidating cross-machine comparison. | Prompt rates plausible on one machine, impossible on the other. | Pinned and recorded on both. |
| 3 | **Device placement persistence.** Ollama retains a loaded model with its original placement for the keep_alive window, so a CPU-only run followed by a GPU run re-measured the CPU copy. | One model reported 0% VRAM and figures identical to the CPU-only condition, while every other model used the GPU. | Per-model VRAM residency reporting. | `unload_all()` evicts every model with `keep_alive: 0` before measuring. |
| 4 | **Prompt prefix caching.** Repeating an identical prompt skips prompt processing almost entirely. | Reported rates of 58,000 to 72,000 tok/s. | Implausible magnitude. | Unique marker **prepended** to each prompt. Appending is insufficient: the cache is on the prefix, so an appended marker leaves the whole prefix cached. |
| 5 | **Thermal ramp confounded with model order.** First model measured cool, last measured throttled. | 63°C at start of run, 85°C at end. | Per-run temperature recording. | Cool-down gate before **every repetition**, aborting rather than continuing if the target is not reached. |
| 6 | **Thermal throttling from a stock fan curve.** Clock capped at 1.70 GHz. | Fan at cooling state 2 of 4 at the 85°C limit. | `cooling_device0/cur_state` against `max_state`. | `FAN_TEMP0..3` lowered in EEPROM. ARM frequency now recorded per request. |
| 7 | **Context length defaults.** The CLI loaded a 128K context, inflating the model footprint from 2 GB to 18 GB and forcing 80% onto CPU. | 22-second load times, 20% GPU residency. | `ollama ps` reporting `CONTEXT 131072`. | `num_ctx` pinned in config and passed explicitly on every call. |
| 8 | **Unapplied device placement on the embedding model.** `time_embedding()` sent no options, so `--cpu-only` never reached it. | A "CPU-only" embedding latency measured with the model in VRAM. | VRAM residency reported alongside the timing. | Options passed through; residency asserted per model. |
| 9 | **Host power profile.** Laptop on Balanced produced roughly a quarter of Best Performance throughput. | Unexplained variance between runs. | Power profile recorded per run. | Recorded in the environment capture. |
| 10 | **Aggregate resource reporting.** GPU residency summed across all loaded models, so an embedding model on the GPU made a CPU-only generation run appear 46% offloaded. | Contradictory offload figure in an explicitly CPU-only condition. | Inspection. | Residency reported per model only; the aggregate was removed. |

Eight of these ten produce numbers that are internally consistent and superficially reasonable. Only errors 1 and 6 were caught by anything other than deliberate instrumentation, and error 1 was caught by hearing a fan.

**The general lesson for edge benchmarking:** measure through the same interface the application uses, record the full environment with every run, and treat any measurement you did not explicitly instrument as unverified.

## Known limitations of these measurements

- **ARM frequency sampling is slightly racy.** The reading is taken immediately after a request returns, so it occasionally catches the clock already dropping to the 1.5 GHz idle state. The majority of readings sit at 2.15 to 2.37 GHz and throttle flags were clear, so the conclusion holds, but sampling mid-request would be cleaner.
- **The laptop ran on the Balanced power profile.** The CPU-only condition should be repeated on Best Performance before the final RQ4 figures are reported.
- **`laptop_cpu_v2` was recorded with a dirty working tree.** The provenance block records this. It is usable but not exactly reproducible from its commit alone, and will be repeated alongside the power profile correction.
- **The Pi runs at 82 to 85°C throughout.** Throttle flags are clear, so this is within the soft limit, but it is close to it. Ambient temperature will affect reproduction.

## Implications for the experimental design

1. **Answer quality is hardware-invariant** given identical weights, seed and options. Quality experiments (RQ1, RQ2, RQ3) run on the laptop in minutes. Only RQ4 requires the Pi. To be verified by running an identical question subset on both and reporting output agreement.
2. **`top_k` and chunk size are cost drivers** and should be treated as experimental variables, not fixed settings.
3. **Verification runs on the smaller model.** Many short calls benefit most from faster prefill.
4. **The embedding model stays on CPU** in every condition.
5. **`phi3:mini` is reported as laptop-only** in the model comparison, with tokenizer efficiency given as the reason for exclusion from the edge condition.
6. **Latency is reported as two components**, prompt processing and generation rate, rather than wall time alone. Wall time confounds hardware speed with how many tokens a model chose to emit.
7. **Thermal state and clock are recorded per question** during evaluation runs, allowing latency to be plotted against elapsed session time. Burst and sustained performance then come from a single run with no manipulation.

## Estimated evaluation cost

Per verified question on the Pi, five extracted claims, `top_k: 4`:

| Configuration | Estimate |
|---|---|
| llama3.2:3b generating and verifying | 3.8 min |
| llama3.2:3b generating, llama3.2:1b verifying | **3.1 min** |
| llama3.2:1b throughout | 2.0 min |

Thirty questions in the asymmetric configuration is under two hours. One hundred is a single overnight run.
