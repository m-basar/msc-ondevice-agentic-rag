# Hardware Benchmarking

Baseline inference measurements for both machines, and the measurement errors found along the way. Written for Chapter 3 (experimental setup) and Chapter 5 (analysis).

All figures produced by `scripts/benchmark.py`, measured through the same HTTP API the pipeline uses, with generation options pinned from `config.json`. Raw JSON with full environment capture is in `results/benchmarks/`.

## Test conditions

| | Laptop | Raspberry Pi 5 |
|---|---|---|
| CPU | AMD Ryzen 7 7840HS, 8C/16T, 3.8 GHz | Cortex-A76, 4C, 2.4 GHz nominal |
| RAM | 15.3 GB | 15.8 GB |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB | None |
| Architecture | x86_64 | aarch64 |
| Cooling | Laptop internal | Official Active Cooler |

Prompt: 903 tokens of real knowledge base evidence, matching the size of an actual retrieval-augmented request. `num_ctx: 4096`, `num_predict: 400`, `temperature: 0.1`.

## Laptop, GPU

| Model | Generation | Wall | VRAM resident |
|---|---|---|---|
| llama3.2:1b | 118.09 tok/s | 0.94s | 100% |
| gemma:2b | 96.54 tok/s | 0.93s | 100% |
| qwen2.5:3b | 75.99 tok/s | 1.86s | 100% |
| phi3:mini | 59.93 tok/s | 6.84s | 100% |

Laptop CPU-only, llama3.2:3b: 22.20 tok/s generation, 3.83s wall.

Embedding, `nomic-embed-text`: 30.6 ms median on GPU, 62.7 ms CPU-only.

## Raspberry Pi 5, CPU only

Measured twice, before and after upgrading Ollama. Nothing else changed.

| Model | Prompt 0.18.3 | Prompt 0.32.6 | Wall 0.18.3 | Wall 0.32.6 |
|---|---|---|---|---|
| llama3.2:1b | 18.87 tok/s | **42.84** | 75.0s | **43.5s** |
| llama3.2:3b | 5.67 tok/s | **21.27** | 181.5s | **62.3s** |
| qwen2.5:3b | 6.18 tok/s | **22.03** | 185.4s | **69.1s** |
| phi3:mini | 3.74 tok/s | **22.82** | 448.1s | **194.6s** |
| gemma:2b | 8.30 tok/s | **46.46** | 129.6s | **32.5s** |

Embedding, `nomic-embed-text`: 79.0 ms median.

**Both Pi measurement sets were taken while the CPU was thermally throttled at 1.7 GHz, 71% of nominal.** See the thermal section below.

## Findings

### 1. Runtime version dominated hardware capability

Upgrading Ollama from 0.18.3 to 0.32.6 improved Pi prompt processing by 3.7x for llama3.2:3b and 6.1x for phi3:mini, with no hardware change. phi3:mini went from unusable at 448 seconds per call to merely slow.

Any edge benchmark that does not pin and report the inference runtime version is close to meaningless. This is the single largest effect observed in this study, larger than the difference between several of the models.

### 2. Prompt processing, not generation, is the edge bottleneck

On the laptop, a 903-token prompt is processed in milliseconds. On the Pi it dominates end-to-end latency. Generation rate, the figure normally quoted in model comparisons, is the less important of the two for retrieval-augmented workloads.

This makes `top_k` and chunk size first-order design decisions on edge hardware rather than tuning details.

### 3. Tokenizer efficiency is a deployment constraint

The same evidence encodes to 903 tokens under the Llama tokenizer, 952 under Qwen, 960 under Gemma, and **1,160 under Phi-3**. Phi-3 pays a 28% penalty on every call, compounding with its slower processing.

Parameter count alone does not predict edge cost.

### 4. Smaller models help disproportionately at the prompt stage

llama3.2:1b processes prompts 2.0x faster than llama3.2:3b on the Pi, while generating only 1.7x faster. Because prompt processing dominates, the practical speedup is larger than the parameter ratio suggests. This is the empirical basis for the asymmetric design in which verification, which is many short calls, runs on a smaller model than generation.

### 5. Thermal design, not compute, sets sustained edge performance

The Pi reached 85°C and entered frequency capping **within a single 40-second generation**, dropping the ARM clock from 2.4 GHz to 1.7 GHz.

Diagnosis: the fan was running at 3,489 RPM in cooling state 2 of 4 while at the thermal limit. The stock fan curve does not ramp aggressively enough for sustained inference. Corrected by lowering the EEPROM `FAN_TEMP0..3` thresholds.

Every figure above was therefore measured at 71% of nominal clock. Retaining them gives a natural experiment: identical hardware and software, two clock speeds, differing only by a fan curve.

## Threats to measurement validity encountered

Each of these produced plausible-looking but wrong numbers. They are documented because none is obvious, and because a benchmark that looks reasonable is the hardest kind of error to catch.

| # | Error | How it presented | Detection | Mitigation |
|---|---|---|---|---|
| 1 | **Endpoint substitution.** VS Code Remote-SSH auto-forwarded the Pi's Ollama port to the laptop's localhost. All "laptop" measurements executed on the Pi. | Laptop and Pi reported near-identical rates. Model loaded 100% CPU despite a working GPU. | The Pi's fan audibly spinning up during a laptop command. | Disable auto port forwarding. `model_store_fingerprint` recorded per run: a hash of the responding server's model digests, so substitution is visible in the results file. |
| 2 | **Runtime version mismatch** between machines. | Different prompt-cache behaviour, invalidating cross-machine comparison. | Prompt rates plausible on one machine, impossible on the other. | Pin and record Ollama version on both. |
| 3 | **Device placement persistence.** Ollama retains a loaded model with its original placement for the keep_alive window. A CPU-only run followed by a GPU run re-measured the CPU copy. | One model reported 0% VRAM and figures identical to the CPU-only condition, while every other model used the GPU. | Per-model VRAM residency reporting. | `unload_all()` evicts every model with `keep_alive: 0` before measuring. |
| 4 | **Prompt prefix caching.** Repeating an identical prompt skips prompt processing almost entirely. | Reported rates of 58,000 to 72,000 tokens per second. | Implausible magnitude. | Unique marker **prepended** to each prompt. Appending is insufficient: the cache is on the prefix. |
| 5 | **Thermal ramp confounded with model order.** First model measured cool, last measured throttled. | 63°C at start of run, 85°C at end. | Per-run temperature recording. | Cool-down gate between models; per-run temperature and throttle state recorded. |
| 6 | **Context length defaults.** The CLI loaded a 128K context, inflating the model footprint from 2 GB to 18 GB and forcing 80% onto CPU. | 22-second load times, 20% GPU residency. | `ollama ps` reporting `CONTEXT 131072`. | `num_ctx` pinned in config and passed explicitly on every call. |
| 7 | **Host power profile.** Laptop on the Balanced plan produced roughly a quarter of its Best Performance throughput. | Unexplained variance between runs. | Power profile recorded per run. | Recorded in the environment capture. |
| 8 | **Aggregate resource reporting.** GPU residency summed across all loaded models, so an embedding model on the GPU made a CPU-only generation run appear 46% offloaded. | Contradictory offload figure in an explicitly CPU-only condition. | Inspection. | Residency reported per model. |

Six of these eight produce numbers that are internally consistent and superficially reasonable. Only errors 1 and 5 were caught by anything other than deliberate instrumentation, and error 1 was caught by hearing a fan.

The general lesson for edge benchmarking: measure through the same interface the application uses, record the full environment with every run, and treat any measurement you did not explicitly instrument as unverified.

## Implications for the experimental design

1. **Answer quality is hardware-invariant** given identical weights, seed and options. Quality experiments (RQ1, RQ2, RQ3) run on the laptop in minutes. Only RQ4 requires the Pi. To be verified by running an identical question subset on both and reporting output agreement.
2. **`top_k` and chunk size are cost drivers**, not tuning details, and should be treated as experimental variables.
3. **Verification runs on the smaller model.** Many short calls benefit most from faster prompt processing.
4. **phi3:mini is excluded from the edge condition** and reported as laptop-only, with tokenizer efficiency given as the reason.
5. **Latency is reported as two components**, prompt processing time and generation rate, rather than wall time alone. Wall time confounds hardware speed with how many tokens a model chose to emit.
6. **Thermal state is recorded per question** during evaluation runs, allowing latency to be plotted against elapsed session time. Burst and sustained performance then come from a single run with no manipulation.
