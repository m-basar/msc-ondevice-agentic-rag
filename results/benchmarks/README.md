# Benchmark run status

Not every file here is valid evidence. Each run is classified below. **Only runs
marked VALID may be cited in the dissertation.**

Superseded runs are retained rather than deleted: they are the evidence that the
measurement errors documented in `docs/BENCHMARKS.md` were real, and deleting
them would remove the audit trail that makes the corrections credible.

| Run | Status | Reason |
|---|---|---|
| `20260807_033309_Rashel_laptop_cpu_perf` | SUPERSEDED | Prompt prefix cache not defeated (marker appended, not prepended). Embedding measured on GPU despite `--cpu-only`. No per-model eviction. |
| `20260807_033714_Rashel_baseline` | SUPERSEDED | Same cache issue. `llama3.2:3b` was still resident from the preceding CPU-only run and was re-measured CPU-pinned at 0% VRAM. |
| `20260807_042801_Rashel_laptop_gpu_clean` | SUPERSEDED | Eviction and cache buster present, but the marker was still appended rather than prepended, so prompt rates remain inflated. No provenance block, no frequency capture. |
| `20260807_041732_agi-pi_pi5_baseline` | SUPERSEDED | Ollama 0.18.3. Thermal ramp confounded with model order: first model measured at 63C, last at 85C. |
| `20260807_050755_agi-pi_pi5_clean` | SUPERSEDED | Ollama 0.32.6 and cool-down between models, but not between repetitions, so second repetitions began at the throttle point. Fan curve not yet corrected, so the board ran frequency-capped throughout. No frequency capture. |

## What a VALID run requires

1. Prefix cache defeated by a marker **prepended** to the prompt
2. Every model evicted before measurement, and per-model VRAM residency reported
3. Cool-down before **every** repetition, aborting rather than continuing if the target is not reached
4. ARM frequency recorded immediately before and after each measured request, with the ratio to nominal
5. Embedding placement controlled, cold start reported separately from warm
6. At least five repetitions
7. Full provenance: corpus, registry, config, script and prompt SHA-256, git commit and dirty state, seed, and the exact command
8. Raspberry Pi EEPROM fan curve corrected (`FAN_TEMP0..3`)

## VALID runs

| Run | Machine | Condition |
|---|---|---|
| `20260807_181428_Basar_laptop_gpu_v2` | Laptop | GPU, all models, 5 reps |
| `20260807_181611_Basar_laptop_cpu_v2` | Laptop | CPU only, llama3.2:3b, 5 reps |
| `20260807_191119_agi-pi_pi5_v2` | Pi 5 | CPU only, all models, 5 reps, cool-to 65C |

All three carry commit `ca511829602f`, corpus SHA `dd51cf741e59...` and seed 42, so they are directly comparable.

Two caveats recorded in the provenance blocks rather than hidden:

- `laptop_cpu_v2` was taken with a **dirty working tree**. It is usable but not exactly reproducible from its commit alone.
- Both laptop runs were taken on the **Balanced** power profile. The CPU-only condition should be repeated on Best Performance before the final RQ4 figures are reported.

Figures from these runs, and only these runs, appear in `docs/BENCHMARKS.md`.
