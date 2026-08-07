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

All eight are in place from commit `HEAD`. No valid run has been recorded yet.
