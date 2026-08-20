# On-Device RAG Assistant with Claim-Level Verification, for SME Knowledge Management

MSc Applied AI dissertation artefact. WMG, University of Warwick.
Md Basar Basar (5753701). Supervisor: Manoj Babu.

**Umbrella research question:** To what extent can an on-device agentic RAG
assistant, with source verification, support private SME knowledge management
and operational decision support within the resource constraints of an edge
device?

The four sub-questions are stated in `docs/PREREGISTRATION.md` section 1 and are
the ones the experiment tests. "Agentic" refers to the verification and revision
loop; the artefact does not perform autonomous planning or tool use.

## This directory is the authoritative implementation

`final_v1/` is the implementation the dissertation reports. The older top-level
`artefact/` directory is a **superseded July 2025 design** and is retained only
as a historical record. It must not be used for the dissertation's claims, for a
demonstration, or for any dashboard: it predates the four-arm design, the
conflict taxonomy, the pre-registration and every result in Chapter 4.

## Pipeline, as implemented

```
retrieval -> grounded generation -> verification (Arm D only)
```

A question is embedded and matched against a pre-built index of document chunks.
The retrieved chunks are assembled into an evidence block and passed to the
generation model, which must answer only from that evidence and cite a chunk
identifier after every statement. In the verified configuration the draft, the
question and the same evidence go to a second model, which audits each claim,
classifies any relationship between the passages, and either returns the draft
unchanged or replaces it. A rule-based categorical confidence level is attached
from the verifier's verdict.

**Not implemented, despite appearing in an earlier design document:** a separate
query-analysis stage, a separate claim-extraction agent, a next-action
suggestion stage, and any calibrated confidence score. Confidence is a declared
mapping in `config.json`; its calibration was not evaluated.

## Models

| Role | Model | Settings |
|---|---|---|
| Generation, all arms | `llama3.2:3b` | temperature 0.1, `num_ctx` 4096, `num_predict` 400, seed 42 |
| Verification, Arm D | `qwen2.5:3b` | temperature 0.0, `num_ctx` 4096, `num_predict` 700, seed 42 |
| Embeddings | `nomic-embed-text` | pinned to CPU during timed runs |

The verifier was selected by the diagnostic protocol in `docs/VERIFIER_PROTOCOL.md`,
recorded as pre-registration amendment 1.10. **No comparison of candidate
generation models is reported.** `llm.candidate_models` in `config.json` lists
models pulled during development; it is not an experimental factor.

## Experimental arms

Four arms over one frozen corpus, forming a tree rooted at B rather than a
ladder.

| Arm | Retrieval | Evidence shown | Verification |
|---|---|---|---|
| A | all documents | identifier and text only | none |
| B | all documents | with status metadata | none |
| C | current documents only | with status metadata | none |
| D | all documents | with status metadata | yes |

**B against D is the confirmatory single-variable contrast** for the
verification layer. C against D changes retrieval mode and verification
together and is reported as a practical comparison, not as an ablation.

## Hardware conditions

Named, never lettered: an earlier lettering collided with the experimental arms,
making "condition B" ambiguous between the laptop CPU condition and the
status-metadata arm.

| Condition | Machine | Placement |
|---|---|---|
| `laptop_gpu` | AMD Ryzen 7 7840HS, 8C/16T, RTX 4050 Laptop, 16 GB, x86_64 | GPU offload |
| `laptop_cpu` | the same laptop | CPU only |
| `pi5_cpu` | Raspberry Pi 5, Cortex-A76 4C, 16 GB, aarch64 | CPU only |

Python 3.13.5 on both machines. H5 is stated over `pi5_cpu` and is scored only
there; the two laptop conditions are descriptive RQ4 figures.

## Layout

| Path | Contents |
|---|---|
| `src/sme_assistant/common/` | Config, LLM client, instrumentation |
| `src/sme_assistant/kb/` | Synthetic SME knowledge base loader |
| `src/sme_assistant/ingest/` | Chunking, embedding, index building |
| `src/sme_assistant/retrieve/` | Vector index and retriever |
| `src/sme_assistant/generate/` | Prompts and grounded generation |
| `src/sme_assistant/verify/` | The contribution: verification and serving rule |
| `src/sme_assistant/evaluation/` | Question set, scoring, harness, analysis |
| `data/` | Knowledge base documents and index |
| `gold/` | Answer key, conflict registry, question set. Not readable by the pipeline |
| `docs/` | Pre-registration, corpus provenance, verifier protocol, dissertation |
| `results/runs/` | Frozen experiment outputs |
| `results/analysis/` | Generated analysis and performance reports |
| `tests/` | Unit, integration, boundary and reported-wording tests |

`src/sme_assistant/agent/` holds orchestration only and contains no separate
agent stages.

Core code uses the Python standard library only. Ollama serves the models over
HTTP. matplotlib is required for figures.

## Install

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/Pi: source .venv/bin/activate
pip install -e ".[dev,plots]"
```

## Regenerating everything reported

```bash
python -m pytest                                    # full test suite
python scripts/kb_summary.py                        # corpus figures
python scripts/analyse_results.py                   # quality analysis
python scripts/analyse_performance.py --index latest_test_performance_pi5_cpu.json
python scripts/make_figures.py                      # Chapter 4 figures
python scripts/make_architecture_figures.py         # Chapter 3 figures
python scripts/make_corpus_doc.py > docs/CORPUS.md  # corpus provenance
python scripts/make_amendment_table.py > docs/dissertation/appendix_amendments.md
```

The four frozen quality runs are a closed list in
`sme_assistant.evaluation.analysis.FROZEN_QUALITY_RUNS`. No run created later
can enter the quality analysis, and performance runs are refused by it outright.

## Ethics

Ethics waiver approved. Synthetic data only, no human participants, no real
organisational data. All inference is local; no request leaves the device.
