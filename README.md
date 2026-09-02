# On-Device RAG Assistant with Claim-Level Verification

An SME document assistant that runs entirely on local hardware, answers only
from retrieved documents, cites a source for every claim, and can pass its draft
answer to a second model for verification.

MSc Applied AI dissertation artefact. WMG, University of Warwick.
Md Basar Basar (5753701). Supervisor: Manoj Babu.

**Research question.** To what extent can an on-device agentic RAG assistant,
with source verification, support private SME knowledge management and
operational decision support within the resource constraints of an edge device?

The four sub-questions are in `docs/PREREGISTRATION.md` section 1. "Agentic"
means the verification and revision stage only; the artefact does not plan,
decompose tasks or select tools.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,plots]"
python scripts/dashboard.py        # http://127.0.0.1:8765
```

That opens the demonstrator. **Frozen Study Replay works immediately with no
models installed.** Live answers need Ollama, see below.

## How it works

```
retrieval -> grounded generation -> verification (Arm D only)
```

A question is embedded and matched against a pre-built index of document chunks.
The retrieved chunks become an evidence block for the generation model, which
must answer only from that evidence and cite a chunk identifier after every
statement. In the verified configuration the draft, the question and the same
evidence go to a second model, which audits each claim, classifies any
relationship between the passages, and either returns the draft unchanged or
replaces it. A rule-based confidence level is attached from the verifier's
verdict.

## Running the demonstrator

The dashboard was built **after** the evaluation. It contributes no evidence to
any hypothesis and writes nothing into any frozen run directory. See
pre-registration amendment 1.27.

```bash
python scripts/dashboard.py                 # http://127.0.0.1:8765
python scripts/dashboard.py --port 8080     # different port
python scripts/dashboard.py --host 0.0.0.0  # reachable on the local network
```

It binds to localhost by default because it serves internal documents and has no
authentication. The server is Python's standard-library `http.server` and no
asset is fetched from the network.

Two modes, chosen on opening and never shown together.

| | Frozen Study Replay | Live Assistant |
|---|---|---|
| Answers come from | the four committed quality runs | this device, now |
| Arms | A, B, C and D side by side | D only |
| Needs Ollama | no | yes |
| Scored | during the experiment | never |

Live mode needs Ollama running with both models pulled:

```bash
ollama serve
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
```

Start-up prints the readiness of each mode. Where live mode is unavailable the
page says why and disables the question box.

On a Raspberry Pi 5 expect roughly three minutes per live answer; the measured
mean is 174 seconds. Frozen replay is instant on any machine, which makes it the
sensible choice for a demonstration.

## Reproducing the reported results

```bash
python -m pytest                                    # full test suite
python scripts/kb_summary.py                        # corpus figures
python scripts/analyse_results.py                   # quality analysis
python scripts/analyse_performance.py --index latest_test_performance_pi5_cpu.json
python scripts/make_figures.py                      # Chapter 4 figures
python scripts/make_architecture_figures.py         # Chapter 3 figures
python scripts/make_corpus_doc.py > docs/CORPUS.md
python scripts/make_amendment_table.py > docs/dissertation/appendix_amendments.md
python scripts/make_verifier_appendix.py > docs/dissertation/appendix_verifier_classification.md
python scripts/compare_index_architectures.py > docs/dissertation/appendix_index_architectures.md
```

Before regenerating figures, run `python scripts/figure_provenance.py` to print
the FreeType and font versions on the current machine. The versions that drew the
committed images are in `docs/dissertation/figures/FIGURE_ENVIRONMENT.json` and
pinned as the `figures` extra in `pyproject.toml`. Both figure scripts take
`--out`, so a check can regenerate into a temporary directory without touching
what is committed.

`data/index.json` is not committed. It is a build artefact and
`scripts/build_index.py` reproduces it from the corpus. Builds are not
byte-identical across architectures because `nomic-embed-text` under Ollama is
not bit-reproducible between x86-64 and ARM64. Appendix E measures what that
costs.

## The experiment

Four arms over one frozen corpus, forming a tree rooted at B rather than a
ladder.

| Arm | Retrieval | Evidence shown | Verification |
|---|---|---|---|
| A | all documents | identifier and text only | none |
| B | all documents | with status metadata | none |
| C | current documents only | with status metadata | none |
| D | all documents | with status metadata | yes |

**B against D is the confirmatory single-variable contrast.** C against D changes
retrieval mode and verification together and is reported as a practical
comparison, not an ablation.

| Role | Model | Settings |
|---|---|---|
| Generation, all arms | `llama3.2:3b` | temperature 0.1, `num_ctx` 4096, `num_predict` 400, seed 42 |
| Verification, Arm D | `qwen2.5:3b` | temperature 0.0, `num_ctx` 4096, `num_predict` 700, seed 42 |
| Embeddings | `nomic-embed-text` | pinned to CPU during timed runs |

| Condition | Machine | Placement |
|---|---|---|
| `laptop_gpu` | AMD Ryzen 7 7840HS, 8C/16T, RTX 4050 Laptop, 16 GB, x86_64 | GPU offload |
| `laptop_cpu` | the same laptop | CPU only |
| `pi5_cpu` | Raspberry Pi 5, Cortex-A76 4C, 16 GB, aarch64 | CPU only |

Python 3.13.5 on both machines. Conditions are named rather than lettered, so
that "condition B" cannot be confused with Arm B. H5 is stated over `pi5_cpu` and
scored only there; the two laptop conditions are descriptive RQ4 figures.

## Layout

| Path | Contents |
|---|---|
| `src/sme_assistant/common/` | Config, LLM client, instrumentation |
| `src/sme_assistant/kb/` | Synthetic SME knowledge base loader |
| `src/sme_assistant/ingest/` | Chunking, embedding, index building |
| `src/sme_assistant/retrieve/` | Vector index and retriever |
| `src/sme_assistant/generate/` | Prompts and grounded generation |
| `src/sme_assistant/verify/` | The contribution: verification and serving rule |
| `src/sme_assistant/agent/` | Orchestration only, no separate agent stages |
| `src/sme_assistant/evaluation/` | Question set, scoring, harness, analysis |
| `data/` | Knowledge base documents and index |
| `gold/` | Answer key, conflict registry, question set. Not readable by the pipeline |
| `docs/` | Pre-registration, corpus provenance, verifier protocol, dissertation draft |
| `results/runs/` | Frozen experiment outputs |
| `results/analysis/` | Generated analysis and performance reports |
| `results/retrieval/` | Development-split retrieval evaluations |
| `tests/` | Unit, integration, boundary and reported-wording tests |

Core code uses the Python standard library only. Ollama serves the models over
HTTP. matplotlib is required for figures.

## Scope and boundaries

Each of these exists because the alternative would overclaim.

**This directory is the implementation the dissertation reports.** The older
top-level `artefact/` directory is a superseded July 2025 design, kept only as a
historical record. It predates the four-arm design, the conflict taxonomy, the
pre-registration and every result in Chapter 4.

**The submitted dissertation is a PDF held by WMG and is not in this
repository.** `docs/dissertation/` holds the working draft it was written from,
roughly 31,000 words against about 14,700, because the cut to the word limit was
made in Word and never carried back. The markdown is therefore not a copy of what
was examined. Where prose and data disagree,
`results/analysis/hypotheses.json` governs the hypothesis decisions and
`results/analysis/performance_latest_test_performance_pi5_cpu.json` governs the
Raspberry Pi timings. The draft is retained because four `scripts/make_*.py`
generators and `tests/test_chapter_claims.py` read it.

**Not implemented, despite appearing in an earlier design document:** a separate
query-analysis stage, a separate claim-extraction agent, a next-action suggestion
stage, and any calibrated confidence score. Confidence is a declared mapping in
`config.json` and its calibration was not evaluated.

**No comparison of candidate generation models is reported.**
`llm.candidate_models` in `config.json` lists models pulled during development;
it is not an experimental factor. The verifier was selected by the diagnostic
protocol in `docs/VERIFIER_PROTOCOL.md`, recorded as amendment 1.10.

**`results/retrieval/` holds development-split evaluations only, and they are
unreported.** The nine tracked files are the laptop runs behind the retrieval
calibration in amendment 1.3, which fixed `top_k` at 6 and `min_similarity` at
0.30 before any test-split run. The Raspberry Pi produced six of its own, which
are listed by name in `.gitignore` so they cannot be staged by accident and are
left in place on that device. Nothing cites them and they contributed to none of
H1 to H5. They are ignored to keep `results/` free of device-local output that a
careless `git add -A` would commit, not to conceal them, which is why they are
named individually.

**The four frozen quality runs are a closed list** in
`sme_assistant.evaluation.analysis.FROZEN_QUALITY_RUNS`. No later run can enter
the quality analysis, and performance runs are refused outright. Both the four
quality runs and the six performance runs carry content digests in
`sme_assistant.evaluation.authenticity`, checked before any number is read. The
two tables are separate, so a timing run cannot be read as a quality run.

## Ethics

Ethics waiver approved. Synthetic data only, no human participants, no real
organisational data. All inference is local; no request leaves the device.

## Licence

Code is MIT licensed, see `LICENSE`. The written material, synthetic corpus,
question set and frozen results are CC BY 4.0, see `LICENSE-DOCS`, covering
`docs/`, `data/kb/`, `gold/`, `results/` and `figures/`.

The corpus describes Northgate Kitchenware Ltd, a fictional company. It contains
no real business or personal data. Identifiers use ranges reserved for fiction
and all email addresses use the reserved `.invalid` domain.
