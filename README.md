# On-Device Agentic RAG Assistant for SME Knowledge Management

MSc Applied AI dissertation artefact. WMG, University of Warwick.
Md Basar Basar (5753701). Supervisor: Manoj Babu.

**Research question:** To what extent can an on-device agentic RAG assistant, with source verification and confidence flagging, support private SME knowledge management and operational decision support within the resource constraints of an edge device?

## Pipeline

```
query analysis -> retrieval -> grounded generation ->
claim extraction -> evidence verification -> confidence and risk flagging ->
next-action suggestion
```

## Layout

| Path | Contents |
|---|---|
| `src/sme_assistant/common/` | Config, LLM client, instrumentation |
| `src/sme_assistant/kb/` | Synthetic SME knowledge base loader |
| `src/sme_assistant/ingest/` | Chunking, embedding, index building |
| `src/sme_assistant/retrieve/` | Vector store and retriever |
| `src/sme_assistant/generate/` | Prompts and grounded generation |
| `src/sme_assistant/verify/` | The contribution: verification and confidence |
| `src/sme_assistant/agent/` | End-to-end orchestration |
| `src/sme_assistant/evaluation/` | Test set, metrics, harness, plots |
| `data/` | Knowledge base documents, index, test set |
| `results/runs/` | Experiment outputs |
| `tests/` | Unit and integration tests |

Core code uses the Python standard library only. Ollama serves the models over HTTP. matplotlib is needed for plots.

## Install

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/Pi: source .venv/bin/activate
pip install -e ".[dev,plots]"
```

## Hardware

| | Laptop | Raspberry Pi 5 |
|---|---|---|
| CPU | AMD Ryzen 7 7840HS, 8C/16T, 3.8 GHz | Cortex-A76, 4C, 2.4 GHz |
| RAM | 16 GB | 16 GB |
| GPU | NVIDIA RTX 4050 Laptop | None |
| Architecture | x86_64 | aarch64 |
| Python | 3.13.5 | 3.13.5 |

Experiment conditions: A = laptop GPU, B = laptop CPU only, C = Pi 5 CPU only.

## Ethics

Ethics waiver approved. Synthetic data only, no human participants, no real organisational data.
