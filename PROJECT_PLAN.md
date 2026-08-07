# Project Plan

**Title:** A Privacy-Preserving On-Device Agentic Assistant for SME Knowledge Management and Operational Decision Support

**Student:** Md Basar Basar (5753701), MSc Applied AI, WMG, University of Warwick
**Supervisor:** Manoj Babu
**Module:** ES9U9-60-2025
**Period:** 25 June 2026 to submission

---

## 1. Research question

To what extent can an on-device agentic RAG assistant, with source verification and confidence flagging, support private SME knowledge management and operational decision support within the resource constraints of an edge device?

### Sub-questions

| | Question |
|---|---|
| RQ1 | What answer quality can small language models achieve over an SME knowledge base when constrained to on-device execution? |
| RQ2 | Does a claim-level verification layer reduce unsupported output compared with a standard RAG baseline? |
| RQ3 | Do the confidence flags correlate with actual answer correctness, and are they calibrated? |
| RQ4 | What are the latency, memory and thermal costs of running the full pipeline on an edge device relative to a laptop? |

The contribution is the verification and confidence-flagging layer running on-device, and its empirical evaluation. The retrieval pipeline itself is standard and exists to support that contribution.

---

## 2. Timeline

### Completed

| Period | Work |
|---|---|
| Late June 2026 | Topic selection and research proposal. Supervisor meeting. Ethics form prepared and submitted. |
| Week of 2 Jul | Project plan, research framework, literature review search strategy. Reading log started. |
| Week of 8 Jul | System architecture designed and diagrammed. Chapters 1 and 2 drafted (Introduction, Literature Review). |
| Week of 16 Jul | Chapter 3 drafted (Methodology). Reference list assembled. First artefact implementation. |
| 25 Jul | Raspberry Pi 5 environment configured. System and artefact backups taken. |
| 7 Aug | Artefact rebuild begun: full reimplementation for verifiable understanding of every component. Environment standardised across both machines (Python 3.13.5, Ollama). |

### Remaining

| Stage | Work | Target |
|---|---|---|
| 1 | Synthetic SME knowledge base and loader | Aug |
| 2 | Ingestion: chunking, embedding, index building | Aug |
| 3 | Retrieval: vector store and top-k search | Aug |
| 4 | Grounded generation with source citations | Aug |
| 5 | Verification layer: query analysis, claim extraction, evidence checking, confidence and risk flagging | Aug |
| 6 | Evaluation harness: test set, metrics, run comparison, plots | Aug |
| 7 | Experiments across all three hardware conditions | Aug |
| - | Chapters 4 to 7 (Results, Analysis, Discussion, Conclusion) | Aug |
| - | Full draft to supervisor | TBC |
| - | Revision, formatting, submission | TBC |

---

## 3. System design

```
question
   |
   v  query analysis
   v  retrieval over local index
   v  source-grounded generation with citations
   v  claim extraction
   v  evidence verification
   v  confidence scoring and risk flagging
   v  next-action suggestion
   |
answer + citations + confidence flag + suggested action
```

All stages execute locally. No data leaves the device.

---

## 4. Evaluation design

Four experiments, run over a fixed test question set covering answerable, partially answerable and unanswerable questions.

| # | Experiment | Answers |
|---|---|---|
| 1 | Model comparison across candidate SLMs | RQ1 |
| 2 | Ablation: pipeline with and without the verification layer | RQ2 |
| 3 | Confidence calibration against ground-truth correctness | RQ3 |
| 4 | Hardware comparison across three conditions | RQ4 |

### Hardware conditions

| Condition | Platform | Accelerator |
|---|---|---|
| A | Laptop, AMD Ryzen 7 7840HS, 16 GB | NVIDIA RTX 4050 Laptop GPU |
| B | Laptop, AMD Ryzen 7 7840HS, 16 GB | CPU only |
| C | Raspberry Pi 5, Cortex-A76, 16 GB | CPU only |

Condition B exists to isolate the effect of hardware from the effect of GPU acceleration. Comparing A with C alone would confound the two.

### Controls

Python version, Ollama version, model digests, configuration file and random seed are held identical across conditions. Architecture, memory bandwidth and thermal headroom are the independent variables. Configuration is hashed and recorded with every run so results are traceable to the parameters that produced them.

### Metrics

| Dimension | Measures |
|---|---|
| Answer quality | Correctness against gold answers, groundedness in retrieved sources |
| Verification value | Unsupported-claim detection rate, refusal accuracy on unanswerable questions, confidence calibration |
| Resource cost | End-to-end and per-stage latency, tokens per second, peak memory, CPU temperature |

---

## 5. Dissertation structure

| Chapter | Word budget | Status |
|---|---|---|
| Abstract | 300 | Not started |
| 1. Introduction | 1,200 | Drafted |
| 2. Literature Review | 3,800 | Drafted |
| 3. Methodology | 2,300 | Drafted |
| 4. Results | 2,300 | Awaiting experimental data |
| 5. Analysis | 2,300 | Awaiting experimental data |
| 6. Discussion | 2,000 | Not started |
| 7. Conclusion | 1,000 | Not started |

Target: 15,000 to 15,500 words. Harvard referencing.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Thermal throttling on the Pi distorts latency measurements | Temperature logged at every stage; throttle state recorded per run; treated as a finding rather than an error |
| Pipeline too slow on the Pi for practical use | Smaller models and shorter contexts available as fallbacks; a slow but working deployment remains a valid result |
| Scope creep during implementation | Feature set fixed at the seven stages listed above; no additions after Stage 5 |
| Writing compressed into the final weeks | Chapters 1 to 3 already drafted; Results written as data arrives rather than afterwards |
| Data loss | Version controlled and pushed to a private remote; dissertation held on institutional storage |

---

## 7. Ethics

Ethics waiver approved on the basis of secondary and synthetic data only. No human participants, no personal data, no real organisational documents. The knowledge base is entirely fabricated for evaluation purposes.
