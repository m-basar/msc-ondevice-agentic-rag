> Working draft. Revised 19 August 2026 to bring the research questions,
> objectives and contribution into line with the experiment as executed and the
> project's pre-registration. Bibliographic details must still be checked against
> the originals before submission. Delete this note before transferring into the
> WMG template.

# Chapter 2: Literature Review

## 2.1 Introduction

This chapter reviews the five bodies of literature on which the project
rests: SME knowledge management and AI adoption (Section 2.2); small
language models and edge deployment (Section 2.3); retrieval-augmented
generation and its agentic extensions (Section 2.4); and hallucination,
verification and confidence calibration (Section 2.5). Sources were
identified through Google Scholar, Scopus, the ACM Digital Library, IEEE
Xplore and arXiv, using search strings combining terms such as "small
language models", "on-device" or "edge" inference,
"retrieval-augmented generation", "agentic RAG", "hallucination
detection", "confidence calibration" and "SME AI adoption",
followed by citation chaining from key surveys. For the fast-moving
technical themes, coverage concentrates on work published from 2020
onwards; earlier foundational studies are retained where later work
builds directly on them. Peer-reviewed venues are preferred, but arXiv
preprints are included where they represent the current state of the
art, and are identified as preprints in the reference list. The review
is organised as an argument rather than a catalogue: each of Sections
2.2 to 2.5 closes by identifying what that literature leaves unresolved,
and Section 2.6 draws the threads together into the specific gap this
project addresses.

## 2.2 SMEs, AI adoption and knowledge management

Knowledge management research treats organisational knowledge as a
strategic resource whose value depends on processes and systems for
creating, storing, transferring and applying it (Alavi and Leidner,
2001). In SMEs much of this machinery is absent. Successive literature
reviews spanning two decades find that small firms manage knowledge
informally and reactively, that practice is constrained by shortages of
time, money and specialist skills, and that knowledge storage, retention
and utilisation are both the least developed and the least researched
parts of the knowledge management cycle (Durst and Edvardsson, 2012;
Massaro et al., 2016; Durst, Edvardsson and Foli, 2023). Where SMEs do
adopt supporting tools, they favour cheap, familiar, general-purpose
systems over dedicated knowledge management platforms (Cerchione and
Esposito, 2017). The practical consequence is documented knowledge that
exists but is not reliably used: policies and procedures sit in shared
folders while day-to-day answers are given from memory.

Artificial intelligence is frequently proposed as the remedy, but the
adoption evidence tempers optimism. SMEs adopt AI at markedly lower
rates than large firms, and recent reviews converge on a consistent set
of barriers: implementation cost, absent in-house expertise, weak data
foundations, unclear return on investment, and concerns over privacy,
security and governance (OECD, 2025; Oldemeyer, Jede and Teuteberg,
2025; Sánchez, Calderón and Herrera, 2025). The privacy barrier has a
hard regulatory edge. The GDPR's principles of data minimisation and
purpose limitation, and its restrictions on transfers of personal data,
apply with full force to firms that lack legal departments (Regulation
(EU) 2016/679), and the EU Artificial Intelligence Act adds phased,
risk-based obligations on providers and deployers of AI systems
(Regulation (EU) 2024/1689). For a small firm, the simplest defensible
posture for sensitive operational material is often that it should not
leave the premises at all.

What the adoption literature offers far less often is an architectural
response to the barriers it diagnoses. Its remedies are predominantly
managerial: readiness frameworks, adoption roadmaps, skills programmes
and governance guidance (Oldemeyer, Jede and Teuteberg, 2025; Sánchez,
Calderón and Herrera, 2025). Local or on-premises deployment is
occasionally mentioned as a mitigation for privacy concerns, but
concrete system designs evaluated in an SME context are scarce. This
project's setting follows directly: an assistant over the firm's own
documents that is inexpensive to run, requires no specialist
administration and keeps data on site. Whether such a system can be made
trustworthy enough to inform operational decisions is the question the
remaining themes address.

## 2.3 Small language models and edge deployment

For most of the recent history of language models, capability has been
treated as a function of scale, which placed useful systems exclusively
in the cloud. The current generation of small language models revises
that assumption. Surveys of models in roughly the 100-million to
five-billion parameter range chart rapid capability gains per parameter,
driven by better data curation, distillation and architecture choices
(Lu et al., 2024; Xu et al., 2024). Several model families now target
the edge explicitly: Phi-3-mini demonstrated that a 3.8-billion
parameter model could run usefully on a phone (Abdin et al., 2024);
Llama 3.2 ships 1B and 3B variants intended for on-device use (Meta AI,
2024; Grattafiori et al., 2024); and the Qwen2.5 family extends down to
0.5B (Yang et al., 2024). Belcak et al. (2025) push the argument
further, contending that most invocations inside agentic systems are
narrow, repetitive subtasks for which small models are not merely
adequate but operationally and economically preferable.

Feasibility on edge hardware rests equally on model compression and
inference runtimes. Eight-bit quantisation preserves quality at roughly
half the memory of sixteen-bit weights (Dettmers et al., 2022), and
post-training methods such as GPTQ and AWQ push weights to three or four
bits with modest degradation (Frantar et al., 2023; Lin et al., 2024),
which brings a three-billion parameter model into approximately two
gigabytes of memory. CPU-oriented runtimes, most prominently llama.cpp
and its packaged distribution Ollama, execute such quantised models on
commodity ARM processors without a discrete accelerator (Gerganov, 2025;
Ollama, 2025). Zheng et al. (2025) organise this fast-growing field into
a lifecycle of resource-efficient model design, pre-deployment
compression and runtime optimisation, and note the heterogeneity of edge
hardware as a persistent complication.

Empirical studies on single-board computers confirm that deployment is
feasible while exposing its costs. Husom et al. (2025) benchmark
twenty-eight quantised models from the Ollama library on a Raspberry Pi
4, showing that quantisation level materially shifts energy consumption,
latency and output accuracy at once, and that the trade-off is
task-dependent. Across the surveyed literature, CPU-only throughput for
multi-billion parameter models sits in the range of a few tokens per
second, with memory ceilings and thermal throttling under sustained load
as recurring constraints (Xu et al., 2024; Zheng et al., 2025). The
Raspberry Pi 5 with 16GB of memory used in this project is
representative of the class: sufficient memory for a quantised 3B model
alongside an embedding model and vector index, but no accelerator and
finite thermal headroom.

This literature establishes that generation runs on edge hardware, but
its evaluation lens is narrow. Studies benchmark models in isolation on
generic reasoning or question-answering datasets and report system
metrics of throughput, memory and energy (Husom et al., 2025). What they
rarely evaluate is an application-level pipeline in which retrieval,
generation and any reliability machinery operate together, measured for
the trustworthiness of answers as well as their speed. Feasibility of
the parts does not establish feasibility of a dependable whole. That
composite question defines the hardware side of this project's gap.

## 2.4 Retrieval-augmented generation and agentic RAG

Retrieval-augmented generation couples a generative model with a
retriever over an external corpus, so that answers draw on
non-parametric knowledge that can be inspected, cited and updated
without retraining (Lewis et al., 2020). The supporting stack is mature
and compact enough for local use: dense retrievers outperform lexical
search for open-domain question answering (Karpukhin et al., 2020),
sentence-level embedding models run comfortably on CPUs (Reimers and
Gurevych, 2019), and approximate nearest-neighbour libraries make
similarity search inexpensive at the scale of an SME document collection
(Johnson, Douze and Jégou, 2021). For private knowledge bases RAG is the
canonical design: the corpus stays local, answers can carry provenance,
and updating knowledge means re-indexing documents rather than
retraining a model.

The field has moved quickly from this foundation. Gao et al. (2023)
trace an evolution from naive RAG through advanced and modular variants,
incorporating query rewriting, re-ranking and iterative retrieval. The
limits are equally well documented. Retrieval demonstrably reduces
hallucination in knowledge-grounded generation (Shuster et al., 2021),
yet benchmark studies show that models remain fragile in the presence of
noisy passages, integrate multiple sources imperfectly, and fail at
negative rejection, answering confidently when the evidence does not
contain the answer (Chen et al., 2024). Long-context research adds that
relevant evidence may be under-used depending on where it appears in the
prompt (Liu et al., 2024), a finding that binds tightly when the small
context windows affordable on an edge device force aggressive truncation
of retrieved material.

The agentic turn treats the language model as a controller that plans,
acts and observes rather than as a single-pass generator. ReAct
interleaves reasoning traces with tool actions (Yao et al., 2023), and
Reflexion adds verbal self-critique across attempts (Shinn et al.,
2023). Applied to retrieval, these patterns yield agentic RAG: pipelines
that decide when and what to retrieve, decompose complex queries, and
reflect on whether the evidence gathered is sufficient before answering
(Singh et al., 2025). Closest to this project are the self-correcting
variants. Self-RAG fine-tunes a model to emit reflection tokens that
critique its own retrieval and generations (Asai et al., 2024), while
corrective RAG interposes a lightweight retrieval evaluator and triggers
corrective actions, including web search, when retrieved evidence is
judged unreliable (Yan et al., 2024).

These systems demonstrate that generate-then-verify loops improve
reliability, but their assumptions do not transfer to the setting
studied here. Self-RAG requires specialised fine-tuning of the deployed
model, which is outside the resources of a lightweight SME deployment.
Corrective RAG's fallback is web search, unavailable by design in a
private offline system. More generally, the agentic RAG literature
evaluates almost exclusively with cloud-scale models for which an
additional model call is cheap (Singh et al., 2025); multi-call loops
multiply precisely the resource an edge device lacks. Emerging on-device
RAG work, meanwhile, concentrates on efficiency, including retrieval
acceleration and caching, rather than on the trustworthiness of answers
(Xu et al., 2024; Zheng et al., 2025). What is missing is an agentic
pipeline whose verification behaviour has been designed for, and
empirically measured under, single-device small-model constraints.

## 2.5 Hallucination, verification and confidence calibration

Hallucination is the central obstacle to trusting generative systems in
operational settings. Surveys of natural language generation distinguish
intrinsic hallucination, which contradicts the source, from extrinsic
hallucination, which cannot be verified against it (Ji et al., 2023).
Taxonomies specific to large language models separate factuality errors,
conflicts with world knowledge, from faithfulness errors, conflicts with
the provided context or instructions (Zhang et al., 2023; Huang et al.,
2025). For a RAG assistant the faithfulness dimension is the operative
one: every claim in an answer should be supported by the retrieved
evidence. The confined, known corpus of an SME knowledge base makes such
checking unusually tractable, one of the few respects in which the
small-firm setting makes the technical problem easier rather than
harder.

One family of detection methods exploits consistency across samples.
SelfCheckGPT samples multiple stochastic generations and treats
disagreement between them as a hallucination signal (Manakul, Liusie and
Gales, 2023), building on the observation that self-consistency across
sampled reasoning paths tracks correctness (Wang et al., 2023). Semantic
uncertainty sharpens this by clustering samples that mean the same thing
before computing entropy over meanings (Kuhn, Gal and Farquhar, 2023),
and semantic entropy has since been shown to detect a broad class of
confabulations (Farquhar et al., 2024). The effectiveness of this family
is well evidenced; so is its cost. Five to twenty generations per answer
is a billing inconvenience at cloud scale, but on an edge CPU generating
a few tokens per second it converts a thirty-second answer into several
minutes, which is fatal for interactive use.

A second family verifies claims directly against evidence. FActScore
decomposes generated text into atomic facts and validates each against a
source corpus (Min et al., 2023). RAG-specific evaluation frameworks
such as RAGAS operationalise faithfulness and context quality as
automated metrics (Es et al., 2024), and benchmark work quantifies how
often RAG systems answer despite insufficient evidence (Chen et al.,
2024). This family maps naturally onto RAG, because the evidence against
which claims must be checked has already been retrieved; verification
reduces to an entailment judgement between generated claims and source
passages. In practice, however, implementations typically delegate that
judgement to a strong external model acting as judge, reintroducing
exactly the cloud dependency an on-device system exists to remove.

Verification decides whether an answer is supported; calibration decides
whether the system's expressed confidence can be believed. The
foundational result is that modern neural networks are systematically
miscalibrated, with expected calibration error and temperature scaling
as standard diagnosis and remedy (Guo et al., 2017). For language models
specifically, larger models show meaningful self-knowledge when asked to
judge their own outputs (Kadavath et al., 2022), models can be trained
to express calibrated uncertainty in words (Lin, Hilton and Evans,
2022), yet verbalised confidence elicited from instruction-tuned models
tends towards systematic overconfidence (Xiong et al., 2024). Selective
prediction research frames the practical goal: a system should answer
when it is likely to be right and abstain otherwise (Kamath, Jia and
Liang, 2020). A recent survey organises uncertainty quantification
methods partly by computational efficiency and identifies efficient
estimation as an open challenge (Liu et al., 2025). For operational
decision support, calibration is what makes a confidence flag
actionable: a low-confidence label must correspond to a materially
higher error rate, and refusals must land on the questions the corpus
cannot answer.

Three regularities emerge across this theme. First, reliability is
consistently bought with computation: extra samples, extra model calls,
or a large judge model. Second, the methods are designed and evaluated
on cloud-scale models; the calibration behaviour of small, heavily
quantised models operating inside a RAG pipeline is barely reported.
Third, verification is used mainly as offline evaluation tooling rather
than as a runtime safeguard inside a deployed pipeline. Whether claim-level
verification remains effective, and affordable, when it runs as a second small
model on hardware where every forward pass is expensive, is an open empirical
question. That question is the core of this project, examined through RQ2, RQ3
and RQ4. Confidence calibration is reviewed here as part of the trustworthiness
literature and informs the design, but the present study does not measure the
calibration of its own confidence mechanism; Chapter 3 states that boundary and
Chapter 7 proposes the measurement as further work.

## 2.6 Research gap

Each literature reviewed above is mature on its own terms, and each
supplies this project with something specific. The SME literature
supplies the motivation: firms with real knowledge management needs that
cloud AI cannot serve on privacy, regulatory and cost grounds (Section
2.2). The edge deployment literature supplies feasibility of the
components: quantised small models generating usefully on single-board
hardware (Section 2.3). The agentic RAG literature supplies the
interaction patterns, including self-correction loops that improve
reliability at cloud scale (Section 2.4). The verification and
calibration literature supplies the mechanisms and the metrics by which
trustworthiness can be engineered and measured (Section 2.5).

Their intersection, however, is unoccupied. No study identified in this
review integrates claim-level verification into a fully on-device agentic RAG
pipeline and then measures both the reliability benefit and the resource cost of
doing so on edge hardware; none does so in an SME knowledge management setting,
and none isolates the benefit against the cheap alternative of a document
metadata filter. The
nearest works fall short on one axis or another: self-correcting RAG
assumes fine-tuning or web access (Asai et al., 2024; Yan et al., 2024),
edge benchmarking stops at isolated model performance (Husom et al.,
2025), and verification and uncertainty methods leave their edge-scale
cost unexamined (Manakul, Liusie and Gales, 2023; Liu et al., 2025).
This absence was checked against the field's principal surveys (Singh
et al., 2025; Zheng et al., 2025; Liu et al., 2025) and the claim is
deliberately falsifiable; it will be revisited if contrary work emerges
before submission. The project responds with a Design Science Research
artefact evaluated under a pre-registered analysis plan (Hevner et al., 2004;
Peffers et al., 2007): a four-arm comparison isolating evidence format,
retrieval mode and verification, manually scored under blinding against a
corpus of typed planted conflicts, and benchmarked across laptop and Raspberry
Pi 5 hardware. The four arms answer the four sub-questions set out in Chapter
1.

## 2.7 Summary

This chapter traced a single argument through five literatures. SMEs
hold operational knowledge they struggle to use and face privacy,
regulatory and cost barriers that make cloud AI assistants unattractive.
Small language models, quantisation and efficient runtimes now make
local generation feasible on commodity edge hardware.
Retrieval-augmented generation grounds that generation in the firm's
own documents, and agentic patterns add self-correction, but the
reliability mechanisms proven at cloud scale assume resources an edge
device does not have. Verification and confidence calibration methods
exist in abundance, yet their effectiveness and cost when compressed
onto a single small-model device are unmeasured, and it is exactly this
combination that operational decision support requires. The project
occupies that intersection: it builds a fully on-device, verified agentic RAG
assistant and measures what the verification layer delivers, what it costs, and
whether it beats the far cheaper alternative of filtering on document status. The next chapter sets out the
methodology, architecture and evaluation design through which this is
done.
