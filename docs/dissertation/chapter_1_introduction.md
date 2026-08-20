> Working draft. Revised 19 August 2026 to bring the research questions,
> objectives and contribution into line with the experiment as executed and the
> project's pre-registration. Bibliographic details must still be checked against
> the originals before submission. Delete this note before transferring into the
> WMG template.

# Chapter 1: Introduction

## 1.1 Background and context

Small and medium-sized enterprises (SMEs) account for the overwhelming
majority of firms and a substantial share of employment and value added
across OECD economies (OECD, 2025). Like larger organisations, they
accumulate operational knowledge: policies, standard operating
procedures, product documentation, compliance guidance and customer
support material. Unlike larger organisations, they rarely have the
systems or staff to manage it. The knowledge management literature has
repeatedly shown that small firms manage knowledge informally, that
documented knowledge is scattered and under-used, and that storage,
retention and utilisation are among the least developed knowledge
management practices in SMEs (Alavi and Leidner, 2001; Durst and
Edvardsson, 2012; Massaro et al., 2016; Durst, Edvardsson and Foli,
2023). In daily operations this materialises as staff answering
questions from memory, inconsistent application of policy, and time lost
searching documents.

Large language models (LLMs) appear well suited to this problem.
Combined with retrieval-augmented generation (RAG), they can answer
natural-language questions over an organisation's own documents (Lewis
et al., 2020). Yet the dominant delivery model is the cloud API, which
sits awkwardly with SME constraints. Sending internal policies,
personnel material or customer records to third-party services raises
confidentiality and data protection concerns under the UK and EU General
Data Protection Regulation, whose data minimisation and transfer rules
weigh heavily on firms without legal teams (Regulation (EU) 2016/679),
while the phased obligations of the EU Artificial Intelligence Act add
further compliance uncertainty (Regulation (EU) 2024/1689). Studies of
AI adoption in SMEs consistently identify cost, skills shortages, data
governance and trust as leading barriers (OECD, 2025; Oldemeyer, Jede
and Teuteberg, 2025; Sánchez, Calderón and Herrera, 2025).

Two recent developments change the calculus. Small language models
(SLMs) in the one-to-four billion parameter range now deliver useful
instruction-following capability (Abdin et al., 2024; Lu et al., 2024;
Meta AI, 2024; Yang et al., 2024), and quantisation together with
efficient inference runtimes allows them to run on commodity edge
hardware costing under £150 (Frantar et al., 2023; Lin et al., 2024;
Zheng et al., 2025). An assistant that runs entirely where the data
lives, with no per-query fees and no data leaving the premises, is now
technically plausible for an SME.

## 1.2 Problem statement

Feasibility of generation, however, is not the same as trustworthiness
of answers. RAG grounds generation in retrieved documents, and retrieval
demonstrably reduces hallucination, but it does not eliminate it: models
still produce claims that are unsupported by, or contradict, the
retrieved evidence, and they struggle to decline questions their sources
cannot answer (Shuster et al., 2021; Chen et al., 2024; Huang et al.,
2025). This matters acutely in the intended setting. If an employee asks
about a refund threshold, a safety procedure or a contractual notice
period, a fluent but unsupported answer can translate directly into
operational, financial or legal harm. The risk is sharpened by the fact
that the small, heavily quantised models an edge device can host are
generally more error-prone than their cloud-scale counterparts (Lu et
al., 2024; Husom et al., 2025).

The research community has produced mechanisms that address exactly this
failure mode: verifying generated claims against evidence, estimating
confidence, and abstaining when confidence is low (Manakul, Liusie and
Gales, 2023; Min et al., 2023; Farquhar et al., 2024; Liu et al., 2025).
However, these methods were designed for, and evaluated on, large models
with abundant compute, and they typically require multiple samples or
additional model calls per answer. Whether they remain effective and
affordable when the entire pipeline must fit on a single edge device is
largely unexamined. The result is a practical gap: on-device assistants
exist, and verification methods exist, but there is no systematic
evidence on what happens when the two are integrated under real resource
constraints. This dissertation addresses that gap.

## 1.3 Research question and objectives

The main research question is: to what extent can an on-device agentic RAG
assistant, with source verification, support private SME knowledge management
and operational decision support within the resource constraints of an edge
device?

Four sub-questions decompose it. They were refined during development and fixed
in the project's pre-registration before the frozen confirmatory experimental
runs were executed; Chapter 3 records that boundary and the reasons for the
refinement.

1. **RQ1 (grounded answering).** Can a retrieval-augmented assistant running
   entirely on consumer and single-board hardware answer SME policy questions
   with verifiable citations?

2. **RQ2 (value of verification).** Does an explicit verification layer improve
   the handling of contradictory evidence beyond what document metadata alone
   achieves?

3. **RQ3 (abstention).** Does the verification layer improve appropriate
   abstention on questions the corpus cannot answer?

4. **RQ4 (edge feasibility).** What is the latency and thermal cost of the
   verification layer on a Raspberry Pi 5, and is it tolerable for the intended
   use?

RQ2 carries the contribution. RQ1 establishes that the baseline works at all,
RQ3 tests a second failure mode, and RQ4 establishes feasibility on the intended
deployment platform, the Raspberry Pi 5.

Six objectives operationalise these questions.

1. Review the literature on small language models at the edge, agentic RAG,
   hallucination and verification, and confidence estimation, to establish the
   state of the art and confirm the research gap.

2. Design a system architecture integrating local retrieval, source-grounded
   generation and claim-level verification into a single pipeline deployable on
   a Raspberry Pi 5 (16GB).

3. Implement the artefact using local small language models over a synthetic SME
   knowledge base containing deliberately planted contradictions and deliberate
   knowledge gaps, in line with the approved ethics position (no primary data,
   no real business or personal data, no human participants).

4. Develop an evaluation framework comprising a held-out question set spanning
   answerable, partially answerable, unanswerable and conflicting-evidence
   cases, with metrics covering conflict handling, answer correctness, citation
   quality, appropriate abstention, latency and thermal behaviour.

5. Pre-register the hypotheses, comparisons and decision rules, then evaluate
   the artefact empirically against RQ1 to RQ4 through a four-arm comparison and
   cross-platform benchmarking on laptop and Raspberry Pi 5 hardware.

6. Critically analyse the results to answer the main research question and
   derive practical deployment guidance for SMEs, acknowledging limitations and
   threats to validity.

## 1.4 Contribution and dissertation structure

The intended contribution is one of integration and empirical evidence
rather than algorithmic novelty. Retrieval-augmented generation, claim
verification and confidence estimation each exist in prior work; what
does not exist is systematic evidence about their combination under edge
constraints. The project therefore contributes, first, a working
demonstration that claim-level verification can run entirely on-device within
SME-realistic hardware limits, and second, quantified evidence of the trade-off
between what the verification layer delivers in answer quality and what it costs
in latency and resources. A categorical confidence mechanism is implemented but
its calibration is not evaluated, so no claim is made for it here; measuring it
is proposed as further work in Chapter 7. Secondary contributions are a reusable
evaluation design, comprising the synthetic knowledge base with its typed
planted conflicts, the held-out question set and the pre-registered analysis
plan, and practical deployment guidance for SMEs considering private
assistants.

The project follows the Design Science Research methodology, in which
knowledge is generated through the construction and rigorous evaluation
of an artefact (Hevner et al., 2004; Peffers et al., 2007). The
verification layer is treated as effective if it measurably improves the
handling of contradictory evidence beyond what document metadata alone achieves,
at a resource cost that still permits interactive use. The criterion, the
comparison and the threshold were fixed in advance, and the commitment to report
a null result as a null result was made in writing before the experiment ran.

The dissertation proceeds as follows. Chapter 2 reviews the literature
across five themes and establishes the research gap. Chapter 3 sets out
the methodology, system architecture, knowledge base and evaluation
design. Chapter 4 reports implementation and experimental results.
Chapter 5 analyses those results against the sub-questions. Chapter 6
discusses the findings, their implications for SMEs and the study's
limitations, and Chapter 7 concludes with contributions and future work.
