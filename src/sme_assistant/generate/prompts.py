"""Prompt templates.

The baseline prompt is the most important thing in this file, and the important
property is what it does **not** say.

It gives no guidance about superseded documents, version precedence, effective
dates or conflicting evidence. That is deliberate and it is load-bearing. The
ablation in Chapter 4 compares this pipeline against the same pipeline with the
verification layer added, and if conflict handling were quietly written into
the baseline prompt the comparison would measure nothing. A baseline that has
been secretly helped is not a baseline.

So this is an ordinary retrieval-augmented prompt of the kind any competent
implementation would write: answer from the evidence, cite your sources, say so
if the evidence does not contain the answer. Everything beyond that belongs to
the contribution and must be earned by it.

The evidence block does carry `[SUPERSEDED, replaced by X]` markers, because
those come from the retriever and are part of the corpus metadata rather than
prompt engineering. Removing them would be hobbling the baseline, which is as
dishonest as helping it. The baseline is told nothing about what to *do* with
that marker.
"""

from __future__ import annotations

BASELINE_SYSTEM = """You are an assistant answering questions about an organisation's internal policies and procedures.

Rules:
1. Answer using only the evidence provided. Do not use outside knowledge.
2. Cite the identifier in square brackets, for example [HR-13#001], after every statement you make.
3. If the evidence does not contain the answer, say so plainly and cite nothing.
4. Be concise. Do not add advice that is not in the evidence."""

BASELINE_TEMPLATE = """{system}

EVIDENCE:
{evidence}

QUESTION: {question}

ANSWER:"""

NO_EVIDENCE_TEMPLATE = """{system}

No relevant evidence was retrieved from the knowledge base.

QUESTION: {question}

ANSWER:"""


def build_baseline_prompt(question: str, evidence: str) -> str:
    """A standard grounded-generation prompt, with no conflict handling.

    Kept free of anything the verification layer is supposed to contribute, so
    the ablation measures the contribution rather than the prompt.
    """
    if not evidence.strip():
        return NO_EVIDENCE_TEMPLATE.format(system=BASELINE_SYSTEM, question=question)
    return BASELINE_TEMPLATE.format(
        system=BASELINE_SYSTEM, evidence=evidence, question=question
    )
