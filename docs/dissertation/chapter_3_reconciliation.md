# Chapter 3 reconciliation note

**19 August 2026, actioned.** `Methodology_DRAFT.rtf` of 16 July described a
design the experiment did not follow. This note listed every claim that no
longer matched. **Chapter 3 has since been rewritten as
`chapter_3_methodology.md`, and Chapters 1 and 2 have been revised as
`chapter_1_introduction.md` and `chapter_2_literature_review.md`.** The note is
retained as the record of what was wrong and what was done about it. The RTF
drafts are now legacy and should not be transferred into the template.

Nothing here is a defect in the experiment. The draft simply predates
amendments 1.1 to 1.29, and the design moved a long way under them.

## Serious: the draft rules out the method that was used

**Grading.** Section 3.5 states that grading is automatic against gold keys,
and gives two reasons for rejecting manual judging: infeasibility at scale, and
the circularity of using a language model to judge language-model output. The
study's **primary** metrics are manual, blinded and three-point: conflict
handling and answer correctness both. Automatic citation metrics are secondary.
The reasoning in the draft is sound against *model-based* judging, which the
study also does not use, but as written it excludes the reviewer-based scoring
that produced the headline results.

Chapter 3 must instead describe: the three-point rubric, the blinding procedure
(`evaluation.run_writer.write_review_sheet`), the intra-rater reliability
measurement, the abstention re-pass, and the single-reviewer limitation.

**Research questions.** The draft answers the RQ1 to RQ4 of
`RESEARCH_FRAMEWORK.md` (model comparison, hallucination reduction, calibration,
edge feasibility). The pre-registration states a different RQ1 to RQ4
(verifiable citations, conflicting evidence beyond metadata, abstention,
latency and thermal cost on the Pi). Chapter 4 answers the second set. One set
has to go, and it cannot be the one the experiment was run against.

**Pre-registration absent.** The draft does not mention the pre-registration,
the decision rule of section 5, the tuning and reported split enforced in code,
the stopping rules, or the pre-registration amendments. These are the
strongest methodological features of the project and currently appear
nowhere in the dissertation.

## Design described but not built

**Two arms, not four.** Section 3.3 describes a verified arm and a baseline arm
differing in query analysis, refusal gate, verification and flagging. The
experiment ran four arms, A to D, differing in retrieval mode and evidence
format as well as verification, forming a tree rooted at B. Arm C, the
metadata-filter baseline, is the arm that makes the result honest and has no
counterpart in the draft at all.

**Confidence flagging.** Section 3.3 describes a weighted score (support 0.60,
retrieval 0.25, coverage 0.15), a contradiction penalty of 0.4, thresholds of
0.70 and 0.45 mapping to HIGH, MEDIUM and LOW, and an answer-relevance cap.
**None of these values appears anywhere in the source.** They survive in
`config.json` as a dead block. What the runs used is the categorical policy in
`verification.confidence`: a verdict maps to a confidence level directly. The
whole paragraph needs rewriting against the code.

**Calibration experiment.** Section 3.5 describes Experiment 3 as expected
calibration error over ten bins with reliability diagrams. No calibration
analysis exists in `results/`. RQ3 as executed is abstention, tested by H4.

**Model comparison.** Section 3.3 names three candidate generation models and
section 3.5 makes comparing them Experiment 1. No model comparison appears in
the reported results. `llama3.2:3b` generates in every arm; `qwen2.5:3b`
verifies in Arm D only, and was chosen by the diagnostic protocol of amendment
1.10, not by a comparison experiment. `config.json` lists five candidates, and
`phi3:mini` rather than Phi-3.5-mini.

**Refusal gate.** Section 3.3 says the system refuses when the best retrieval
similarity falls below 0.32. The threshold is 0.30 and, per amendment 1.3.2, it
refuses nothing: the answerable and unanswerable score distributions overlap
completely, so no threshold separates them. Abstention is the verification
layer's job, which is what H4 was stated to test. The draft describes a gate
that does not fire.

**Abstention text.** Section 3.3 attributes the refusal sentence to the
generator. Amendment 1.7.6 moved it to a fixed template written in source and
applied by the verification layer. That change is also what defeated the
blinding on thirteen items (amendment 1.13), so it has to be described
correctly for the limitation to make sense.

## Figures to update

| Section | Draft says | Actual |
|---|---|---|
| 3.4 | 32 documents | **38** documents, 141 chunks |
| 3.4 | 8 categories, counts given | recount against `results/corpus_manifest.json`, which is itself stale at 138 chunks |
| 3.5 | 60 questions: 30 answerable, 15 partial, 15 unanswerable | **109** questions in 53 groups; reported test split **68** in **32** groups |
| 3.5 | three categories | conflict, factual, partial, synthesis, unanswerable |
| 3.3 | top four chunks | `top_k: 6` |
| 3.3 | similarity threshold 0.32 | `min_similarity: 0.30` |
| 3.3 | thirty automated tests | **767** at the time of writing. Chapter 3 no longer quotes a count, because it rises with every correction and a number in prose is stale by the next commit (amendment 1.30.8) |
| 3.5 | Pi 5 and laptop | three conditions: `laptop_gpu`, `laptop_cpu`, `pi5_cpu`, placement enforced and observed |

## Missing entirely

The conflict families are the substance of the study and do not appear in the
draft. Chapter 3 needs: the four conflict types (`version_supersession`,
`mutually_exclusive`, `stricter_looser`, `compatible`), the 15 reported and 8
tuning families, why `compatible` families exist as negative controls, the five
deliberate gap topics, and the family as the unit of analysis.

## What survives unchanged

Section 3.2 on Design Science Research, including the advance commitment to
report a negative result, which the study then honoured. Section 3.6 on ethics.
The chunking parameters (180 words, one-sentence overlap). The choice of Ollama
and local-only inference. The architectural motivation for verification. These
can be carried over with light editing.

## Found while writing Chapter 3

**The organisation has a different name.** The draft calls the fictional firm
"Bramley and Finch Ltd" with 48 staff. The corpus describes **Northgate
Kitchenware Ltd**, and `docs/CORPUS.md` records why its identifiers were changed
to ranges reserved for fiction after an earlier version used a company
registration number in valid Companies House format.

**Two pipeline stages were never built.** The package docstring, and Chapter 1
objective 2, describe a six-stage pipeline including query analysis and
next-action suggestion. Neither exists in the source. The served pipeline is
retrieval, generation and, in the verified arm, verification. Chapter 1's
objective and Chapter 3's architecture section have both been corrected, and the
stale docstring in `src/sme_assistant/__init__.py` should be fixed too.

**`docs/CORPUS.md` is itself stale**, giving 37 documents where the index gives
38. The file says figures should be regenerated rather than transcribed, which is
what Chapter 3 does, but the stale number should be corrected at source.

## Still outstanding

- `src/sme_assistant/__init__.py` docstring still describes six stages.
- `docs/CORPUS.md` says 37 documents; `scripts/kb_summary.py` reports 38.
- `config.json` retains a dead `confidence.weights` block that appears nowhere
  in the source. Removing it, or marking it dead in the file, would stop the next
  reader assuming it is live.
- References needed for Chapter 3 that are not yet in `references.md`: a source
  for pre-registration as a methodological device, and Cohen (1960) for the
  agreement coefficient.
