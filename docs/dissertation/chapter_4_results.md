# Chapter 4: Results

> Working draft, 19 August 2026. Written against `docs/PREREGISTRATION.md` and
> the committed analysis outputs, not against the 16 July methodology draft,
> which describes an earlier design. Chapter 3 is to be reconciled with the
> experiment as executed before Chapter 5 is written. Delete this note before
> transferring into the WMG template.

## 4.1 Introduction

This chapter reports what the evaluation measured. It is organised by
hypothesis, in the order the hypotheses were registered, so that each result
can be read directly against the prediction and the decision rule fixed before
any arm was run. Section 4.2 states the evaluation as it was actually executed.
Section 4.3 establishes how far the measurements can be trusted, before any
verdict is given, because several of them are limited in ways that change how
the results should be read. Sections 4.4 to 4.9 take the hypotheses in turn.
Section 4.10 reports the one primary metric that carries no hypothesis.
Interpretation, comparison with the literature and implications for deployment
are held for Chapter 5.

## 4.2 The evaluation as executed

Four arms were run over a single frozen corpus of 38 documents and 141 chunks,
sharing the index, the embedding model, the generation model, the seed and the
retrieval parameters. Arm A is naive retrieval-augmented generation, showing
identifiers and text only. Arm B adds document status metadata to the evidence
block. Arm C filters superseded documents out before ranking. Arm D adds the
verification layer to the same evidence Arm B sees. The arms form a tree rooted
at B rather than a ladder, so B against D is the single-variable contrast that
isolates verification, and C against D changes retrieval mode and verification
together and is reported as a practical comparison rather than as an ablation.

The reported sample is the held-out test split: 68 questions in 32 groups per
arm, comprising 45 conflict questions across 15 families, 10 unanswerable
questions across 5 deliberate gap topics, 8 factual, 3 synthesis and 2 partial
questions. The unit of analysis is the family or group, not the question. Every
claim of difference is made at that level, and the sample size cited is the
number of families.

All four quality runs used `llama3.2:3b` for generation, with `qwen2.5:3b` as
the verifier in Arm D, executed through Ollama on the development laptop. The
runs are closed to a named list and are provenance-checked on every read, so a
later run cannot enter the analysis. Timing was measured separately, in
performance-only executions that compute no quality metric at all.

## 4.3 Measurement quality and analysis integrity

The primary conflict metric is a manual three-point rubric applied by one
reviewer. Its reliability was measured rather than assumed. The review sheet
contained 58 groups of byte-identical answers scored independently, blind, at
separations of up to 250 items in a shuffled sheet. The rubric score agreed in
58 of 58 groups, with no divergences at all.

The `abstained` flag did not hold up as well. It drifted monotonically with
position through the first pass, and a second focused pass was run over all 272
items in a separately seeded order under a rule fixed in advance. The two
passes agree on 253 of 272 items, a rate of 0.930, and the second pass is the
reported value throughout. The second pass was internally consistent on all 58
duplicate groups where the first managed 52. One divergence remains
unreconciled across the whole sheet: items 27 and 221 are the same answer to
CONF-04-Q2 with the same rubric score, differing only on whether a conflict was
asserted. It falls outside the denominator of every hypothesis stated over that
field, and any descriptive figure depending on it is reported both ways.

The blinding is **partial and self-monitored**, and is reported as such.
Thirteen of the 272 items carry the system's abstention template verbatim, a
string only the verified arm can produce. Recognising it once attaches an
opaque code to that arm, and the code then identifies all 68 of its items.
Unblinding was self-reported on none of the 266 items where the question was
put, and that figure is not evidence that the blinding held: those thirteen
items were identifiable on sight regardless. The affected arm is the
contribution, and the contrast it protects is the confirmatory one, so the
control is not asserted.

Two further constraints apply throughout. With fifteen families no
significance test is credible, and none is computed. A difference is reported
as supported only when the paired mean difference exceeds 0.25 on the
three-point scale **and** the direction holds in 3 of 4 supersession families
or 6 of 8 live-disagreement families; meeting one criterion alone is
suggestive, and meeting neither is not supported, including where the point
estimate favours the contribution. No confidence interval is computed anywhere,
so no arm is described as equivalent to another.

## 4.4 H1: supersession conflicts

H1 predicted A < B ~ C ~ D on the four `version_supersession` families, on the
reasoning that a status marker alone should resolve them and the verification
layer should add nothing.

**Table 4.1** Conflict handling on the supersession families, three-point scale,
4 families and 12 questions per arm

| Arm | Mean | A vs arm, paired | Direction |
|---|---:|---:|---:|
| A | 1.1667 | - | - |
| B | 1.4167 | +0.2500 | 2 / 4 |
| C | 1.5833 | +0.4167 | 2 / 4 |
| D | 1.2500 | +0.0833 | 1 / 4 |

Both legs fail. On the superiority leg, B exceeds A by exactly 0.250, which
does not exceed the 0.25 threshold, and the direction holds in 2 of the 4
families against the 3 required. C meets the effect criterion but not the
direction criterion and is recorded as suggestive. D barely separates from A at
all. On the within-margin leg, C and B sit within the margin and so do D and B,
but D and C differ by 0.3333 in C's favour, with C higher in 3 of 4 families
and 1 tied. That contrast changes retrieval mode and verification together, so
the difference cannot be attributed to either alone; the confound limits
attribution rather than erasing the observation.

The superseded citation rate measures the same four families directly. It is a
pre-registered primary metric that carries no hypothesis, and under amendment
1.25 it is reported without a verdict.

**Table 4.2** Answers citing a withdrawn document as authority, 12 questions in
4 families per arm

| Arm | Answers | Rate | Families with any |
|---|---:|---:|---:|
| A | 6 / 12 | 0.500 | 2 / 4 |
| B | 2 / 12 | 0.167 | 2 / 4 |
| C | 0 / 12 | 0.000 | 0 / 4 |
| D | 1 / 12 | 0.083 | 1 / 4 |

Therefore, H1 was not supported, on both legs.

## 4.5 H2: live policy disagreements

H2 is the confirmatory hypothesis and the quantitative case for the
contribution. It predicted A ~ B ~ C < D on the eight families where two
current documents disagree, three `mutually_exclusive` and five
`stricter_looser` pooled, on the reasoning that neither a status filter nor a
status marker can help when both documents are in force. The pre-registered
confirmatory contrast is D against B.

**Table 4.3** Conflict handling on the pooled live-disagreement families,
three-point scale, 8 families and 24 questions per arm

| Arm | Mean | Contrast with D | Paired difference | Direction |
|---|---:|---|---:|---:|
| A | 0.2917 | D vs A | -0.0417 | 0 / 8 |
| B | 0.3333 | **D vs B** | **-0.0833** | **0 / 8** |
| C | 0.2917 | D vs C | -0.0417 | 0 / 8 |
| D | 0.2500 | | | |

The confirmatory contrast fails on both criteria and runs in the opposite
direction to the prediction. D does not exceed B on a single one of the eight
families: it ties on six and is worse on two, CONF-12 and CONF-14. Leaving out
any one family in turn leaves the paired difference negative in every fold,
with a spread of 0.0476 across the eight, so the estimate is stable and no
single family is driving it.

The pooled result hides a split between the two subtypes, which keep separate
rubrics because they demand different behaviour. On the five `stricter_looser`
families, where naming the stricter course satisfies both documents, the arms
score 0.4667, 0.5333, 0.4667 and 0.4000 for A, B, C and D. On the three
`mutually_exclusive` families, where no single action satisfies both documents
and the correct response is to surface both and escalate, **every arm scored
zero on every family**. No configuration tested handled a mutual exclusion
correctly even once, so the pooled contrast between the arms rests entirely on
the stricter-looser material.

That stability is not evidence for the null. With eight families the study
could not have detected a small true effect, and a stable estimate of a small
negative number is not a demonstration that the true difference is zero. All
four arms also score low in absolute terms on this material, below one point on
a three-point scale, which bounds what any contrast between them can show.

Therefore, H2 was not supported.

## 4.6 H2c: over-detection on the negative controls

H2c was stated as a directional prediction against the contribution: that on
the three `compatible` families, which are negative controls, Arm D would
assert conflicts where none exist more often than Arm B. The pre-registration
states that a null result here would be a genuinely strong finding.

No arm produced a single false conflict. All four sit at 0 of 3 families and 0
of 9 questions. The prediction is falsified in the direction that favours the
system: the verification layer did not over-detect.

This is reported with its limitation attached. Zero events were observed
everywhere, across 3 control families and 9 questions per arm, which is a floor
rather than a measurement: the design could not have detected moderate
over-detection had it occurred. The CONF-04-Q2 divergence noted in section 4.3
lies outside this denominator, CONF-04 being a supersession family, so it does
not move these figures.

Therefore, H2c was not supported, and the failure of this prediction is a
favourable result for the system rather than an adverse one.

## 4.7 H3: citation validity overstates citation quality

H3 predicted that in every arm the rate at which cited passages actually
contain the claim would fall below the rate at which cited identifiers are real
and retrieved. Figures follow the convention established before unsealing:
restricted to claim-making answers, since an abstention cites nothing by
design, with the group as the unit of inference.

**Table 4.4** Citation metrics, group level, claim-making answers only

| Arm | Validity | Support | Completeness | Claim-making n |
|---|---:|---:|---:|---:|
| A | 0.8333 | 0.6667 | 0.7176 | 68 |
| B | 0.8177 | 0.6766 | 0.6748 | 68 |
| C | 0.8333 | 0.4561 | 0.6404 | 68 |
| D | 0.7989 | 0.6528 | 0.6875 | 55 |

Support falls below validity in all four arms, by between 0.14 and 0.38. The
gap is widest in Arm C. Arm D's denominator is 55 rather than 68 because
thirteen of its answers were abstentions. Citation support is a lower bound by
construction and is reported as such, so the true gap is no smaller than the
figures shown. A sensitivity analysis restricting validity and support to a
common denominator returns the same direction in every arm.

Therefore, H3 was supported.

## 4.8 H4: abstention on the deliberate gaps

H4 predicted that Arm D would abstain appropriately more often than A, B and C,
on the reasoning that a structured `INSUFFICIENT_EVIDENCE` verdict is either
present or absent whereas the baseline can only express refusal in free text.
The unit is the gap topic abstained on throughout, and the `abstained` field is
the second pass described in section 4.3.

**Table 4.5** Appropriate abstention on the 5 gap topics, 10 questions per arm

| Arm | Questions | Gap topics abstained on throughout |
|---|---:|---:|
| A | 8 / 10 | 3 / 5 |
| B | 10 / 10 | 5 / 5 |
| C | 9 / 10 | 4 / 5 |
| D | 10 / 10 | 5 / 5 |

D exceeds A and C and ties B, both at the ceiling of 5 of 5 topics. H4 as
stated requires D to exceed all three. The margin over A and C is real and is
reported, but it cannot be separated from the ceiling: with five topics there
was no room for D to exceed a baseline that abstained on every one.

Therefore, H4 was not supported.

## 4.9 H5: the latency cost of verification

H5 predicted that on the Raspberry Pi 5 the verified arm would take between 1.5
and 2.5 times as long as the unverified baseline, on the reasoning that one
verification pass over a short prompt should cost less than a full second
answer. Only the `pi5_cpu` condition carries an H5 verdict; the laptop
conditions are reported as descriptive RQ4 figures.

**Table 4.6** Mean end-to-end latency per answer, 68 questions per arm per
condition

| Condition | Arm B | Arm D | of which verification | D / B |
|---|---:|---:|---:|---:|
| Laptop, GPU | 1.09 s | 4.01 s | 2.92 s | 3.68 |
| Laptop, CPU | 6.64 s | 20.88 s | 14.24 s | 3.15 |
| **Pi 5, CPU** | **54.84 s** | **174.17 s** | **119.33 s** | **3.18** |

The observed ratio of 3.18 sits above the upper bound of the predicted range.
The ratio is also stable across three platforms whose decode rates differ by
more than twenty-fold, from 77.2 tokens per second on the laptop GPU to 3.28 on
the Pi, which indicates the cost is structural rather than a property of the
target device. The workload accounts for it: over the Pi run the verifier
prompt averaged 1,934 tokens against the draft's 900, and the verifier
generated 179 output tokens against the draft's 46, under a `num_predict` of
700 against 400.

Thermal behaviour was recorded rather than assumed. The Pi reported itself
throttled on 64 of 68 draft questions and on all 68 verifier questions, at a
mean CPU temperature of 87.1 and 88.2 degrees Celsius respectively and a
maximum of 90.3. Both arms are affected, so the ratio holds, but the absolute
latencies are those of a throttled machine.

Therefore, H5 was not supported.

## 4.10 Answer correctness

Answer correctness is the second pre-registered primary metric carrying no
hypothesis, scored on the same three-point rubric against each question's
required and forbidden claims. It applies to the 13 questions in 12 groups that
ask for a direct answer or an answer flagging a gap, and no threshold is
applied to it.

**Table 4.7** Answer correctness, three-point scale, 13 questions in 12 groups
per arm

| Arm | Question level | Group level |
|---|---:|---:|
| A | 1.6154 | 1.6667 |
| B | 1.5385 | 1.5833 |
| C | 1.4615 | 1.5417 |
| D | 1.0769 | 1.0833 |

Arm D is the weakest of the four on the questions that contain no conflict at
all. The shortfall is located rather than characterised here: D scored zero on
`FACT-account-hold`, `FACT-forklift-preuse` and `FACT-written-warning-duration`,
and on each of the three it declined to answer a question that A, B and C
answered correctly. A fourth loss, on `SYN-faulty-warranty-return`, is shared
with B and C. No refusal rate over answerable questions is computed, because
section 4 of the pre-registration defines abstention over the gaps only and a
metric over the complement would be one invented after the data.

## 4.11 Summary

**Table 4.8** Hypothesis verdicts under the pre-registered decision rule

| | Prediction | Verdict |
|---|---|---|
| H1 | A < B ~ C ~ D on 4 supersession families | not supported |
| H2 | A ~ B ~ C < D on 8 live-disagreement families | not supported |
| H2c | D over-detects on 3 compatible controls | not supported |
| H3 | citation support < validity in every arm | supported |
| H4 | D abstains more than A, B and C on 5 gap topics | not supported |
| H5 | latency(D) is 1.5x to 2.5x latency(B) on the Pi 5 | not supported |

Of the five hypotheses about the verification layer, one was supported, and it
concerns a property of citation reporting shared by every arm rather than an
effect of verification. The confirmatory contrast for the contribution, D
against B on the live-disagreement families, ran in the opposite direction to
the prediction on every family. The layer's measured cost is a factor of 3.18
in end-to-end latency on the target device. Chapter 5 examines why.

---

**Sources.** All figures in this chapter are regenerated by
`scripts/analyse_results.py` and `scripts/analyse_performance.py` from the
frozen runs. Hypothesis decisions, levels, contrasts, sensitivity analyses and
the two primary metrics of sections 4.4 and 4.10:
`results/analysis/hypotheses.json`. Latency, token rates and thermal state:
`results/analysis/performance_latest_test_performance_{laptop_gpu,laptop_cpu,pi5_cpu}.json`;
prompt and output token counts: the corresponding `answers.jsonl`. Scoring
reliability: `results/manual/consistency.json`,
`results/manual/abstention_agreement.json` and
`results/manual/drift_report.json`. Hypotheses, metric definitions and the
decision rule: `docs/PREREGISTRATION.md` sections 3 to 5; the limitations in
section 4.3 are recorded in amendments 1.13, 1.14 and 1.16, and the two primary
metrics of sections 4.4 and 4.10 in amendment 1.25.
