# Pre-registration of hypotheses and analysis plan

**Project:** A Privacy-Preserving On-Device Agentic Assistant for SME Knowledge
Management and Operational Decision Support
**Author:** Md Basar Basar, 5753701, MSc Applied AI, WMG, University of Warwick
**Supervisor:** Manoj Babu
**Registered:** 8 August 2026

---

## Why this document exists and when it was written

This is written **before the verification layer exists**, before any question
set has been run, and before any result has been seen for any arm on any
conflict family. The only generation results in hand are the four pilot answers
from Stage 4, recorded in `docs/PILOT_STAGE4.md` and permanently excluded from
every reported set.

The reason for writing it now is narrow and practical. A system's author is the
worst possible judge of whether an ambiguous result supports their contribution,
because the space of defensible analyses is large enough that one of them will
usually favour the system. Choosing the analysis in advance removes that
freedom. Where an analysis is chosen after the fact, this document is amended
with the date and the reason, and the amendment is reported in the
dissertation rather than quietly folded in.

State frozen at registration:

| | |
|---|---|
| Corpus | `3060ae540015` (38 documents) |
| Chunk set | `5fc6a227c0a1` (138 chunks) |
| Conflict families | 9 reported, 4 tuning |
| Question set | version 1.1, 78 questions, 42 groups |
| Tests | 271 passing |

Superseded by amendment 1.1 of the same date. See section 8. The values above are the amended ones; the originals are recorded in the amendment.

---

## 1. Research questions

**RQ1.** Can a retrieval-augmented assistant running entirely on consumer and
single-board hardware answer SME policy questions with verifiable citations?

**RQ2.** Does an explicit verification layer improve the handling of
contradictory evidence beyond what document metadata alone achieves?

**RQ3.** Does the verification layer improve appropriate abstention on questions
the corpus cannot answer?

**RQ4.** What is the latency and thermal cost of the verification layer on a
Raspberry Pi 5, and is it tolerable for the intended use?

RQ2 is the contribution. RQ1 establishes that the baseline works at all, RQ3
tests a second failure mode, and RQ4 establishes feasibility.

---

## 2. Experimental arms

All four arms share the corpus, the chunk set, the index, the embedding model,
the generation model, the seed and the retrieval parameters.

**The arms are a tree rooted at B, not a ladder.** An earlier version of this
document claimed that adjacent arms differ in one respect each. They do not:
C to D changes both the retrieval mode and the verification layer, so a C to D
difference is not attributable to verification alone. The single-variable
contrasts are:

| Contrast | Variable | What it answers |
|---|---|---|
| **A vs B** | evidence format | Does exposing status metadata help at all? |
| **B vs C** | retrieval mode | Does filtering out withdrawn documents help beyond showing their status? |
| **B vs D** | verification | **The clean test of the contribution.** |
| C vs D | retrieval mode *and* verification | The practical comparison against the cheap filter. Reported, but not treated as isolating verification. |

**B versus D is the confirmatory contrast for H1 and H2.** C versus D is
reported alongside as the practical question a practitioner would ask, and is
described as such rather than as an ablation.

| Arm | Retrieval | Evidence shown | Verification | What it isolates |
|---|---|---|---|---|
| **A** | `ALL` | `PLAIN` | none | Naive RAG. Identifier and text only, no status marker, no dates. |
| **B** | `ALL` | `WITH_STATUS` | none | The value of exposing metadata in the prompt. |
| **C** | `CURRENT_ONLY` | `WITH_STATUS` | none | The metadata-filter baseline: drop superseded documents before ranking. |
| **D** | `ALL` | `WITH_STATUS` | yes | The contribution. |

Arm C is the arm that matters most to the honesty of the result. A three-line
filter on `status == "current"` resolves every supersession conflict without any
reasoning at all. If the dissertation reported only A against D it would credit
the verification layer with work a metadata filter does for free. Arm C exists
to take that credit away.

---

## 3. Hypotheses

Directional and stated before any data.

### H1 - Supersession conflicts are largely solved by metadata

On the four `version_supersession` families:

> **A < B ≈ C ≈ D**

Arm A, which never sees a status marker, cites withdrawn policy more often than
the other three. Arms B, C and D do not differ meaningfully from one another.

**Rationale.** The Stage 4 pilot showed a 3B model resolving a supersession
conflict correctly when the marker was present. If that reproduces, the
verification layer adds nothing here, and saying so in advance is what stops it
being reported as a success later.

**What would falsify it.** D exceeding B by more than the pre-specified
threshold in section 5.

### H2 - Conflicts between two live documents are not

On the five `current_current` families:

> **A ≈ B ≈ C < D**

Arms A, B and C perform similarly and poorly. Arm D is better.

**Rationale.** No metadata distinguishes the documents. Both are current, both
are in force, neither supersedes the other. A filter cannot help, a status
marker cannot help, and only reasoning over the claims themselves can detect
that the evidence disagrees. This is the entire quantitative case for the
contribution.

**What would falsify it.** D not exceeding B by the threshold in section 5. This is the hypothesis most likely to fail and the
one that decides whether the dissertation reports a positive or a null result.

### H3 - Citation validity overstates citation quality

Across all arms:

> **citation support < citation validity**

The rate at which cited passages actually contain the claim is lower than the
rate at which cited identifiers are real and retrieved.

**Rationale.** Observed once already, in the pilot: an answer citing `IT-03#001`
for a deadline that appears only in `IT-03#002` passed every check of the time.
H3 predicts that was not a one-off.

### H4 - Abstention is unreliable without structure

On the deliberate gaps:

> **D abstains appropriately more often than A, B or C**

**Rationale.** The baseline can only express refusal in free text, which is why
refusal detection is currently a heuristic and reported as diagnostic only. A
structured `INSUFFICIENT_EVIDENCE` verdict is either present or absent.

### H5 - Verification costs roughly one extra generation

On the Raspberry Pi 5:

> **latency(D) is between 1.5x and 2.5x latency(B)**

**Rationale.** Arm D adds one verification pass per answer over a short prompt
with a small `num_predict`. Prefill dominates on the Pi, so the added cost
should be less than a full second answer but more than trivial. A ratio above
2.5x indicates the verifier is doing more work than intended; below 1.5x
suggests it is not verifying much.

---

## 4. Metrics

### Primary

| Metric | Definition | Applies to |
|---|---|---|
| **Conflict handling** | Manual, blinded, three-point: 2 correct, 1 partial, 0 wrong. Correct means the answer states the governing position and, where unresolvable, discloses the disagreement. | Conflict families |
| **Superseded citation rate** | Fraction of answers citing a withdrawn document as authority | Supersession families |
| **Appropriate abstention** | Fraction of unanswerable questions where the system declines rather than answers | Gaps |
| **Answer correctness** | Manual, blinded, three-point, scored against the question's `required_claims` and `forbidden_claims`. Added under amendment 1.1: RQ1 asks whether the assistant answers correctly, and no metric measured that. | Factual, partial and synthesis questions |

### Secondary

| Metric | Source |
|---|---|
| Citation validity | `GroundedAnswer.has_valid_citation_ids` |
| Citation support | `evaluation.answer_scoring.citation_support` |
| Citation completeness | `evaluation.answer_scoring.citation_completeness` |
| Hallucinated citation rate | `GroundedAnswer.hallucinated_citations` |
| End-to-end latency | `GroundedAnswer.wall_seconds` |
| Peak CPU temperature and throttle state | `common.hostinfo` |

The primary conflict metric is **manual and blinded**. Automatic scoring cannot
judge whether an answer adequately discloses a disagreement, and pretending
otherwise would be the weakest part of the evaluation. The blinding procedure is
`evaluation.run_writer.write_review_sheet`: answers pooled across arms,
shuffled, arm labels replaced by opaque codes, key written to a separate file
and not opened until scoring is complete.

Citation support is a **lower bound** by construction and is reported as such.

---

## 5. Analysis plan

### Unit of analysis

The **conflict family**, not the question. Three paraphrases of one family are
one observation, and enforced as such by
`evaluation.question_set.validate_question_set` and
`evaluation.aggregate.aggregate`. Both question-level and family-level figures
are reported; every claim of difference uses the family level, and the sample
size cited is the number of families.

### Held-out evaluation, not cross-validation

The nine reported families form a **fixed held-out set**, scored once, with
results macro-averaged by family. This was called "leave-one-family-out
cross-validation", which was wrong: nothing is trained on the remaining folds,
so there is no model being validated. The name implied a resampling procedure
the study does not perform.

A leave-one-family-out calculation is still available in
`QuestionSet.leave_one_family_out` and is reported as a **sensitivity
analysis**: it shows how far the overall figure moves when any single family is
removed, which is the relevant question when the sample is nine. It is not
presented as validation.

### Tuning and contamination

Prompt wording, similarity thresholds, verifier output format and every other
design decision are tuned **only** on the tuning families and the non-conflict
development questions. The tuning families are **TUNE-01** (returns window, OPS-02 against CS-03) and
**TUNE-02** (accident record review timescale, OPS-08 against REG-01). Both are
`current_current`, both are marked `split: "tuning"` in the registry, and
neither appears in any reported result. The boundary is enforced in code rather
than by intention: `evaluation.question_set` refuses a reported family in the
development split and refuses a tuning family in the test split, so a
contaminating question set cannot be written or loaded.

They were chosen to match the *shape* of the reported families - a quantitative
threshold and a timescale, a general policy disagreeing with a specific
procedure - without duplicating any fact that appears in CONF-01 to CONF-09, so
that tuning against them generalises without leaking a specific answer.

### Comparison procedure

Paired by family. For each family, the per-family mean is computed for each arm,
and differences are taken within family before averaging. Standard errors are
computed over family means.

### Threshold for claiming a difference

With nine families, no conventional significance test is credible, and this
document will not pretend otherwise. A difference is reported as **supported**
when both hold:

1. the paired mean difference exceeds **0.25** on the three-point conflict scale
   (that is, better than a quarter step per family on average), and
2. the direction holds in at least **4 of the 5** `current_current` families for
   H2, or **3 of the 4** supersession families for H1.

A difference meeting one criterion but not the other is reported as
**suggestive**. A difference meeting neither is reported as **not supported**,
including when the point estimate favours the contribution.

### Multiplicity

Five hypotheses are tested. No correction is applied, because no p-values are
computed and the thresholds above are not significance tests. The number of
hypotheses is stated so a reader can discount accordingly.

---

## 6. Stopping rules

These exist so that a decision to stop or extend cannot be made on the basis of
a result.

- **The question set is frozen** once written and validated. Questions are not
  added, removed or reworded after any arm has been run on them. A question
  found to be defective after running is reported and excluded, with the reason,
  not silently replaced.
- **Runs are not repeated to obtain a better result.** A run is repeated only
  for a recorded technical fault: a crash, a timeout, a thermal event, or an
  environment mismatch detected by the manifest. The reason is recorded in the
  run directory and the discarded run is retained.
- **The verifier is not modified after seeing conflict-family results.** If it
  is, the modification and its date are recorded here, all reported results are
  re-run, and the earlier results are reported alongside as an amendment.
- **Manual scoring is completed before the blinding key is opened.**
- **A null result is reported.** If H2 is not supported, the dissertation
  reports that the verification layer did not improve on the metadata baseline,
  and analyses why. This is stated in advance because it is the outcome under
  most pressure to be reframed.

---

## 7. What would make this study wrong

Recorded now rather than discovered by an examiner.

| Threat | Status |
|---|---|
| Corpus is synthetic and single-organisation | Accepted. External validity limited; stated in the limitations chapter. |
| Nine families is a small sample | Accepted and central. Drives the family-level analysis, the paired comparison and the refusal to compute p-values. |
| Conflicts were planted by the author | Partly mitigated: CONF-07 was an accidental contradiction found in the corpus, not planted. The rest are deliberate. |
| The author scores the answers | Mitigated by blinding. Not eliminated: one rater, no inter-rater agreement. A second rater on a sample would be better and is proposed as further work. |
| Superseded documents are 13 of 133 chunks | Reported as a corpus property. A different ratio would change the retrieval baseline. |
| Automatic citation support is a lower bound | Stated in every result payload via `is_lower_bound`. |
| The Pi throttles under sustained load | Measured, not assumed. Temperature and throttle state are recorded at both ends of every run. |
| Ollama endpoint could be tunnelled to the wrong machine | Detected once already. Every run records `model_store_fingerprint` and host, and the build-time difference between machines is checked. |

---

## 8. Amendments

None at registration. Any change below this line carries a date and a reason,
and is reported in the dissertation.

---

# Amendment 1.1 - 8 August 2026

Made **before any arm was run on any question**, in response to an independent
read-only review of commit `9dae3f9`. No result exists for any arm on any
family, so nothing below could have been chosen to suit an outcome. Every change
is recorded here rather than folded into the text above, except where the text
above has been corrected and the original is reproduced in this amendment.

## 1.1.1 Two reported families were contaminated by the Stage 4 pilot

**Found.** `docs/PILOT_STAGE4.md` states that its four questions "belong
permanently to the development split". Its section headings name **CONF-01**
(mileage) and **CONF-05** (lost laptop). Version 1.0 of the question set placed
both in the test split, and `CONF-01-Q1` was the pilot question verbatim. The
lost-laptop pilot answer is what motivated `evaluation/answer_scoring.py`, so
CONF-05 did not merely get looked at, it changed the architecture.

**Done.** Both moved to `split: "tuning"` in the registry, keeping their
identifiers so the move is visible rather than tidied away. Two replacements
were planted:

| New family | Type | Risk | Documents | Conflict |
|---|---|---|---|---|
| **CONF-10** | version_supersession | medium | CS-04 / CS-14 | Stage 1 complaint resolution, 3 or 2 working days |
| **CONF-11** | current_current | high | GEN-03 / OPS-05 | Whether a visitor needs safety footwear in the warehouse |

The reported set is again **4 supersession and 5 `current_current`**, so the
thresholds in section 5 stand unchanged. CONF-11 was made high risk because
CONF-05 was the only high-risk `current_current` family and its removal would
otherwise have left that class entirely low and medium.

The tuning split now holds CONF-01, CONF-05, TUNE-01 and TUNE-02.

## 1.1.2 The question set scored paraphrases against a shared fact list

**Found.** Version 1.0 gave each family three questions that asked genuinely
different things - password length, rotation period and SMS authentication -
and one family-level `gold_facts` list. A concise correct answer about rotation
would have been marked wrong for omitting "14 characters", and an answer
reciting the length while ignoring the question would have passed.

**Done.** Each family now has **one focal claim and three genuine paraphrases
of it**. `gold_facts` is replaced by per-question `required_claims`,
`forbidden_claims` and `acceptable_variants`. The manual 0/1/2 criteria are
defined per expected behaviour in `SCORING_RUBRICS` and written into the
question set file, so the rubric in force cannot drift from the scores it
produced. A test asserts that all paraphrases of a family share one claim set.

The remaining disputed facts stay in the registry. They are simply not what
these questions measure.

## 1.1.3 The sample size was misstated

**Found.** The 42-group figure spans both splits.

**Done.** Section 5 now states the reported sample as **26 test groups**, with
the per-hypothesis denominator given alongside each claim. The 42-group figure
describes the artefact and is never used as a sample size.

## 1.1.4 Leave-one-family-out is not cross-validation

**Found.** Nothing is trained on the remaining folds, so there is no model being
validated. The name implied a resampling procedure the study does not perform.

**Done.** Renamed to fixed held-out evaluation with family-level macro
averaging. The leave-one-family-out calculation is retained and reported as a
**sensitivity analysis**, which is the honest use of it at n=9.

## 1.1.5 The arms are a tree, not a ladder

**Found.** Section 2 claimed adjacent arms differ in one respect each. C to D
changes both retrieval mode and verification.

**Done.** Section 2 now gives the single-variable contrasts explicitly. **B
versus D is the confirmatory contrast** for H1 and H2. C versus D is reported as
the practical comparison against the cheap metadata filter and is described as
such, not as an ablation.

## 1.1.6 RQ1 had no correctness metric

**Found.** RQ1 asks whether the assistant answers correctly. Every metric
measured citation behaviour, conflict handling or abstention. Nothing measured
whether an ordinary answer was right.

**Done.** Answer correctness added as a primary metric for factual, partial and
synthesis questions, scored manually and blind against `required_claims` and
`forbidden_claims`.

## 1.1.7 Hardware conditions collided with experimental arms

**Found.** `config.json` labelled hardware conditions A, B and C while the
ablation used arms A, B, C and D. "Condition B" was ambiguous.

**Done.** Hardware conditions renamed to `laptop_gpu`, `laptop_cpu` and
`pi5_cpu`. No letters.

## 1.1.8 Provenance was recorded but never checked

**Found.** The question set stored a corpus hash and never compared it. It
stored no chunk-set hash at all, which is the one that matters: expected chunk
identifiers keep resolving after a chunker change while pointing at different
text. Tests confirmed the identifiers existed, not that the chunks contained the
claims. The run writer recorded neither the question set, the registry nor this
document.

**Done.**

- `check_provenance` compares corpus, chunk-set and registry hashes on load and
  refuses a mismatch. A missing recorded hash is refused, not skipped.
- The question set records the chunk-set hash.
- The builder asserts that every expected chunk contains its family's anchor
  text, and that figures in required claims appear in the expected evidence.
- The run manifest records `question_set_sha256`, `registry_sha256` and
  `preregistration_sha256`.

## 1.1.9 Arm A recorded evidence it was never shown

**Found.** `RunWriter.record` called `evidence_text()` with no argument, which
defaults to `WITH_STATUS`. Arm A is defined by receiving evidence *without*
status markers, so its records stored supersession markers the model never saw,
and the saved prompt and saved evidence contradicted each other.

**Done.** Evidence is rendered in the arm's own format, and an unrecognised
format now raises rather than silently defaulting.

## 1.1.10 The blinding was defeatable

**Found.** The blinded review sheet included the evidence block. Arm A's
evidence has no status markers, so its presence or absence identified the arm on
sight. A reviewer noticing the pattern once had deanonymised the whole sheet.

**Done.** The evidence block is removed from the review sheet. Conflict handling
and answer correctness are scored from the answer against the question's rubric,
which is what the reviewer judges in any case. Citation support is scored
automatically from the full record, where the evidence is still stored verbatim.
`group_id` is retained in the sheet so manual scores can be macro-averaged; it
carries no arm signal.

## 1.1.11 The chunker had a dead guard

**Found.** The short-tail merge was guarded by `if True`, so it merged
unconditionally while the comment above it described a rule that was not
enforced.

**Assessed.** Measured across all documents, every tail merge joins immediately
adjacent sibling sections and none reaches `max_words`, and the merged chunk
already names both sections. The behaviour was correct; the comment was wrong.
Refusing to merge would leave 20 stub chunks, one of them 11 words, which is a
worse retrieval unit than a correctly labelled merge.

**Done.** A real guard replaces `if True`: the tail must be contiguous with the
chunk before it and the result must fit inside `max_words`. Both hold for every
current document, so the chunk set is unchanged. Two regression tests cover the
merge and the refusal. Section metadata is now included in the chunk-set
fingerprint, which a relabelling with unchanged text previously slipped past.

## 1.1.12 Conflict-handling behaviour is now one consistent rule

**Found.** The CONF-05 gold answer recommended the shorter one-hour deadline as
a safe interim action while other families only surfaced and escalated.

**Done.** One rule across all families: **surface both positions, name both
documents, state that neither supersedes the other, and escalate.** The
assistant does not recommend an interim action. This is a design decision, not a
finding, and it is the item most worth confirming with the supervisor before
Stage 5, because a decision-support system that never recommends the
conservative option is arguably less useful than one that does. If the rule
changes, it changes for every family and this amendment records the change.

## What was not changed

The hypotheses, the thresholds in section 5, the stopping rules and the primary
metrics are unchanged. The family counts that the thresholds depend on were
deliberately restored rather than the thresholds relaxed.

## State after this amendment

| | Before | After |
|---|---|---|
| Corpus | `14295c5db36d`, 37 documents | `3060ae540015`, 38 documents |
| Chunk set | `feccf6cccf2b`, 134 chunks | `5fc6a227c0a1`, 138 chunks |
| Reported families | 9, two contaminated | 9, none contaminated |
| Tuning families | 2 | 4 |
| Question set | 72 questions, 40 groups | 78 questions, 42 groups |
| Reported sample | misstated as 40 groups | 26 test groups |
| Tests | 249 | 271 |

No arm has been run. Stage 5 has not begun.
