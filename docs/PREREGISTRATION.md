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
| Corpus | `63a92324d734` (38 documents) |
| Chunk set | `2c92533c2f86` (141 chunks) |
| Conflict families | 15 reported (4 supersession / 3 mutually exclusive / 5 stricter-looser / 3 compatible), 8 tuning |
| Question set | version 1.5, 109 questions, 53 groups |
| Reported sample | 68 questions, **32 test groups** |
| Retrieval | `top_k: 6`, `min_similarity: 0.30`, calibrated on development |
| Tests | 291 passing |

Superseded by amendments 1.1 to 1.5. The values
above are the amended ones; the originals are recorded in each amendment.

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

### H2 - Live policy disagreements need reasoning

**Confirmatory.** On the eight families where two current documents disagree -
three `mutually_exclusive` and five `stricter_looser` pooled:

> **A ≈ B ≈ C < D**

**Rationale.** Both documents are current, both are in force, neither
supersedes the other. A status filter cannot help and a status marker cannot
help. Only reasoning over the claims themselves detects that the trade counter
cannot open at both 08:00 and 08:30, or that quoting 72 hours for a breach
report breaches another live document's 24. This is the quantitative case for
the contribution.

**Why pooled.** H2a and H2b were separate until amendment 1.5. Repeated
classification failures showed that sorting a family into one subtype or the
other is unreliable on this corpus: a threshold conditioned on the split would
be conditioned on a judgement that has not held up. Pooling makes the
confirmatory claim rest on the distinction that *is* robust - whether two live
documents disagree - and leaves the finer sort to description.

**The subtypes remain, and keep separate rubrics.** They demand different
behaviour: `mutually_exclusive` surfaces both and picks neither;
`stricter_looser` names the safe course. Merging the hypothesis does not merge
the scoring, and per-subtype figures are reported as descriptive breakdowns
alongside the pooled result.

**What would falsify it.** D not exceeding B by the threshold in section 5.
This is the hypothesis most likely to fail and the one that decides whether the
dissertation reports a positive or a null result.

### H2c - Verification over-detects

On the three `compatible` families, which are negative controls:

> **false-conflict rate(D) > false-conflict rate(B)**

The verification layer flags a conflict where none exists more often than the
baseline does.

**Rationale.** This is stated as a directional prediction *against* the
contribution, on purpose. A component built to find conflicts will find them
where they are not, and a system that reports a conflict between the petty cash
procedure and the procurement policy is worse than one that answers plainly.
Nothing in the earlier design measured this, so a verifier that shouted
"conflict" at every document pair would have scored perfectly on every other
hypothesis.

**What would falsify it.** D's false-conflict rate being no worse than B's,
which would be a genuinely strong result and should be reported as such.

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
| **False-conflict rate** | Fraction of `compatible` families where the answer asserts that the documents contradict each other, or declines on those grounds. Added under amendment 1.2. Without it, over-detection is invisible and a verifier that flagged every document pair would score perfectly. | Compatible negative controls |
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

## 1.1.12 Conflict-handling behaviour: surface and escalate, never recommend

**Found.** The CONF-05 gold answer recommended the shorter one-hour deadline as
a safe interim action, while every other family only surfaced the disagreement
and escalated. Two rules were in force at once.

**Decided.** One rule across all families: **state both positions, name both
documents, say that neither supersedes the other, and escalate.** The assistant
does not recommend which to follow.

This was initially left open as a question for the supervisor. It is settled
here instead, because the deciding argument is a property of the corpus rather
than a matter of taste. A "recommend the conservative reading" rule requires
that a conservative reading exists, and across the five reported
`current_current` families it usually does not:

| Family | Conservative reading? | |
|---|---|---|
| CONF-06, backup retention | **No** | Keeping data is safe for business continuity and unsafe for data protection. "Safe" points both ways. |
| CONF-07, £800 approver | **No** | Two approval routes. Neither is safer; obtaining both is slower, not safer. |
| CONF-08, trade counter hours | **No** | Neither set of hours is safer than the other. |
| CONF-09, access review cadence | Weak | Six months is arguably more diligent, but nothing is at risk in the interval. |
| CONF-11, visitor safety footwear | **Yes** | Always requiring footwear is unambiguously the safer reading. |

One family of five has a clear conservative reading. A rule that fires on one
family and not the other four cannot be scored on a single rubric, and Arm D's
score would then depend on how many risk-asymmetric conflicts happen to be in
the set rather than on how well it handles conflict. That is a measurement
defect, not a preference.

The surface-and-escalate rule applies uniformly to all five, so it is scoreable
on one rubric, and it is also the behaviour the system's stated purpose
supports: a recommendation the corpus does not contain is not a verifiable
answer, which is the property the whole design exists to provide.

**Recorded as a limitation.** For risk-asymmetric conflicts such as CONF-11, a
clearly labelled interim recommendation would be more useful to a real SME user
than escalation alone, and a system that never offers one is less helpful than
it could be. Detecting risk asymmetry automatically, rather than by the author's
judgement, is the natural extension and is proposed as further work in the
limitations chapter. It is not attempted here because the judgement would be
mine, applied to families I designed, and unfalsifiable at this sample size.

A test asserts that no `surface_both_and_qualify` gold answer contains
recommendation language, so the rule cannot drift back.

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

---

# Amendment 1.2 - 8 August 2026

Made **before any arm was run on any question**, in response to a second
independent read-only review. No result exists for any arm on any family.

Amendment 1.1 corrected how the study was measured. This one corrects what it
was measuring, which is the more serious of the two.

## 1.2.1 Two reported families were not contradictions

**Found.** CONF-07 and CONF-09 were registered as `current_current` conflicts.
Neither is a conflict.

**CONF-07.** GEN-04 reads "written approval from the department head **and**,
where the cost exceeds £750, from the Finance Manager". That is cumulative.
FIN-03 places £501 to £2,500 with the department head. Obtaining both approvals
satisfies both documents; GEN-04 is simply stricter for IT equipment.

Worse, the registry recorded GEN-04's position as *"the Finance Manager, because
the cost exceeds £750"*, dropping the conjunction that makes the requirement
cumulative and turning a stricter rule into a competing one. **The gold data
misrepresented the document.** The anchor check passed because the shortened
phrase is a literal substring: a substring can be truthful and still mislead
once the words in front of it are cut away. That is a limitation of anchor
validation, now recorded in the family's note.

**CONF-09.** IT-11 requires a six-monthly review by each department head across
all systems. IT-02 requires an annual review by the Data Protection Lead of
systems holding personal data. Different reviewer, different scope, different
cadence. Both happen.

**Why this mattered more than a miscount.** A verifier scored on that set would
have been rewarded for flagging policies that agree. Over-detection is the
failure a conflict detector most needs to avoid, and the study was set up to
reward it.

## 1.2.2 One type became four

`current_current` was one label over three different situations. The
distinguishing question is whether one action can satisfy both documents, and
whether an answer can still be wrong.

| Type | One action satisfies both? | Can an answer be wrong? | Behaviour | Reported |
|---|---|---|---|---|
| `version_supersession` | n/a | Yes | `cite_current_only` | 4 |
| `mutually_exclusive` | **No** | Yes | `surface_both_and_escalate` | 5 |
| `stricter_looser` | Yes | **Yes** | `prefer_stricter_and_escalate` | 3 |
| `compatible` | Yes | **No** | `answer_without_flagging_conflict` | 3 |

The second review treated `stricter_looser` as "not a conflict". That is too
harsh. A user told "you have 24 hours" who reports at hour 20 has breached
IT-02's stricter rule. The action is compatible; the **answer** is unsafe. Only
`compatible` is genuinely benign.

Expected behaviour is now **derived** from the type in
`ConflictFamily.expected_behaviour`, not declared per question. When CONF-07 and
CONF-09 were reclassified, their questions could not keep asserting the old
rule.

## 1.2.3 Eight families planted

Reported: CONF-12 and CONF-13 (`mutually_exclusive`), CONF-14, CONF-15 and
CONF-16 (`stricter_looser`), CONF-17 (`compatible`). Tuning: TUNE-03
(`mutually_exclusive`) and TUNE-04 (`compatible`), so development covers every
type.

CONF-13 is high risk by design: two fire assembly points, which means a roll
call cannot account for everyone. CONF-11 and CONF-13 keep the high-risk band
populated after CONF-05 moved to tuning.

CONF-17 and TUNE-04 required **no corpus edit**. FIN-02 already states that
petty cash "is not an alternative to the procurement process in FIN-03", and
PRD-01 and PRD-02 already give different washing advice for different products.
A negative control found rather than planted is the better kind.

## 1.2.4 Negative controls must be discoverable

A `compatible` family now must carry both a `reconciliation` and
`reconciliation_anchors`: the literal corpus text showing why the apparent
disagreement is not one. A control whose reconciliation is only asserted in the
registry is a trick question; one a reader could discover is a fair test.

## 1.2.5 H2 split into H2a, H2b and H2c

H2c predicts that the verification layer **over-detects**, which is a
directional prediction against the contribution, stated on purpose. Its
falsification would be a strong result.

Per-type thresholds are in section 5. H2b requires unanimity across its three
families, because a majority rule over three is decided by one family's noise. A
smaller class earns a stricter rule, not a looser one.

## 1.2.6 Amendment 1.1.12 reopened and replaced

1.1.12 ruled out recommending a conservative reading, on the ground that only
one of five families had one. That count was taken over a set containing two
non-conflicts, and it collapsed a distinction the taxonomy now makes.

**Replaced with a type-dependent rule:**

| Type | Rule |
|---|---|
| `mutually_exclusive` | Surface both, pick neither, escalate. There is no safe reading; the trade counter cannot open at two times. |
| `stricter_looser` | Name the stricter figure as the safe course, state the other and its document, escalate. |
| `compatible` | Answer plainly. Do not flag a conflict. |

The stricter document is recorded in the registry as `stricter`, so the
conservative reading is a **property of the family**, not a judgement made at
answer time. That is what makes it scoreable on one rubric per type, which is
the objection that sank the earlier attempt.

## 1.2.7 The tuning boundary ran only one way

**Found.** The check refused a test question expecting a chunk carrying a
tuning family's disputed fact, but nothing refused a development question
expecting a chunk carrying a **reported** family's. Tuning against the passage
that decides a reported result is the same contamination in the other
direction.

**Done.** `check_no_reported_fact_reaches_the_development_split` added. It
immediately found four questions written minutes earlier under this amendment:
`FACT-order-picking-cutoff` on OPS-01#003 (CONF-15), `FACT-fire-assembly` on
OPS-07#003 (CONF-13), `SYN-visitor-evacuation` on GEN-03#002 and OPS-07#003
(CONF-11, CONF-13), and `FACT-development-conversation` on HR-06#002 (TUNE-03).
All four replaced, each carrying a note saying why.

## 1.2.8 Scoring and provenance gaps

| Found | Done |
|---|---|
| `answer_scoring` stripped only bracketed identifiers, so a bare `IT-04` in prose still yielded the quantity "04" | General identifier regex, with regression tests for all four forms |
| Full-mark conflict criterion did not require escalation, contradicting 1.1.12 | Escalation is in the criterion for both escalating types |
| `answer_directly` mixed content correctness with citation quality, so the metric named answer correctness could not report correctness | Content only. Citation validity, support and completeness are reported alongside |
| A test-split run could be written with no corpus, index, question set or registry | `RunWriter` refuses a test-split run with missing provenance |
| `question_set_sha256` hashed only the records, leaving metadata and rubrics outside it | Hashes the complete file |
| `Index.load` trusted the declared chunk-set hash | Recomputes from the stored chunks first, so a tampered file is refused before anything else is checked |
| The blinded sheet gave no rubric to score against | Required claims, forbidden claims, acceptable variants and the scoring criteria are included. They come from the question set, are identical across arms, and carry no signal about which system produced the answer |

## What was not changed

The 0.25 effect threshold, the stopping rules, the blinding procedure and the
commitment to report a null result are unchanged. The family counts the
thresholds depend on were restored by planting rather than by relaxing the
thresholds.

## State after this amendment

| | Before 1.2 | After 1.2 |
|---|---|---|
| Corpus | `3060ae540015`, 38 documents | `3f761d41b44b`, 38 documents |
| Chunk set | `5fc6a227c0a1`, 138 chunks | `03e5d42bfdfb`, 140 chunks |
| Conflict types | 2 | 4 |
| Reported families | 9, two of them not conflicts | 15, all correctly typed |
| Negative controls | none | 3 reported, 1 tuning |
| Question set | 78 questions, 42 groups | 103 questions, 51 groups |
| Reported sample | 26 test groups | **32 test groups** |
| Tests | 271 | 276 |

No arm has been run. Stage 5 has not begun.

---

# Amendment 1.3 - 8 August 2026

Retrieval calibration. Made **before any arm was run**, on the development split
only, using `scripts/evaluate_retrieval.py`. The script refuses the test split.

`top_k: 4` and `min_similarity: 0.32` had been carried since the Stage 4 pilot,
where they were chosen from four questions. This is the first time either has
been measured.

## 1.3.1 top_k raised from 4 to 6

| | k=4 | k=6 | k=8 |
|---|---|---|---|
| Strict recall | 0.667 | **0.815** | 0.852 |
| Lenient recall | 0.963 | 0.963 | 0.963 |
| MRR | 0.889 | | |
| `version_supersession` pair recall | 1.00 | 1.00 | 1.00 |
| `stricter_looser` pair recall | 0.56 | **0.89** | 0.89 |
| `compatible` pair recall | 0.67 | **1.00** | 1.00 |

At k=4, nearly half the `stricter_looser` conflicts were never assembled, so
H2b would have failed on retrieval rather than on verification. The curve
flattens after 6 and k=8 buys 0.037 of strict recall for 33% more evidence
tokens on every Pi prompt.

## 1.3.2 min_similarity lowered from 0.32 to 0.30, and reported as a finding

| | Range of top-1 similarity |
|---|---|
| Answerable, n=27 | 0.5548 to 0.8515 |
| Unanswerable, n=6 | 0.5555 to 0.6113 |

**All six unanswerable questions scored above the lowest answerable one.** The
distributions overlap completely, so no threshold separates them. At 0.32
nothing is refused. The first threshold that refuses any unanswerable question
is 0.60, which also wrongly refuses a real one, and every step from there trades
one error directly for the other.

This is a **finding, not a tuning failure**, and it is reported as one:
cosine similarity over a 768-dimension embedding cannot distinguish "the corpus
does not cover this" from "the corpus covers this". A question about pensions
retrieves the HR documents with respectable similarity because it is
recognisably an HR question; nothing in the geometry knows that the answer is
absent.

It is evidence that *one particular* refusal mechanism does not work. It is
**not** support for H4, which compares a structured verdict against the
baseline's free-text refusal, and neither has been measured. An earlier version
claimed the stronger thing. Corrected under 1.4. The threshold is set to
0.30, below every observed score, so it refuses nothing and stops pretending to.
Abstention becomes the verifier's job, which is what H4 predicts it must be.

**Consequence for Arm A, B and C.** Those arms lose the *similarity* refusal
path. They retain the instruction in `BASELINE_SYSTEM`: "If the evidence does
not contain the answer, say so plainly and cite nothing." An earlier version of
this amendment said they had no refusal mechanism at all, which my own prompt
contradicts. Corrected under 1.4. That is the honest baseline: a retrieval-augmented pipeline without
verification cannot abstain, and reporting a refusal rate produced by a
threshold that fires at random would have overstated it.

## 1.3.3 Two representative mutually_exclusive tuning families

`mutually_exclusive` conflict-pair recall read **0.00 at every k up to 6**. That
row was one family, three questions: TUNE-03.

TUNE-03 is atypical. Its HR-06 side sits in a 122-word chunk headed
*"Professional development; Study leave; Availability during training"*, with
the core-hours sentence at the end. A query about core hours has no reason to
rank it. Every one of the five reported `mutually_exclusive` families has the
disputed fact in a topically matched chunk on both sides.

So the 0.00 is probably a property of that family rather than of the type -
**but that cannot be proved without measuring the families the protocol
forbids looking at.** Two representative families were planted instead:

| | Documents | Conflict | Both sides topically matched |
|---|---|---|---|
| TUNE-05 | HR-01 / GEN-01 | Leave year, April-March or January-December | `Leave year` in both |
| TUNE-06 | OPS-02 / OPS-03 | RA number valid 21 or 14 days | `Return authorisation` / `Authorisation validity` |

Development now has three `mutually_exclusive` families and nine questions.
TUNE-03 is kept rather than fixed: a family whose disputed fact is buried in an
unrelated chunk is a realistic hard case, and retaining it alongside two
representative ones separates "this type is hard" from "this planting was hard".

### Resolution, measured

| Family | Type | Pair recall @4 | @6 | @8 |
|---|---|---|---|---|
| TUNE-05 | `mutually_exclusive` | **1.00** | 1.00 | 1.00 |
| TUNE-06 | `mutually_exclusive` | **1.00** | 1.00 | 1.00 |
| TUNE-03 | `mutually_exclusive` | 0.00 | 0.00 | 0.33 |
| CONF-05 | `stricter_looser` | 0.33 | **1.00** | 1.00 |
| TUNE-02 | `stricter_looser` | 1.00 | 1.00 | 1.00 |
| TUNE-01 | `stricter_looser` | 0.33 | 0.67 | 0.67 |
| CONF-01 | `version_supersession` | 1.00 | 1.00 | 1.00 |
| TUNE-04 | `compatible` | 0.67 | **1.00** | 1.00 |

The type-level 0.67 was two families at 1.00 and one at zero, not three
middling ones. **Superseded by amendment 1.4.** TUNE-03 and TUNE-06 were both misclassified,
so this table compares one genuine mutual exclusion against two families that
were not. It supports no conclusion about the type, and the sentence "Stage 5
proceeds" is withdrawn.

TUNE-03's 0.00 was read as evidence that a buried policy cannot be retrieved.
Amendment 1.4 withdraws that reading: TUNE-03 was never a conflict, so retrieval
was behaving correctly.

TUNE-01 plateaus at 0.67 and does not reach 1.00 by k=10. One of its three
paraphrases never assembles OPS-02 with CS-03. Recorded, not fixed: a family
that is hard for one phrasing out of three is a realistic case and removing it
would flatter the measurement.

## What this does not license

The calibrated values were chosen on development questions and are now frozen.
They are not revisited after seeing any test result. If retrieval turns out to
be the limiting factor on the test split, that is reported as a limitation, not
corrected by retuning.

## State after this amendment

| | Before 1.3 | After 1.3 |
|---|---|---|
| Corpus | `3f761d41b44b` | `63a92324d734` |
| Chunk set | `03e5d42bfdfb`, 140 chunks | `2c92533c2f86`, 141 chunks |
| top_k | 4 | **6** |
| min_similarity | 0.32 | **0.30** |
| Tuning families | 6 | 8 |
| Question set | 103 questions, 51 groups | 109 questions, 53 groups |
| Tests | 276 | 288 |

No arm has been run. Stage 5 has not begun.

---

# Amendment 1.4 - 13 August 2026

Third independent review. Amendment 1.2 built a four-type taxonomy to stop
families being misclassified. Within a day I misclassified four families using
it. This amendment corrects them and, more importantly, removes the conditions
that let it happen.

**"Stage 5 proceeds" in amendment 1.3 is withdrawn.**

## 1.4.1 Four families were typed by intuition and typed wrongly

| Family | Was | Is | The fact that decides it |
|---|---|---|---|
| **CONF-11** | mutually_exclusive | `stricter_looser` | Wear safety footwear on every visit. GEN-03 grants a permission, not an obligation, so this satisfies both. |
| **TUNE-06** | mutually_exclusive | `stricter_looser` | Use the RA number within 14 days. Inside OPS-03's 14 and OPS-02's 21. |
| **TUNE-03** | mutually_exclusive | `compatible` | 09:30-16:00 applies while working remotely, 10:00-16:30 while in training. Different circumstances, not different answers. |
| **CONF-12** | mutually_exclusive | unchanged, now justified | At 35 days overdue the order is taken or refused. No action does both. |

Reported counts move from 4 / 5 / 3 / 3 to **4 / 4 / 4 / 3**, which is a better
balance than the one it replaces. H2a's denominator falls to 4 and H2b's rises
to 4.

## 1.4.2 The condition that allowed it

`compatible` families were already safe: they had to carry a `reconciliation`
the reader could check against the corpus. The other types carried nothing, so
the classification rested on my judgement and nothing tested it.

Two fields now close that gap:

- `satisfying_action` - **required** on every `stricter_looser` family. Name the
  action that satisfies both documents.
- `no_satisfying_action_because` - **required** on every `mutually_exclusive`
  family, and forbidden alongside `satisfying_action`.

Writing "wear safety footwear on every visit, which satisfies both" makes it
obvious that CONF-11 was never mutually exclusive. Being unable to write such a
sentence is what establishes that a family is. `_validate_family_shape` enforces both, so a family
cannot be typed without a stated reason.

**What this does and does not achieve.** The schema validates that a
justification *exists*, not that it is *sound*. CONF-12 passed with a sentence
that denied the antecedent - reading OPS-01's "hold if over 45 days" as licence
to trade below 45 - and amendment 1.5 corrects it. The field makes the
reasoning visible and reviewable, which is worth having; it does not make
misclassification impossible, and an earlier version of this amendment said it
did.

## 1.4.3 Conflict-pair recall measured the wrong thing

It compared **document identifiers**: any chunk from each document counted. A
family could score 1.00 while the passages stating the disagreement never
reached the model, which is the only text a verifier could reason over.

It now requires the chunks carrying **both disputed claims**, located by the
registry anchors. Every figure reported in amendment 1.3 was measured the old
way and is superseded.

## 1.4.4 Overclaims withdrawn

| Claimed in 1.3 | Correction |
|---|---|
| "Retrieval is not the binding constraint on H2a, Stage 5 proceeds" | Withdrawn. The table compared one genuine mutual exclusion against two misclassified families. |
| "TUNE-03 demonstrates a limitation of the whole approach" | Withdrawn. TUNE-03 was not a conflict; retrieval was behaving correctly. |
| "Arms A, B and C now have no refusal mechanism at all" | Wrong, and contradicted by `BASELINE_SYSTEM`, which instructs exactly that. They lose the *similarity* path only. |
| "Direct empirical support for H4" | Withdrawn. It shows one refusal mechanism fails. H4 compares a structured verdict against free-text refusal and neither has been measured. |

## 1.4.5 Also corrected

`TUNE-05-Q3` asked whether leave is lost in March, which is carry-over, not the
leave-year dates it belongs to. `gold/evaluation.json` still said nine reported
families and 53 groups as 51.

## What remains defensible from 1.3

`top_k: 6` stands: the k=8 gain is small and the misses are reported rather than
tuned away. `min_similarity: 0.30` stands **as a conservative floor**, described
as such and not as an answerability classifier.

## Not addressed here

`/api/embeddings` is still the sequential legacy endpoint. Index builds take 5.8
seconds on the laptop and 36 on the Pi, so it is not a bottleneck, and it is
recorded as known rather than fixed.

## State after this amendment

| | Before 1.4 | After 1.4 |
|---|---|---|
| Reported `mutually_exclusive` | 5, two misclassified | **4, each justified** |
| Reported `stricter_looser` | 3 | **4** |
| Type justification | asserted | **required and tested** |
| Pair recall | document presence | **disputed chunks** |
| Tests | 288 | 289 |

No arm has been run. Stage 5 has not begun.

---

# Amendment 1.5 - 13 August 2026

Final amendment before the gold data is frozen. Made **before any arm was run**.
After this: no new families, no taxonomy changes, no retrieval tuning.

## 1.5.1 CONF-12 reclassified, and the logical error named

CONF-12 was `mutually_exclusive`, justified by the claim that at 35 days overdue
FIN-01 has the account on hold while OPS-01 has not yet held orders, so the
order is either taken or refused.

That **denies the antecedent**. OPS-01 reads "trade orders are automatically
held if any invoice is more than 45 days overdue". That is a *sufficient*
condition for holding, not a requirement to leave orders unheld below it. My
inference treated it as necessary. Holding the account from day 30 under FIN-01
satisfies both documents.

CONF-12 is `stricter_looser`, stricter = FIN-01.

**Reported counts: 4 supersession / 3 mutually exclusive / 5 stricter-looser /
3 compatible.**

## 1.5.2 What the justification field actually does

Amendment 1.4 said a misclassification is "now a validation failure". It is not.
The schema validates that a justification **exists**, not that it is **sound**.
CONF-12 passed the check with an invalid inference.

The field is still worth having: it puts the reasoning in the file where a
reader can test it, which is how this error was found. But the claim has been
corrected in 1.4 and is stated accurately here.

## 1.5.3 H2a and H2b merged for the confirmatory analysis

A trigger was written down before this review: if another misclassification
survived the justification field, the taxonomy was too fine for this corpus and
the subtypes should be pooled. It survived, so the trigger fires.

**H2 is now one hypothesis over the eight families where two live documents
disagree**, threshold Δ > 0.25 with the direction holding in **6 of 8**.

Two decisions were being conflated and are now separated:

| Decision | Outcome |
|---|---|
| How answers are **scored** | Subtypes keep separate rubrics. `mutually_exclusive` surfaces both and picks neither; `stricter_looser` names the safe course. These are different behaviours and one rubric cannot score both. |
| What is **claimed** | Pooled. The confirmatory threshold rests on "two live documents disagree", which has held up, rather than on the sort between subtypes, which has not. |

Per-subtype figures are reported as descriptive breakdowns with no threshold of
their own.

## 1.5.4 Rubric drift repaired

| | Was | Now |
|---|---|---|
| CONF-11 | `stricter_looser` since 1.4 but the rubric never required the safe course | Requires "wear safety footwear on every warehouse visit" and escalation |
| TUNE-06 | Same, and its gold answer did not escalate | Requires "use the RA within 14 days" and escalation |
| CONF-12 | Rubric written for a mutual exclusion | Rewritten for stricter/looser, naming 30 days as the operative point |
| CONF-06-Q3 | Asked about a deletion request while the gold concerned expiry of a retention period | Reworded to expiry |

## 1.5.5 Platform-sensitive ranking

Laptop and Pi rankings differ at k=3 on near-tied similarity scores, while
agreeing at every configured k. Recorded as platform-sensitive ranking rather
than as a defect or as identical execution.

## Freeze

Retrieval stands: `top_k: 6`, `min_similarity: 0.30` as a conservative floor,
not an answerability classifier. No index rebuild.

**The gold data is frozen at this commit.** No arm has been run on any question.
Stage 5 begins next.

| | |
|---|---|
| Corpus | `63a92324d734`, 38 documents |
| Chunk set | `2c92533c2f86`, 141 chunks |
| Reported families | 15: **4 / 3 / 5 / 3** |
| Confirmatory H2 denominator | **8 pooled** |
| Question set | 109 questions, 53 groups; reported sample 68 questions in 32 test groups |
| Tests | 291 passing |

---

# Amendment 1.6 - 13 August 2026

Made **before any test-split arm was run**, and before the diagnostic protocol
in `docs/VERIFIER_PROTOCOL.md` was executed. This amendment changes what Arm D
does, so it is recorded rather than treated as a bug fix.

The gold data, the corpus, the chunk set, the question set and the retrieval
settings are untouched. The freeze at Amendment 1.5 stands.

## 1.6.1 Arm D was rewriting answers it had no complaint about

Development pilot 02 revised six answers. In five of the six the verifier had
reported `relationship: no_relationship` with every claim `SUPPORTED`: it found
nothing wrong and rewrote the answer anyway. Two of those rewrites were served
and were worse than the drafts they replaced.

| Question | Draft | Served instead |
|---|---|---|
| CONF-01-Q3 | "You may claim 55 pence per mile [HR-13#001]" | "The answer under review does not address a conflict between passages." |
| TUNE-06-Q1 | "A Returns Authorisation number is valid for 14 days [OPS-03#002]" | "The validity period ... is not explicitly stated in the provided evidence. However, it can be inferred that ..." |

The first is commentary on the review served to the user as the answer. The
second withdraws a correct cited fact and replaces it with an inference, which
is the specific behaviour grounded generation exists to prevent.

The revision guard did not stop either. It checked that cited identifiers
resolve and that citations name passages rather than documents. Both checks
pass **vacuously** on an answer that cites nothing, which is what both of these
were. The guard was written against the failure mode I imagined - invented
citations - and was silent on the degenerate case.

## 1.6.2 Why this matters more than the two bad answers

Arms B and D share a draft so that any difference between them is attributable
to the verification layer. That is the whole basis of the confirmatory contrast
for H2. Three of the six revisions were cosmetic: a citation moved to the other
side of a full stop, an "According to" removed. Serving those puts differences
into the B-versus-D comparison that verification did not cause, and the
comparison then partly measures rewording.

## 1.6.3 The rule, as now implemented

A revision is served only when the verification records a reason for one:

1. a conflict was detected, or
2. a claim verdict is `CONTRADICTED` or `INSUFFICIENT_EVIDENCE`, or
3. the draft cites evidence that was never retrieved, or cites a document
   rather than a passage.

Otherwise the draft stands unchanged and `revision_rejected` records why.

Condition 3 was added after the first version of the rule blocked a revision
that repaired a miscitation - the one correction the layer is best placed to
make. A rule that prevented it would have been worse than no rule.

Separately, a revision that cites **no** passages while replacing a draft that
cited evidence is rejected, unless every claim verdict is
`INSUFFICIENT_EVIDENCE`. Withdrawing an unsupported claim legitimately leaves
nothing to cite; quietly stripping citations from a supported answer does not.

Under this rule all five unwarranted pilot 02 revisions are withheld and the
sixth, an abstention, is served.

## 1.6.4 A verification layer that finds nothing is now a no-op

Stated as a design commitment rather than left implicit. It is testable, and it
is the property that makes B versus D interpretable.

## 1.6.5 The stopping rule is executable and committed in advance

`src/sme_assistant/evaluation/stopping_gate.py` computes the development
decision instead of leaving it to be read off a table. It reports five
quantities and returns one of `PROCEED`, `REVISE`, `STOP` or `DEFECT`.

Detection and classification are scored separately and deliberately. The pilot
diagnostic found `qwen2.5:3b` flagging three of four families while correctly
classifying one; a single accuracy figure would have concealed that, and the
two findings call for different responses.

A family counts as detected only when a **majority of its three paraphrases**
is detected. The paraphrases exist to test robustness to wording, so accepting
one in three would defeat the reason they are there.

The rule was committed before the run that tests it, and it fired `DEFECT` on
pilot 02 the first time it ran. That is what surfaced 1.6.1. The pilot 02
decision stands as recorded; the rule was not relaxed to clear it.

## 1.6.6 Every served revision names its warrant

`revision_warrant` records **which** of the four conditions applied, on the
record, for any revision actually served. It is empty when the draft stood.

"It must have had a reason" is not an audit trail. A rewrite whose
justification has to be reconstructed afterwards cannot be checked, and the
whole of 1.6.1 is a case of exactly that going unnoticed.

## 1.6.7 A narrow warrant does not license a wide change

Condition 3 above creates a hole on its own: if a misplaced identifier grants a
warrant, an unrelated prose rewrite could be laundered through it.

So when citation repair is the **only** warrant, the content must survive.
Similarity is measured over content tokens, with citations and function words
removed, and must reach **0.90**.

The measure is deliberately not character-level. A character measure is
length-sensitive: deleting "According to" scores 0.71 on a one-line answer and
0.96 on a paragraph, so one threshold would permit or refuse the identical edit
depending on how long the answer happened to be.

The threshold is calibrated on the six revisions pilot 02 produced, which score
1.000, 1.000, 1.000, 0.696, 0.211 and 0.000. The gap between the pure citation
moves and the genuine rephrase is 0.30 wide, so the threshold is not fitted to
a boundary case: any value from 0.75 to 0.99 separates the same groups. Stated
here because a threshold chosen on development data and then reported as if it
were principled is a small dishonesty that compounds.

## 1.6.8 What "frozen" has to cover

The protocol runner originally refused only when `docs/VERIFIER_PROTOCOL.md`
had uncommitted changes. That was theatre. The 96 calls are produced by the
verifier prompt, the parsing rules, the retrieval settings and the gold data,
none of which live in `docs/`. It now refuses unless `src/`, `gold/`,
`config.json`, the harness and the protocol are all committed, and it records
the config, corpus, chunk-set and registry fingerprints alongside the commit.

## 1.6.9 Pilot 03 is a controlled re-run, not a fresh one

Arm B is **not** regenerated. Its pilot 02 drafts and retrieval are replayed
from the recorded run, and the reconstruction is checked against the
`evidence_sha256` each answer carries. A rebuild that quietly differed would
leave every downstream number looking plausible while comparing two different
experiments.

Regenerating B would draw a second sample from the generator, and any pilot 02
to pilot 03 difference would then mix the revision fix with generator variation
with no way to separate them.

## What was not changed

The verifier prompt. This amendment fixes what is done with the verifier's
output, not what it is asked. Prompt revision 3 remains conditional on the
diagnostic protocol indicating that a prompt can fix the detection failure.

## State after this amendment

| | Before 1.6 | After 1.6 |
|---|---|---|
| Revisions served | any that parsed | **only when warranted** |
| Uncited revision of a cited answer | served | **rejected unless an abstention** |
| Development stopping rule | read off a table | **computed and committed** |
| H5 hardware label | machine that summarised | **machine recorded in the run** |
| Arm D without Arm B | warning, run proceeded | **refused unless marked exploratory** |
| Tests | 357 | **404** |

## What pilot 02 still supports

An earlier draft of this amendment said pilot 02 was superseded and that its
numbers meant nothing until re-run. That was an overstatement in the careless
direction rather than the flattering one, but it is still wrong, and discarding
valid evidence is not a form of rigour.

**Pilot 02 failed, and its served-answer outcomes were invalidated by a
revision-control defect.** That is the accurate statement. The defect sits in
what was done with the verifier's output, so everything decided before that
point is unaffected:

| Pilot 02 evidence | Status |
|---|---|
| Served answers, revision rate, final-answer citation metrics | **Invalid** for the amended Arm D |
| Zero conflict detections across 18 genuine-conflict questions | Still valid |
| Four parse failures | Still valid |
| Raw verification outputs and inferred relationships | Still valid |
| Prompt and evidence contents | Still valid |
| Token counts, latency, and the truncation result | Still valid |
| The gate result that exposed the revision defect | Still valid |

The null detection result therefore stands as evidence going into the
diagnostic protocol. Pilot 03 re-establishes the served-answer measurements
only.

No test-split arm has been run. The verifier is not frozen.

---

# Amendment 1.7 - 13 August 2026

Made **before any test-split arm was run** and before the diagnostic protocol
was executed.

This amendment exists in two layers. The first is a claim I made and had to
withdraw within the day. The second is what is actually established. Both are
recorded, because the withdrawal is the more instructive of the two.

## 1.7.1 A claim made and withdrawn: "the verifier is not reproducible"

I ran a check that repeated each of 12 recorded prompts three times **back to
back**, found 4 of 12 changing their raw output, concluded that the verifier was
not reproducible at a fixed seed, wrote that every reported figure must be a
mean over repeats, and expanded the diagnostic protocol from 96 calls to 288.

The same output contained the evidence against that conclusion:

| | |
|---|---|
| call 1 matched the recorded pilot 03 run | **12 / 12** |
| calls 2 and 3 agreed with each other | 12 / 12 |
| output changed under immediate repetition | 4 / 12 |

**Every first call reproduced the earlier session exactly.** The changes appear
only when the identical prompt is sent again immediately, which no run of this
protocol performs. My script printed "matches 8/12" because it required all
three calls to equal the recording, folding "a fresh prompt reproduces" together
with "adjacent calls are stable" and reporting neither.

The accurate description of that artefact is **adjacent-repeat raw-output
variability: 4 of 12**. It is preserved under that name.

The error is worth naming precisely because it does not fit the pattern of the
earlier ones. The previous overstatements ran towards a tidier positive story.
This one ran the other way, towards a dramatic negative finding about on-device
reproducibility, and I wrote a section arguing it belonged in the findings
chapter. A bias towards the interesting result is the same failure as a bias
towards the flattering one; it is merely harder to notice because it feels like
rigour.

## 1.7.2 What is actually established

Nothing about the reproducibility of any reported metric. That has not been
measured, and a hash of the raw text cannot measure it: a reordered JSON key or
a reworded rationale changes the hash and changes no result.

What is open:

* Pilots 02 and 03 differed on **24 of 41** raw outputs with byte-identical
  prompts. Two explanations remain live and neither is excluded. Pilot 02
  predates option recording, so its effective options cannot be read from the
  record. Arm D also ran after A, B and C in pilot 02 and alone in pilot 03, so
  the preceding workload on the server differed.
* Whether detection, classification, verdicts, parse status or the served
  answer move between runs is unknown.

## 1.7.3 The corrected instrument

`scripts/check_determinism.py` now:

1. runs **complete passes in protocol order**, never adjacent repeats, so a
   repeat meets the server in the state a real run does;
2. replays the **model and options recorded in the run**, not the current
   configuration, which has changed since;
3. **parses every response** and compares raw text, relationship, detection,
   claim verdicts, parse and validation status, invented evidence, whether a
   revision was served, the served answer and its citations, each separately;
4. reports pass-1-against-recording and pass-against-pass apart, because they
   answer different questions.

## 1.7.4 The decision rule, fixed before the measurement

| Finding | Response |
|---|---|
| Raw text varies, every reported outcome stable | Protocol stays at **96 calls**. Prose-level variability is documented as a finding in its own right. |
| Any reported outcome varies | **Three complete 96-call blocks.** R0 to R5 applied independently per block; three outcomes reported with mean and range. |

**Blocks are not observations.** Three blocks are three results. Calls within a
block are not independent of each other, and pooling 288 calls into one
denominator would inflate every proportion in the analysis.

The protocol is therefore back to **96 calls, one block**, until the corrected
check says otherwise.

## 1.7.5 A trap that would read as reproducibility

Holds regardless of the above. The aggregate relationship counts were
**identical** across pilots 02 and 03, 36 / 4 / 1 both times, while four
questions moved: two one way, two the other, cancelling exactly.

Read from totals that is perfect stability with a tenth of the questions having
changed answer. Stability is reported per prompt and never inferred from an
aggregate.

## 1.7.6 The abstention is now written by the system

The rule in 1.6 rejected an uncited revision that stated a figure. It caught
pilot 03's "the 14-day validity period" and it would not have caught "the
authorisation remains valid for two weeks", which asserts the same thing with
no digit in it.

Detecting assertions inside free prose is unbounded work: the space of ways to
state a fact has no edge, and each patch closes one and leaves the rest.

So when every claim is `INSUFFICIENT_EVIDENCE` and the revision cites no
passage, the served answer is a **fixed template written in the source**, not
the model's prose. The verifier's finding still decides whether to abstain. It
no longer chooses the words, and there is nothing left to smuggle a claim into.

## 1.7.7 Citation repair requires exact content equality

The 0.90 similarity threshold from 1.6.7 is withdrawn. It was a tolerance for
content change with no principle behind the number, and "mostly the same claim"
is not a standard.

When citation repair is the only warrant, the content tokens of the revision
must equal those of the draft exactly. A model that cannot reproduce the
sentence has its revision refused and the draft stands, which is what Arm B
would have served anyway. Nothing is lost against the baseline.

## 1.7.8 The stopping thresholds, stated rather than only coded

`evaluate_gate` had thresholds that appeared in no document. An executable rule
whose numbers live only in source is not pre-committed, because nobody can
check it against what was promised. On the development split, 6 genuine
families and 2 controls:

| Decision | Condition |
|---|---|
| `DEFECT` | any invalid revision served. Vetoes everything below. |
| `STOP` | every control falsely detected (2 of 2), **or** genuine detected <= 1 of 6 |
| `PROCEED` | genuine detected >= 4 of 6 **and** genuine classified >= 3 of 6 |
| `REVISE` | anything else. One prompt revision remains and it is the last. |

Expressed as fractions of the families present, so the same rule reads
correctly on either split without being rewritten to suit the outcome.

## 1.7.9 The protocol precondition is now enforced

Section 7 of the protocol said it runs only after a pilot serves no invalid
revision. Nothing checked it, which made it a wish. `verifier_protocol.py` now
evaluates the gate against the latest development Arm D run and refuses.

## State after this amendment

| | Before 1.7 | After 1.7 |
|---|---|---|
| Reproducibility of reported metrics | asserted absent | **unmeasured, instrument corrected** |
| Protocol size | 96, briefly 288 | **96, one block** |
| Determinism schedule | adjacent repeats | **complete passes** |
| Abstention wording | model's prose, checked | **system template** |
| Citation-repair tolerance | 0.90 similarity | **exact content equality** |
| Stopping thresholds | code only | **documented** |
| Protocol precondition | documented only | **enforced** |
| Tests | 404 | **419** |

Pilot 03's served-answer measurements are superseded by 1.7.6 and 1.7.7. Its
detection result, zero on every genuine family, agrees with pilot 02's.

---

# Amendment 1.8 - 14 August 2026

Made after the corrected reproducibility measurement and **before** the
diagnostic protocol was run. The measurement was committed on its own, ahead of
this amendment, so the evidence precedes the decision in the history and not
only in the prose.

## 1.8.1 The measurement

41 prompts, three complete passes in protocol order, replaying the model and
options recorded in pilot 03 rather than the current configuration.

| Outcome | pass 1 vs recorded | pass to pass |
|---|---|---|
| raw text | 41 / 41 | 26 / 41 |
| **conflict detection** | **41 / 41** | **41 / 41** |
| relationship | 41 / 41 | 38 / 41 |
| claim verdicts | 41 / 41 | 35 / 41 |
| parse status | 41 / 41 | 39 / 41 |
| validation failures | 41 / 41 | 36 / 41 |
| invented evidence | 41 / 41 | 40 / 41 |
| revision served | 41 / 41 | 36 / 41 |
| served answer | 41 / 41 | 36 / 41 |
| citations | 41 / 41 | 38 / 41 |

**Cross-session reproduction of a fresh prompt is exact on every outcome.**

The statement to use, and the only one supported:

> Binary conflict detection was reproducible, while relationship labels,
> structural validation and served-answer outcomes showed block-level
> variability.

Not "the verifier is unreproducible", which is what I said a day earlier on
weaker evidence. The confirmatory metric did not move on a single prompt of 41.

## 1.8.2 The pre-declared rule fires

Secondary outcomes that feed reported figures do move, so the protocol becomes
**three complete blocks of 96 calls**. R0 to R5 are applied independently to
each block and the three outcomes reported with their mean and range.

**288 calls are three results, not 288 observations.** Calls within a block are
not independent of each other. Pooling them would inflate every denominator and
present three runs as a sample of 288.

## 1.8.3 The declared gate is restored

Amendment 1.7.8 documented the thresholds that were in the code. It should have
noticed that they were not the thresholds that had been declared. The code
permitted 4 of 6 detections, tolerated one control false positive, and ignored
parse failures and citation completeness entirely. All three are looser than
what was promised, and all three were written after pilots 02 and 03.

Documenting a weakened threshold is not the same as pre-registering it.
Restored:

| Condition | Requirement |
|---|---|
| Detection | >= 5 of 6 genuine families |
| Control false positives | 0 of 2 |
| Parse failures | <= 2 of the answers |
| Citation completeness | within 0.05 of Arm B, at group level |
| Invalid revisions served | 0 |

Every condition must hold to PROCEED.

## 1.8.4 The abstention template applies whenever the verifier abstains

1.7.6 applied it only when the model also omitted citations, which left the
hole open from the other side. "The authorisation remains valid for two weeks
[OPS-03#002]" carries a citation, would have passed, and asserts the very claim
the verdict just called unsupported. Attaching a passage to an assertion the
verifier does not believe is worse than asserting it bare, because it looks
grounded.

The condition is now `is_abstention` alone.

## 1.8.5 The gate's rule, in one sentence

Every served revision cites a passage that resolves, unless it is exactly the
abstention template. No detection of assertions, no inspection of prose.

## 1.8.6 Safeguards that were decorative

| Was | Now |
|---|---|
| `--models`, `--repeats` and an override flag on the protocol runner | removed; a design changeable from the command line was never pre-registered |
| `git()` ignored the exit status | fails closed; a failed `git status` returns empty stdout, which read as a clean tree - the one answer that lets an unfrozen experiment run |
| R0 to R5 in prose, a table printed beneath them | executed in code, per block |
| Families with the pair absent counted | excluded, as section 3 always required |
| Design assumed | 8 families and 3 paraphrases each asserted before any call |
| Chunk ids and raw output recorded | question, draft, evidence text and hash, prompt and hash, effective options |

## 1.8.7 Pilot 04

Pilot 03's stored answers still contain the invalid served revision.
Re-parsing them inside a diagnostic did not produce a clean recorded arm run,
and the enforced precondition reads the recorded run. Pilot 04 replays Arm B's
pilot 02 drafts again under the amended rules and must report zero invalid
served revisions before the protocol runs.

## State after this amendment

| | Before 1.8 | After 1.8 |
|---|---|---|
| Reproducibility | unmeasured | **measured; detection stable, secondary outcomes vary** |
| Protocol | 96, one block | **288, three blocks, rules per block** |
| Gate thresholds | weakened post-pilot | **as declared** |
| Abstention template | only when uncited | **whenever abstaining** |
| Protocol overrides | three flags | **none** |
| Git guard | failed open | **fails closed** |
| Decision rules | prose | **executed** |
| Tests | 419 | **428** |

---

# Amendment 1.9 - 14 August 2026

Pilot 04 passed structural containment and, in doing so, exposed a measurement
defect created by amendment 1.8 itself.

## 1.9.1 What pilot 04 actually shows

| Check | Result |
|---|---|
| Structurally invalid revisions | **0** - pass |
| Genuine families detected | **0 / 6** - fail |
| Compatible false positives | **0 / 2** - pass |
| Parse failures | **3 / 41** - exceeds the declared limit of 2 |
| Citation completeness against B | **unavailable** - Arm B absent |

All **25 served revisions are the abstention template**. There is not one
substantive corrected answer in the run. Fifteen of the eighteen
genuine-conflict questions were refused, along with correct drafts including
the mileage rate and the dishwasher answer.

That is a real verifier failure: the model judged supported claims
insufficient. The template stops it asserting something ungrounded; it does not
make the misjudgement disappear, and nothing here suppresses it.

## 1.9.2 The measurement defect 1.8 introduced

Serving a template changed three reported figures for reasons that have nothing
to do with what they claim to measure.

| Figure | Reported | Actual |
|---|---|---|
| Citation validity | 0.342, mislabelled | **three figures, reported together** |
| Refusal | 2 / 41 | **25 / 41** served abstentions |
| "Invalid revisions" | 0 | 0 **structurally**; 21 semantically wrong |

**0.875 does not replace 0.342.** They measure different things, and an
earlier draft of this amendment said "0.875, not 0.342" as though the second
were simply an error. It was mislabelled, not wrong.

| Figure | Pilot 04 | What it answers |
|---|---|---|
| Conditional citation validity | 14/16 = **0.875** | of the answers that made a claim, how many cited validly |
| Claim-making coverage | 16/41 = **0.390** | of the questions asked, how many were answered at all |
| Grounded answer coverage | 14/41 = **0.341** | of the questions asked, how many got a validly cited answer |
| Abstention rate | 25/41 = **0.610** | |
| End-to-end false refusal | 21/35 = **0.600** | answerable questions refused |
| Correct refusal | 4/6 = **0.667** | unanswerable questions refused |

Reporting the conditional figure alone rewards selective abstention: a verifier
that refuses everything it would have got wrong drives 0.875 towards 1.00 while
grounded coverage falls. At a 61% abstention rate that distortion is not
hypothetical. The three are printed together and the gate refuses to show one
without the others.

An abstention cites nothing by design, so including abstentions in citation
validity measures how often the verifier refused and prints it under the
heading of citation behaviour. The prose refusal heuristic missed the template
entirely, because the template does not phrase itself the way a model phrases a
refusal - a measurement that has to recognise its own output.

Corrections:

1. `served_abstention` is recorded **structurally** on the verification, not
   recovered by matching prose.
2. Abstention is split: **21 of 35 answerable questions falsely refused**, 4 of
   6 unanswerable correctly refused. "25 refusals" reports neither.
   The 21 are split again by how much of the expected evidence arrived. The
   first version of this split used ``wanted & got`` - at least one expected
   chunk - which is the retrieval evaluation's **lenient** rule, and reported
   the result as "the verifier had the evidence". That overstates what can be
   laid at the verifier's door.

   | Expected evidence retrieved | False refusals |
   |---|---|
   | All | **16** - the verifier's error; it held everything and refused |
   | Some, but not all | **2** - mixed; it refused on half a disagreement |
   | None | **1** - retrieval's error, not the verifier's |
   | None declared | **2** - nothing to check against |

   The two partial cases are TUNE-01-Q3, which expected `OPS-02#001` and
   `CS-03#001` and received only the second, and TUNE-03-Q3, which expected
   `HR-04#003` and `HR-06#002` and received only the first.

   All 21 are system-level failures. **16 are clearly the verifier's.** The 2
   with no declared expectation satisfy "all required evidence present"
   vacuously and are counted separately rather than folded into the 16, which
   would have inflated the attributable figure to 18.
3. Citation metrics are computed over claim-making answers, with abstentions
   counted separately.
4. The gate field is `structurally_invalid_revisions_served`. It was never a
   semantic check and the name implied it was.
5. The gate reports **every** unmet condition, including parse failures and any
   it could not evaluate. A condition skipped for want of data used to read as
   a condition passed, so a missing Arm B now withholds PROCEED rather than
   being silently absent from the verdict.

## 1.9.3 The conclusion, at the width the evidence supports

The STOP wording said "this is the null result. Freeze the verifier and report
it." That claims a finding about verification as an approach. What was tested is
one model, one prompt revision, one evidence window. Qwen has not been run and
neither has the isolated-pair condition.

The gate now says:

> Stop further prompt tuning for llama3.2:3b at k=6; run the precommitted
> model/window diagnostic.

And the defensible statement of the result is:

> `llama3.2:3b`, revision-2 prompting and six chunks failed the development
> gate through zero conflict detection and extensive false abstention.

Not a null result for Qwen, and not for the verification approach.

## 1.9.4 No further changes before the diagnostic

The verifier prompt and the serving rule are frozen as they stand. Both have
been changed twice in two days, each time for a good reason, and a third change
before the diagnostic would mean the diagnostic tested something that had never
been measured end to end.

## State after this amendment

| | Before 1.9 | After 1.9 |
|---|---|---|
| Abstention | inferred from prose | **recorded structurally, split by answerability** |
| Citation validity | 0.342, abstentions included | **0.875, claim-making answers** |
| Gate field name | `invalid_revisions_served` | **`structurally_invalid_revisions_served`** |
| Unmet conditions | first failure only | **all, including unevaluable ones** |
| STOP wording | "the null result" | **one model, one prompt, one window** |
| Tests | 428 | **432** |

---

# Amendment 1.10 - 14 August 2026

The diagnostic protocol ran to completion at commit `001c82f`, 288 calls in
three complete blocks, after the precondition confirmed pilot 04 served no
structurally invalid revision. Artefact:
`results/diagnostics/20260814_043323_verifier_protocol.json`.

## 1.10.1 The result

Detection and correct classification, family-level majority within each block,
identical in all three blocks:

| Verifier | Evidence | Detected | Classified |
|---|---|---|---|
| llama3.2:3b | full, six chunks | 0 / 6 | 0 / 6 |
| llama3.2:3b | oracle pair | 2 / 6 | 1 / 6 |
| qwen2.5:3b | full, six chunks | 2 / 6 | 1 / 6 |
| **qwen2.5:3b** | **oracle pair** | **5 / 6** | **3 / 6** |

**R3 fired: model.** Arm D becomes a Llama-generated answer audited by Qwen.

## 1.10.2 What this is not

It is not "Qwen solves it". Read across the table rather than at the best cell:

* Qwen reaches the threshold **only when the disputed pair is isolated**. On
  the deployed six-chunk window it detects 2 of 6.
* Llama stays below threshold even with the oracle pair, so this is not a
  window problem alone.
* **Both matter.** Model capability and evidence dilution each account for part
  of the gap, and neither explanation is sufficient by itself.

Classification lags detection badly. Of Qwen's 24 detections on
`stricter_looser` families under the oracle condition, **18 were called
`mutually_exclusive`** and 6 were correct. It finds the disagreement and
misnames its kind, which matters because the two demand different behaviour:
one surfaces both positions and picks neither, the other names the safe course.

The defensible statement:

> Qwen demonstrated substantially greater conflict-detection capability than
> Llama under controlled evidence, but full-context performance remained
> limited by evidence dilution and subtype misclassification.

## 1.10.3 A control that was excluded, and what it hides

The reported control false-positive figure in the `full` condition is `0/1`,
not `0/2`. TUNE-03's disputed pair was not retrieved, so the family was
excluded under section 3 of the protocol, which is correct: a verifier shown
one side has nothing to detect.

But the exclusion conceals a real result. In the `full` condition Qwen flagged
**TUNE-03 on 6 of 9 observations** - a majority false positive on a compatible
control - while Llama flagged it 0 of 9.

Recorded here descriptively because the pre-registered rule properly excludes
it and I will not amend a rule after seeing its output. It qualifies the
reading of R3: Qwen detects more, and it also over-detects more. On the one
control the protocol could evaluate, TUNE-04, both models were clean at 0/9.

H2c predicted exactly this over-detection, and it now has development evidence
behind it.

## 1.10.4 Invalid evidence identifiers are one-sided

9 of 288 calls cited a chunk that was never supplied. **All nine were Llama
under the oracle condition**; Qwen produced none. Given a two-chunk window
Llama invented identifiers, which is a worse failure than silence and is worth
reporting alongside its low detection rate.

## 1.10.5 The three blocks were unnecessary, and that is a finding

**Zero of 96 prompts differed across the three blocks.** Raw text, relationship,
detection, classification and parse status all agreed at 1.00. The 288 calls
produced precisely what 96 would have.

The block count came from the corrected reproducibility check on pilot 03's
prompts, which found relationship, verdicts and served answers moving between
passes. On this prompt set, under these conditions, nothing moved.

Recorded rather than tidied away. It says the variability measured on pilot 03
does not generalise across prompt sets, and it is a further argument against
the non-reproducibility claim I made and withdrew in Amendment 1.7. The cost of
having run three blocks is eight minutes; the cost of having assumed stability
would have been an unexamined assumption in the results chapter.

## 1.10.6 What changes, and what does not

| | |
|---|---|
| Verifier model | **`qwen2.5:3b`** |
| Generator | unchanged, `llama3.2:3b` |
| Verifier prompt | **unchanged**, revision 2 |
| Serving rule | **unchanged** |
| Retrieval | **unchanged**, `top_k: 6`, `min_similarity: 0.30` |
| Evidence condition at runtime | **full retrieval** |

**The oracle pair does not become Arm D's retrieval.** Selecting it requires
knowing which passages carry the disputed claims, which is the question the
system exists to answer. It was an instrument and it stays one.

Pi latency for Arm D must now include **model-loading cost**: two 3B models
will not stay resident together on 8 GB, so a verification call may pay a load
that the laptop figures do not show.

## 1.10.7 Next, in order

1. A development Arm D over the 41 questions with Qwen as verifier, replaying
   the same Arm B drafts from pilot 02. Only the verifier model differs from
   pilot 04, so the comparison is clean.
2. Read the stopping gate on that run.
3. Freeze the verifier **only after** that evaluation, and commit the freeze
   before any test-split run.

## State after this amendment

| | Before 1.10 | After 1.10 |
|---|---|---|
| Verifier model | `llama3.2:3b` | **`qwen2.5:3b`** |
| Basis | pilot evidence | **288-call pre-registered protocol, R3** |
| Block variability | expected | **none observed, 0/96** |
| H2c over-detection | predicted | **observed on the excluded control** |
| Tests | 437 | 437 |

No test-split arm has been run. The verifier is not frozen.

---

# Amendment 1.11 - 14 August 2026

A fail-closed validation amendment only. **No change to the prompt, the model,
the corpus, the retrieval settings or the taxonomy.** Pilot 05's raw outputs
are reparsed under the corrected validation; no model call is made.

## 1.11.1 Every Qwen response omitted the claim audit

Not the eleven unrevised detections. **All 41.** Including all 14 detections
and all 3 revisions.

The parser read `payload.get("claims") or []`, which collapses two different
things: a verifier that audited claims and found none, and a response that
never performed the audit at all. The second is a schema violation and it went
through silently 41 times.

## 1.11.2 The prompt taught the omission

The verifier prompt mentions `claims` **once**, in the schema block. Its
**three worked examples all omit the field.**

So the prompt requires the audit in one place and demonstrates skipping it in
three. Attributing the omission to Qwen alone is unsound: what failed is the
configuration of Qwen **plus** revision-2 prompting **plus** a permissive
parser, and the prompt is not the least of the three.

The prompt is **not** being changed here. It is frozen for the same reason it
was frozen before the diagnostic, and this is recorded as a known confound
rather than fixed mid-analysis.

## 1.11.3 The validated helper was never wired in

`cites_a_passage` requires a bracketed identifier with an ordinal, matching
`extract_citations`. It was written after pilot 03, given a regression test,
and used in exactly one place - `asserts_uncited_quantity`.

The serving decision still used `CHUNK_ID_RE`, the bare-identifier pattern. So
pilot 05 served:

> "The answer under review is incorrect. The correct timeframe ... as outlined
> in both CS-03#001 and OPS-02#001."

in place of a correctly cited draft. It reads as cited; the pipeline recorded
zero citations for it.

**This is the third instance of the same failure**: `Generation.options`
declared and never populated, `require_clean_pilot` reading a renamed field,
and now a validated helper never called from the path it was written for. Each
looked correct in review and each was inert. Worth stating plainly in the
methodology chapter: a check that exists is not a check that runs.

## 1.11.4 What changes

1. A bare identifier in prose no longer satisfies the citation requirement.
   `cites_a_passage` is wired into the serving decision.
2. A `final_answer` is not served when the claim audit is missing or empty. A
   relationship label alone does not license replacing an answer.
3. `claim_audit_complete` is recorded on every verification, separately from
   the relationship, because "named a relationship without auditing" and
   "audited and found nothing" are different failures.
4. The DEFECT message now states its scope: served-answer results are
   compromised, **inference results are not.** Detection, classification and
   the control false-positive rate are read from the verifier's own output
   before anything is served. The previous wording said the detection figures
   "mean nothing until it is fixed", which discarded valid evidence.

## 1.11.5 Pilot 05 reparsed

| | As recorded | Under corrected validation |
|---|---|---|
| Revisions served | 3 | **0** |
| Complete claim audits | not measured | **0 / 41** |
| Conflicts detected | 14 | **14, unchanged** |

Detection is unchanged because it is read before anything is served. That is
the point of 1.11.4.

## 1.11.6 The gate still fails, and not because of this

None of the above rescues pilot 05. Against the declared gate:

| Condition | Required | Pilot 05 |
|---|---|---|
| Genuine families detected | >= 5 / 6 | **2 / 6** |
| Correctly classified | - | 1 / 6 |
| Control false positives | 0 / 2 | **1 / 2** |
| Parse failures | <= 2 | 0 / 41 |

Qwen detects considerably more than Llama and still fails on sensitivity,
subtype classification and false detection. The validation repair changes what
is served; it cannot change 2/6 into 5/6.

The defensible finding:

> With revision-2 prompting, Qwen produced relationship labels but omitted the
> required per-claim audit on all 41 development calls. Schema-incomplete
> worked examples and permissive parsing confound attribution to the model
> alone. Independently of that, Qwen failed the declared development gate:
> 2/6 families detected, 1/6 correctly classified, 1/2 compatible controls
> falsely flagged.

## State after this amendment

| | Before 1.11 | After 1.11 |
|---|---|---|
| Missing claims audit | silently accepted | **validation failure, recorded** |
| Revision on a label alone | served | **refused** |
| Bare identifier as citation | accepted at serving | **refused** |
| DEFECT scope | "the figures mean nothing" | **served-answer only** |
| Tests | 437 | **442** |

No test-split arm has been run. The verifier is not frozen.

---

# Amendment 1.12 - 14 August 2026

The final development result. Prompt revision 3 was committed at `0f0111c`
**before** pilot 06 ran. The budget is now exhausted.

## 1.12.1 Pilot 06 against the declared gate

Arm D, development split, 41 questions, Arm B's pilot 02 drafts replayed so
only the verifier differs.

| Declared condition | Required | Pilot 06 | |
|---|---|---|---|
| Genuine families detected | >= 5 / 6 | **4 / 6** | **FAIL** |
| Control false positives | 0 / 2 | 0 / 2 | pass |
| Parse failures | <= 2 | 0 / 41 | pass |
| Citation completeness vs B | within 0.05 | **-0.009** | pass |
| Structurally invalid revisions | 0 | 0 | pass |

**Four of five conditions pass. Detection alone defeats the gate**, and no
revisions remain. `DECISION: FAIL`.

## 1.12.2 What revision 3 did and did not demonstrate

It targeted one thing: the worked examples omitted the `claims` field the
schema required.

| | Pilot 05 | Pilot 06 |
|---|---|---|
| **Claim audits complete** | **0 / 41** | **41 / 41** |
| Conflicts detected (families) | 2 / 6 | 4 / 6 |
| Correctly classified | 1 / 6 | 2 / 6 |
| Control false positives | 1 / 2 | 0 / 2 |

Only the first line is attributable to revision 3. It is the quantity the
change was aimed at, it moved from nothing to complete, and no other
explanation is available.

**Detection rose from 2/6 to 4/6 after revision 3. It is not claimed that
revision 3 caused it.** The examples always demonstrated the relationship
label correctly, so there is no mechanism connecting the change to detection,
and n=1 either side. It is reported as a sequence, not a cause.

## 1.12.3 Everything else pilot 06 shows

* **Classification stayed weak: 2 / 6.** The verifier finds a disagreement more
  often than it names its kind, and the two demand different behaviour.
* **Eight of 35 answerable questions were falsely refused.** Of those, five had
  every expected passage retrieved, two had part of it, one had no expectation
  declared. The five are the verifier's error outright.
* **None of the six unanswerable questions received the abstention.** The
  system refuses answerable questions and answers unanswerable ones, which is
  close to the opposite of the intended behaviour and bears directly on RQ3.
* **Two invented evidence identifiers**, on CONF-05-Q2 (`GEN-04#001`) and
  GAP-export-Q2 (`ANSWER_UNDER_REVIEW`, which is not an identifier at all).
  Validation contained both: the draft was served in each case and neither
  reached a user. Contained is not absent, and they are reported.

## 1.12.4 False refusals are not a gate condition

Amendment 1.9 added them to the unmet list. That put a condition into the
declared gate after seeing the data the gate was judging, which is the move the
pre-registration exists to prevent, and I made it while correcting someone
else's reporting.

Removed. The declared gate has five conditions and this is not one. The count
is reported prominently because it matters to RQ3; it does not enter the
verdict. It would not have changed this outcome, and it would have if detection
had reached 5/6.

## 1.12.5 The budget

Three prompt revisions were available. All three are spent. Revision 3 was
declared the last before it was made, and the gate no longer offers a REVISE
verdict: `PROMPT_REVISIONS_REMAINING = 0` in the source, asserted by a test.

**There is no pilot 07.** No further change to the prompt, the model, the
retrieval settings, the corpus, the taxonomy or the serving rules.

## 1.12.6 The defensible statement

> A verification layer using `qwen2.5:3b` over `llama3.2:3b` drafts, with
> revision-2 prompting corrected for a schema defect, six retrieved chunks and
> `min_similarity: 0.30`, detected 4 of 6 genuine conflict families on the
> development split against a pre-declared threshold of 5, correctly
> classifying 2. It produced no false conflicts on the compatible controls, no
> parse failures, and citation completeness within 0.009 of the unverified
> baseline. It falsely refused 8 of 35 answerable questions and abstained on
> none of the 6 unanswerable ones. **The layer did not meet its declared
> development gate.**

What is not claimed: that verification cannot work, that a larger model would
fail, or that this is the ceiling for the approach. What was tested is one
configuration, arrived at through a pre-registered model diagnostic and three
declared prompt revisions.

## 1.12.7 State

No test-split arm has been run. The verifier is frozen next, at the
configuration above, and the freeze is committed before any test run. The test
split is evaluated **once**, and nothing is tuned from its results.

| | |
|---|---|
| Verifier | `qwen2.5:3b` |
| Generator | `llama3.2:3b` |
| Prompt | revision 3, sha256 `2d3bdaae...` |
| Retrieval | `top_k: 6`, `min_similarity: 0.30` |
| Development verdict | **FAIL, 4/6 detection against 5/6 required** |
| Revisions remaining | **0** |
| Tests | 451 |

---

# Amendment 1.13 - 14 August 2026

Recorded while manual scoring is in progress. A small number of items had been
judged before the flag below existed; the exact count is read from the log
rather than restated here, for the reason given in 1.13.4.

## 1.13.1 The blinding is defeated, and not on thirteen items

**Found.** Thirteen of the 272 entries in `results/manual/review_sheet.jsonl`
are the verbatim `ABSTENTION_TEXT` constant from
`src/sme_assistant/verify/schema.py`. They span 13 questions in 12 groups, and
all thirteen carry a single opaque code.

Only the verified arm can produce that string. `schema.py` assigns it as the
final answer when every claim returns `INSUFFICIENT_EVIDENCE`, so an unverified
arm has no path to it. The reviewer wrote the constant.

The count that matters is not thirteen. Opaque codes are stable across the
whole sheet, at 68 items each. Recognising the template once attaches a code to
the verified arm, and that code then identifies **all 68 of its items**,
including every conflict question. A thirteen-item trigger unblinds a quarter
of the sheet.

This is the failure of 1.1.10 repeated. That amendment removed the evidence
block for exactly this reason: blinding that a careful reader can defeat is
worse than none, because it is reported as a control.

## 1.13.2 Amendment 1.7.6 created it and the blinding check was not revisited

1.7.6 replaced the model's abstention prose with a fixed template written in
the source, so that no claim could be smuggled into free text. That was the
right decision for the serving rule and it is not withdrawn.

Its side effect is a byte-exact arm signature in the answer field, and the
answer field is the one thing the review sheet has to carry.

The build-time audit missed it because it was a list of known give-aways: arm
labels, model names, opaque codes, evidence status markers. It tested for the
leaks already known rather than for the property, so a constant introduced five
amendments later was never going to appear on it. The blinding was audited
against its own docstring.

## 1.13.3 It is recorded, not repaired

Two reasons the sheet is not rebuilt.

Item numbers are positions in a shuffled file and judgements already key on
them, so rebuilding would leave recorded scores pointing at different answers.
`manual_scoring.open_session` refuses a changed sheet for this reason.

More fundamentally, redaction would destroy the items. On those thirteen the
judgement *is* whether declining to answer was the right response to the
question asked, scored against whatever rubric that question carries. Replacing
the text with a placeholder leaves nothing to score, and scoring a paraphrase
of it scores something the system did not say.

There is no version of this in which the thirteen items are both blind and
judgeable. The leak stands and is reported.

## 1.13.4 The unblinding rate is measured rather than assumed

A per-item flag is added to the scoring tool. `i` records that the reviewer
believes they can identify which system produced the answer. It is tri-state:
items scored before the flag existed are recorded as **not asked**, not as
"no". Item 3 is among them, and item 3 is one of the thirteen.

The denominator is **derived, not declared.** Those lines simply have no
`arm_identified` field, and `load_judgements` reads an absent field as "not
asked" rather than as False, so the rate is computed over the items where the
question was actually put. Writing the count into this document instead would
make it wrong as soon as another item was scored, and a pre-registration whose
numbers rot is worse than one that says where to look. `score_answers.py
status` prints it.

**The flag does not ask which arm.** Prompting for a guess on every item is an
invitation to speculate, and would raise the unblinding it is meant to measure.
Blinding indices are conventionally collected after assessment for this reason.

**The resulting figure is a self-report and bounds nothing in either
direction.** A reviewer who believes they can identify the arm may be wrong,
which overstates it. A reviewer who has absorbed the pattern without noticing
will not flag it, which understates it. It is reported as what it is: the rate
at which the reviewer was aware of being unblinded. That is more than this
study previously had and less than a verified measurement.

## 1.13.5 What this does to the reported result

The affected arm is the contribution. B versus D is the confirmatory contrast
for H1 and H2, D is the arm the reviewer built and wants to win, and this
document says so in those words. Blinding was the control protecting that
contrast, and on 68 of 272 items it is now known to be defeasible.

Nothing here establishes that the reviewer did identify the arm. That is what
1.13.4 measures. What is established is that the control cannot be asserted, so
it will not be asserted.

Conflict handling and answer correctness are therefore reported with the
blinding described as **partial and self-monitored**, with the unblinding rate
stated alongside them rather than in a footnote. If that rate is high, the
manual metrics are reported as unblinded, and the confirmatory weight falls
back to the automatic metrics, which is less weight than the manual ones were
designed to carry. That consequence is accepted in advance here so it cannot be
negotiated later.

## What was not changed

* The review sheet and the key. Both are byte-identical to the files built at
  272 items, and the sheet hash is recorded in `results/manual/session.json`.
* The existing judgements. Not rescored on account of this, not deleted, not
  back-filled with a flag their reviewer was never shown.
* The frozen verifier, the four test runs, `SCORING_RUBRICS` and the arm
  definitions.
* Amendment 1.7.6. The template stays. The defect is in the blinding audit, not
  in the serving rule.

## 1.13.6 State

| | |
|---|---|
| Review sheet | 272 items, unchanged |
| Leaked items | 13 verbatim `ABSTENTION_TEXT`, one opaque code |
| Items exposed through that code | 68 |
| Blinding | **partial, self-monitored** |
| Unblinding flag denominator | items carrying the field, derived from the log |
| Items judged before the flag existed | recorded as not asked, not as no |
| Tests | 505 |
