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
