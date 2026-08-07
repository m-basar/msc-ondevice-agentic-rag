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
| Corpus | `14295c5db36d` (37 documents) |
| Chunk set | `feccf6cccf2b` (134 chunks) |
| Registry | `50e697b201fe` |
| Conflict families | 9 reported, 2 tuning |
| Tests | 249 passing |

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
the generation model, the seed and the retrieval parameters. They differ in one
respect each, so any difference between adjacent arms is attributable.

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

**What would falsify it.** D exceeding B or C by more than the pre-specified
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

**What would falsify it.** D not exceeding the best of A, B and C by the
threshold in section 5. This is the hypothesis most likely to fail and the
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

### Cross-validation

Leave-one-family-out over the nine reported families. Folds are constructed by
family, never by question.

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
