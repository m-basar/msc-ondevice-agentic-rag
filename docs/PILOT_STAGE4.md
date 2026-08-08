# Stage 4 Pilot: baseline generation, development records

**These are development observations, not test results.** They influenced the
design of the experiment and must never be reported as unseen evaluation data.

Specifically, these four outputs caused three design changes:

1. The no-status-marker arm (A) was added, because result 1 showed the marker
   was doing most of the work.
2. The chunker's section attribution was rewritten, because inspecting these
   results exposed sibling headings rendered as a hierarchy.
3. Refusal detection was demoted to a diagnostic, because result 4 showed a
   correct refusal that the detector missed.

Any question that shapes a system cannot afterwards test it. These four
questions belong permanently to the development split.

## Provenance

| | Value |
|---|---|
| Commit | `f274f60` |
| Corpus SHA-256 | `dd51cf741e59...` **(historical: the corpus at the time of the pilot, not the current one)** |
| Chunks at the time | **147** (the corpus now produces 133; chunk identifiers below refer to the 147-chunk layout and are not comparable to current ones) |
| Index | 147 chunks, 768 dimensions, `nomic-embed-text` |
| Model store fingerprint | `e39bc25c0edd` (identical on both machines) |
| Generation model | `llama3.2:3b` |
| Ollama | 0.32.6 on both machines |
| Options | `temperature 0.1, num_ctx 4096, num_predict 400` |
| **Seed** | **Not sent.** `project.seed` was recorded in provenance but never passed to Ollama. These outputs are therefore not exactly reproducible. Fixed after this pilot. |
| Retrieval | `top_k 4`, `min_similarity 0.32` |
| Laptop | Ryzen 7 7840HS, RTX 4050, Ollama 0.32.6, Balanced power profile |
| Pi 5 | aarch64, CPU only, corrected fan curve, **85.3°C at 1.50 GHz, throttled on all four queries** |

Laptop and Pi produced **identical answers and identical citations** on all
four questions. Retrieval scores differed by at most 0.0007.

---

## 1. Mileage rate — supersession conflict (CONF-01)

**Question:** what is the mileage rate for business travel?

**Retrieved:** HR-13#001 (0.7791), HR-03#001 (0.7071, SUPERSEDED), HR-13#002 (0.6803), HR-03#002 (0.6064, SUPERSEDED)

**Answer:**
> According to [HR-13#001], the mileage rate for business travel is as follows:
> * 55 pence per mile for the first 10,000 business miles in a tax year
> * 25 pence per mile thereafter.

**Audit:** cited `HR-13#001`; no hallucinated citations; **no superseded document cited**; grounded.

**Observation.** Half the evidence was a withdrawn policy stating 40 pence, and
the model ignored it. It had two signals: the `[SUPERSEDED, replaced by HR-13]`
marker in the evidence block, and HR-03's own opening line, "This version has
been withdrawn. See HR-13 for the policy in force."

**Implication.** Clearly-marked supersession appears to be handled by a 3B model
without assistance. If this holds across the development set, the four
supersession families will show little difference between arms, and the
`current_current` families become the entire quantitative case for the
contribution. This is the observation that motivated hypothesis H1.

**Same question, `current_only`:** HR-03 excluded, same answer, same citation.

## 2. Lost laptop — live conflict between two current documents (CONF-05)

**Question:** how quickly must I report a lost laptop?

**Retrieved:** IT-03#001 (0.6909), IT-03#002 (0.6806), GEN-04#003 (0.6214), IT-04#004 (0.5963)

**Answer:**
> You must report a lost laptop to the IT helpdesk within 1 hour of becoming aware [IT-03#001].

**Audit:** cited `IT-03#001`; grounded; `GEN-04#003` retrieved and uncited.

**Observation.** GEN-04 states 24 hours and IT-03 states 1 hour. Both are
current. Both were retrieved. The model asserted one deadline without
qualification and never mentioned the other. `RetrievalResult.conflicts` was
empty, correctly: no metadata distinguishes two live documents.

**A second failure in the same answer.** The "within 1 hour" figure is in
**IT-03#002**, the Timescales table. The model cited **IT-03#001**. The claim is
true and the citation points at the wrong chunk. This is invisible under the
Stage 4 metrics, which count a citation as valid if the identifier was
retrieved. It is the reason citation *support* must be measured separately from
citation validity.

**Implication.** This is the failure mode the verification layer exists to
address, observed rather than hypothesised. `current_only` would return the same
four chunks and be equally unable to resolve it.

## 3. Pension scheme — deliberate gap, unanswerable

**Question:** what is the company pension scheme?

**Retrieved:** HR-01#001 (0.5462), HR-13#001 (0.5435), IT-02#001 (0.5407), HR-06#003 (0.5392)

**Answer:**
> I cannot provide an answer about the company pension scheme as it is not present in the provided evidence.

**Audit:** no citations; `looks_like_refusal` returned **False**.

**Two separate observations.**

The model refused correctly. Retrieval did not: the best score of 0.5462 is
above the `0.32` threshold, so evidence was supplied and the refusal came from
the model rather than the pipeline.

The evaluator failed. `looks_like_refusal` matched neither "cannot provide an
answer" nor "not present in the provided evidence". Keyword matching over free
text is not a sound basis for scoring refusal, and this is now a regression
test. Final refusal scoring uses a predefined rubric or structured system
output, with the regex retained only as a diagnostic.

**Threshold calibration.** Across these four questions:

| Question | Best score | Answerable | Refused by threshold |
|---|---|---|---|
| Mileage rate | 0.7791 | Yes | No, correct |
| Lost laptop | 0.6909 | Yes | No, correct |
| Pension scheme | 0.5462 | **No** | **No, incorrect** |

`nomic-embed-text` produces a compressed similarity range on this corpus, so an
absolute cosine threshold has little room to operate. `0.32` is far too low and
was never calibrated. It must be selected from development data, and the
alternative of letting verification make the refusal decision should be
compared against it.

## 4. Raspberry Pi timing

| Question | Generation | Device |
|---|---|---|
| Mileage rate | 51.30s | 85.3°C @1.50GHz, throttled |
| Mileage rate, current only | 31.63s | 85.3°C @1.50GHz, throttled |
| Lost laptop | 38.14s | 85.3°C @1.50GHz, throttled |
| Pension scheme | 30.87s | 85.3°C @1.50GHz, throttled |

**These are sustained real-use figures, not a thermally controlled benchmark.**
The queries ran back to back with no cool-down, which is what continuous use
looks like. The controlled benchmark, which cools to 65°C between measurements,
recorded 2.20 to 2.37 GHz.

Burst performance is therefore around 2.3 GHz and sustained performance around
1.5 GHz, a 35% clock reduction, on identical hardware and software with the
corrected fan curve already applied. Both figures are valid; they answer
different questions and must be labelled accordingly.

## Status

| | |
|---|---|
| Split | **Development. Permanently excluded from the test set.** |
| Reproducible | No: seed was not sent to the model |
| Chunk identifiers | Refer to the 147-chunk layout at commit `f274f60`, superseded by the current 133-chunk layout |
| Citable as | Development observations and motivation for design decisions |
| Not citable as | Baseline performance, arm comparison, or any final result |


---

## What was done about this, 8 August 2026

This document says that questions which shape a system cannot afterwards test
it. Version 1.0 of the question set placed **CONF-01** and **CONF-05** in the
test split anyway, and `CONF-01-Q1` was the mileage question above, verbatim.

Both families were moved to `split: "tuning"` under **pre-registration
amendment 1.1**, and two replacements were planted so the reported set keeps
four supersession and five `current_current` families:

| Moved to tuning | Replaced by |
|---|---|
| CONF-01, mileage | CONF-10, complaint response deadlines (CS-04 to CS-14) |
| CONF-05, lost laptop | CONF-11, visitor safety footwear (GEN-03 against OPS-05) |

`sme_assistant.evaluation.question_set` now refuses a question set that places a
reported family in the development split or a tuning family in the test split,
so the specific mistake this document warned about cannot be made silently
again.

The corpus and chunk hashes recorded above are historical. The pilot outputs are
kept as they were, unreproducible and excluded from every reported result.
