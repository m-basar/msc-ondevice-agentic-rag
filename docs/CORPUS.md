# Knowledge Base Provenance

How the synthetic corpus was produced, what was checked, and what is deliberately wrong. Written for Chapter 3 and for anyone attempting to reproduce the experiment.

Legal and factual content last reviewed: **7 August 2026**.

## The organisation is fictional

**Northgate Kitchenware Ltd does not exist.** It is a fabricated wholesale kitchenware distributor invented for this research. No real organisation, employee, customer or supplier is represented.

The first version of this corpus used identifiers that looked real, including an eight-digit company registration number in valid Companies House format. Numbers in that format are allocated to real companies, so the corpus asserted things about a real legal entity while claiming to be entirely fabricated. That contradicted the ethics declaration and has been corrected.

All identifiers now use ranges reserved for exactly this purpose:

| Identifier type | Value used | Reserved by |
|---|---|---|
| Telephone numbers | `01632 960 xxx` | Ofcom, reserved for drama and fiction |
| Email domain | `northgate-kitchenware.invalid` | RFC 2606, `.invalid` can never be registered |
| Company registration | None quoted | Removed; the corpus states that no company of this name is registered |
| Postcode | `XX1 4LP` | `XX` is not an allocated UK postcode area |
| Address | Unit 12, Halden Industrial Park, Middleton | Fictional composite address |

The address is a composite rather than an invention. "Middleton" is a common English place name borne by several real settlements, so it is not claimed here as fictional; the unit number, the industrial park and the postcode are invented, and the postcode uses an unallocated area code so the combined address cannot resolve to a real location.

**Nothing in this corpus should be treated as a real address, number or organisation.**

## Composition

Thirty-seven Markdown documents across eight categories. Each carries front matter recording identifier, title, category, version, effective date, status, and where relevant its supersession relationship and withdrawal date.

Run `python scripts/kb_summary.py` for current figures. Do not transcribe them into the dissertation by hand; regenerate them.

## How the documents were written

Documents were authored directly as static Markdown rather than produced by a generator script. They are research materials, and a reviewer must be able to read exactly what the system was given.

Content is internally consistent: opening hours in GEN-02 match the delivery cut-offs in CS-11, the fire assembly point in OPS-07 matches REG-01, and every in-text cross-reference resolves. The last of these is enforced by the loader rather than trusted.

## Deliberate defects, and only deliberate ones

The corpus contains planted contradictions. It must not contain accidental ones.

Everything deliberate is registered in `gold/conflicts.json`. `validate_against_corpus` fails the test suite if a superseded document exists that no family accounts for, if a declared conflicting value is not present in the document text, if a topic declared absent has reappeared, or if a partial gap has lost its near-miss evidence.

This distinction matters because gold answers are written against the corpus. An accidental contradiction that reaches the test set becomes an error the evaluation cannot see: the system is marked wrong for being right, and every downstream number is quietly corrupted.

## The gold data boundary

`gold/conflicts.json` holds expected answers, prohibited assertions and the list of unanswerable topics. **No component of the inference pipeline may read it.**

Three mechanisms enforce this rather than one:

1. **Gold data locations live in a separate configuration file.** `gold/evaluation.json` is loaded only by `sme_assistant.evaluation.config`. The runtime `Config` object built from `config.json` has no key that leads there, so there is no config lookup at all, correct or otherwise, that reaches the answer key.
2. The loader lives in `sme_assistant.evaluation`, not `sme_assistant.kb`, so the boundary is visible in the import graph.
3. `tests/test_no_oracle_leakage.py` parses every module in the inference packages and fails if any imports the registry, imports from the evaluation package, contains a literal path to gold data, or attempts a config lookup for an evaluation key.

An earlier design placed evaluation paths under `config.evaluation`, alongside runtime paths in the same file. A reviewer observed that this demonstrated the absence of *current* leakage while leaving the route open: an inference module could have called `config.path("evaluation.conflicts")` and read the answer key, importing nothing and containing no literal path, so the tests would have passed. Splitting the files removes the possibility rather than testing for its absence.

Without this separation the system would be scoring against its own answer key.

## Evaluation protocol

Declared once, in `gold/evaluation.json`, and asserted by a test:

- **Fixed held-out evaluation.** The nine reported families are scored once. This was called leave-one-family-out cross-validation, which was wrong: nothing is trained on the remaining folds, so there is no model being validated. A leave-one-family-out calculation is retained as a sensitivity analysis, showing how far the overall figure moves when any single family is removed.
- **Macro-averaged by family.** Paraphrased questions drawn from one family test the same document pair and are not independent observations. The reported sample is 26 test groups, not the 42 groups the artefact contains.
- **Reported and tuning families are separate.** Families carry `split: reported` or `split: tuning`. Tuning families exist so prompt wording, thresholds and verifier output can be developed against real conflicts without inspecting a family that is later scored. A reported family may not appear in the development split, and a tuning family may not appear in the test split.

## Conflict design

Thirteen registered families: **nine reported** and **four tuning**. The type
distinction is the important part.

| Type | Reported | Resolvable by a metadata filter? |
|---|---|---|
| `version_supersession` | 4 | **Yes.** Filtering on `status == current` resolves them with no reasoning at all. |
| `current_current` | 5 | **No.** Both documents are live, both authoritative, no metadata field ranks them. |

An earlier version of this corpus contained only supersession conflicts. That was a design flaw: an examiner could reasonably ask why a claim-level verification layer is needed when three lines of filtering achieve the same result. Supersession families alone cannot demonstrate that the contribution adds anything.

Two of these began as accidental contradictions found during review. Rather than harmonising them away, they were promoted to registered conflicts, because they are exactly the class a filter cannot touch and exactly what happens in real organisations when two departments draft policy separately. CONF-07 is one such: GEN-04 and FIN-03 disagree about who approves an £800 purchase, and neither mentions the other.

### Reported `current_current` families

| Family | Risk | Documents | Conflict |
|---|---|---|---|
| CONF-06 | medium | IT-04 vs REG-02 | An annual backup is kept seven years, while records must be deleted at the end of their retention period including from backups. |
| CONF-07 | low | GEN-04 vs FIN-03 | An £800 IT purchase needs Finance Manager approval, or department head approval. |
| CONF-08 | low | GEN-02 vs CS-11 | The trade counter opens 08:00 to 17:00, or 08:30 to 16:30. |
| CONF-09 | medium | IT-11 vs IT-02 | Access rights are reviewed every six months, or annually. |
| CONF-11 | high | GEN-03 vs OPS-05 | A visitor needs safety footwear in the warehouse at all times, or is exempt for escorted visits under fifteen minutes. |

### Reported `version_supersession` families

| Family | Risk | Documents | Conflict |
|---|---|---|---|
| CONF-02 | high | HR-02 → HR-12 | Statutory Sick Pay from the fourth qualifying day, or the first. |
| CONF-03 | medium | IT-01 → IT-11 | Passwords of 8 characters rotated every 90 days, or 14 characters with no rotation. |
| CONF-04 | low | CS-01 → CS-11 | Free delivery over £50, or over £75. |
| CONF-10 | medium | CS-04 → CS-14 | A stage 1 complaint resolved in 3 working days, or 2. |

### Tuning families

Never reported. CONF-01 and CONF-05 were moved here on 8 August 2026 under
pre-registration amendment 1.1, because both were Stage 4 pilot questions and
the lost-laptop pilot answer directly motivated
`evaluation/answer_scoring.py`. A question that shaped the system cannot
afterwards test it.

| Family | Type | Documents | Conflict |
|---|---|---|---|
| CONF-01 | version_supersession | HR-03 → HR-13 | Mileage at a flat 40p, or 55p then 25p. |
| CONF-05 | current_current | GEN-04 vs IT-03 | Lost equipment reported within 24 hours, or within 1 hour. |
| TUNE-01 | current_current | OPS-02 vs CS-03 | A returns window of 30 days, or 28. |
| TUNE-02 | current_current | OPS-08 vs REG-01 | An accident book entry reviewed within two working days, or five. |

Correct behaviour on a `current_current` family is not to pick one. It is to surface both, name both documents, state that neither supersedes the other, cap confidence at low, and escalate. The assistant does not recommend an interim safe action; that rule is applied consistently across every family and is recorded in pre-registration amendment 1.1.

`tests/test_conflicts.py::test_registry_contains_conflicts_a_metadata_filter_cannot_solve` fails if the count of reported `current_current` families drops below four.

### Evaluation implication

Three arms should be compared, not two:

1. Naive retrieval over all documents
2. Retrieval filtered to current documents only
3. The full verification pipeline

Arm 2 is the cheap obvious alternative to this project's contribution. Arm 3 must beat it on the `current_current` families, or the contribution is not justified.

### Independence

Multiple paraphrased questions drawn from one family are not independent observations: they test the same document pair. Results must be macro-averaged by family, not micro-averaged by question.

## Legal and factual review

The following were corrected at the reviews of 7 August 2026:

| Document | Issue | Correction |
|---|---|---|
| HR-02 | SSP stated as beginning on the fourth qualifying day with a Lower Earnings Limit test. Both changed on 6 April 2026. | Retained as a superseded version; HR-12 created with first-day payment and no earnings threshold. Registered as CONF-02. |
| HR-13 | Claimed 45p per mile matches the HMRC approved rate. The AMAP rate rose to 55p for the first 10,000 miles from 6 April 2026, the first change since 2011/12. | Updated to 55p/25p with the effective date stated. |
| REG-01 | Written health and safety policy threshold given as "more than five" employees. | Corrected to "five or more". |
| OPS-06 | Blanket 12-month LOLER examination interval, while OPS-05 directed staff to a man-riding cage. Equipment used to lift people requires 6-monthly examination. | Both intervals stated, person-lifting configuration on the shorter cycle. |
| OPS-05 | Implied routine use of a non-integrated cage for stock access. HSE treats these as exceptional. | Routine picking moved to an integrated-platform order picker; cage restricted to occasional exceptional access under a same-day work-at-height assessment. |
| OPS-03 | Ran the statutory refund deadline only from receipt of goods, and blocked all refunds on physical inspection. | Three refund routes separated. Statutory cancellation runs from receipt or evidence of dispatch, whichever is earlier, and may fall due before inspection. |
| OPS-03 | Applied an automatic 15% restocking deduction. | Replaced with a deduction reflecting assessed diminished value, evidenced and explained, and not applied to statutory cancellations. |
| OPS-01 vs FIN-01 | Credit hold at 60 days in one document, 30 in the other. | Aligned on 30 days, FIN-01 as authority. |
| GEN-02 vs HR-01 | Site declared closed on all bank holidays while HR-01 granted lieu days. | GEN-02 now describes a reduced peak-season despatch shift. |
| FIN-02 | Dated 2025 while citing HR-13, an identifier created in 2026. | Effective date moved. |
| OPS-01 | Incomplete email address `sales@`. | Completed with the reserved `.invalid` domain. |

**Legal statements in this corpus are accurate as at the review date and are not legal advice.** They exist to make the documents realistic.

Where a superseded document states something no longer legally correct, that is the point. HR-02 and IT-01 are deliberately out of date and registered as such.

## Deliberate gaps

Eight topics are fully absent, verified by keyword probes stored in the registry rather than in code. Two are partially present, which is the harder category: the topic is named somewhere but the answer is not there.

One correction worth noting. "International shipping" was originally declared fully absent, but CS-11 explicitly states that the company does not ship outside the United Kingdom, which answers the availability question. Only the **export documentation procedure** is genuinely absent. Questions must target the procedure, not the availability, or they are not unanswerable at all.

## The superseded proportion is not representative

Four of thirty-seven documents are superseded, roughly 11%. A real SME knowledge base would not carry that proportion of stale policy alongside its replacements in the same searchable index.

The proportion is elevated deliberately so conflict handling can be **measured** rather than illustrated. One pair supports a case study; it does not support a rate. This is a threat to external validity and must be stated in the discussion, not buried.

## Superseded documents are indexed, not filtered

Excluding superseded documents at ingestion would make the conflict problem disappear. They are indexed anyway, because detecting conflicting evidence is the capability under evaluation and cannot be demonstrated if the conflict never reaches the model.

The obvious objection, why not just filter, is addressed by the `current_current` families and by arm 2 of the evaluation design above.

## Reproducibility

Every document is hashed with SHA-256, and the corpus carries a combined fingerprint derived from those hashes. It changes if and only if document content changes, and is independent of file timestamps and read order.

Full-length hashes are stored; truncation to twelve characters happens only at the point of display.

Each run manifest records:

- Full corpus SHA-256
- Conflict registry SHA-256
- Configuration SHA-256
- Git commit, branch, and whether the working tree was dirty
- Schema versions
- Legal review date
- Host, hardware, thermal and Ollama environment, including the model store fingerprint

```
python scripts/kb_summary.py --manifest > results/corpus_manifest.json
```

A dirty working tree is recorded rather than ignored: a run made with uncommitted changes is not reproducible from its commit alone, and that should be visible.

## Continuous integration

`.github/workflows/tests.yml` runs the full test suite and the corpus validator on every push to `main`. Corpus validation is a separate step from the unit tests so that a failure is unambiguous: it means a document was edited into a state that contradicts the registry, not that code is broken.
