# Knowledge Base Provenance

How the synthetic corpus was produced, what was checked, and what is deliberately wrong.

**This file is generated.** Run `python scripts/make_corpus_doc.py > docs/CORPUS.md`.
Every count is read from the registry, the question set, the index and the corpus
itself; nothing here is transcribed. Do not edit it by hand.

Legal and factual content last reviewed: **7 August 2026**.

## The authoritative implementation

`final_v1/` is the implementation the dissertation reports. The older top-level
`artefact/` directory is a **superseded July 2026 design**, retained only as a
historical record. It predates the four-arm design, the conflict taxonomy, the
pre-registration and every reported result, and must not be used for dissertation
claims, for a demonstration, or for any dashboard.

## The organisation is fictional

**Northgate Kitchenware Ltd does not exist.** It is a fabricated wholesale
kitchenware distributor invented for this research. No real organisation,
employee, customer or supplier is represented.

An early version used identifiers that looked real, including an eight-digit
company registration number in valid Companies House format. Numbers in that
format are allocated to real companies, so the corpus asserted things about a
real legal entity while claiming to be entirely fabricated. That contradicted the
ethics declaration and was corrected. All identifiers now use ranges reserved
for exactly this purpose.

| Identifier type | Value used | Reserved by |
|---|---|---|
| Telephone numbers | `01632 960 xxx` | Ofcom, reserved for drama and fiction |
| Email domain | `northgate-kitchenware.invalid` | RFC 2606, `.invalid` can never be registered |
| Company registration | None quoted | Removed; the corpus states that no company of this name is registered |
| Postcode | `XX1 4LP` | `XX` is not an allocated UK postcode area |
| Address | Unit 12, Halden Industrial Park, Middleton | Fictional composite address |

The address is a composite rather than an invention. "Middleton" is a common
English place name borne by several real settlements, so it is not claimed here as
fictional; the unit number, the industrial park and the postcode are invented, and
the postcode uses an unallocated area code so the combined address cannot resolve
to a real location.

**Nothing in this corpus should be treated as a real address, number or organisation.**

## Composition

**38 Markdown documents, 11,374 words, 141 chunks.** 33 documents are
current and 5 are superseded, 13.2 per cent of the corpus. Each carries front
matter recording identifier, title, category, version, effective date, status and,
where relevant, its supersession relationship and withdrawal date.

| Category | Documents |
|---|---:|
| CS | 6 |
| FIN | 3 |
| GEN | 4 |
| HR | 8 |
| IT | 5 |
| OPS | 8 |
| PRD | 2 |
| REG | 2 |
| **Total** | **38** |

Documents were authored directly as static Markdown rather than produced by a
generator script. They are research materials, and a reviewer must be able to
read exactly what the system was given.

Content is internally consistent except where a contradiction is registered:
opening hours match delivery cut-offs, the fire assembly point matches across
documents, and every in-text cross-reference resolves. The last of these is
enforced by the loader rather than trusted.

## Deliberate defects, and only deliberate ones

The corpus contains planted contradictions. It must not contain accidental ones.

Everything deliberate is registered in `gold/conflicts.json`. `validate_against_corpus`
fails the test suite if a superseded document exists that no family accounts for,
if a declared conflicting value is not present in the document text, if a topic
declared absent has reappeared, or if a partial gap has lost its near-miss evidence.

This distinction matters because gold answers are written against the corpus. An
accidental contradiction that reaches the test set becomes an error the evaluation
cannot see: the system is marked wrong for being right, and every downstream
number is quietly corrupted.

## The gold data boundary

`gold/conflicts.json` holds expected answers, prohibited assertions and the list of
unanswerable topics. **No component of the inference pipeline may read it.**

Three mechanisms enforce this rather than one.

1. **Gold data locations live in a separate configuration file.** `gold/evaluation.json`
   is loaded only by `sme_assistant.evaluation.config`. The runtime `Config` object
   built from `config.json` has no key that leads there, so there is no config
   lookup at all, correct or otherwise, that reaches the answer key.
2. The loader lives in `sme_assistant.evaluation`, not `sme_assistant.kb`, so the
   boundary is visible in the import graph.
3. `tests/test_no_oracle_leakage.py` parses every module in the inference packages
   and fails if any imports the registry, imports from the evaluation package,
   contains a literal path to gold data, or attempts a config lookup for an
   evaluation key.

An earlier design placed evaluation paths under `config.evaluation`, alongside
runtime paths in the same file. That demonstrated the absence of *current*
leakage while leaving the route open: an inference module could have called
`config.path("evaluation.conflicts")` and read the answer key, importing nothing
and containing no literal path, so the tests would have passed. Splitting the
files removes the possibility rather than testing for its absence.

## Evaluation protocol

Declared in `gold/evaluation.json` and in `docs/PREREGISTRATION.md` section 5, and
asserted by tests.

- **Fixed held-out evaluation.** The 15 reported families are scored once. This was
  once called leave-one-family-out cross-validation, which was wrong: nothing is
  trained on the remaining folds, so there is no model being validated. A
  leave-one-family-out calculation is retained as a sensitivity analysis.
- **Macro-averaged by family.** Paraphrased questions drawn from one family test the
  same document pair and are not independent observations. The reported sample is
  **32 test groups**, not the 109 questions or 53 groups the artefact contains.
- **Reported and tuning families are separate.** Families carry `split: reported` or
  `split: tuning`. Tuning families exist so prompt wording, thresholds and verifier
  output can be developed against real conflicts without inspecting a family that
  is later scored. A reported family may not appear in the development split, and a
  tuning family may not appear in the test split; the question-set loader enforces
  both.

## Conflict design

23 registered families: **15 reported** and **8 tuning**. The type distinction is
the important part, and it is the part this document previously got wrong. An
earlier taxonomy had two types, one of them `current_current`; pre-registration
amendments 1.2, 1.4 and 1.5 replaced it with the four below, reclassified four
families that had been typed by intuition, and pooled two of the types for the
confirmatory analysis.

| Type | Reported | Correct behaviour | Resolvable by a metadata filter? |
|---|---:|---|---|
| `version_supersession` | 4 | Cite the current document; the superseded one is not authority. | **Yes.** Filtering on `status == current` resolves them with no reasoning at all. |
| `mutually_exclusive` | 3 | Surface both, name neither as correct, and escalate. | **No.** Both documents are live and no single action satisfies both. |
| `stricter_looser` | 5 | Name the stricter course, which satisfies both documents. | **No.** Both are live; the stricter course satisfies both, but no metadata field says which is stricter. |
| `compatible` | 3 | Answer plainly and assert no conflict. | **Not applicable.** Negative controls. There is no conflict to resolve, and asserting one is the failure. |

An early version of this corpus contained only supersession conflicts. That was a
design flaw: an examiner could reasonably ask why a claim-level verification layer
is needed when three lines of filtering achieve the same result. Supersession
families alone cannot demonstrate that the contribution adds anything.

Some families began as accidental contradictions found during review. Rather than
harmonising them away, they were promoted to registered conflicts, because they
are exactly the class a filter cannot touch and exactly what happens in real
organisations when two departments draft policy separately.

### Reported `version_supersession` families

| Family | Risk | Documents | Conflict |
|---|---|---|---|
| CONF-02 | high | HR-02 -> HR-12 | first payable day of Statutory Sick Pay: fourth qualifying day, three unpaid waiting days (HR-02) against first qualifying day, no waiting days (HR-12). |
| CONF-03 | medium | IT-01 -> IT-11 | minimum password length: 8 characters with complexity requirements (IT-01) against 14 characters, three random words encouraged (IT-11). |
| CONF-04 | low | CS-01 -> CS-11 | free delivery threshold: £50 excluding VAT (CS-01) against £75 excluding VAT (CS-11). |
| CONF-10 | medium | CS-04 -> CS-14 | stage 1 complaint resolution deadline: 3 working days (CS-04) against 2 working days (CS-14). |

### Reported `mutually_exclusive` families

| Family | Risk | Documents | Conflict |
|---|---|---|---|
| CONF-06 | medium | IT-04 vs REG-02 | how long data survives once its retention period has expired: an annual backup of all business systems is kept for seven years (IT-04) against records are deleted at the end of their retention period, including from backups at the next rollover (REG-02). |
| CONF-08 | low | GEN-02 vs CS-11 | trade counter opening hours: Monday to Friday, 08:00 to 17:00 (GEN-02) against Monday to Friday, 08:30 to 16:30 (CS-11). |
| CONF-13 | high | OPS-07 vs OPS-05 | where staff assemble on hearing the fire alarm: the north corner of the staff car park (OPS-07) against the main gate, for warehouse staff (OPS-05). |

### Reported `stricter_looser` families

| Family | Risk | Documents | Conflict |
|---|---|---|---|
| CONF-11 | high | GEN-03 vs OPS-05 | whether a visitor needs safety footwear to enter the warehouse: not required for escorted visits under fifteen minutes on marked walkways (GEN-03) against mandatory for everyone at all times, including visitors passing through (OPS-05). |
| CONF-12 | medium | OPS-01 vs FIN-01 | how many days overdue an invoice must be before orders are held: more than 45 days overdue (OPS-01) against 30 days overdue, account placed on hold (FIN-01). |
| CONF-14 | medium | CS-14 vs CS-02 | how quickly a customer complaint or damage report is acknowledged: the same working day (CS-14) against within two working days (CS-02). |
| CONF-15 | low | OPS-01 vs CS-11 | the latest time an order can be placed and still ship the same day: released before 13:00 (OPS-01) against placed and paid before 14:00 (CS-11). |
| CONF-16 | high | IT-02 vs REG-02 | how quickly a suspected personal data breach is reported to the Data Protection Lead: within 24 hours of discovery (IT-02) against within 72 hours (REG-02). |

### Reported `compatible` families

| Family | Risk | Documents | Conflict |
|---|---|---|---|
| CONF-07 | low | GEN-04 vs FIN-03 | who approves an IT equipment purchase of £800: the department head, and additionally the Finance Manager because the cost exceeds £750 (GEN-04) against the department head, under the £501 to £2,500 band (FIN-03). |
| CONF-09 | medium | IT-11 vs IT-02 | how often user access rights are reviewed: every six months, by each department head (IT-11) against annually, by the Data Protection Lead (IT-02). |
| CONF-17 | low | FIN-02 vs FIN-03 | who approves a £30 operational purchase: the Finance Manager, for petty cash above £25 (FIN-02) against the line manager, for procurement up to £500 (FIN-03). |

### Tuning families

Never reported. CONF-01 and CONF-05 were moved here on 8 August 2026 under
pre-registration amendment 1.1, because both were Stage 4 pilot questions and the
lost-laptop pilot answer directly motivated `evaluation/answer_scoring.py`. A
question that shaped the system cannot afterwards test it.

| Family | Type | Documents | Conflict |
|---|---|---|---|
| CONF-01 | `version_supersession` | HR-03 -> HR-13 | mileage rate for the first 10,000 business miles: 40 pence per mile, flat rate (HR-03) against 55 pence per mile, then 25 pence thereafter (HR-13). |
| CONF-05 | `stricter_looser` | GEN-04 vs IT-03 | time limit for reporting a lost or stolen company device: within 24 hours, to the IT helpdesk (GEN-04) against within 1 hour of becoming aware (IT-03). |
| TUNE-01 | `stricter_looser` | OPS-02 vs CS-03 | how many days after delivery an unwanted item may be returned: 30 days (OPS-02) against 28 days (CS-03). |
| TUNE-02 | `stricter_looser` | OPS-08 vs REG-01 | how long the Health and Safety Coordinator has to review an accident book entry: two working days (OPS-08) against five working days (REG-01). |
| TUNE-03 | `compatible` | HR-04 vs HR-06 | the core hours during which an employee must be contactable: 09:30 to 16:00 (HR-04) against 10:00 to 16:30 (HR-06). |
| TUNE-04 | `compatible` | PRD-01 vs PRD-02 | whether detergent may be used when washing by hand: warm soapy water, for knives (PRD-01) against hot water and a stiff brush, a little washing-up liquid on a well-seasoned pan (PRD-02). |
| TUNE-05 | `mutually_exclusive` | HR-01 vs GEN-01 | the dates of the company leave year: 1 April to 31 March (HR-01) against 1 January to 31 December (GEN-01). |
| TUNE-06 | `stricter_looser` | OPS-02 vs OPS-03 | how long a Returns Authorisation number stays valid: 21 days (OPS-02) against 14 days (OPS-03). |

### Evaluation implication

**Four arms are compared, not three.** The arms form a tree rooted at Arm B rather
than a ladder, so that each contrast changes one thing.

| Arm | Retrieval | Evidence shown | Verification |
|---|---|---|---|
| A | all documents | identifier and text only | none |
| B | all documents | with status metadata | none |
| C | current documents only | with status metadata | none |
| D | all documents | with status metadata | yes |

Arm C is the cheap obvious alternative to this project's contribution: a filter on
document status resolves every supersession family with no reasoning at all. **B
against D is the confirmatory single-variable contrast** for verification. C
against D changes retrieval mode and verification together and is reported as a
practical comparison, not as an ablation.

## Deliberate gaps

8 topics are fully absent, verified by keyword probes stored in the registry
rather than in code. 2 are partially present, which is the harder category: the
topic is named somewhere but the answer is not there.

| Topic | Kind |
|---|---|
| pensions and auto-enrolment | fully absent |
| maternity, paternity and shared parental leave | fully absent |
| redundancy and notice periods | fully absent |
| company car and vehicle allowance schemes | fully absent |
| share options and bonus schemes | fully absent |
| export documentation procedure | fully absent |
| flexible working requests | fully absent |
| staff purchase and employee discount | fully absent |
| probationary period | partially present, named in HR-04 |
| grievance procedure | partially present, named in HR-05 |

One correction worth noting. "International shipping" was originally declared
fully absent, but CS-11 explicitly states that the company does not ship outside
the United Kingdom, which answers the availability question. Only the **export
documentation procedure** is genuinely absent. Questions must target the
procedure, not the availability, or they are not unanswerable at all.

## The superseded proportion is not representative

5 of 38 documents are superseded, 13.2 per cent. A real SME knowledge base would
not carry that proportion of stale policy alongside its replacements in the same
searchable index.

The proportion is elevated deliberately so conflict handling can be **measured**
rather than illustrated. One pair supports a case study; it does not support a
rate. This is a threat to external validity and is stated in the discussion rather
than buried.

## Superseded documents are indexed, not filtered

Excluding superseded documents at ingestion would make the conflict problem
disappear. They are indexed anyway, because detecting conflicting evidence is the
capability under evaluation and cannot be demonstrated if the conflict never
reaches the model. The obvious objection, why not just filter, is exactly what Arm
C tests, and it is answered by the families a filter cannot touch.

## Reproducibility

Every document is hashed with SHA-256, and the corpus carries a combined
fingerprint derived from those hashes. It changes if and only if document content
changes, and is independent of file timestamps and read order. Full-length hashes
are stored; truncation to twelve characters happens only at the point of display.

Each run manifest records the full corpus, registry and configuration hashes, the
git commit and branch and whether the working tree was dirty, schema versions, the
legal review date, and the host, hardware, thermal and Ollama environment
including the model store fingerprint. A dirty working tree is recorded rather than
ignored: a run made with uncommitted changes is not reproducible from its commit
alone, and that should be visible.

```
python scripts/kb_summary.py --manifest > results/corpus_manifest.json
```

## Continuous integration

`.github/workflows/tests.yml` runs the full test suite and the corpus validator on
every push to `main`. Corpus validation is a separate step from the unit tests so
that a failure is unambiguous: it means a document was edited into a state that
contradicts the registry, not that code is broken.
