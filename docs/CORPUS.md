# Knowledge Base Provenance

Documentation of how the synthetic corpus was produced, what was checked, and what is deliberately wrong. Written for Chapter 3 and for anyone attempting to reproduce the experiment.

## The organisation is fictional

**Northgate Kitchenware Ltd does not exist.** It is a fabricated wholesale kitchenware distributor invented for this research. No real organisation, employee, customer or supplier is represented.

The first version of this corpus used identifiers that looked real, including an eight-digit company registration number in valid Companies House format. Numbers in that format are allocated to real companies, so the corpus asserted things about a real legal entity while claiming to be entirely fabricated. That contradicted the ethics declaration and has been corrected.

All identifiers now use ranges reserved for exactly this purpose:

| Identifier type | Value used | Reserved by |
|---|---|---|
| Telephone numbers | `01632 960 xxx` | Ofcom, reserved for drama and fiction |
| Email domain | `northgate-kitchenware.invalid` | RFC 2606, `.invalid` is reserved and can never be registered |
| Company registration | None quoted | Removed entirely; the corpus states the company is unregistered |
| Postcode | `XX1 4LP` | `XX` is not an allocated UK postcode area |
| Address | Unit 12, Halden Industrial Park, Middleton | Invented site and town name |

**Nothing in this corpus should be treated as a real address, number or organisation.**

## Composition

Thirty-seven Markdown documents across eight categories, averaging around 285 words. Each carries front matter recording its identifier, title, category, version, effective date, status, and where relevant its supersession relationship and withdrawal date.

Run `python scripts/kb_summary.py` for current figures. Do not transcribe them by hand into the dissertation; regenerate them.

## How the documents were written

Documents were authored directly as static Markdown rather than produced by a generator script. They are research materials, and a reviewer must be able to read exactly what the system was given. A generator would add indirection without adding anything the experiment needs.

Content was written to be internally consistent: opening hours in GEN-02 match the delivery cut-offs in CS-11, the fire assembly point in OPS-07 matches the arrangements in REG-01, and every in-text cross-reference resolves to a document that exists. The last of these is enforced by the loader rather than trusted.

## Deliberate defects, and only deliberate ones

The corpus contains planted contradictions. It must not contain accidental ones.

Everything deliberate is registered in `data/conflicts.json`: four supersession families spanning HR, finance, information security and customer service, plus the list of topics that are absent by design. `validate_against_corpus` fails the test suite if a superseded document exists that no family accounts for, or if a topic declared absent has crept back into the corpus.

This distinction matters because gold answers are written against the corpus. An accidental contradiction that reaches the test set becomes an error the evaluation cannot see: the system would be marked wrong for being right, or right for being wrong, and every downstream number would be quietly corrupted.

## Legal and factual review

Corpus content was reviewed against current UK guidance on **7 August 2026**. The following were corrected at that review:

| Document | Issue | Correction |
|---|---|---|
| HR-02 | Stated SSP begins on the fourth qualifying day with a Lower Earnings Limit test. Both changed on 6 April 2026. | Retained as a superseded version; HR-12 created reflecting first-day payment and removal of the earnings threshold. Registered as conflict family CONF-02. |
| REG-01 | Stated a written health and safety policy is required for "more than five" employees. | Corrected to "five or more". |
| OPS-06 | Applied a blanket 12-month LOLER thorough examination interval, while OPS-05 directs staff to use a man-riding cage. Equipment used to lift people requires examination every 6 months. | Both intervals now stated, with the person-lifting configuration on the shorter cycle. |
| OPS-03 | Ran the 14-day statutory refund deadline only from receipt of returned goods. It also runs from the customer supplying evidence of dispatch, whichever is earlier. | Both triggers now stated. |
| OPS-01 vs FIN-01 | Credit hold applied at 60 days overdue in one document and 30 in the other. | Aligned on 30 days, with FIN-01 as the authority. |
| GEN-02 vs HR-01 | Site declared closed on all bank holidays while HR-01 granted lieu days to warehouse staff rostered on them. | GEN-02 now describes a reduced peak-season despatch shift, making both statements true. |
| FIN-02 | Dated 2025 while citing HR-13, an identifier created in 2026. | Effective date moved to the same annual review cycle. |

**Legal statements in this corpus are accurate as at the review date and are not legal advice.** They exist to make the documents realistic, not to be relied upon.

Where a superseded document states something that is no longer legally correct, that is the point: HR-02 and IT-01 are deliberately out of date and are registered as such.

## The superseded proportion is not representative

Four of thirty-seven documents are superseded, roughly 11% of the corpus. A real SME knowledge base would not carry that proportion of stale policy sitting alongside its replacements in the same searchable index.

The proportion is elevated deliberately so that conflict handling can be **measured** rather than merely illustrated. One superseded pair supports a case study; it does not support a rate. This is a threat to external validity and must be stated as such in the discussion, not buried.

## Superseded documents are indexed, not filtered

Excluding superseded documents at ingestion time would make the conflict problem disappear. They are indexed anyway, because detecting conflicting evidence is the capability under evaluation and it cannot be demonstrated if the conflict never reaches the model.

This is a research decision with an obvious objection: why not just filter? The comparison between filtering and verifying is a legitimate additional experimental arm and would strengthen the claim if time allows.

## Reproducibility

Every document is hashed with SHA-256, and the corpus carries a combined fingerprint derived from those hashes. The fingerprint changes if and only if document content changes, and is independent of file timestamps and read order.

Both the corpus fingerprint and the configuration fingerprint are recorded in every experimental run, so any table in the results chapters can be traced to the exact inputs that produced it.

```
python scripts/kb_summary.py --manifest > results/corpus_manifest.json
```
