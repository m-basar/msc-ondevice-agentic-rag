# Appendix D: Verifier relationship classification

Supporting detail for the exploratory section 4.13. **Post-hoc and exploratory**,
governed by pre-registration amendment 1.29: the pattern was inspected before
the rule was written, no threshold is applied, no verdict is reached and no
chance baseline is computed. Source is the frozen Arm D quality run
`20260814_055018_D_test` and nothing else.

Two measures are reported and never combined. **Detection** is binary: did the
verifier report any conflict relationship. **Exact classification** asks whether
the reported relationship equalled the declared type mapped through
`DECLARED_TO_INFERRED`, which is used unmodified.

## D.1 Confusion matrix

Rows are the declared type from the registry, columns the relationship the
verifier returned. The cell that would be an exact match is emboldened.

| Declared type | `insufficient` | `mutually_exclusive` | `no_relationship` | `stricter_looser` | `supersession` |
|---|---:|---:|---:|---:|---:|
| Version supersession | 2 | 4 | 2 | - | **4** |
| Mutually exclusive | 3 | **1** | 5 | - | - |
| Stricter-looser | 1 | 4 | 8 | **1** | 1 |
| Compatible (controls) | - | 4 | 4 | - | 1 |

Three observations, offered as description.

The verifier **never returned `contextually_compatible`**, so one of its six
categories went unused across the whole run, and it is the one the three
compatible control families call for.

`no_relationship` is the most common answer overall, including on families where
two documents visibly disagree.

On the compatible controls the verifier reported a conflict relationship on five
of nine questions. That is its internal conclusion and **not** what H2c measures;
H2c reads the reviewer's judgement of the served answer and records zero false
conflicts. The internal conclusion did not reach the answer.

## D.2 Per family

`Pair` counts the questions where the chunks carrying both sides of the focal
disputed fact were retrieved, computed with `anchor_chunks` and
`pair_is_present`.

| Family | Declared | Expected relationship | Questions | Pair | Detected | Exact |
|---|---|---|---:|---:|---:|---:|
| CONF-02 | `version_supersession` | `supersession` | 3 | 3 | 2 | 0 |
| CONF-03 | `version_supersession` | `supersession` | 3 | 3 | 1 | 1 |
| CONF-04 | `version_supersession` | `supersession` | 3 | 3 | 2 | 0 |
| CONF-06 | `mutually_exclusive` | `mutually_exclusive` | 3 | 2 | 0 | 0 |
| CONF-07 | `compatible` | `contextually_compatible` | 3 | 3 | 1 | 0 |
| CONF-08 | `mutually_exclusive` | `mutually_exclusive` | 3 | 2 | 1 | 1 |
| CONF-09 | `compatible` | `contextually_compatible` | 3 | 3 | 3 | 0 |
| CONF-10 | `version_supersession` | `supersession` | 3 | 3 | 3 | 3 |
| CONF-11 | `stricter_looser` | `stricter_looser` | 3 | 3 | 2 | 0 |
| CONF-12 | `stricter_looser` | `stricter_looser` | 3 | 2 | 1 | 0 |
| CONF-13 | `mutually_exclusive` | `mutually_exclusive` | 3 | 2 | 0 | 0 |
| CONF-14 | `stricter_looser` | `stricter_looser` | 3 | 0 | 1 | 0 |
| CONF-15 | `stricter_looser` | `stricter_looser` | 3 | 1 | 0 | 0 |
| CONF-16 | `stricter_looser` | `stricter_looser` | 3 | 2 | 2 | 1 |
| CONF-17 | `compatible` | `contextually_compatible` | 3 | 1 | 1 | 0 |

## D.3 One illustrative case

`CONF-02-Q1` asks when Statutory Sick Pay starts being paid. The corpus holds
HR-02, withdrawn, saying the fourth qualifying day, and HR-12, current, saying
the first. Both were retrieved and HR-02 carried a `[SUPERSEDED]` marker in the
evidence block.

Arm D's frozen record classifies the relationship as `mutually_exclusive`
where the declared type maps to `supersession`. In its claim audit it
marks the claim drawn from the current document `CONTRADICTED` and records the
withdrawn document's claim as `SUPPORTED`.

The served answer was nonetheless correct, because the verifier returned the
draft unchanged. The failure is confined to the internal audit and is invisible
in the answer the user receives, which is the reason it went unreported until
the demonstrator displayed the audit alongside the answer.

**This is one question.** It illustrates the pattern in the tables above; it
does not establish it, and no argument in this dissertation rests on it alone.
