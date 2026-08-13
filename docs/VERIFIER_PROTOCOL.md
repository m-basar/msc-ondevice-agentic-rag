# Verifier diagnostic protocol

**Written and committed before the protocol was run.** Commit the file, then
run `scripts/verifier_protocol.py`. If this document and the results appear in
the same commit, the pre-commitment is worthless and should not be believed.

Development split only. Nothing here is reported as a result. Nothing here
touches the test split, and the oracle condition defined below is a diagnostic
instrument that must never become Arm D's deployed retrieval.

---

## 1. What this is for

Development pilot 02 detected **zero conflicts across eighteen genuine-conflict
questions**. On CONF-01-Q1 the verifier received both HR-13#001 and HR-03#001,
with the SUPERSEDED marker present, and answered that the evidence does not
mention mileage rates for business travel.

That is not a wording problem. Three explanations fit, and they call for
entirely different responses:

| | Explanation | If true, the response is |
|---|---|---|
| **Model** | `llama3.2:3b` cannot do structured cross-passage reasoning | Amend Arm D to a different verifier model, and report the substitution and its cost |
| **Dilution** | six chunks and roughly 3,000 characters bury the two that matter | A retrieval finding, not a verification finding: the layer can reason but retrieval does not isolate the pair |
| **Task** | the task is beyond a 3B model on this hardware however it is framed | A null result, reported honestly, which is the primary contribution if it holds |

The protocol separates these. It cannot separate them if detection and
classification are collapsed into one number, so they are scored separately
throughout.

## 2. Design

**96 verifier calls.** 8 development families x 3 paraphrases x 2 models x 2
evidence conditions.

### Families

All eight development families, every paraphrase. No selection.

| Family | Declared type | Role |
|---|---|---|
| CONF-01 | `version_supersession` | genuine |
| CONF-05 | `stricter_looser` | genuine |
| TUNE-01 | `stricter_looser` | genuine |
| TUNE-02 | `stricter_looser` | genuine |
| TUNE-05 | `mutually_exclusive` | genuine |
| TUNE-06 | `stricter_looser` | genuine |
| **TUNE-03** | `compatible` | **negative control** |
| **TUNE-04** | `compatible` | **negative control** |

All three paraphrases are used because the stopping rule in
`src/sme_assistant/evaluation/stopping_gate.py` requires a majority of three,
and a one-paraphrase diagnostic cannot be read against it. It also removes the
risk that one unrepresentative wording decides the outcome.

The two compatible controls are included in **every** condition. Detection
figures on the genuine families are uninterpretable without them: a verifier
that flags everything scores perfectly on conflicts.

### Models

`llama3.2:3b` (the current verifier) and `qwen2.5:3b`.

`phi3:mini` is deliberately excluded. The pilot diagnostic already showed it
weaker than Qwen, and it is not part of the decision being made, which is
whether to retain Llama or amend Arm D to Qwen. Adding a third model would add
48 calls and turn model selection into shopping for whichever performs best.

### Evidence conditions

| Condition | Evidence given to the verifier |
|---|---|
| `full` | the deployed retrieval: `top_k: 6`, `min_similarity: 0.30` |
| `oracle_pair` | **only** the chunks anchoring both sides of the disputed fact the question asks about |

`oracle_pair` uses `anchor_chunks()` from `scripts/evaluate_retrieval.py`, the
same function that computes conflict-pair recall, so the two measurements refer
to the same set of chunks.

**The oracle condition is an instrument, not a candidate design.** Selecting it
requires already knowing which fact is disputed and which passages carry it,
which is the question the system is supposed to answer. It exists solely to ask
whether the verifier can do the reasoning **when the reasoning is all that is
left to do**. It is development-diagnostic only: never at runtime, never on the
test split, and it is not a retrieval configuration that could be adopted.

The draft answer is generated once per (question, condition) from the same
evidence, and both models audit the same draft. Regenerating per model would
let generator sampling differences appear as verifier differences.

## 3. Recorded outputs

Pre-committed, recorded separately, never combined into a single score.

| Output | Definition |
|---|---|
| **Binary conflict detection** | `relationship` in {`supersession`, `mutually_exclusive`, `stricter_looser`}. `contextually_compatible` is **not** a detection: it is the verifier declining to flag |
| **Exact relationship classification** | inferred `relationship` equals the registry's declared type under the fixed mapping in `stopping_gate.DECLARED_TO_INFERRED` |
| **Parse success** | the response was valid JSON conforming to the schema and survived fail-closed validation |
| **Evidence-ID validity** | every chunk identifier the verifier cites was present in the evidence it was given |
| **Family-level majority** | at least 2 of the 3 paraphrases, applied to detection and to classification separately |
| **Compatible-control false positives** | control families detected by majority |
| **`pair_present`** | in the `full` condition, whether both anchor chunks were actually retrieved |

`pair_present` is the confound check. If the disputed pair is absent from the
full retrieval, the verifier was never given the chance to detect anything, and
a `full` versus `oracle_pair` difference measures retrieval rather than
reasoning. Any reading of the two conditions that ignores it is invalid.

Behavioural correctness - whether the answer names the safe course, surfaces
both positions, or escalates - is **not** scored here. It is manual, blinded,
and against the rubric in the question set. No automatic proxy for it appears
in this protocol.

## 4. Decision rules

Written before the run. Evaluated in order; the first that matches decides.
Read only against families where `pair_present` holds in the `full` condition.

**R0 - controls first.** If a model detects both control families by majority
in a condition, its detection figures in that condition are uninterpretable and
are reported as such. Its genuine-family detections are not credited.

**R1 - task.** Neither model detects a majority of genuine families in either
condition.
→ **The null result.** Freeze the verifier as it stands and report it. No
prompt revision 3. The finding is that a 3B-class model on this hardware does
not perform structured cross-passage conflict detection reliably, which is a
substantive and publishable answer to the research question.

**R2 - dilution.** Detection is at least 3 families higher under `oracle_pair`
than under `full`, for the same model, on families where `pair_present` holds.
→ A **retrieval** finding, not a verification one. Report that the layer can
reason over an isolated pair and that the deployed retrieval does not isolate
it. Arm D is not redesigned around the oracle, because the oracle is not
available at runtime.

**R3 - model.** Qwen detects a majority of genuine families in a condition
where Llama does not, and R0 does not apply to Qwen.
→ Amend Arm D: Llama-generated answer audited by Qwen. Recorded as a
pre-registration amendment. Pi latency must then include model-loading cost,
because two 3B models will not stay resident together.

**R4 - prompt.** A model detects a majority of genuine families but classifies
fewer than half correctly.
→ Detection works and naming does not, which is prompt-shaped rather than a
capability ceiling. **Prompt revision 3 is permitted, and it is the last one.**
If revision 3 does not clear R4, R1 applies and the null result is reported.

**R5 - anything else.** Report the observed pattern without a further
intervention. The revision budget is not extended by finding a rule that was
not anticipated.

### The budget, stated plainly

One prompt revision remains, available only under R4. Two have already been
spent. A third revision conditional on nothing would let the prompt be tuned
until the null result disappeared, and a result obtained that way would be a
description of the development split rather than a finding.

## 5. What a null result is worth here

If R1 fires, the dissertation reports that a privacy-preserving on-device
assistant using 3B-class models can ground answers and cite passages, and does
**not** reliably detect conflicts between them, with the boundary located
precisely: which subtypes, under which evidence conditions, at what cost.

That is a genuine contribution to the research question. The alternative - a
prompt revised until the development split cooperates - would be worth less and
would not survive the test split.

## 6. Provenance

Run `python scripts/verifier_protocol.py` after this file is committed.

Committing this document is **not** sufficient, and an earlier version of the
guard that checked only this file was theatre. The 96 calls are produced by the
verifier prompt, the parsing rules, the retrieval settings and the gold data,
none of which live in `docs/`. An uncommitted change to any of them would
influence every result while the protocol sat frozen in git looking
authoritative.

The script therefore refuses unless all of these are committed:

| Path | Why it decides the outcome |
|---|---|
| `src/` | the verifier prompt, the parsing and validation rules, retrieval |
| `gold/` | the families, the questions, the declared types classification is scored against |
| `config.json` | models, `top_k`, `min_similarity`, verification options |
| `scripts/verifier_protocol.py` | this harness |
| `scripts/evaluate_retrieval.py` | `anchor_chunks`, which builds the oracle condition |
| `docs/VERIFIER_PROTOCOL.md` | the design and the decision rules |

`results/` is not checked, because it is where this run writes. Refusing on it
would only produce a habit of reaching for a bypass flag.

The output records the commit, the prompt hash, and the config, corpus,
chunk-set and registry fingerprints, alongside every raw response.

## 7. Precondition

This protocol runs only after pilot 03 reports **zero invalid served
revisions** in the stopping gate. Pilot 03 is a controlled re-run: Arm B is
replayed from its pilot 02 drafts rather than regenerated, so the only thing
that changes is the revision rule under test.

Diagnosing a verifier while a known defect sits in the path that serves its
output would attribute the defect's effects to the model.

The stopping gate is separate and applies to arm runs, not to this protocol.
This protocol decides what Arm D is; the gate decides whether Arm D's
development run is worth carrying to the test split.
