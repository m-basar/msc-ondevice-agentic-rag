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

**288 verifier calls: three blocks of 96.** 8 development families x 3
paraphrases x 2 models x 2 evidence conditions, run as **three complete
blocks**.

The block count was set by the measurement in section 2a, not by preference.
The decision rules are applied **independently to each block** and the three
outcomes reported with their mean and range. Three blocks are **three results,
not 288 independent observations**: calls within a block are not independent of
each other, and pooling them would inflate every denominator in the analysis.

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

### 2a. Reproducibility: measured, and what it showed

The first attempt to settle this was misread, and both the error and the
correction are recorded because the error decided a design.

`seed: 42`, `temperature: 0.0`, one host, one Ollama build, one model store.
A check repeated each of 12 recorded prompts three times **back to back** and
found 4 of 12 changing their raw output. I reported that as the verifier being
unreproducible and expanded the protocol to 288 calls on the strength of it.

That was wrong, and the same output contained the evidence against it:

| | |
|---|---|
| call 1 matched the recorded run | **12 / 12** |
| calls 2 and 3 agreed with each other | 12 / 12 |
| output changed under immediate repetition | 4 / 12 |

Every first call reproduced the earlier session exactly. The changes appear
only when the same prompt is sent again immediately, which no run of this
protocol ever does. The script reported "matches 8/12" because it required all
three calls to equal the recording, which folded "a fresh prompt reproduces"
together with "adjacent calls are stable" and reported neither.

The accurate statement of that artefact is **adjacent-repeat raw-output
variability: 4 / 12**. It says nothing about whether a reported outcome moves,
because a reordered JSON key or a reworded rationale changes the hash and
changes no result.

**The corrected measurement.** 41 prompts, three complete passes in protocol
order, replaying the model and options recorded in the run:

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
Within a session, block-level variability appears in the relationship label,
the structural validation and the served answer.

The accurate statement, and the one to use in the write-up:

> Binary conflict detection was reproducible, while relationship labels,
> structural validation and served-answer outcomes showed block-level
> variability.

Not "the verifier is unreproducible". The confirmatory metric did not move on a
single prompt of 41; several secondary outcomes did, and they feed reported
figures, which is why the pre-declared rule fires.

Pilots 02 and 03 also differed on 24 of 41 raw outputs with byte-identical
prompts. Pilot 02 predates option recording, so an options difference cannot be
excluded from the record, and Arm D ran after A, B and C in pilot 02 and alone
in pilot 03. Neither explanation is ruled out and neither needs to be: the
protocol now runs blocks regardless.

`scripts/check_determinism.py` now runs **complete passes in protocol order**,
replays the options and model recorded in the run rather than the current
config, and **parses every response**, comparing raw text, relationship,
detection, claim verdicts, parse and validation status, invented evidence,
whether a revision was served, the served answer and its citations, each
separately.

Its two comparisons answer different questions and are reported apart: pass 1
against the recorded run asks whether a fresh prompt reproduces across
sessions; pass against pass asks whether it reproduces within one.

**The decision this feeds, fixed in advance:**

| Finding | Response |
|---|---|
| Raw text varies, every reported outcome stable | Keep 96 calls. Document prose-level variability as a finding in its own right. |
| Any reported outcome varies | Three complete 96-call blocks. R0 to R5 applied independently per block, three outcomes reported with mean and range. |

**A trap worth naming either way.** The aggregate relationship counts were
identical across pilots 02 and 03, 36 / 4 / 1 both times, while four questions
moved: two one way, two the other, cancelling. Read from totals that is perfect
stability with a tenth of the questions having changed answer. Stability is
reported per prompt and never inferred from an aggregate.

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
| **Family-level majority** | a majority of the family's 3 paraphrases, computed **within a block**, applied to detection and classification separately |
| **Compatible-control false positives** | control families detected by majority |
| **`pair_present`** | in the `full` condition, whether both anchor chunks were actually retrieved |
| **Repeat agreement** | per prompt, whether the repeats agreed, reported separately for raw text, relationship, parse status, classification and detection |

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

**R00 - stability first.** If the protocol runs more than one block, R0 to R5
are applied to each block separately and the block outcomes reported with their
mean and range. Any outcome whose block agreement is below 1.00 is a
distribution and is reported as a proportion with its spread, never as a count.
Blocks are never pooled into one denominator.

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
