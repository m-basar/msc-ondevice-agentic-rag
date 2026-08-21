# Chapter 5: Analysis

## 5.1 Introduction

Chapter 4 reported what the evaluation measured. This chapter asks why those
results occurred, and answers the four research questions of section 1.3 in
turn. It is organised by research question rather than by hypothesis, because a
research question is answered by several measurements at once and because the
most informative pattern in this study cuts across three hypotheses.

Nothing here recomputes a reported figure. Every number is read from
`results/analysis/hypotheses.json`, from the Raspberry Pi performance report, or
from the frozen Arm D run, and each is cited to the section of Chapter 4 that
reports it. Section 5.6 introduces one post-hoc reading of fields already
recorded in the frozen run. It is labelled as such, carries no threshold and no
verdict, and revisits no hypothesis. Comparison with the literature, the
implications for an SME considering deployment, and the study's limitations are
held for Chapter 6.

The short answer to the main research question is that the pipeline works, the
cheap intervention works, and the expensive one does not pay for itself on this
evidence. The rest of this chapter establishes why.

## 5.2 RQ1: can the assistant answer with verifiable citations?

**Yes, with a qualification that H3 makes precise: the citations are real, and
they support the claim less often than their being real would suggest.**

Citation validity sits between 0.7989 and 0.8333 at group level in all four arms
(section 4.8, Table 4.6). Validity is answer-level and all-or-nothing: an answer
counts only when every identifier it cites is real and was retrieved. Roughly
four answers in five therefore cite nothing invented and nothing absent from the
evidence block. Under the common-eligibility sensitivity analysis, which
restricts every arm to answers where support is defined so that the two metrics
share a denominator, validity rises to between 0.9524 and 1.0000. On the
questions where citation quality can be assessed at all, identifier fabrication
is close to absent.

Support is the weaker property, at 0.4561 to 0.6766 at group level. The gap
between the two is the finding H3 predicted, and it held in every arm under both
conventions, which makes H3 the one supported hypothesis in the study. The
reason it holds is structural rather than incidental. Retrieval selects chunks
by embedding similarity to the question; generation cites the chunk it drew on;
the support measure asks whether that passage contains the claim's quantities.
Similarity is not entailment, and a passage can be the most similar chunk in the
corpus while carrying a related figure rather than the one asserted. The measure
is a lower bound on citation error and therefore an upper bound on true support
(section 4.8), so the real gap is at least as wide as the table shows.

This matters more than a null result would, because the standard defence of
retrieval-augmented generation is that a cited answer is a checkable answer. It
is checkable only if someone checks. **An SME reading these answers would see
identifiers that resolve, in a system where a third of the checkable ones do not
carry the claim they are attached to.** A citation that resolves is evidence of
provenance, not of correctness, and this study measured how far apart those two
properties sit in practice.

The second half of RQ1 is whether the assistant answers correctly. On the 13
questions in 12 groups that contain no conflict and ask for a direct answer,
answer correctness at group level is 1.6667 for A, 1.5833 for B, 1.5417 for C
and 1.0833 for D on the three-point rubric (section 4.11, Table 4.9). The three
retrieval arms are close, and the gap between them and Arm D is large. Section
5.4 shows that the whole of Arm D's shortfall is three questions on which it
declined to answer, and section 5.6 shows what caused it to decline.

The honest limit on RQ1 is the corpus. Thirty-eight synthetic documents with
planted contradictions are a controlled instrument, not a sample of real SME
document estates, and the citation figures describe this corpus under one model
at one size. What transfers is the shape of the result, not its magnitude.

## 5.3 RQ2: does verification improve on metadata alone?

**No, on this evidence, and the reason is more interesting than the verdict.**

The confirmatory contrast is Arm D against Arm B, which is the single-variable
comparison: D sees the same evidence block, replays B's draft, and differs only
by the verification pass. On the eight live-disagreement families of H2, D
exceeded B on none: six ties and two families where D was worse, a paired
difference of -0.0833 where the decision rule required +0.25 and agreement in
six families of eight (section 4.6). Leave-one-family-out resampling moves that
figure between -0.0952 and -0.0476, so it is not the product of a single family
(section 4.6). On the four supersession families of H1, D sits below B by
0.1667, within the pre-specified 0.25 margin and in B's favour (section 4.5).
The verification layer did not lose badly. It did not win at all.

Arm C is the comparison that gives the result its edge. A three-line filter on
document status drove citations of withdrawn documents from 6 of 12 questions in
Arm A to 0 of 12, and scored highest of the four arms on the supersession
families at 1.5833 (section 4.5, Tables 4.2 and 4.3). Arm B, which
merely places the status in the prompt and leaves the model to act on it, reached
2 of 12. Verification reduced B's two to 1 of 12: it removed the withdrawn
citation on `CONF-10-Q1` and left the one on `CONF-04-Q2` in place. **Of the two
withdrawn-document citations the verified arm inherited, it caught one and
missed one, at three times the latency; the filter caught both by never
retrieving them.**

That contrast is the study's central practical finding, and the explanation is
that supersession is decidable without reading anything. Whether a document has
been withdrawn is a property of the document record, not of its prose, so the
cheapest possible mechanism is sufficient and a language model adds only the
opportunity to disagree with a fact it was given. The conflicts where reading
would be necessary are the live disagreements of H2, where two current documents
say different things and no metadata field separates them. That is precisely
where the verifier failed to help.

Section 4.13 shows why. On the twelve supersession questions the verifier
detected a relationship on eight and named the correct one on four, exact on a
majority in one family of four. On the twenty-four live-disagreement questions it
detected seven and named two, exact on a majority in none of the eight families.
Restricting to the fourteen questions where the chunks carrying both sides of the
disputed fact were actually retrieved, it detected four and named none. The label
`contextually_compatible` was never returned once in forty-five questions,
including the nine compatible controls where it was the correct answer.

Two mechanisms follow from that, and they are different in kind.

**The first is a retrieval ceiling.** On the supersession families both sides of
the disputed fact were retrieved on all twelve questions; on the live-
disagreement families, on only fourteen of twenty-four. A verifier cannot
adjudicate a disagreement whose second side never reached it. Verification is
downstream of retrieval and inherits its recall as an upper bound, so on more
than a third of the questions the layer was asked to arbitrate a dispute it could
not see. The design placed the verifier after a six-chunk retrieval step and
assumed sufficiency without measuring it, which is a design error this study can
now name.

**The second is a classification failure that the ceiling does not explain.** On
the fourteen questions where both sides were present, exact classification was
zero. The verifier could frequently tell that something was wrong and could
almost never say what. A three-billion-parameter model asked to place a pair of
passages into one of five relationship types performed at a level that supports
no downstream decision, and the confusion table shows the shape of the error:
`mutually_exclusive` was the model's default guess for disagreement of any kind,
returned on four of the nine compatible controls as well.

H2c is the remaining piece and it is a floor rather than a finding. Zero false
conflicts were asserted on the controls by any arm, so the prediction that D
would over-detect is not supported, and cannot be: a denominator of three
families and nine questions with no events in any arm has no power to
distinguish anything (section 4.7). What can be said is narrow. The served
answers did not assert conflicts on the compatible families, while the
verifier's internal field reported a relationship on five of the nine. The
system's outward behaviour was more conservative than its internal state, which
is a property of the serving rule rather than of the verifier.

## 5.4 RQ3: does verification improve abstention?

**It reaches the ceiling, and it does not stop there.**

H4 was not supported for a reason that is a property of the instrument. Arm B
abstained on all ten gap questions and all five gap topics; Arm D did the same.
A hypothesis requiring D to exceed all three baselines cannot be satisfied
against a baseline at 5 of 5, and with five topics there was no headroom for it
to try (section 4.9, Table 4.7). The margin over A at 3 of 5 topics and C at 4
of 5 is real and is reported, but it cannot be separated from the ceiling. The
design fault is one of sizing rather than of concept: five deliberate gaps were
enough to show that abstention happens and too few to compare two arms that both
saturate. A larger and harder gap set, including questions the corpus answers
only partially, would have been able to separate them.

The more useful result is the one the hypothesis did not ask about. Abstention
was not scarce in this system. It was abundant, and Arm D produced the most of
it. The reviewer judged 27 of Arm D's 68 answers to decline to commit, against
17 for Arm B, 16 for Arm C and 12 for Arm A. Those counts are descriptive, carry
no threshold and are not a refusal rate: section 4.11 declines to compute a rate
over answerable questions because that denominator was not pre-registered, and
that reasoning is respected here.

The decomposition is what matters. Arm D serves Arm B's draft unchanged unless
the verifier objects, so any difference between them is an intervention. Of Arm
D's 27, thirteen were served through the structured abstention path and fourteen
were prose refusals inherited unchanged from Arm B's draft. **Verification did
not reduce the refusals the draft arrived with; it added ten of its own.** On
three of those ten the corpus contained a direct and unambiguous answer, which A,
B and C each gave correctly: `FACT-account-hold`, `FACT-forklift-preuse` and
`FACT-written-warning-duration`. Those three questions are the whole of Arm D's
deficit on answer correctness in section 5.2.

So the verification layer did improve the *form* of abstention. Where Arm B
refuses in free prose, Arm D returns a structured verdict a caller can act on,
and that is a real engineering property. What it did not do is improve the
*decision* to abstain, which is what H4 was about and what an SME would care
about. A system that declines correctly on all five gaps and also declines on
three questions its own corpus answers has not become more discriminating. It
has become more cautious, and caution without discrimination is indistinguishable
from a lower answer rate.

## 5.5 RQ4: what does it cost on the Pi, and is that tolerable?

On the Raspberry Pi 5, mean end-to-end latency per answer rose from 54.84
seconds in Arm B to 174.17 in Arm D, of which 119.33 was the verification pass:
a ratio of 3.18 against a predicted 1.5 to 2.5 (section 4.10, Table 4.8). H5 was
not supported, and it was wrong in the expensive direction.

The mechanism is token count rather than device speed, and the run records
separate the two. Both stages decode at essentially the same rate on the same
hardware: 3.281 tokens per second for the draft model and 3.235 for the verifier.
At a fixed decode rate, latency is a linear function of tokens produced, so the
2.18-fold cost of verification relative to drafting is a statement about how much
text each stage handles. Section 4.10 gives the counts: the verifier prompt
averaged 1,934 tokens against the draft's 900, and the verifier generated 179
output tokens against the draft's 46. A claim audit that must restate each claim,
its verdict and its supporting passage is simply a larger generation task than
the answer it audits.

The same ratio appears on all three platforms: 3.68 on the laptop GPU, 3.15 on
the laptop CPU, 3.18 on the Pi, across decode rates differing more than
twenty-fold. That is consistent with an overhead which is structural rather than
a property of the target device, although three conditions cannot establish it
and no attempt is made here to claim more.

Thermal behaviour was recorded rather than assumed, and it changes how the
absolute numbers should be read. The Pi reported itself throttled on 64 of 68
draft questions and on all 68 verifier questions, at mean CPU temperatures of
87.1 and 88.2 degrees and a maximum of 90.3. **These are the latencies of a
throttled machine under sustained load, which is the realistic condition for the
intended use rather than a defect of the measurement.** A passively cooled Pi 5
answering questions continuously will be hot within minutes and will stay hot.
Both arms ran under throttling, the verifier stage slightly hotter and more
consistently so, and the unthrottled ratio was not measured, so it cannot be
assumed that throttling affected the two stages equally. Memory was never the
constraint: 15.83 GB total with 6.01 GB free at the tightest point. The binding
resource is single-threaded decode on four Arm cores.

**Is it tolerable?** Section 4.10 defers that judgement to this chapter, and the
answer depends entirely on the interaction model.

For an interactive question and answer service, no. Three minutes is far past
the point where a user waits rather than leaves, and the complete 68-question
Arm D run took 2 hours 15 minutes against 1 hour 2 minutes for Arm B. For
asynchronous use it is a different question: a policy query submitted and
answered within five minutes, or a nightly batch over a day's queue, is entirely
feasible on a 16 GB Pi 5 drawing a few watts, with no document leaving the
premises. That is a real deployment position and Chapter 6 develops it.

But the tolerance question cannot be settled on latency alone, because
tolerability is a ratio and this study measured both sides of it. What the
additional 119 seconds per answer bought, over 68 questions, was one corrected
withdrawn-document citation, one revision that made an answer worse, and ten
refusals of which seven cost a correct or partially correct answer. **On this
corpus, at this model size, the verification layer is not expensive relative to
its benefit; it is expensive relative to a benefit that is close to zero and
partly negative.** That is a stronger and more useful conclusion than a latency
figure on its own.

## 5.6 What the verifier actually did: a post-hoc reading

The four answers above all turn on the same underlying behaviour, which the
frozen Arm D run records directly. This section reads those recorded fields and
sets them against the rubric scores. **It is post-hoc and descriptive. No
threshold is applied, no verdict is drawn, and no hypothesis is revisited.** It
is included because it explains three separate null results with one mechanism,
and because the fields it reads were written by the system during the frozen run
rather than derived afterwards.

Arm D serves Arm B's draft unchanged unless the verifier revises it, so the
records partition the 68 questions exactly.

**Table 5.1** What the verification pass did on each of the 68 test questions,
from the `verification` block of run `20260814_055018_D_test` and the manual
rubric scores of both arms. Counts are of questions, and the rubric change is
Arm D's score minus Arm B's on the same question.

| What the verifier did | Questions | Rubric score unchanged | Higher | Lower |
|---|---:|---:|---:|---:|
| Served Arm B's draft unchanged | 52 | 52 | 0 | 0 |
| Revised after detecting a conflict | 3 | 1 | 1 | 1 |
| Revised to a structured abstention | 13 | 6 | 0 | 7 |
| **Total** | **68** | **59** | **1** | **8** |

Three readings follow.

**The layer was mostly inert.** On 52 of 68 questions the verifier found nothing
to change, and the answer an SME would receive from Arm D is byte-for-byte the
answer Arm B produced, after a pass costing 119 seconds. Inertness is not a
fault in itself, since a verifier with no complaint should say nothing, and the
serving rule of amendment 1.6 exists to enforce exactly that. It does mean the
cost is paid on every question and the benefit is available on very few.

**Where it acted on a detected conflict, it acted rarely and inconsistently.**
The verifier reported `conflict_detected` on 22 of the 68 questions but revised
the answer on only three of them, all classified as supersession. Those three
produced the study's single improvement, `CONF-10-Q1`, where Arm B cited a
withdrawn document and scored 0 and Arm D did not and scored 2; one degradation,
`CONF-10-Q2`, revised from 2 to 1; and one revision that left the score at 1.
This is the layer doing the job it was designed for, on three questions, with a
net of one.

**Where it acted without detecting a conflict, it withheld the answer.** All
thirteen structured abstentions were served on questions where the verifier had
reported no conflict and returned the relationship `no_relationship`. The
abstention path was not a conflict response at all: it fired on the judgement
that the evidence was insufficient, and on ten of those thirteen Arm B had
committed to an answer. Seven lost rubric points as a result, including the three
plain factual questions of section 5.4. The claim audit completed on all 68
answers, so this is not a parsing failure or a degenerate output; the verifier
read the evidence, produced a complete audit, and concluded it could not answer
questions that a document in front of it answered.

That single mechanism accounts for Arm D being weakest on answer correctness
(RQ1), for it failing to beat Arm B on the confirmatory contrast (RQ2), and for
its abstention behaviour ceasing to be discriminating (RQ3). It is one behaviour
with three measured consequences, which is why the four research questions do
not have four independent answers.

## 5.7 Threats to this analysis

The analysis above is only as good as the measurements underneath it, and four
limits bear on it directly. Chapter 6 treats the study's limitations in full.

**One reviewer, and blinding that was partially defeated.** All rubric scores
come from a single reviewer under the arm-blinding of amendment 1.13. That
blinding failed on the thirteen items carrying the abstention template verbatim,
which structurally identified one arm. Those thirteen items are precisely the
structured abstentions that section 5.6 analyses, so the items carrying most of
this chapter's explanatory weight are the items whose arm the reviewer could
identify. Duplicate-group agreement was 58 of 58 on the rubric score and Cohen's
kappa was 0.8198 on the abstention flag (section 4.3, Table 4.1), which
establishes consistency but not independence: a reviewer can be consistent with
themselves and wrong in the same direction twice.

**Family counts are small and the sensitivity analysis says so.** H1 rests on
four families, and leave-one-family-out moves the B against A difference from
-0.1111 to +0.4444, a range of 0.5556 around a threshold of 0.25 (section 4.5).
Any statement about H1's margin is a statement about four families. H2's eight
families are more stable, moving only 0.0476 across folds, which is why the
conclusion in section 5.3 is drawn more firmly there.

**One model pair, chosen by a documented protocol and not by a sweep.**
Qwen2.5 3B was selected over Llama 3.2 3B on a conflict-detection diagnostic
under amendment 1.10, and the whole of RQ2's negative result is a result about
that model at that size performing that task. A larger verifier, or a
task-specific fine-tune, might classify relationships the way this one could not,
and nothing here establishes otherwise.

**Section 5.6 is post-hoc.** The intervention table reads fields recorded during
the frozen run, but the decision to read them was taken after the verdicts were
known. It is offered as explanation, not as evidence, and it changes no verdict
in Chapter 4.

## 5.8 Summary

RQ1 is answered positively with a measured qualification: the assistant produces
answers whose citations resolve, and whose citations support the claim
substantially less often, a gap that held in every arm and is the study's one
supported hypothesis. RQ2 is answered negatively: verification did not improve
on document metadata, because supersession is decidable without reading and live
disagreement requires both sides to be retrieved and correctly classified, and
the pipeline achieved neither reliably. RQ3 is answered negatively for a reason
partly of instrument design and partly of behaviour: the baseline was already at
ceiling, and the verified arm's additional abstentions were not discriminating.
RQ4 is answered with a figure and a judgement: 3.18 times the latency, 174
seconds per answer on a throttled Pi 5, tolerable for asynchronous use and not
for interactive use, and not repaid by what the layer delivered on this corpus.

The main research question asked whether a privacy-preserving on-device agentic
assistant can handle contradictory SME documentation well enough to be useful.
It can, and the component that achieves it is document status metadata rather
than the agentic verification layer this project set out to evaluate. Chapter 6
sets that against the literature and draws out what it means for an SME.
