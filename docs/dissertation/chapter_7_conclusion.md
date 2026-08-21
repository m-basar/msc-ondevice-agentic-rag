# Chapter 7: Conclusion

## 7.1 The problem and what was done about it

Small firms hold operational knowledge in documents they cannot reliably use,
and the cloud assistants that would make that knowledge accessible are the ones
they have the strongest reasons to refuse. This project asked whether a
privacy-preserving assistant running entirely on hardware an SME can afford can
handle the specific failure that makes such document estates dangerous:
contradictory evidence, where two documents in the corpus say different things
and one of them is out of date.

The response was a Design Science Research artefact evaluated under a
pre-registered analysis plan (Hevner et al., 2004; Peffers et al., 2007;
Nosek et al., 2018). A four-arm comparison over one frozen synthetic corpus
isolated the variable of interest: Arm A is naive retrieval-augmented
generation, Arm B adds document status metadata to the evidence, Arm C filters
withdrawn documents out before ranking, and Arm D adds a claim-level
verification pass over the evidence Arm B sees, replaying Arm B's draft so that
the difference between them is the verification pass alone. Sixty-eight held-out
questions in 32 groups were scored manually under blinding against rules fixed
before any arm was run, and latency was measured separately on a laptop and on a
Raspberry Pi 5.

## 7.2 What the study found

**RQ1, grounded answering.** The assistant answers SME policy questions with
citations that resolve, and with support that lags well behind. Citation
validity is 0.7989 to 0.8333 at group level and rises to 0.9524 to 1.0000 once
the arms share a denominator, while citation support is 0.4561 to 0.6766. The
gap held in every arm and is the study's one supported hypothesis, H3. A
citation that resolves shows the system did not invent a source; it does not
show the source says what the answer says.

**RQ2, the value of verification.** The verification layer did not improve on
document metadata. On the pre-registered confirmatory contrast, Arm D against
Arm B on the eight live-disagreement families, D was better on none: six ties
and two families worse. On the supersession families a three-line metadata
filter reduced citations of withdrawn documents to zero, where the verified arm
reached one of twelve. The mechanism is two-part: verification is downstream of
retrieval and inherited its ceiling, with both sides of the disputed fact
reaching the verifier on only fourteen of twenty-four live-disagreement
questions, and on those fourteen the verifier detected four conflicts and
classified none correctly.

**RQ3, abstention.** The verified arm abstained on all five deliberate gap
topics, tying a baseline already at that ceiling, so H4 could not be supported
as stated. The behaviour behind the tie matters more than the tie: all thirteen
of the verifier's structured abstentions were served on questions where it had
reported no conflict, ten of them on questions the unverified arm had answered,
and three of those on plain factual questions the corpus answers directly.
Abstention became more frequent without becoming more discriminating.

**RQ4, edge feasibility.** Verification cost 3.18 times the baseline latency on
the Raspberry Pi 5, 174.17 seconds per answer against 54.84, of which 119.33 was
the verification pass, against a predicted 1.5 to 2.5 times. The ratio held
within a narrow band across three platforms whose decode rates differ more than
twentyfold, and the mechanism is token count rather than device speed: both
stages decode at about 3.2 tokens per second, and the verifier handles roughly
twice the tokens. The board ran throttled at 87 to 88 degrees for the whole
measurement, which is the realistic condition for sustained use rather than a
defect of it.

**The main research question.** A privacy-preserving on-device assistant can
handle contradictory SME documentation well enough to be useful, and the
component that achieves it is document status metadata rather than the agentic
verification layer this project set out to evaluate. The criterion stated in
section 1.4 was that verification would count as effective if it measurably
improved the handling of contradictory evidence beyond what metadata alone
achieves, at a resource cost still permitting interactive use. It failed both
halves, and the criterion was written down before the experiment so that this
sentence could be written afterwards.

## 7.3 Contributions

**A negative result that survives its cheap baseline.** The primary
contribution is empirical: claim-level verification, implemented as a second
small model inside an on-device RAG pipeline, did not improve conflict handling
over a document metadata filter, and cost 3.18 times the latency on the target
device. The comparison against Arm C is what makes this a finding rather than a
disappointment. Against naive retrieval alone the verified arm would have looked
like an improvement.

**A mechanism, not just an outcome.** The study locates why. Verification
inherits retrieval's recall as a ceiling; a prompted 3B verifier can detect that
evidence is troubled and cannot classify how; and a verifier operating as a
runtime gate rather than an offline metric converts its own errors into withheld
answers. Sixteen of sixty-eight drafts were revised, thirteen of them into
abstentions on questions where no conflict had been detected. One behaviour
accounts for three separate null results.

**A demonstration of feasibility with a measured price.** The artefact runs end
to end on a 16GB Raspberry Pi 5 with no network dependency: retrieval,
generation, claim-level verification and a browser interface, at 174 seconds per
answer under sustained thermal throttling. That is a working answer to whether
such a pipeline fits on the hardware, and a quantified answer to what the
verification layer costs there.

**A reusable evaluation design.** The synthetic knowledge base with typed
planted conflicts, the held-out question set with its per-question rubrics, the
four-arm tree rooted at B, and the pre-registered analysis plan with its
decision rules are reusable independently of the artefact. The four-arm tree in
particular is the design decision that made the result interpretable: without an
arm isolating the cheap alternative there is no way to tell an improvement from
a purchase.

**A methodological observation.** Across 37 amendments the same defect recurred
twelve times: a rule stated in the pre-registration and not enforced in the
code, then cited as though it were. The countermeasure that worked was to treat
every stated property as a test to be written, and to distrust any test that
could not be made to fail. That is offered as transferable practice for
pre-registered software-artefact studies.

The claim made in section 1.4 that a categorical confidence mechanism is
implemented but not evaluated stands unchanged. No claim is made for its
calibration anywhere in this dissertation.

## 7.4 Recommendations for practice

For an SME considering a private document assistant:

1. **Curate document status before buying anything.** The measured benefit
   depends entirely on documents carrying an accurate current-or-withdrawn
   field. That is an organisational task, and it is the highest-return step
   available.
2. **Filter on status at retrieval time.** The cheapest mechanism tested was
   also the most effective one on the conflict type that dominates real
   document estates.
3. **Route live disagreements to a person.** No arm handled two current
   documents that contradict each other; every arm scored zero on all three
   families of that type. Detect the situation and escalate it rather than
   asking the system to decide.
4. **Deploy asynchronously.** Submit-and-collect or overnight batch, not an
   interactive help desk, at the latencies a single-board device delivers.
5. **Treat citations as pointers, not warrants.** Around a third of checkable
   citations did not carry the claim attached to them.
6. **Do not display a confidence label you have not evaluated.**

## 7.5 Further work

**Measure the confidence mechanism.** The categorical confidence level is
implemented and unevaluated, as declared in section 1.4. Measuring whether a
low-confidence label corresponds to a materially higher error rate is the most
direct extension of this work, and the established apparatus applies: expected
calibration error and reliability diagrams (Guo et al., 2017), against the
finding that verbalised confidence from instruction-tuned models tends towards
overconfidence (Xiong et al., 2024).

**Fix the retrieval ceiling before testing verification again.** The single
most consequential confound in this study is that on ten of twenty-four
live-disagreement questions the verifier never saw both sides. A retrieval stage
that explicitly seeks contradicting evidence, rather than the six most similar
chunks, would test verification rather than testing verification and retrieval
together. This is a prerequisite for any replication, not an enhancement.

**Test whether classification is a capability or a scale problem.** The verifier
detected conflicts and could not name them. Whether a 7B or 14B verifier crosses
that threshold, and at what latency on the same board, is directly measurable
with the instrument already built. The four-arm design and the frozen question
set can be reused unchanged.

**Fine-tuning, as a separate study.** Self-RAG's reflection tokens are learned
rather than prompted (Asai et al., 2024). Whether a small verifier fine-tuned on
typed conflict examples can classify relationships a prompted one cannot is a
worthwhile question and a different project, with its own data collection,
training budget and evaluation. It is named here because this study's negative
result is evidence about prompted verification specifically and should not be
read as evidence about the fine-tuned kind.

**Independent scoring, and users.** A second blinded scorer would remove the
study's clearest measurement limitation. Beyond that, no measurement here
touches whether people trust, verify or act on these answers, or whether an
over-refusing assistant is abandoned faster than an over-answering one. That
question needs an SME, a real corpus and its staff.

**A real corpus.** The planted conflicts were typed and counted so that the
phenomenon could be measured. What proportion of a real firm's document estate
carries each conflict type, and whether status metadata exists to filter on, is
unknown and determines whether the recommendation in section 7.4 is cheap or
impossible.

## 7.6 Concluding remarks

The project set out to test whether an agentic verification layer could make a
small on-device assistant trustworthy enough to answer questions from
contradictory company documents. It cannot, at this model size, at this cost,
on this evidence. What can is a filter on a metadata field that most document
management systems already have and most small firms do not maintain.

That is a smaller answer than the one the project hoped for and a more useful
one than a positive result would have been at the same level of rigour, because
it was fixed in advance what would count as success, the cheap alternative was
measured alongside the expensive one, and the rules were not moved when the
answer arrived. The contribution is not that verification failed. It is that the
study was built so that its failure could be seen, located and priced.
