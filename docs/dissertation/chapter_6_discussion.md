# Chapter 6: Discussion

## 6.1 Introduction

Chapter 5 answered the four research questions from the evidence. This chapter
asks what those answers mean. Section 6.2 sets the findings against the
literature reviewed in Chapter 2, including the three regularities that chapter
identified and the gap it declared. Section 6.3 draws out what an SME
considering a private assistant should take from the results, and what it should
not. Section 6.4 states the study's limitations and threats to validity.
Section 6.5 reflects on the method itself, which is the part of this project
most likely to outlast its findings.

No new measurement is introduced here and no figure is recomputed. Where a
number appears it is cited to the section of Chapter 4 that reports it or the
section of Chapter 5 that analyses it.

## 6.2 The findings against the literature

### 6.2.1 Reliability was bought with computation, and none was delivered

Section 2.5 concluded that across the verification and uncertainty literature
"reliability is consistently bought with computation": extra samples, extra
model calls, or a large judge model. That reading led directly to this project's
design choice. Sampling-based detection needs five to twenty generations per
answer (Manakul, Liusie and Gales, 2023; Farquhar et al., 2024), which on an
edge CPU generating a few tokens per second converts a thirty-second answer into
several minutes. A single verification pass over already-retrieved evidence is
the cheapest member of that family, and it is the one this project implemented.

The result is that the cheapest available option still cost 3.18 times the
baseline latency, 174 seconds per answer on the Raspberry Pi 5 (section 4.10),
and returned one corrected withdrawn-document citation across 68 questions
against seven questions where a correct or partially correct answer became a
refusal (section 5.6). The regularity in section 2.5 holds in the direction it
predicts and fails in the direction it assumes: computation was spent, and
reliability did not follow. That is not a refutation of the literature, which
reports these methods working at cloud scale with far larger models. It is
evidence that the exchange rate between computation and reliability is not
constant across model scale, and that a mechanism can be affordable in principle
and useless in practice at the size where it is affordable.

### 6.2.2 Negative rejection, inverted

Chen et al. (2024) find that RAG systems fail at negative rejection: they answer
confidently when the retrieved evidence does not contain the answer. Selective
prediction states the goal precisely, that a system should answer when it is
likely to be right and abstain otherwise (Kamath, Jia and Liang, 2020).

The verified arm in this study failed the same objective from the opposite side.
It abstained on all five deliberate gap topics, which is the behaviour both
literatures ask for, and it also abstained on three plain factual questions the
corpus answers directly and that the three unverified arms each answered
correctly (sections 4.11 and 5.4). All thirteen of its structured abstentions
were served on questions where the verifier had reported no conflict, so the
refusals were not a conflict response but an insufficiency judgement, and on ten
of them Arm B had committed to an answer (section 5.6).

**This is the more useful contribution of the two.** Over-answering is the
documented failure mode and the one the design guarded against; what the
experiment found is that adding a small verifier to guard against it converts
the failure into over-refusal, which is less visible and, in an operational
setting, not obviously less harmful. An assistant that answers wrongly can be
corrected by the person reading it. An assistant that declines to answer
questions its own corpus answers will simply stop being used.

### 6.2.3 Why Self-RAG fine-tunes and corrective RAG reaches for the web

Section 2.4 identified the two nearest systems and the assumptions that put them
out of reach here. Self-RAG fine-tunes the deployed model to emit reflection
tokens (Asai et al., 2024); corrective RAG interposes a retrieval evaluator and
falls back to web search when the evidence is judged unreliable (Yan et al.,
2024). This project deliberately did neither, using an off-the-shelf quantised
3B model prompted to verify (Yang et al., 2024).

The diagnostic in section 4.13 suggests why those two design choices exist. The
verifier detected that something was wrong reasonably often and could almost
never say what: on the fourteen live-disagreement questions where the chunks
carrying both sides of the disputed fact were actually retrieved, it detected
four and classified none correctly, and the label `contextually_compatible` was
never returned once in forty-five questions. Detection without classification
supports no downstream decision, which is exactly the capability fine-tuning
supplies in Self-RAG and exactly the judgement corrective RAG declines to make
locally by fetching more evidence instead.

The finding therefore reads as empirical support for a design assumption those
papers make implicitly. A prompted small model can be asked whether the evidence
is troubled; it cannot usefully be asked what kind of trouble it is in.

### 6.2.4 Verification as a runtime safeguard, not as evaluation tooling

The third regularity in section 2.5 was that verification appears mainly as
offline evaluation tooling rather than as a runtime safeguard inside a deployed
pipeline. Frameworks such as RAGAS operationalise faithfulness as an automated
metric computed after the fact (Es et al., 2024), and FActScore decomposes
generated text for scoring rather than for serving (Min et al., 2023).

This project ran the check inside the pipeline, on the device, before the answer
was served, and the serving rule made the verifier's output consequential: a
verifier finding nothing is a no-op, and every served revision names its warrant.
That distinction turned out to matter more than expected. As an evaluation
metric, a verifier that flags 22 of 68 answers is informative. As a runtime
gate, the same verifier changed the served answer on 16 questions and made it
worse on eight of the nine where the score moved (section 5.6). **An evaluation
metric that is wrong costs a misleading number; a runtime gate that is wrong
costs the user an answer.** The literature's preference for the offline use is,
on this evidence, a reasonable one at small model scale, and moving verification
into the serving path raises the accuracy bar it must clear.

### 6.2.5 Citations that resolve are not citations that support

Retrieval demonstrably reduces hallucination in knowledge-grounded generation
(Shuster et al., 2021), and provenance is the property that makes a RAG answer
auditable. H3 is the study's one supported hypothesis and it qualifies that
promise: citation validity sits between 0.7989 and 0.8333 at group level, while
support sits between 0.4561 and 0.6766, and the gap held in every arm under both
conventions (section 4.8). Under common eligibility, validity reaches 0.9524 to
1.0000: identifier fabrication is close to absent, and the weakness is entirely
in whether the cited passage carries the claim.

The gap is structural rather than incidental, and section 5.2 gives the
mechanism: retrieval selects by embedding similarity (Karpukhin et al., 2020;
Reimers and Gurevych, 2019), the generator cites what it drew on, and similarity
is not entailment. The finding is worth stating plainly because it cuts against
how RAG provenance is usually presented. A resolvable citation is evidence that
the system did not invent a source. It is not evidence that the source says what
the sentence in front of it says, and in this study roughly a third of the
checkable citations did not.

### 6.2.6 What the declared gap now contains

Section 2.6 declared the gap deliberately falsifiably: no identified study
integrates claim-level verification into a fully on-device agentic RAG pipeline
and measures both the reliability benefit and the resource cost on edge
hardware, in an SME knowledge management setting, isolating the benefit against
the cheap alternative of a document metadata filter.

That gap is now occupied by a negative result, and the metadata comparison is
the reason the result means anything. Had the study compared Arm D only against
naive retrieval, the verified arm would have looked like an improvement on
withdrawn-document citations, from 6 of 12 questions to 1 of 12 (section 4.5).
The three-line filter of Arm C reached 0 of 12 without a model call. **A
contribution that survives the cheap baseline is the only kind worth claiming,
and this one did not survive it.** Reporting that is the study's principal
finding rather than its disappointment.

## 6.3 What this means for an SME

**The deployable configuration is the metadata filter, not the verification
layer.** For the conflict type that dominates real document estates, a policy
superseded by a newer policy, Arm C reduced citations of withdrawn documents to
zero and scored highest of the four arms on the supersession families at 1.5833
(section 4.5). It does so with no second model, no additional generation and no
extra prompt tokens. Arm C was not timed on the Raspberry Pi, where only Arms B
and D were measured, so no latency figure is claimed for it; what can be said
from the architecture is that it contains the same single generation pass as
Arm B and no second model call.

**The precondition is document status metadata, and that is an organisational
task rather than a technical one.** Every result in this study that favours the
cheap mechanism depends on documents carrying an accurate current-or-withdrawn
status. An SME whose shared drive contains six versions of the same handbook
with no status field gains nothing from Arm C, and the knowledge management
literature suggests this is the normal condition rather than the exception:
small firms manage knowledge informally, and storage and retention are the least
developed parts of the cycle (Durst and Edvardsson, 2012; Massaro et al., 2016;
Durst, Edvardsson and Foli, 2023). The practical recommendation is therefore
unglamorous. Curate the status field first. The retrieval improvement that field
enables is larger than anything the verification layer delivered.

**Live disagreements should be routed to a person, not resolved by the system.**
No arm handled them. Conflict-handling means on the eight live-disagreement
families range from 0.25 to 0.3333 on a three-point scale, and every arm scored
zero on all three `mutually_exclusive` families, where two current documents
directly contradict each other (sections 4.6 and 4.4). The design implication is
to detect the situation rather than adjudicate it: flagging that two current
documents disagree is a retrieval-side property that can be computed from which
documents were retrieved, whereas deciding which one governs was beyond the
verifier tested here.

**Plan for asynchronous use.** At 174 seconds per answer on a throttled Pi 5,
with a complete 68-question run taking 2 hours 15 minutes (section 5.5), an
interactive help desk is not the deployment. A submit-and-collect model, or an
overnight batch over a day's queue, is entirely feasible on a 16GB board drawing
a few watts, with no document leaving the premises. That posture also answers
the barrier the adoption literature identifies most consistently for small firms:
privacy, security and governance concerns compounded by the absence of in-house
legal and technical expertise (OECD, 2025; Oldemeyer, Jede and Teuteberg, 2025).
Under the GDPR's data minimisation and purpose limitation principles
(Regulation (EU) 2016/679), and the phased obligations of the EU Artificial
Intelligence Act (Regulation (EU) 2024/1689), an assistant whose corpus never
leaves the building is the simplest defensible posture available to a firm
without a legal department.

**Do not promise a confidence label that has not been evaluated.** The
implemented mechanism assigns a categorical confidence level by rule. Its
calibration was not measured, no claim is made for it, and the interface labels
it rule-based wherever it appears. The literature is unambiguous that verbalised
confidence from instruction-tuned models tends towards systematic overconfidence
(Xiong et al., 2024) and that neural networks are miscalibrated by default (Guo
et al., 2017). A confidence flag becomes actionable only when a low-confidence
label corresponds to a materially higher error rate, and nothing in this study
establishes that it does.

**Check the citations.** Roughly four answers in five cite only real, retrieved
identifiers, and roughly a third of the checkable citations do not carry the
claim attached to them (section 4.8). An SME deploying this should treat the
citation as a pointer to where to look, not as a warrant that the answer is
right.

## 6.4 Limitations and threats to validity

Section 5.7 stated the four limits bearing directly on Chapter 5's analysis.
This section takes the study as a whole.

**Construct validity.** Conflict handling and answer correctness are manual
three-point rubric judgements, and the rubric operationalises constructs that
have no standard measure. Citation support is measured automatically by checking
whether a cited passage contains the claim's quantities, which is a necessary
and not a sufficient condition, making the reported figure an upper bound on
true support (section 4.8). Appropriate abstention is measured over five
deliberate gap topics, and the ceiling reached by two arms means the measure
could not discriminate between them (section 4.9). Latency is wall-clock time
per answer, which is what a user waits for and is therefore the right construct
for RQ4, but it says nothing about throughput under concurrent load.

**Internal validity.** The single-variable contrast between Arms B and D is
clean: D sees the same evidence, replays B's draft and differs only by the
verification pass. The comparison between C and D is not, because it changes
retrieval mode and verification together, and it is reported throughout as a
practical comparison rather than as an ablation. One further confound is
specific to the verification test and is arguably the most consequential
limitation in the study: **on ten of the twenty-four live-disagreement questions
the chunks carrying both sides of the disputed fact were never retrieved**
(section 4.13). On those questions the verifier was asked to adjudicate a
dispute it could not see, so H2 tests the verification layer and the retrieval
configuration together. The result is a valid statement about this pipeline and
a weaker statement about verification as such.

**External validity.** The corpus is 38 synthetic documents with typed planted
conflicts, built to make the phenomenon measurable rather than to resemble a
particular firm's document estate. The question set is 68 held-out questions in
32 groups. One model pair was used, `llama3.2:3b` for generation and
`qwen2.5:3b` for verification, selected by a documented diagnostic rather than a
sweep (amendment 1.10), at one quantisation, on one board. Every quantitative
claim in this dissertation is a claim about that configuration. What is likely
to transfer is the shape of the findings, that verification is downstream of
retrieval, that detection is easier than classification, and that a runtime gate
must clear a higher bar than an offline metric; the magnitudes should not be
assumed to.

**Conclusion validity.** No inferential test is computed anywhere in this
study, by design and by pre-registration: with four and eight families the
sample cannot support one, and the decision rules are operational thresholds
rather than significance tests (section 4.3). Leave-one-family-out resampling is
reported instead, and it shows H1's margin to be fragile, moving from -0.1111 to
+0.4444 across folds against a threshold of 0.25, while H2's is stable within
0.0476 (sections 4.5 and 4.6). Statements about H2 are therefore made more
firmly than statements about H1. H2c is a floor rather than a result: zero
false-conflict events were observed in every arm, so the prediction could not
have been supported by this design at this sample size (section 4.7).

**Measurement independence.** All manual scores come from one reviewer, the
author. Duplicate-group agreement was 58 of 58 on the rubric score and Cohen's
kappa 0.8198 on the abstention flag across 272 items (section 4.3), which
establishes self-consistency and not independence. The arm-blinding was
structurally defeated on the thirteen items carrying the abstention template
verbatim (amendment 1.13), and those are precisely the items carrying most of
Chapter 5's explanatory weight. A second independent scorer is the single
cheapest improvement available to a replication.

**No human users.** The study measures answers, not use. Whether an SME employee
would trust, verify or act on these answers, and whether an over-refusing
assistant is abandoned faster than an over-answering one, are empirical
questions this design cannot reach.

## 6.5 Reflection on the method

The pre-registration is the part of this project most likely to be useful to
someone else, and it deserves a critical rather than a celebratory reading.

It worked in the sense that matters. The decision rule for every hypothesis, the
thresholds, the confirmatory contrast, the split on which tuning was permitted
and the commitment to report a null result as a null result were all fixed in
writing before the confirmatory runs, and every subsequent change is recorded
with its date, its reason and a statement of what it did not change (Nosek et
al., 2018). Five of the six hypotheses were not supported. **The value of having
written the rules down first is precisely that this outcome is reportable rather
than negotiable**, and a reader can check that the rules were not moved by
reading the amendment record in Appendix B against the commit history.

It also cost more than expected, in two ways worth naming. The amendment record
runs to 37 amendments across 227 numbered entries, and a substantial fraction of
them correct not the design but the gap between what the document claimed and
what the code enforced. The pattern recurred often enough to be the project's
own methodological finding: a rule that is written down and not enforced where
it matters is worth less than no rule, because it will be quoted as though it
were enforced. Twelve instances are recorded. The habit that eventually caught
them was to treat every stated property as a test to be written, and to distrust
any test that could not be made to fail.

The second cost is that pre-registration constrains what can be reported, not
only what can be claimed. Section 4.11 declines to compute a rate of refusal
over answerable questions because that denominator was not registered in
advance, even though such a rate would have been the most direct expression of
this study's most interesting finding. Chapter 5 reports counts instead and
explains why. That is the correct trade, but it is a trade, and a study designed
today with the same evidence would register that denominator at the outset.

## 6.6 Summary

The results place a negative finding in a gap the literature had left open, and
the metadata baseline is what gives it weight: the verification layer did not
beat a three-line filter on document status, at 3.18 times the latency. Against
the reviewed literature, the study confirms that reliability is bought with
computation and shows that at 3B scale the purchase can fail to deliver; it
inverts the documented over-answering failure into over-refusal; and it supplies
empirical grounds for the design assumptions behind Self-RAG's fine-tuning and
corrective RAG's external fallback. For an SME the practical guidance is to
curate document status metadata, filter on it, route live disagreements to a
person, plan for asynchronous use, treat citations as pointers rather than
warrants, and promise nothing about a confidence label that has not been
evaluated. Chapter 7 states the contributions and what should be done next.
