# Appendix B: Pre-registration amendment record

All 30 amendments to `docs/PREREGISTRATION.md`, in order. Numbers, dates and
sub-entry counts are read from that document by `scripts/make_amendment_table.py`;
the summaries are written for this appendix. Each amendment in the source carries
its own reason, its evidence, and a statement of what it did **not** change.

Phases are taken from the commit history. The frozen four-arm test run is
`4ba79da`, 14 August 07:13; the blinding key was opened at `ed65f22`, 15 August
01:42; the quality analysis was signed off at `be55077`.

| # | Date | Entries | What it changed |
|---|---|---:|---|
| | | | **Phase A: Development, before the frozen confirmatory runs** |
| 1.1 | 8 August | 12 | Pilot-contaminated families excluded; per-question rubrics; the arms restated as a tree rather than a ladder; a correctness metric added; blinding and provenance defects recorded. |
| 1.2 | 8 August | 8 | The conflict taxonomy replaced one type with four; three compatible families added as negative controls; eight further families planted; H2 split into H2a, H2b and H2c. |
| 1.3 | 8 August | 3 | Retrieval calibrated on the development split alone: top_k 4 to 6, min_similarity 0.32 to 0.30. No threshold separates answerable from unanswerable questions, and that is reported as a finding. |
| 1.4 | 13 August | 5 | Four families had been typed by intuition and typed wrongly. Reclassified, the enabling condition named, conflict-pair recall corrected, and the overclaims it had supported withdrawn. |
| 1.5 | 13 August | 5 | CONF-12 reclassified; H2a and H2b merged into a single pooled H2 because the subtype judgement was not reliable enough to condition a threshold on; rubric drift repaired; gold data frozen. |
| 1.6 | 13 August | 9 | Arm D was rewriting answers it had no complaint about. A serving rule was implemented so that a verifier finding nothing is a no-op, every served revision names its warrant, and the stopping rule became code. |
| 1.7 | 13 August | 9 | A reproducibility claim made and withdrawn; the corrected instrument and its decision rule fixed before measurement; the abstention text moved into the source so no claim could be smuggled into free prose. |
| 1.8 | 14 August | 7 | The pre-declared development gate restored and made to fire; decorative safeguards made real; the abstention template applied wherever the verifier abstains. |
| 1.9 | 14 August | 4 | Pilot 04 passed containment and exposed a measurement defect that amendment 1.8 had introduced. Coverage is now reported beside every conditional figure and false refusals are attributed. |
| 1.10 | 14 August | 7 | The verifier diagnostic protocol selected Qwen2.5 3B over Llama 3.2 3B on conflict detection. What the result does not establish is recorded alongside it. |
| 1.11 | 14 August | 6 | Every Qwen response had omitted the claim audit because the prompt taught the omission and the validated helper was never wired in. Both fixed; the system now fails closed on a missing audit. |
| 1.12 | 14 August | 7 | Pilot 06 against the declared gate. The gate fails, the budget is spent, and the defensible statement about the verifier is written at the width the evidence supports. |
| | | | **Phase B: Manual scoring, after the runs and before unsealing** |
| 1.13 | 14 August | 6 | The blinding was defeated on thirteen items carrying the abstention template verbatim, exposing all 68 items of one arm. Recorded rather than repaired, with the unblinding rate measured rather than assumed. |
| 1.14 | 14 August | 8 | Manual scoring completed. The rubric agreed on 58 of 58 duplicate groups; the abstention flag drifted with position, so a re-pass was run under a rule fixed in advance and became the reported value. |
| | | | **Phase C: Hardware boundary and execution, after the quality analysis** |
| 1.15 | 15 August | 5 | The hardware runs declared performance-only, before any were executed, so that a timing run could not produce a quality figure. |
| 1.16 | 15 August | 5 | The enforcement 1.15 claimed did not exist; the four frozen quality runs are now a closed list. Four analysis corrections, including a post-unsealing denominator reverted and 'equivalent' withdrawn. |
| 1.17 | 15 August | 7 | The runner gate, placement enforcement and observation, Arm D's stage reporting, and H5 restricted to the Raspberry Pi 5, all corrected and tested before any hardware execution. |
| 1.18 | 15 August | 6 | Validation hardened before any latency value was read, with the reason for hardening after the runs rather than before stated explicitly. |
| 1.19 | 15 August | 9 | Seven fail-open paths closed before reading latency, including a rejected run whose latency was computed anyway and a rejection record that invited destroying the evidence. |
| 1.20 | 15 August | 9 | The preflight was testing the wrong model, warming the run it protected, and could not be exercised without spending a run. Made standalone and per timed stage. |
| 1.21 | 15 August | 7 | Model names canonicalised against what Ollama reports, placement actually applied rather than only observed, cleanup run on the path where it mattered, and magnitudes stripped from rejection records. |
| 1.22 | 15 August | 6 | The first live preflight passed all three placement checks and its cleanup check failed. The failure is retained, and the reading that would explain it away is declined for want of evidence. |
| 1.23 | 15 August | 3 | The unload budget had assumed a retention it never stated, and one field was doing three jobs. Both corrected before the instrument was rerun. |
| 1.24 | 15 August | 3 | The unload question closed by measurement. The analyser's success path had transposed two arguments and no test had ever reached it; fixed, made keyword-only, and covered end to end. |
| | | | **Phase D: Write-up corrections** |
| 1.25 | 19 August | 6 | Two pre-registered primary metrics, answer correctness and superseded citation rate, had been scored and frozen but never aggregated. The rule was fixed before either figure was computed. |
| 1.26 | 19 August | 6 | Cohen's kappa was quoted in working notes but existed in no file; implemented and tested. Figure-generation rules declared, then seven reporting defects corrected following independent review. |
| | | | **Phase E: Post-evaluation demonstrator, no evidence contributed** |
| 1.27 | 20 August | 5 | Declared the boundary for a post-evaluation dashboard before it was built: two separated modes, live output confined to Arm D, replay read-only over the frozen runs, and no evidence contributed to any hypothesis. |
| 1.28 | 20 August | 6 | Corrected the dashboard against its own amendment. 1.27 described a write path the implementation does not have; replay was joining records without enforcing that all four arms answered every question over the same corpus; the claim audit showed only supporting evidence and read as an adjudication rather than as recorded model output; live questions travelled in the URL. |
| | | | **Phase F: Exploratory analysis of already-frozen data** |
| 1.29 | 20 August | 5 | Post-hoc exploratory diagnostic of the frozen verifier's internal relationship classification, kept separate from binary conflict detection. No threshold, no verdict, no hypothesis revisited. |
| | | | **Phase G: Review corrections, no experimental change** |
| 1.30 | 20 August | 12 | Eight corrections after a second review, seven of them rules this document stated and the code did not enforce: 1.28's claim that replay failed closed and that live questions used POST was false when written, 1.29's principal denominator contradicted its own rule against a pooled headline, and 1.26's figures were not byte-reproducible. Enforcement added, the pooled total withdrawn, Appendix D generated. 1.30.11 records the same defect recurring in this amendment's own test, which claimed an isolation the code could not perform and overwrote four committed figures. |

**30 amendments, 196 numbered sub-entries.** Phase A amendments precede the
frozen confirmatory runs and could and did change the design. Nothing from Phase B
onwards could: the runs were complete before Phase B opened. Phases B and D govern
how the already-frozen data are scored, aggregated and reported; Phase C concerns
the separate hardware experiment; Phase E concerns a demonstrator built after all
evidence was frozen, which contributes none and is scored nowhere; Phase F is
exploratory analysis of the frozen data, carrying no threshold and no verdict; and
Phase G corrects how all of the above are enforced and reported, changing no
hypothesis, no verdict and no frozen file.
