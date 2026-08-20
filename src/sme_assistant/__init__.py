"""Privacy-preserving on-device RAG assistant for SME knowledge management.

MSc Applied AI dissertation artefact, WMG, University of Warwick.
Md Basar Basar (5753701).

Implemented pipeline:

    retrieval -> grounded generation -> verification (Arm D only)

Verification audits each claim in the draft against the same evidence the
generator saw, classifies any relationship between the retrieved passages, and
either returns the draft unchanged or replaces it. A rule-based categorical
confidence level is attached from the verifier's verdict.

Two stages named in an earlier design were never built and are not part of this
artefact: a separate query-analysis stage and a next-action suggestion stage.
"Agentic" refers to the verification and revision loop, not to autonomous
planning or tool use. Confidence is a declared mapping, not a calibrated score,
and its calibration was not evaluated.
"""

__version__ = "0.1.0"
