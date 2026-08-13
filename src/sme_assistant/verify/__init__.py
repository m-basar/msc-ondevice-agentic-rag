"""The verification layer: Arm D, and the contribution.

**No module in this package may read evaluation data.** The verifier sees the
question, the retrieved chunks and ordinary document metadata. It infers
whether the evidence disagrees and what kind of disagreement it is. Reading a
declared relationship would make conflict detection a dictionary lookup and the
result a restatement of the registry.
"""

from .schema import (
    CONFLICTING_RELATIONSHIPS,
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE,
    SUPPORTED,
    VALID_RELATIONSHIPS,
    VALID_VERDICTS,
    ClaimVerdict,
    Verification,
    VerificationError,
)
from .verifier import VerifiedAnswer, Verifier, build_verification_prompt

__all__ = [
    "CONFLICTING_RELATIONSHIPS", "CONTRADICTED", "INSUFFICIENT_EVIDENCE",
    "SUPPORTED", "VALID_RELATIONSHIPS", "VALID_VERDICTS", "ClaimVerdict",
    "Verification", "VerificationError", "VerifiedAnswer", "Verifier",
    "build_verification_prompt",
]
