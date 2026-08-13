"""Scoring an answer against the evidence it cited.

This module exists because of one observation in the Stage 4 pilot. Asked how
quickly a lost laptop must be reported, the model answered:

    "You must report a lost laptop to the IT helpdesk within 1 hour of
    becoming aware [IT-03#001]."

The answer is correct. The citation is not. "Within 1 hour" appears in
IT-03#002, the Timescales table; IT-03#001 is the list of what counts as an
incident. Every metric available at the time passed this answer: the identifier
was real, it had been retrieved, nothing was invented.

An answer whose citations do not support its claims is unverifiable by the
reader, which for a system whose purpose is verifiable answers is a failure
even when the answer is right. Three distinct things therefore need separate
measurement:

``citation validity``      were the cited identifiers real and retrieved
``citation support``       does the cited passage contain the claim's content
``citation completeness``  does every substantive claim carry a citation

The first is structural and lives in the generator. The other two are here,
because they are evaluation.

Why this can be automated
-------------------------
Policy answers are dominated by specific quantities: 55 pence, 1 hour, 25 days,
14 characters, £130. A claim's *salient tokens* are these quantities together
with the words that qualify them. If a claim asserts "1 hour" and the chunk it
cites contains no such figure, the citation does not support the claim, and no
gold answer is needed to establish that.

This is a **necessary** condition, not a sufficient one. A citation can contain
the right number and still be the wrong source, and a purely qualitative claim
has no salient tokens to check. Support scored here is therefore a lower bound
on citation error, reported as such, and complemented by manual review on a
sample. It is not a replacement for a gold-answer comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# A quantity and the word that gives it meaning: "55 pence", "1 hour",
# "10,000 business", "£130 per". The unit carries the meaning, so both parts
# must be present in a chunk for it to support the claim.
QUANTITY_RE = re.compile(
    r"(?P<value>£?\d[\d,]*(?:\.\d+)?%?)\s*(?P<unit>[A-Za-z][A-Za-z-]{1,15})?"
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*\-•])")
CITATION_RE = re.compile(r"\[([A-Z]{2,4}-\d{2}(?:#\d{1,3})?)\]")

# A document identifier anywhere in the text, bracketed or not. Stripping
# only the bracketed form left the bare form intact, so "IT-04 retains annual
# backups for seven years [IT-04#002]" still yielded the quantity "04" with
# unit "retains". The same class of bug as the original citation-marker one,
# in a second place, found in review on 8 August 2026.
DOCUMENT_ID_RE = re.compile(r"\b[A-Z]{2,4}-\d{2}(?:#\d{1,3})?\b")

# Words that carry no discriminating power as a unit.
STOP_UNITS = {
    "and", "or", "of", "in", "the", "a", "an", "to", "for", "is", "are", "was",
    "were", "be", "per", "at", "on", "as", "by", "with", "from", "that", "this",
}


@dataclass(frozen=True)
class Claim:
    """One sentence of an answer, with whatever it cited."""

    text: str
    citations: tuple[str, ...]
    quantities: tuple[tuple[str, str], ...]

    @property
    def is_substantive(self) -> bool:
        """Whether this sentence asserts something checkable.

        A sentence with no quantity is not necessarily unsubstantive, but it is
        not checkable by this method, and counting it as supported would
        overstate the metric.
        """
        return bool(self.quantities)


@dataclass(frozen=True)
class ClaimScore:
    claim: Claim
    supported_by: tuple[str, ...]
    unsupported_quantities: tuple[str, ...]

    @property
    def is_cited(self) -> bool:
        return bool(self.claim.citations)

    @property
    def is_supported(self) -> bool:
        return self.is_cited and not self.unsupported_quantities


@dataclass(frozen=True)
class AnswerScore:
    """Citation support and completeness for one answer."""

    claims: tuple[ClaimScore, ...]

    @property
    def checkable(self) -> tuple[ClaimScore, ...]:
        return tuple(c for c in self.claims if c.claim.is_substantive)

    @property
    def citation_completeness(self) -> float | None:
        """Fraction of checkable claims that carry any citation at all.

        Deliberately not "how many retrieved chunks were cited". A generator is
        not expected to cite every passage it was given; it is expected to
        attribute every claim it makes.
        """
        checkable = self.checkable
        if not checkable:
            return None
        return round(sum(1 for c in checkable if c.is_cited) / len(checkable), 4)

    @property
    def citation_support(self) -> float | None:
        """Fraction of cited checkable claims whose citation contains the claim.

        This is the measure that fails the lost-laptop answer.
        """
        cited = [c for c in self.checkable if c.is_cited]
        if not cited:
            return None
        return round(sum(1 for c in cited if c.is_supported) / len(cited), 4)

    @property
    def unsupported_claims(self) -> tuple[ClaimScore, ...]:
        return tuple(c for c in self.checkable if c.is_cited and not c.is_supported)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_count": len(self.claims),
            "checkable_claim_count": len(self.checkable),
            "citation_completeness": self.citation_completeness,
            "citation_support": self.citation_support,
            "unsupported": [
                {
                    "claim": c.claim.text,
                    "cited": list(c.claim.citations),
                    "quantities_not_found": list(c.unsupported_quantities),
                }
                for c in self.unsupported_claims
            ],
            "method": "quantity_overlap",
            "is_lower_bound": True,
            "note": (
                "Support is a necessary condition checked automatically over "
                "quantities. A citation containing the right figure may still be "
                "the wrong source, and qualitative claims are not checkable this "
                "way. Treat as a lower bound on citation error and complement "
                "with manual review on a sample."
            ),
        }


def extract_quantities(text: str) -> tuple[tuple[str, str], ...]:
    """Quantities and their units, as (value, unit) pairs.

    Document identifiers are stripped first, bracketed or not. ``[IT-03#001]``
    otherwise yields the quantities "03" and "001", which no chunk contains, so
    every cited claim would be scored unsupported and the metric would read zero
    everywhere while appearing to work.

    Stripping only the bracketed form is not enough. Models routinely write the
    identifier in running prose as well: "IT-04 retains annual backups for seven
    years [IT-04#002]" yielded "04" from the bare identifier long after the
    bracketed one was handled. The bare form is stripped for the same reason and
    is covered by its own regression test.
    """
    text = DOCUMENT_ID_RE.sub(" ", CITATION_RE.sub(" ", text))
    found: list[tuple[str, str]] = []
    for match in QUANTITY_RE.finditer(text):
        value = match.group("value")
        unit = (match.group("unit") or "").lower()
        if unit in STOP_UNITS:
            unit = ""
        pair = (value, unit)
        if pair not in found:
            found.append(pair)
    return tuple(found)


def split_claims(answer: str) -> tuple[Claim, ...]:
    """Split an answer into sentences, each with its citations and quantities.

    Bullet points count as claims. Models answering policy questions frequently
    produce a lead sentence carrying the citation followed by bulleted figures,
    and treating the whole block as one claim would hide which figure was
    attributed to what.
    """
    claims: list[Claim] = []
    pending_citations: tuple[str, ...] = ()

    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for sentence in SENTENCE_SPLIT_RE.split(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            citations = tuple(dict.fromkeys(CITATION_RE.findall(sentence)))
            if citations:
                pending_citations = citations
            # A bullet under a cited lead sentence inherits its citation: the
            # model attributed the block, not each line.
            effective = citations or (
                pending_citations if line.startswith(("*", "-", "•")) else ()
            )
            claims.append(Claim(
                text=sentence,
                citations=effective,
                quantities=extract_quantities(sentence),
            ))
    return tuple(claims)


def _chunk_supports(quantity: tuple[str, str], chunk_text: str) -> bool:
    value, unit = quantity
    lowered = chunk_text.lower()
    if value.lower() not in lowered:
        return False
    return not unit or unit in lowered


def score_answer(answer: str, chunk_texts: Mapping[str, str]) -> AnswerScore:
    """Score an answer's citations against the text of the chunks it cited.

    ``chunk_texts`` maps chunk or document identifier to text. A citation of a
    document rather than a chunk is checked against that document's chunks
    combined, since the model attributed the document as a whole.
    """
    scored: list[ClaimScore] = []
    for claim in split_claims(answer):
        cited_text = " \n ".join(
            chunk_texts.get(identifier, "") for identifier in claim.citations
        )
        missing = tuple(
            f"{value} {unit}".strip()
            for value, unit in claim.quantities
            if not _chunk_supports((value, unit), cited_text)
        )
        supported_by = tuple(
            identifier for identifier in claim.citations
            if identifier in chunk_texts
            and all(_chunk_supports(q, chunk_texts[identifier]) for q in claim.quantities)
        )
        scored.append(ClaimScore(
            claim=claim,
            supported_by=supported_by,
            unsupported_quantities=missing if claim.citations else (),
        ))
    return AnswerScore(claims=tuple(scored))


def chunk_text_map(retrieval) -> dict[str, str]:
    """Build the identifier-to-text map from a retrieval result.

    Document-level identifiers map to that document's retrieved chunks joined,
    because a citation of ``[HR-13]`` attributes the document rather than a
    passage.
    """
    texts: dict[str, str] = {}
    by_document: dict[str, list[str]] = {}
    for scored in retrieval:
        texts[scored.chunk_id] = scored.chunk.text
        by_document.setdefault(scored.chunk.doc_id, []).append(scored.chunk.text)
    for doc_id, parts in by_document.items():
        texts[doc_id] = "\n".join(parts)
    return texts
