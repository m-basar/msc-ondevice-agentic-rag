"""The conflict registry.

Every contradiction in the knowledge base is either deliberate and declared
in ``data/conflicts.json``, or it is a defect. There is no third category.

This module exists because of a specific failure mode. During review of the
first corpus, several genuine contradictions were found between documents
that were meant to agree, sitting alongside one contradiction that was
planted on purpose. Nothing in the repository distinguished them. Gold
answers written against that corpus would have encoded the accidents as if
they were the design, and the evaluation would have measured the wrong thing
while looking entirely healthy.

Validating the registry against the corpus closes that gap: a superseded
document that no family accounts for now fails the test suite.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import KnowledgeBase


class ConflictRegistryError(RuntimeError):
    """Raised when the registry and the corpus disagree."""


@dataclass(frozen=True)
class ConflictFamily:
    """One planted conflict, and what the system is expected to do about it."""

    family_id: str
    name: str
    domain: str
    risk_level: str
    superseded_document: str
    current_document: str
    conflicting_facts: tuple[dict[str, str], ...]
    must_cite: tuple[str, ...]
    must_not_assert: tuple[str, ...]

    @property
    def documents(self) -> tuple[str, str]:
        return (self.superseded_document, self.current_document)


class ConflictRegistry:
    """Loaded conflict manifest."""

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source
        self.families = tuple(
            ConflictFamily(
                family_id=entry["id"],
                name=entry["name"],
                domain=entry["domain"],
                risk_level=entry["risk_level"],
                superseded_document=entry["superseded_document"],
                current_document=entry["current_document"],
                conflicting_facts=tuple(entry["conflicting_facts"]),
                must_cite=tuple(entry.get("must_cite", ())),
                must_not_assert=tuple(entry.get("must_not_assert", ())),
            )
            for entry in data["families"]
        )

    def __len__(self) -> int:
        return len(self.families)

    def by_id(self, family_id: str) -> ConflictFamily:
        for family in self.families:
            if family.family_id == family_id:
                return family
        raise ConflictRegistryError(f"No conflict family {family_id!r}")

    def for_document(self, doc_id: str) -> ConflictFamily | None:
        for family in self.families:
            if doc_id in family.documents:
                return family
        return None

    @property
    def expected_answer_policy(self) -> dict[str, str]:
        return self._data["expected_answer_policy"]

    @property
    def fully_absent_topics(self) -> tuple[str, ...]:
        return tuple(self._data["deliberate_gaps"]["fully_absent"])

    @property
    def partial_topics(self) -> tuple[dict[str, str], ...]:
        return tuple(self._data["deliberate_gaps"]["partially_present"])

    def summary(self) -> dict[str, Any]:
        return {
            "family_count": len(self.families),
            "by_risk": {
                level: sum(1 for f in self.families if f.risk_level == level)
                for level in ("high", "medium", "low")
            },
            "domains": sorted({f.domain for f in self.families}),
            "fact_count": sum(len(f.conflicting_facts) for f in self.families),
            "fully_absent_topics": len(self.fully_absent_topics),
            "partial_topics": len(self.partial_topics),
        }


def load_conflicts(path: Path | str) -> ConflictRegistry:
    source = Path(path)
    if not source.exists():
        raise ConflictRegistryError(f"Conflict registry not found: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConflictRegistryError(f"{source} is not valid JSON: {exc}") from exc
    for key in ("families", "expected_answer_policy", "deliberate_gaps"):
        if key not in data:
            raise ConflictRegistryError(f"{source}: missing top-level key {key!r}")
    return ConflictRegistry(data, source)


# Keyword probes for the deliberate gaps. Deliberately broad: the test should
# fail if a topic creeps into the corpus later, even obliquely.
GAP_PROBES: dict[str, tuple[str, ...]] = {
    "pensions and auto-enrolment": ("auto-enrol", "workplace pension", "pension scheme", "pension contribution"),
    "maternity, paternity and shared parental leave": ("maternity", "paternity", "parental leave", "adoption leave"),
    "redundancy and notice periods": ("redundan", "notice period", "period of notice"),
    "company car and vehicle allowance schemes": ("company car", "car allowance", "vehicle allowance"),
    "share options and bonus schemes": ("share option", "share scheme", "bonus scheme", "annual bonus"),
    "international shipping and export documentation": ("export documentation", "customs declaration", "international shipping", "commodity code"),
    "flexible working requests": ("flexible working request", "statutory flexible working"),
    "grievance procedure": ("grievance procedure", "raise a grievance", "grievance hearing"),
}


def validate_against_corpus(registry: ConflictRegistry, kb: KnowledgeBase) -> None:
    """Check the registry describes the corpus that is actually on disk.

    Four classes of failure are caught here:

    1. A family names a document that does not exist
    2. A family's documents do not carry the supersession relationship claimed
    3. A superseded document exists in the corpus that no family accounts for
    4. A topic declared absent has appeared in the corpus
    """
    for family in registry.families:
        for doc_id in family.documents:
            if not kb.has(doc_id):
                raise ConflictRegistryError(
                    f"{family.family_id}: names unknown document {doc_id!r}"
                )

        old = kb.by_id(family.superseded_document)
        new = kb.by_id(family.current_document)

        if old.status != "superseded":
            raise ConflictRegistryError(
                f"{family.family_id}: {old.doc_id} is registered as the superseded "
                f"document but its status is {old.status!r}"
            )
        if not new.is_current:
            raise ConflictRegistryError(
                f"{family.family_id}: {new.doc_id} is registered as the current "
                f"document but its status is {new.status!r}"
            )
        if old.superseded_by != new.doc_id:
            raise ConflictRegistryError(
                f"{family.family_id}: {old.doc_id} points at "
                f"{old.superseded_by!r}, not {new.doc_id!r}"
            )
        if not family.conflicting_facts:
            raise ConflictRegistryError(
                f"{family.family_id}: declares no conflicting facts, so nothing can be tested"
            )

    registered = {doc_id for family in registry.families for doc_id in family.documents}
    unaccounted = [doc.doc_id for doc in kb.superseded() if doc.doc_id not in registered]
    if unaccounted:
        raise ConflictRegistryError(
            "Superseded documents not declared in the conflict registry: "
            f"{sorted(unaccounted)}. Either register them as deliberate conflicts "
            "or remove them from the corpus."
        )

    corpus_text = "\n".join(doc.body for doc in kb).lower()
    for topic in registry.fully_absent_topics:
        probes = GAP_PROBES.get(topic)
        if probes is None:
            raise ConflictRegistryError(
                f"No keyword probes defined for declared gap {topic!r}; "
                "add them to GAP_PROBES so the gap can be verified"
            )
        hits = [p for p in probes if re.search(re.escape(p), corpus_text)]
        if hits:
            raise ConflictRegistryError(
                f"Topic {topic!r} is declared absent but the corpus matches {hits}. "
                "Unanswerable test questions on this topic would no longer be unanswerable."
            )
