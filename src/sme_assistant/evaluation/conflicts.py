"""The conflict registry. Evaluation gold data.

Every contradiction in the knowledge base is either deliberate and declared in
``gold/conflicts.json``, or it is a defect. There is no third category.

This module exists because of a specific failure mode. During review of the
first corpus, several genuine contradictions were found between documents that
were meant to agree, sitting alongside one contradiction that was planted on
purpose. Nothing distinguished them. Gold answers written against that corpus
would have encoded the accidents as if they were the design, and the evaluation
would have measured the wrong thing while looking entirely healthy.

**This is gold data and no part of the inference pipeline may read it.** It
holds expected answers, prohibited assertions, and the list of unanswerable
topics. It lives under ``config.evaluation``, deliberately not under
``config.paths``, and it sits in ``sme_assistant.evaluation`` rather than
``sme_assistant.kb`` so that the boundary is visible in the import graph.
``tests/test_no_oracle_leakage.py`` enforces it.

Four family types
-----------------
``current_current`` was one label covering three different situations, and two
of the families filed under it were not contradictions at all. A verifier
scored on that set would have been rewarded for flagging policies that agree,
which is the failure a conflict detector most needs to avoid. Found in review
and corrected under pre-registration amendment 1.2.

The distinguishing question is whether one action can satisfy both documents,
and whether an answer can still be wrong.

``version_supersession``
    One document explicitly supersedes another. A status filter resolves it
    with no reasoning at all, so these families alone cannot demonstrate that
    verification adds anything.

``mutually_exclusive``
    No action satisfies both. The trade counter cannot open at 08:00 and at
    08:30. Surface both, pick neither, escalate.

``stricter_looser``
    An action satisfies both, but an answer can still be unsafe. Told "24
    hours" when another document says 1 hour, a user who reports at hour 20 has
    breached the stricter document. State the stricter figure as the safe
    course, name both, escalate. Which document is stricter is recorded in the
    registry, so the conservative reading is a property of the family rather
    than a judgement made at answer time.

``compatible``
    Both hold independently and no answer is wrong, only incomplete. A
    six-monthly departmental access review and an annual Data Protection Lead
    review of personal-data systems are two layers, not two answers. These are
    **negative controls**: a verifier that flags one is producing a false
    positive. Without them nothing measures over-detection, and a system that
    shouted "conflict" at every pair of documents would score perfectly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..kb.loader import KnowledgeBase

# Four types, not two. ``current_current`` lumped together three situations
# that need different behaviour from a verifier, and two of the families filed
# under it were not contradictions at all. Found in review, 8 August 2026,
# corrected under pre-registration amendment 1.2.
#
# The distinguishing question is: **can one action satisfy both documents, and
# can an answer still be wrong?**
VALID_TYPES = {
    # One document explicitly supersedes another. A status filter resolves it.
    "version_supersession",
    # No action satisfies both. The trade counter cannot open at 08:00 and
    # 08:30. Surface both, pick neither, escalate.
    "mutually_exclusive",
    # An action satisfies both, but an answer can still be unsafe: told "24
    # hours" when another document says 1 hour, a user who reports at hour 20
    # has breached the stricter document. State the stricter figure as the safe
    # course, name both, escalate.
    "stricter_looser",
    # Both hold independently and no answer is wrong, only incomplete. A
    # six-monthly departmental review and an annual Data Protection Lead review
    # are two layers, not two answers. **Negative control:** a verifier that
    # flags one of these is producing a false positive, and the false-conflict
    # rate is a reported metric.
    "compatible",
}

CONFLICTING_TYPES = VALID_TYPES - {"compatible"}

VALID_RESOLUTIONS = {
    "prefer_current",
    "escalate_unresolved",
    "prefer_stricter_and_escalate",
    "no_conflict",
}

# "reported" families appear in results. "tuning" families exist only so the
# pipeline can be developed against real conflicts without inspecting a family
# that will later be scored. Nothing else distinguishes them: both are planted
# the same way, validated the same way, and present in the same corpus.
VALID_SPLITS = {"reported", "tuning"}


class ConflictRegistryError(RuntimeError):
    """Raised when the registry and the corpus disagree."""


@dataclass(frozen=True)
class ConflictFact:
    """One disputed fact, with the literal text that evidences each position."""

    fact: str
    values: dict[str, str]
    anchors: dict[str, str]


@dataclass(frozen=True)
class ConflictFamily:
    """One planted conflict, and what the system is expected to do about it."""

    family_id: str
    name: str
    conflict_type: str
    domain: str
    risk_level: str
    documents: tuple[str, ...]
    authoritative: str | None
    resolution: str
    conflicting_facts: tuple[ConflictFact, ...]
    must_cite: tuple[str, ...]
    must_not_assert: tuple[str, ...]
    # "reported" or "tuning". Defaulted so every existing family keeps its
    # meaning without the file being rewritten: absence means reported, which
    # is the conservative reading.
    split: str = "reported"
    # For stricter_looser: which document states the safe reading. The
    # conservative answer is a property of the family recorded here, not a
    # judgement made at answer time, so it stays scoreable on one rubric.
    stricter: str | None = None
    # For compatible: why the apparent disagreement is not one, and the
    # literal text in the corpus that shows it. A negative control whose
    # reconciliation is only asserted in the registry is a trick question;
    # one a reader could discover is a fair test.
    reconciliation: str | None = None
    reconciliation_anchors: dict[str, str] = field(default_factory=dict)
    # The fact that decides the type, stated rather than assumed.
    #
    # Four families were typed by intuition and typed wrongly, because the type
    # was asserted at planting time and nothing forced it to be justified.
    # ``compatible`` families were already safe: they had to carry a
    # reconciliation the reader could check. These two fields give the other
    # types the same discipline. Writing "wear safety footwear on every visit,
    # which satisfies both" makes it obvious that CONF-11 was never mutually
    # exclusive; being unable to write it is what proves one is.
    satisfying_action: str | None = None
    no_satisfying_action_because: str | None = None

    @property
    def is_filter_resolvable(self) -> bool:
        """Whether a metadata filter on document status would resolve this.

        Reported in results so the analysis can separate what the contribution
        achieved from what three lines of filtering would have achieved anyway.
        """
        return self.conflict_type == "version_supersession"

    @property
    def is_conflict(self) -> bool:
        """Whether the documents actually disagree.

        False for ``compatible`` families, which exist to be *not* flagged.
        Without them nothing measures over-detection, and a verifier that
        shouted "conflict" at every pair of documents would score perfectly.
        """
        return self.conflict_type in CONFLICTING_TYPES

    @property
    def expected_behaviour(self) -> str:
        """What a correct system does, derived from the type.

        Derived rather than declared per question, so the behaviour cannot
        drift away from the classification that justifies it.
        """
        return {
            "version_supersession": "cite_current_only",
            "mutually_exclusive": "surface_both_and_escalate",
            "stricter_looser": "prefer_stricter_and_escalate",
            "compatible": "answer_without_flagging_conflict",
        }[self.conflict_type]


@dataclass(frozen=True)
class Gap:
    topic: str
    probes: tuple[str, ...]
    note: str | None = None


@dataclass(frozen=True)
class PartialGap:
    topic: str
    mentioned_in: str
    must_contain: str
    missing_detail_probes: tuple[str, ...]
    gap: str
    expected_behaviour: str


class ConflictRegistry:
    """Loaded conflict manifest."""

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source
        self.schema_version = data.get("schema_version", "1.0")

        parsed = tuple(
            ConflictFamily(
                family_id=entry["id"],
                split=entry.get("split", "reported"),
                name=entry["name"],
                conflict_type=entry["type"],
                domain=entry["domain"],
                risk_level=entry["risk_level"],
                documents=tuple(entry["documents"]),
                authoritative=entry.get("authoritative"),
                stricter=entry.get("stricter"),
                reconciliation=entry.get("reconciliation"),
                reconciliation_anchors=dict(entry.get("reconciliation_anchors", {})),
                satisfying_action=entry.get("satisfying_action"),
                no_satisfying_action_because=entry.get("no_satisfying_action_because"),
                resolution=entry["resolution"],
                conflicting_facts=tuple(
                    ConflictFact(
                        fact=f["fact"],
                        values=dict(f["values"]),
                        anchors=dict(f["anchors"]),
                    )
                    for f in entry["conflicting_facts"]
                ),
                must_cite=tuple(entry.get("must_cite", ())),
                must_not_assert=tuple(entry.get("must_not_assert", ())),
            )
            for entry in data["families"]
        )

        # ``families`` is the reported set, not everything in the file. Tuning
        # families exist so that prompt wording, thresholds and verifier output
        # format can be developed against real conflicts without ever touching a
        # family that appears in a result. Making the reported set the default
        # means that code which forgets to filter gets the safe answer rather
        # than a contaminated one; ``all_families`` is what corpus validation
        # uses, because an unregistered contradiction is a problem wherever it
        # sits.
        self.all_families = parsed
        self.families = tuple(f for f in parsed if f.split == "reported")
        self.tuning_families = tuple(f for f in parsed if f.split == "tuning")
        for family in parsed:
            if family.split not in VALID_SPLITS:
                raise ConflictRegistryError(
                    f"{family.family_id}: split {family.split!r} must be one of "
                    f"{sorted(VALID_SPLITS)}"
                )

        gaps = data["deliberate_gaps"]
        self.fully_absent = tuple(
            Gap(topic=g["topic"], probes=tuple(g["probes"]), note=g.get("note"))
            for g in gaps["fully_absent"]
        )
        self.partially_present = tuple(
            PartialGap(
                topic=g["topic"],
                mentioned_in=g["mentioned_in"],
                must_contain=g["must_contain"],
                missing_detail_probes=tuple(g["missing_detail_probes"]),
                gap=g["gap"],
                expected_behaviour=g["expected_behaviour"],
            )
            for g in gaps["partially_present"]
        )

    def __len__(self) -> int:
        return len(self.families)

    def by_id(self, family_id: str) -> ConflictFamily:
        for family in self.all_families:
            if family.family_id == family_id:
                return family
        raise ConflictRegistryError(f"No conflict family {family_id!r}")

    def for_document(self, doc_id: str) -> list[ConflictFamily]:
        """Every family naming this document, reported or tuning.

        A lookup, not a reporting set. A superseded document whose family
        moved to the tuning split is still registered, and code asking
        "is this document accounted for" must see it.
        """
        return [f for f in self.all_families if doc_id in f.documents]

    def of_type(self, conflict_type: str) -> list[ConflictFamily]:
        return [f for f in self.families if f.conflict_type == conflict_type]

    @property
    def confidence_policy(self) -> dict[str, Any]:
        return self._data["confidence_policy"]

    def fingerprint(self) -> str:
        import hashlib

        canonical = json.dumps(self._data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family_count": len(self.families),
            "tuning_family_count": len(self.tuning_families),
            "by_type": {
                t: len(self.of_type(t)) for t in sorted(VALID_TYPES)
            },
            "by_risk": {
                level: sum(1 for f in self.families if f.risk_level == level)
                for level in ("high", "medium", "low")
            },
            "filter_resolvable": sum(1 for f in self.families if f.is_filter_resolvable),
            "requires_reasoning": sum(1 for f in self.families if not f.is_filter_resolvable),
            "domains": sorted({f.domain for f in self.families}),
            "fact_count": sum(len(f.conflicting_facts) for f in self.families),
            "fully_absent_topics": len(self.fully_absent),
            "partial_topics": len(self.partially_present),
            "registry_sha256": self.fingerprint(),
        }


def load_conflicts(path: Path | str) -> ConflictRegistry:
    source = Path(path)
    if not source.exists():
        raise ConflictRegistryError(f"Conflict registry not found: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConflictRegistryError(f"{source} is not valid JSON: {exc}") from exc
    for key in ("families", "confidence_policy", "deliberate_gaps"):
        if key not in data:
            raise ConflictRegistryError(f"{source}: missing top-level key {key!r}")
    return ConflictRegistry(data, source)


# --- validation -------------------------------------------------------------


def _validate_family_shape(family: ConflictFamily) -> None:
    if family.conflict_type not in VALID_TYPES:
        raise ConflictRegistryError(
            f"{family.family_id}: type {family.conflict_type!r} must be one of {sorted(VALID_TYPES)}"
        )
    if family.resolution not in VALID_RESOLUTIONS:
        raise ConflictRegistryError(
            f"{family.family_id}: resolution {family.resolution!r} must be one of "
            f"{sorted(VALID_RESOLUTIONS)}"
        )
    if len(family.documents) < 2:
        raise ConflictRegistryError(f"{family.family_id}: needs at least two documents")
    if not family.conflicting_facts:
        raise ConflictRegistryError(
            f"{family.family_id}: declares no conflicting facts, so nothing can be tested"
        )

    expected_resolution = {
        "version_supersession": "prefer_current",
        "mutually_exclusive": "escalate_unresolved",
        "stricter_looser": "prefer_stricter_and_escalate",
        "compatible": "no_conflict",
    }[family.conflict_type]
    if family.resolution != expected_resolution:
        raise ConflictRegistryError(
            f"{family.family_id}: a {family.conflict_type} family resolves by "
            f"{expected_resolution!r}, not {family.resolution!r}"
        )

    if family.conflict_type == "version_supersession":
        if family.authoritative is None:
            raise ConflictRegistryError(
                f"{family.family_id}: a supersession family must name the authoritative document"
            )
    elif family.authoritative is not None:
        raise ConflictRegistryError(
            f"{family.family_id}: only a supersession family names an authoritative "
            "document; if one document really does take precedence, the conflict is "
            "resolvable and should be modelled as a supersession"
        )

    if family.conflict_type == "stricter_looser":
        # The whole point of this type is that the conservative reading is a
        # property of the family rather than a judgement made at answer time.
        # Without ``stricter`` there is no rule to score against.
        if not family.stricter:
            raise ConflictRegistryError(
                f"{family.family_id}: a stricter_looser family must name which document "
                "states the safe reading, or the expected behaviour is undefined"
            )
        if family.stricter not in family.documents:
            raise ConflictRegistryError(
                f"{family.family_id}: stricter document {family.stricter!r} is not one "
                f"of {list(family.documents)}"
            )
    elif family.stricter:
        raise ConflictRegistryError(
            f"{family.family_id}: only a stricter_looser family names a stricter "
            "document"
        )

    # The test that four misclassifications would have failed.
    if family.conflict_type == "stricter_looser":
        if not family.satisfying_action:
            raise ConflictRegistryError(
                f"{family.family_id}: a stricter_looser family must name the action "
                "that satisfies both documents. If no such action can be written, "
                "the family is mutually_exclusive."
            )
    elif family.conflict_type == "mutually_exclusive":
        if not family.no_satisfying_action_because:
            raise ConflictRegistryError(
                f"{family.family_id}: a mutually_exclusive family must say why no "
                "action satisfies both documents. Four families were typed this way "
                "on intuition and were wrong; being unable to write this sentence is "
                "the check that catches it."
            )
        if family.satisfying_action:
            raise ConflictRegistryError(
                f"{family.family_id}: mutually_exclusive but names an action that "
                "satisfies both, which is a contradiction in terms"
            )

    if family.conflict_type == "compatible":
        # A negative control with no explanation is indistinguishable from a
        # conflict someone gave up on. The reconciliation is what makes it
        # reviewable, and it must be discoverable in the corpus rather than
        # asserted here.
        if not family.reconciliation:
            raise ConflictRegistryError(
                f"{family.family_id}: a compatible family must explain why the apparent "
                "disagreement is not one, or it cannot be distinguished from an "
                "unresolved conflict"
            )
    elif family.reconciliation:
        raise ConflictRegistryError(
            f"{family.family_id}: only a compatible family carries a reconciliation"
        )


def _validate_family_against_corpus(family: ConflictFamily, kb: KnowledgeBase) -> None:
    for doc_id in family.documents:
        if not kb.has(doc_id):
            raise ConflictRegistryError(f"{family.family_id}: names unknown document {doc_id!r}")

    if family.conflict_type == "version_supersession":
        old_ids = [d for d in family.documents if d != family.authoritative]
        new = kb.by_id(family.authoritative)  # type: ignore[arg-type]
        if not new.is_current:
            raise ConflictRegistryError(
                f"{family.family_id}: authoritative document {new.doc_id} has status {new.status!r}"
            )
        for old_id in old_ids:
            old = kb.by_id(old_id)
            if old.status != "superseded":
                raise ConflictRegistryError(
                    f"{family.family_id}: {old_id} should be superseded but is {old.status!r}"
                )
            if old.superseded_by != new.doc_id:
                raise ConflictRegistryError(
                    f"{family.family_id}: {old_id} points at {old.superseded_by!r}, "
                    f"not {new.doc_id!r}"
                )
    else:
        for doc_id in family.documents:
            doc = kb.by_id(doc_id)
            if not doc.is_current:
                raise ConflictRegistryError(
                    f"{family.family_id}: {doc_id} is declared as a live conflicting document "
                    f"but its status is {doc.status!r}. A current_current conflict requires both "
                    "documents to be in force."
                )

    # Evidence anchors. Without these the registry asserts a conflict that may
    # no longer exist in the text, and gold answers would be written against a
    # fiction.
    for fact in family.conflicting_facts:
        if set(fact.values) != set(fact.anchors):
            raise ConflictRegistryError(
                f"{family.family_id}: fact {fact.fact!r} has values for {sorted(fact.values)} "
                f"but anchors for {sorted(fact.anchors)}"
            )
        for doc_id, anchor in fact.anchors.items():
            if doc_id not in family.documents:
                raise ConflictRegistryError(
                    f"{family.family_id}: fact {fact.fact!r} anchors {doc_id!r}, "
                    f"which is not one of {list(family.documents)}"
                )
            body = kb.by_id(doc_id).body
            if anchor not in body:
                raise ConflictRegistryError(
                    f"{family.family_id}: fact {fact.fact!r} claims {doc_id} contains "
                    f"{anchor!r}, but that text is not in the document. Either the document "
                    "changed or the registry is wrong."
                )
        if len({v.strip().lower() for v in fact.values.values()}) < 2:
            raise ConflictRegistryError(
                f"{family.family_id}: fact {fact.fact!r} states the same value for every "
                "document, so there is no conflict to detect"
            )


def validate_against_corpus(registry: ConflictRegistry, kb: KnowledgeBase) -> None:
    """Check the registry describes the corpus that is actually on disk.

    Catches:

    1. A family naming a document that does not exist
    2. A family whose documents do not carry the relationship claimed
    3. A declared conflicting fact whose evidence is not in the text
    4. A superseded document that no family accounts for
    5. A topic declared absent that has appeared in the corpus
    6. A partial gap whose near-miss evidence has gone, or whose missing
       detail has been filled in
    """
    seen_ids: set[str] = set()
    for family in registry.all_families:
        if family.family_id in seen_ids:
            raise ConflictRegistryError(f"Duplicate family id {family.family_id!r}")
        seen_ids.add(family.family_id)
        _validate_family_shape(family)
        _validate_family_against_corpus(family, kb)

    registered = {doc_id for family in registry.all_families for doc_id in family.documents}
    unaccounted = [doc.doc_id for doc in kb.superseded() if doc.doc_id not in registered]
    if unaccounted:
        raise ConflictRegistryError(
            "Superseded documents not declared in the conflict registry: "
            f"{sorted(unaccounted)}. Either register them as deliberate conflicts "
            "or remove them from the corpus."
        )

    corpus_text = "\n".join(doc.body for doc in kb).lower()

    for gap in registry.fully_absent:
        hits = [p for p in gap.probes if re.search(re.escape(p.lower()), corpus_text)]
        if hits:
            raise ConflictRegistryError(
                f"Topic {gap.topic!r} is declared absent but the corpus matches {hits}. "
                "Unanswerable test questions on this topic would no longer be unanswerable."
            )

    for partial in registry.partially_present:
        if not kb.has(partial.mentioned_in):
            raise ConflictRegistryError(
                f"Partial gap {partial.topic!r} cites unknown document {partial.mentioned_in!r}"
            )
        body = kb.by_id(partial.mentioned_in).body.lower()
        if partial.must_contain.lower() not in body:
            raise ConflictRegistryError(
                f"Partial gap {partial.topic!r} expects {partial.mentioned_in} to contain "
                f"{partial.must_contain!r}, but it does not. The near-miss this gap relies on "
                "has gone, and the question is now simply unanswerable rather than partial."
            )
        filled = [p for p in partial.missing_detail_probes if re.search(re.escape(p.lower()), corpus_text)]
        if filled:
            raise ConflictRegistryError(
                f"Partial gap {partial.topic!r} is no longer partial: the corpus now matches "
                f"{filled}, so the detail declared missing has been supplied somewhere."
            )
