"""Knowledge base loading and validation.

The knowledge base is a directory of Markdown files, each carrying a small
block of front matter. Documents are static, version-controlled files rather
than the output of a generator script: they are research materials, and a
reviewer must be able to read exactly what the system was given.

Front matter is parsed with a deliberately restricted parser (flat
``key: value`` pairs only) so that the project keeps zero runtime
dependencies. Anything richer would mean adding PyYAML, which has to be
compiled or wheel-matched on ARM and buys nothing here.

Semantics of the date fields
----------------------------
``effective_date`` is the date on which *this version* of the document took
effect. It is not the date the document identifier was created. Documents
cross-reference each other by identifier, and identifiers persist across
versions, so a document may legitimately reference an identifier whose
current version carries a later effective date. ``withdrawn`` records when a
superseded version ceased to apply.

Fingerprinting
--------------
Every document is hashed, and the corpus as a whole carries a combined
fingerprint. Results files record both the configuration fingerprint and the
corpus fingerprint, so a table in Chapter 4 can be tied to the exact inputs
that produced it. Without the corpus hash, editing a document would silently
change results while the configuration hash stayed the same.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

FRONT_MATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
DOC_ID_PATTERN = re.compile(r"\A[A-Z]{2,4}-\d{2}\Z")

REQUIRED_FIELDS = ("id", "title", "category", "version", "effective_date", "status")
DATE_FIELDS = ("effective_date", "withdrawn")
VALID_STATUSES = {"current", "superseded"}


class KnowledgeBaseError(RuntimeError):
    """Raised when a document is malformed or the corpus is inconsistent."""


@dataclass(frozen=True)
class Document:
    """A single knowledge base document.

    Frozen because documents are read-only inputs to the experiment. If a
    document could be mutated after loading, a run would no longer be
    reproducible from the files on disk. ``metadata`` is wrapped in a
    read-only mapping for the same reason: ``frozen=True`` prevents rebinding
    an attribute but does nothing to stop a caller mutating a dict behind it.
    """

    doc_id: str
    title: str
    category: str
    version: str
    effective_date: date
    status: str
    body: str
    path: Path
    content_hash: str
    metadata: Mapping[str, str]

    @property
    def word_count(self) -> int:
        return len(self.body.split())

    @property
    def is_current(self) -> bool:
        return self.status == "current"

    @property
    def citation(self) -> str:
        """Short label used when the assistant cites this document."""
        return f"{self.doc_id} {self.title} (v{self.version})"

    @property
    def superseded_by(self) -> str | None:
        return self.metadata.get("superseded_by")

    @property
    def supersedes(self) -> str | None:
        return self.metadata.get("supersedes")

    @property
    def withdrawn(self) -> date | None:
        raw = self.metadata.get("withdrawn")
        return date.fromisoformat(raw) if raw else None


def parse_front_matter(text: str, source: Any) -> tuple[dict[str, str], str]:
    """Split a document into its front matter mapping and its body.

    Returns ``(metadata, body)``. Raises if the front matter block is absent,
    because a document without an identifier cannot be cited, and an answer
    the assistant cannot cite is worthless for this project.

    Duplicate keys are rejected rather than silently overwritten. A document
    carrying two ``status`` lines is ambiguous, and silently taking the last
    one would hide the mistake until it corrupted a result.
    """
    match = FRONT_MATTER_PATTERN.match(text)
    if match is None:
        raise KnowledgeBaseError(f"{source}: missing front matter block")

    metadata: dict[str, str] = {}
    for line_number, raw_line in enumerate(match.group(1).splitlines(), start=2):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise KnowledgeBaseError(
                f"{source}:{line_number}: front matter line is not 'key: value' -> {line!r}"
            )
        key, _, value = line.partition(":")
        key = key.strip()
        if key in metadata:
            raise KnowledgeBaseError(
                f"{source}:{line_number}: duplicate front matter key {key!r}"
            )
        metadata[key] = value.strip()

    body = text[match.end():].strip()
    return metadata, body


def load_document(path: Path) -> Document:
    """Load and validate one document."""
    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    metadata, body = parse_front_matter(raw.decode("utf-8"), path)

    missing = [name for name in REQUIRED_FIELDS if name not in metadata]
    if missing:
        raise KnowledgeBaseError(f"{path}: missing front matter fields {missing}")

    if not body:
        raise KnowledgeBaseError(f"{path}: document body is empty")

    doc_id = metadata["id"]
    if not DOC_ID_PATTERN.match(doc_id):
        raise KnowledgeBaseError(
            f"{path}: document id {doc_id!r} must match CATEGORY-NN, "
            "two to four capitals, hyphen, two digits"
        )

    status = metadata["status"]
    if status not in VALID_STATUSES:
        raise KnowledgeBaseError(
            f"{path}: status {status!r} must be one of {sorted(VALID_STATUSES)}"
        )

    parsed_dates: dict[str, date] = {}
    for name in DATE_FIELDS:
        if name not in metadata:
            continue
        try:
            parsed_dates[name] = date.fromisoformat(metadata[name])
        except ValueError as exc:
            raise KnowledgeBaseError(
                f"{path}: {name} must be ISO format YYYY-MM-DD, got {metadata[name]!r}"
            ) from exc

    if status == "current" and "withdrawn" in parsed_dates:
        raise KnowledgeBaseError(f"{path}: a current document cannot have a withdrawn date")

    return Document(
        doc_id=doc_id,
        title=metadata["title"],
        category=metadata["category"],
        version=metadata["version"],
        effective_date=parsed_dates["effective_date"],
        status=status,
        body=body,
        path=path,
        content_hash=content_hash,
        metadata=MappingProxyType(dict(metadata)),
    )


class KnowledgeBase:
    """The loaded corpus, with the integrity checks the experiment relies on."""

    def __init__(self, documents: list[Document], root: Path) -> None:
        self.documents: tuple[Document, ...] = tuple(documents)
        self.root = root
        self._by_id = MappingProxyType({doc.doc_id: doc for doc in documents})

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def by_id(self, doc_id: str) -> Document:
        if doc_id not in self._by_id:
            raise KnowledgeBaseError(f"No document with id {doc_id!r}")
        return self._by_id[doc_id]

    def has(self, doc_id: str) -> bool:
        return doc_id in self._by_id

    def current(self) -> list[Document]:
        return [doc for doc in self.documents if doc.is_current]

    def superseded(self) -> list[Document]:
        return [doc for doc in self.documents if not doc.is_current]

    def fingerprint(self) -> str:
        """Stable hash of the whole corpus.

        Built from the per-document hashes rather than from a directory walk,
        so it changes if and only if document content changes, and does not
        depend on file modification times or on the order files are read.
        """
        canonical = "\n".join(
            f"{doc.doc_id}:{doc.content_hash}" for doc in sorted(self.documents, key=lambda d: d.doc_id)
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def manifest(self) -> dict[str, Any]:
        """Full provenance record, written into every results file."""
        return {
            "corpus_fingerprint": self.fingerprint(),
            "document_count": len(self.documents),
            "documents": [
                {
                    "id": doc.doc_id,
                    "title": doc.title,
                    "version": doc.version,
                    "status": doc.status,
                    "effective_date": doc.effective_date.isoformat(),
                    "words": doc.word_count,
                    "sha256": doc.content_hash,
                }
                for doc in sorted(self.documents, key=lambda d: d.doc_id)
            ],
        }

    def summary(self) -> dict[str, Any]:
        """Corpus statistics, for the knowledge base description in Chapter 3.

        Reported rather than estimated: the numbers in the methodology chapter
        should be generated by this method, not counted by hand.
        """
        words = [doc.word_count for doc in self.documents]
        return {
            "document_count": len(self.documents),
            "current_count": len(self.current()),
            "superseded_count": len(self.superseded()),
            "superseded_share": round(len(self.superseded()) / len(self.documents), 3),
            "categories": dict(sorted(Counter(d.category for d in self.documents).items())),
            "total_words": sum(words),
            "mean_words": round(sum(words) / len(words), 1) if words else 0,
            "min_words": min(words) if words else 0,
            "max_words": max(words) if words else 0,
            "corpus_fingerprint": self.fingerprint(),
        }


def _validate_identifiers(documents: list[Document]) -> dict[str, Path]:
    seen: dict[str, Path] = {}
    for doc in documents:
        if doc.doc_id in seen:
            raise KnowledgeBaseError(
                f"Duplicate document id {doc.doc_id!r} in {doc.path} and {seen[doc.doc_id]}"
            )
        seen[doc.doc_id] = doc.path
    return seen


def _validate_supersession(documents: list[Document], known: set[str]) -> None:
    """Check supersession links point somewhere, agree with each other, and terminate."""
    by_id = {doc.doc_id: doc for doc in documents}

    for doc in documents:
        target = doc.superseded_by
        if target and target not in known:
            raise KnowledgeBaseError(
                f"{doc.path}: superseded_by points at unknown document {target!r}"
            )
        if doc.status == "superseded" and not target:
            raise KnowledgeBaseError(
                f"{doc.path}: status is 'superseded' but superseded_by is not set"
            )
        if target:
            replacement = by_id[target]
            # The replacement must acknowledge the link, or the pair is only
            # half-wired and a version-aware retriever cannot walk it backwards.
            if replacement.supersedes != doc.doc_id:
                raise KnowledgeBaseError(
                    f"{doc.path}: {doc.doc_id} claims to be superseded by {target}, "
                    f"but {target} does not declare 'supersedes: {doc.doc_id}'"
                )
            if replacement.effective_date <= doc.effective_date:
                raise KnowledgeBaseError(
                    f"{doc.path}: replacement {target} is not effective later "
                    f"({replacement.effective_date} <= {doc.effective_date})"
                )

    # Walk each chain to a terminus, refusing to loop forever on a cycle.
    for doc in documents:
        seen_chain = [doc.doc_id]
        cursor = doc
        while cursor.superseded_by:
            nxt = cursor.superseded_by
            if nxt in seen_chain:
                raise KnowledgeBaseError(
                    f"Supersession cycle detected: {' -> '.join(seen_chain + [nxt])}"
                )
            seen_chain.append(nxt)
            cursor = by_id[nxt]


def _validate_cross_references(documents: list[Document], known: set[str]) -> None:
    """Every document identifier mentioned in a body must resolve.

    These documents cite each other constantly. A reference to a document that
    does not exist sends retrieval down a dead end and, worse, invites the
    model to invent the contents of the missing document.

    The pattern is built from the identifier prefixes actually present in the
    corpus, so unrelated codes in the text such as RIDDOR 2013 or LOLER 1998
    are not mistaken for references.
    """
    prefixes = sorted({doc.doc_id.split("-")[0] for doc in documents}, key=len, reverse=True)
    if not prefixes:
        return
    pattern = re.compile(r"\b(?:" + "|".join(prefixes) + r")-\d{2}\b")

    for doc in documents:
        for reference in sorted(set(pattern.findall(doc.body))):
            if reference == doc.doc_id:
                continue
            if reference not in known:
                raise KnowledgeBaseError(
                    f"{doc.path}: body references unknown document {reference!r}"
                )


def load_knowledge_base(directory: Path | str) -> KnowledgeBase:
    """Load every Markdown document in ``directory`` and validate the corpus.

    Corpus-level checks run here rather than per document because they are
    about relationships between files. A single document cannot know whether
    its identifier is unique, whether the document it points at exists, or
    whether it sits in a supersession cycle.
    """
    root = Path(directory)
    if not root.is_dir():
        raise KnowledgeBaseError(f"Knowledge base directory not found: {root}")

    paths = sorted(root.glob("*.md"))
    if not paths:
        raise KnowledgeBaseError(f"No Markdown documents found in {root}")

    documents = [load_document(path) for path in paths]

    known = set(_validate_identifiers(documents))
    _validate_supersession(documents, known)
    _validate_cross_references(documents, known)

    return KnowledgeBase(documents, root)
