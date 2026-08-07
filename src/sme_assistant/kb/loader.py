"""Knowledge base loading and validation.

The knowledge base is a directory of Markdown files, each carrying a small
block of front matter. Documents are static, version-controlled files rather
than the output of a generator script: they are research materials, and a
reviewer must be able to read exactly what the system was given.

Front matter is parsed with a deliberately restricted parser (flat
``key: value`` pairs only) so that the project keeps zero runtime
dependencies. Anything richer would mean adding PyYAML, which has to be
compiled or wheel-matched on ARM and buys nothing here.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator

FRONT_MATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)

REQUIRED_FIELDS = ("id", "title", "category", "version", "effective_date", "status")
VALID_STATUSES = {"current", "superseded"}


class KnowledgeBaseError(RuntimeError):
    """Raised when a document is malformed or the corpus is inconsistent."""


@dataclass(frozen=True)
class Document:
    """A single knowledge base document.

    Frozen because documents are read-only inputs to the experiment. If a
    document could be mutated after loading, a run would no longer be
    reproducible from the files on disk.
    """

    doc_id: str
    title: str
    category: str
    version: str
    effective_date: date
    status: str
    body: str
    path: Path
    metadata: dict[str, str]

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


def parse_front_matter(text: str, source: Path) -> tuple[dict[str, str], str]:
    """Split a document into its front matter mapping and its body.

    Returns ``(metadata, body)``. Raises if the front matter block is absent,
    because a document without an identifier cannot be cited, and an answer
    the assistant cannot cite is worthless for this project.
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
        metadata[key.strip()] = value.strip()

    body = text[match.end():].strip()
    return metadata, body


def load_document(path: Path) -> Document:
    """Load and validate one document."""
    text = path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(text, path)

    missing = [field for field in REQUIRED_FIELDS if field not in metadata]
    if missing:
        raise KnowledgeBaseError(f"{path}: missing front matter fields {missing}")

    if not body:
        raise KnowledgeBaseError(f"{path}: document body is empty")

    status = metadata["status"]
    if status not in VALID_STATUSES:
        raise KnowledgeBaseError(
            f"{path}: status {status!r} must be one of {sorted(VALID_STATUSES)}"
        )

    try:
        effective = date.fromisoformat(metadata["effective_date"])
    except ValueError as exc:
        raise KnowledgeBaseError(
            f"{path}: effective_date must be ISO format YYYY-MM-DD, got "
            f"{metadata['effective_date']!r}"
        ) from exc

    return Document(
        doc_id=metadata["id"],
        title=metadata["title"],
        category=metadata["category"],
        version=metadata["version"],
        effective_date=effective,
        status=status,
        body=body,
        path=path,
        metadata=metadata,
    )


class KnowledgeBase:
    """The loaded corpus, with the integrity checks the experiment relies on."""

    def __init__(self, documents: list[Document], root: Path) -> None:
        self.documents = documents
        self.root = root
        self._by_id = {doc.doc_id: doc for doc in documents}

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def by_id(self, doc_id: str) -> Document:
        if doc_id not in self._by_id:
            raise KnowledgeBaseError(f"No document with id {doc_id!r}")
        return self._by_id[doc_id]

    def current(self) -> list[Document]:
        return [doc for doc in self.documents if doc.is_current]

    def superseded(self) -> list[Document]:
        return [doc for doc in self.documents if not doc.is_current]

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
            "categories": dict(sorted(Counter(d.category for d in self.documents).items())),
            "total_words": sum(words),
            "mean_words": round(sum(words) / len(words), 1) if words else 0,
            "min_words": min(words) if words else 0,
            "max_words": max(words) if words else 0,
        }


def load_knowledge_base(directory: Path | str) -> KnowledgeBase:
    """Load every Markdown document in ``directory`` and validate the corpus.

    Corpus-level checks run here rather than per document, because they are
    about relationships between files: duplicate identifiers would make
    citations ambiguous, and a ``superseded_by`` pointer to a document that
    does not exist would break the conflicting-evidence test cases.
    """
    root = Path(directory)
    if not root.is_dir():
        raise KnowledgeBaseError(f"Knowledge base directory not found: {root}")

    paths = sorted(root.glob("*.md"))
    if not paths:
        raise KnowledgeBaseError(f"No Markdown documents found in {root}")

    documents = [load_document(path) for path in paths]

    seen: dict[str, Path] = {}
    for doc in documents:
        if doc.doc_id in seen:
            raise KnowledgeBaseError(
                f"Duplicate document id {doc.doc_id!r} in {doc.path} and {seen[doc.doc_id]}"
            )
        seen[doc.doc_id] = doc.path

    known_ids = set(seen)
    for doc in documents:
        target = doc.superseded_by
        if target and target not in known_ids:
            raise KnowledgeBaseError(
                f"{doc.path}: superseded_by points at unknown document {target!r}"
            )
        if doc.status == "superseded" and not target:
            raise KnowledgeBaseError(
                f"{doc.path}: status is 'superseded' but superseded_by is not set"
            )

    return KnowledgeBase(documents, root)
