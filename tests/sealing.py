"""Record a content digest for a stub run, so it authenticates.

Amendment 1.31.2. The analysis and the replay refuse a run whose parsed content
does not match a digest recorded in ``sme_assistant.evaluation.authenticity``.
That is the point of it, and it means a stub built in a temporary directory is
refused before any other check is reached.

Tests exercising the checks *behind* authentication therefore record a digest
for what they wrote. The autouse fixture in ``conftest`` restores the table
after every test, so this leaks nothing, and no production code needs a
test-only escape hatch: a bypass flag in the library would be a hole in exactly
the wall this amendment builds.
"""

from __future__ import annotations

from pathlib import Path

from sme_assistant.evaluation.authenticity import (FROZEN_PERFORMANCE_DIGESTS,
                                                   FROZEN_PERFORMANCE_INDEX_DIGESTS,
                                                   FROZEN_RUN_DIGESTS,
                                                   RunDigest, digest_of,
                                                   read_run_content,
                                                   read_summary)


def seal(directory: Path, name: str | None = None, *,
         performance: bool = False) -> RunDigest:
    """Record a stub run's digests. ``performance`` picks the timing table."""
    directory = Path(directory)
    records, manifest = read_run_content(directory)
    summary = read_summary(directory)
    digest = RunDigest(answers=digest_of(list(records)),
                       manifest=digest_of(manifest),
                       summary=None if summary is None else digest_of(summary))
    table = FROZEN_PERFORMANCE_DIGESTS if performance else FROZEN_RUN_DIGESTS
    table[name or directory.name] = digest
    return digest


def seal_index(path: Path) -> str:
    """Record a stub performance run-index file's digest."""
    import json

    path = Path(path)
    digest = digest_of(json.loads(path.read_text(encoding="utf-8")))
    FROZEN_PERFORMANCE_INDEX_DIGESTS[path.name] = digest
    return digest
