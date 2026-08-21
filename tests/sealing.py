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

from sme_assistant.evaluation.authenticity import (FROZEN_RUN_DIGESTS,
                                                   RunDigest, digest_of,
                                                   read_run_content)


def seal(directory: Path, name: str | None = None) -> RunDigest:
    directory = Path(directory)
    records, manifest = read_run_content(directory)
    digest = RunDigest(answers=digest_of(list(records)),
                       manifest=digest_of(manifest))
    FROZEN_RUN_DIGESTS[name or directory.name] = digest
    return digest
