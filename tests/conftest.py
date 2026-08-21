"""Shared fixtures.

Amendment 1.31.2 made the frozen quality runs content-authenticated: the
analysis and the replay refuse a run whose parsed content does not match a
digest recorded in ``sme_assistant.evaluation.authenticity``. That is the point
of it, and it means a stub run built in a temporary directory is refused before
any other check is reached.

Tests that exercise the checks *behind* authentication therefore have to record
a digest for what they wrote. The autouse fixture below snapshots the digest
table around every test and restores it afterwards, so a stub can seal itself
without leaking into the next test, and so no production code needs a
test-only escape hatch. A bypass flag in the library would be a hole in exactly
the wall this amendment builds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sme_assistant.evaluation.authenticity import (FROZEN_PERFORMANCE_DIGESTS,
                                                   FROZEN_PERFORMANCE_INDEX_DIGESTS,
                                                   FROZEN_RUN_DIGESTS)


@pytest.fixture(autouse=True)
def isolate_run_digests():
    """Restore every recorded digest table after each test, whatever it did.

    All three: the quality runs, the performance runs and the performance
    run-index files. Amendment 1.32.3 added the second and third, and a fixture
    restoring only the first would let a sealed stub leak into the next test.
    """
    tables = (FROZEN_RUN_DIGESTS, FROZEN_PERFORMANCE_DIGESTS,
              FROZEN_PERFORMANCE_INDEX_DIGESTS)
    originals = [dict(table) for table in tables]
    yield FROZEN_RUN_DIGESTS
    for table, original in zip(tables, originals):
        table.clear()
        table.update(original)
