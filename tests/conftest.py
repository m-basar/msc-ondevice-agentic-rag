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

from sme_assistant.evaluation.authenticity import (FROZEN_RUN_DIGESTS,
                                                   RunDigest, digest_of,
                                                   read_run_content)


@pytest.fixture(autouse=True)
def isolate_run_digests():
    """Restore the recorded digests after every test, whatever it did to them."""
    original = dict(FROZEN_RUN_DIGESTS)
    yield FROZEN_RUN_DIGESTS
    FROZEN_RUN_DIGESTS.clear()
    FROZEN_RUN_DIGESTS.update(original)
