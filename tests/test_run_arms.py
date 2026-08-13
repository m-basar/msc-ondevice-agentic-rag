"""Tests for the arm harness.

The arms must differ in exactly the three respects the pre-registration names
and in nothing else. A harness that quietly varied a fourth would make every
comparison uninterpretable, and the difference would not show up in any answer.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_arms import ARMS, arm_definition  # noqa: E402

from sme_assistant.common.config import load_config
from sme_assistant.retrieve.retriever import EvidenceFormat, RetrievalMode


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_the_arms_match_the_pre_registered_definitions(config):
    expected = {
        "A": (RetrievalMode.ALL, EvidenceFormat.PLAIN, False),
        "B": (RetrievalMode.ALL, EvidenceFormat.WITH_STATUS, False),
        "C": (RetrievalMode.CURRENT_ONLY, EvidenceFormat.WITH_STATUS, False),
        "D": (RetrievalMode.ALL, EvidenceFormat.WITH_STATUS, True),
    }
    assert set(ARMS) == set(expected)
    for name, (mode, fmt, verify) in expected.items():
        spec = ARMS[name]
        assert spec["retrieval_mode"] is mode
        assert spec["evidence_format"] is fmt
        assert spec["verification"] is verify


def test_b_and_d_differ_only_in_verification():
    """B versus D is the confirmatory contrast, so it must isolate one thing."""
    b, d = ARMS["B"], ARMS["D"]
    assert b["retrieval_mode"] is d["retrieval_mode"]
    assert b["evidence_format"] is d["evidence_format"]
    assert b["verification"] is not d["verification"]


def test_only_arm_d_records_a_verification_model(config):
    """A verification model on an unverified arm would suggest one ran."""
    assert arm_definition("D", config).verification_model
    for name in ("A", "B", "C"):
        assert arm_definition(name, config).verification_model is None


def test_the_test_split_cannot_be_run_by_accident():
    """A test run produces the reported numbers.

    It should not be startable by editing a default, which is why the flag is
    deliberately awkward rather than a plain --split test.
    """
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/run_arms.py", "--split", "test", "--mock"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "frozen" in result.stderr.lower()
