"""Path resolution for the determinism check.

Trivial to get wrong and invisible until someone runs the script: the first
version globbed ``runs/*`` beneath a path that already ended in ``runs``, so it
matched nothing and reported "no run" on a repository full of runs. A script
that only fails in the operator's hands wastes their time rather than mine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_determinism import digest, latest_run  # noqa: E402


class FakeConfig:
    def __init__(self, root: Path) -> None:
        self._root = root

    def path(self, key: str) -> Path:
        assert key == "paths.results"
        return self._root


@pytest.fixture()
def runs(tmp_path):
    for name in ["20260813_105935_A_dev_pilot02",
                 "20260813_110154_D_dev_pilot02",
                 "20260813_231710_D_dev_pilot03"]:
        (tmp_path / name).mkdir()
    (tmp_path / "latest_dev.json").write_text("{}", encoding="utf-8")
    return FakeConfig(tmp_path)


def test_it_finds_the_most_recent_matching_run(runs):
    assert latest_run(runs, "*_D_dev_*").name == "20260813_231710_D_dev_pilot03"
    assert latest_run(runs, "*_A_dev_*").name == "20260813_105935_A_dev_pilot02"


def test_a_file_is_not_a_run_directory(runs):
    """latest_dev.json sits alongside the runs and is not one.

    A plain glob would return it as the most recent match and the script would
    then fail deeper in, reading a manifest out of a file.
    """
    assert latest_run(runs, "*").is_dir()
    with pytest.raises(SystemExit):
        latest_run(runs, "latest_*.json")


def test_no_match_lists_what_is_there(runs):
    """"No run matching" against a directory full of runs is a useless error."""
    with pytest.raises(SystemExit) as excinfo:
        latest_run(runs, "*_Z_test_*")
    message = str(excinfo.value)
    assert "Available:" in message
    assert "20260813_231710_D_dev_pilot03" in message
    assert "--run" in message


def test_an_empty_results_directory_says_so(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        latest_run(FakeConfig(tmp_path), "*_D_dev_*")
    assert "(none)" in str(excinfo.value)


def test_digest_is_stable_and_short():
    assert digest("hello") == digest("hello")
    assert digest("hello") != digest("world")
    assert len(digest("hello")) == 12
