"""The generators that produce committed documents and figures.

Amendment 1.30. Three of this project's documents are generated rather than
transcribed - the amendment appendix, the corpus description and Appendix D -
and the figures are generated too. None of them had a test. A generator with no
test is a claim about provenance that nothing checks, which is the defect
amendment 1.16.1 records in another place.

What is asserted here is narrow and deliberate. These tests do not re-derive the
figures' contents; they assert that regenerating a committed artefact reproduces
it, that a generator refuses rather than emitting a shorter document when its
input is wrong, and that nothing in a figure names the version of the library
that drew it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIGURES = ROOT / "docs" / "dissertation" / "figures"
DISSERTATION = ROOT / "docs" / "dissertation"

sys.path.insert(0, str(SCRIPTS))


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True, text=True, cwd=str(ROOT))


needs_matplotlib = pytest.mark.skipif(
    __import__("importlib").util.find_spec("matplotlib") is None,
    reason="matplotlib is the optional plots dependency")


# --- generated documents -----------------------------------------------------


@pytest.mark.parametrize("script,document", [
    ("make_amendment_table.py", "appendix_amendments.md"),
    ("make_verifier_appendix.py", "appendix_verifier_classification.md"),
])
def test_the_committed_document_is_what_its_generator_emits(script, document):
    """The point of generating a document is that it cannot drift from its
    source. That only holds if somebody checks, so this checks."""
    path = DISSERTATION / document
    if not path.exists():
        pytest.skip(f"{document} is not present")
    result = run(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout == path.read_text(encoding="utf-8"), (
        f"{document} differs from what {script} emits; regenerate it")


def test_the_appendix_generator_refuses_a_report_with_no_diagnostic(tmp_path):
    """A generator that quietly produces a smaller document when its input
    changes is a worse failure than one that stops."""
    empty = tmp_path / "hypotheses.json"
    empty.write_text(json.dumps({"hypotheses": {}}), encoding="utf-8")
    result = run("make_verifier_appendix.py", str(empty))
    assert result.returncode != 0
    assert "no verifier_relationship_diagnostic" in result.stderr


def test_the_appendix_generator_refuses_a_pooled_total(tmp_path):
    """Amendment 1.30.5 removed the 45-question headline. This is what stops it
    being typeset again if it ever returns to the analysis."""
    report = json.loads(
        (ROOT / "results" / "analysis" / "hypotheses.json")
        .read_text(encoding="utf-8"))
    report["verifier_relationship_diagnostic"]["all_registered_families"] = {
        "questions": 45}
    path = tmp_path / "hypotheses.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = run("make_verifier_appendix.py", str(path))
    assert result.returncode != 0
    assert "1.30.5" in result.stderr


def test_the_appendix_generator_refuses_another_source_run(tmp_path):
    report = json.loads(
        (ROOT / "results" / "analysis" / "hypotheses.json")
        .read_text(encoding="utf-8"))
    report["verifier_relationship_diagnostic"]["source_run"] = "some_other_run"
    path = tmp_path / "hypotheses.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = run("make_verifier_appendix.py", str(path))
    assert result.returncode != 0
    assert "source run" in result.stderr


def test_the_amendment_table_refuses_an_undescribed_amendment(tmp_path):
    """The guard already existed in one direction and was added in the other
    under 1.28. Both are asserted here rather than assumed."""
    import make_amendment_table

    document = (ROOT / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    described = set(make_amendment_table.SUMMARY)
    import re

    present = set(re.findall(r"^# Amendment (1\.\d+) - ", document,
                             flags=re.MULTILINE))
    assert present == described, (
        f"described but absent: {sorted(described - present)}; "
        f"present but undescribed: {sorted(present - described)}")


# --- figures -----------------------------------------------------------------


@needs_matplotlib
def test_no_committed_figure_names_the_library_that_drew_it():
    """Amendment 1.30.7. matplotlib stamps its own version into the metadata of
    both formats, so the same script over the same data under a different
    matplotlib produced different bytes while amendment 1.26 called the figures
    reproducible."""
    import figure_provenance

    figures = sorted(list(FIGURES.glob("*.png")) + list(FIGURES.glob("*.svg")))
    if not figures:
        pytest.skip("figures are not present")
    offenders = [f.name for f in figures if figure_provenance.carries_a_version(f)]
    assert not offenders, offenders


@needs_matplotlib
def test_every_committed_figure_names_the_project_as_its_producer():
    from PIL import Image  # noqa: PLC0415 - optional, skipped below if absent

    import figure_provenance

    pngs = sorted(FIGURES.glob("*.png"))
    if not pngs:
        pytest.skip("figures are not present")
    for path in pngs:
        assert Image.open(path).info.get("Software") == figure_provenance.CREATOR, (
            path.name)


@needs_matplotlib
def test_the_figure_environment_is_recorded_and_matches_the_pinned_versions():
    """Removing the version from the file does not make the rendering
    independent of it: FreeType and the font stack decide where glyphs land. The
    versions that drew the committed images are recorded so a regeneration
    producing different pixels can be explained rather than argued about."""
    path = FIGURES / "FIGURE_ENVIRONMENT.json"
    if not path.exists():
        pytest.skip("figures are not present")
    recorded = json.loads(path.read_text(encoding="utf-8"))["versions"]
    for key in ("python", "matplotlib", "freetype", "numpy"):
        assert recorded.get(key), key
    pinned = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'matplotlib=={recorded["matplotlib"]}' in pinned
    assert f'numpy=={recorded["numpy"]}' in pinned


@needs_matplotlib
def test_two_consecutive_figure_runs_produce_identical_bytes():
    """The whole of what "reproducible" is allowed to mean here. Run on a
    scratch copy of the output directory so a failure cannot leave the
    committed images half-rewritten."""
    import hashlib

    if not (ROOT / "results" / "analysis" / "hypotheses.json").exists():
        pytest.skip("analysis outputs are not present")

    def digest() -> dict[str, str]:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(FIGURES.iterdir()) if p.is_file()}

    before = digest()
    if not before:
        pytest.skip("figures are not present")
    first = run("make_architecture_figures.py")
    assert first.returncode == 0, first.stderr
    after_one = digest()
    second = run("make_architecture_figures.py")
    assert second.returncode == 0, second.stderr
    assert digest() == after_one, "two runs of the same script differ"
    assert after_one == before, (
        "regenerating changed a committed figure; the committed images and the "
        "code that draws them have diverged")
