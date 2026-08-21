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
    ("compare_index_architectures.py", "appendix_index_architectures.md"),
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
    # Pillow writes the PNG bytes. Amendment 1.31.4: it decides the file as
    # surely as FreeType decides the glyphs, and it was neither recorded nor
    # pinned while this project called the figures reproducible.
    for key in ("python", "matplotlib", "freetype", "numpy", "pillow"):
        assert recorded.get(key), key
    pinned = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package in ("matplotlib", "numpy", "pillow"):
        assert f'{package}=={recorded[package]}' in pinned, package


def _current_environment() -> dict[str, str]:
    import figure_provenance

    return figure_provenance.environment()


def _recorded_environment() -> dict[str, str] | None:
    path = FIGURES / "FIGURE_ENVIRONMENT.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["versions"]


@needs_matplotlib
def test_two_consecutive_figure_runs_produce_identical_bytes(tmp_path):
    """The whole of what "reproducible" is allowed to mean without qualification.

    Amendment 1.30.12. The first version of this test regenerated into the
    committed directory while its own docstring said it used a scratch copy, and
    on a machine with different font metrics it overwrote four committed figures.
    That is the defect 1.30.1 is about, committed inside the amendment that
    exists to correct it. ``--out`` now makes the isolation real.

    What is asserted here is machine-local determinism: the same script, on this
    machine, twice, byte for byte. Cross-machine identity is asserted separately
    below and only where the environment matches, because it is not a property
    this design has.
    """
    import hashlib

    if not (ROOT / "results" / "analysis" / "hypotheses.json").exists():
        pytest.skip("analysis outputs are not present")

    def digest(directory: Path) -> dict[str, str]:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(directory.iterdir()) if p.is_file()}

    committed_before = digest(FIGURES) if FIGURES.exists() else {}
    out = tmp_path / "figures"
    for script in ("make_figures.py", "make_architecture_figures.py"):
        result = run(script, "--out", str(out))
        assert result.returncode == 0, result.stderr
    first = digest(out)
    assert first, "the scripts wrote nothing to the output directory"
    for script in ("make_figures.py", "make_architecture_figures.py"):
        result = run(script, "--out", str(out))
        assert result.returncode == 0, result.stderr
    assert digest(out) == first, "two runs of the same scripts differ"

    # The committed directory must be exactly as it was. This is the assertion
    # the previous version claimed and did not make.
    assert digest(FIGURES) == committed_before, (
        "running the figure scripts with --out touched the committed figures")


@needs_matplotlib
def test_the_committed_figures_match_this_machine_when_the_environment_does(tmp_path):
    """Cross-machine byte identity is conditional, and the condition is checked.

    FreeType and the font stack decide where glyphs land. Regenerating on a
    machine whose versions differ from the recorded ones legitimately produces
    different text positions in the SVG and different anti-aliased pixels in the
    PNG, with identical geometry. Asserting byte identity unconditionally
    contradicted `FIGURE_ENVIRONMENT.json`, which says in as many words that a
    regeneration elsewhere may differ.

    So: where the environment matches, a difference is a real divergence between
    the committed images and the code that draws them, and this fails. Where it
    does not match, this skips and names the versions, which is information a
    reader can act on rather than a red failure they must learn to ignore.
    """
    import hashlib

    recorded = _recorded_environment()
    if recorded is None or not (ROOT / "results" / "analysis" / "hypotheses.json").exists():
        pytest.skip("figures or analysis outputs are not present")
    current = _current_environment()
    relevant = ("matplotlib", "freetype", "pillow")
    mismatch = {k: (recorded.get(k), current.get(k))
                for k in relevant if recorded.get(k) != current.get(k)}
    if mismatch:
        pytest.skip(
            "this machine did not draw the committed figures: "
            + ", ".join(f"{k} recorded {r!r}, here {c!r}"
                        for k, (r, c) in mismatch.items())
            + ". Regenerate deliberately if you want them redrawn here, and "
            "commit FIGURE_ENVIRONMENT.json with them.")

    out = tmp_path / "figures"
    for script in ("make_figures.py", "make_architecture_figures.py"):
        result = run(script, "--out", str(out))
        assert result.returncode == 0, result.stderr
    for path in sorted(out.iterdir()):
        committed = FIGURES / path.name
        assert committed.exists(), f"{path.name} is generated but not committed"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            hashlib.sha256(committed.read_bytes()).hexdigest(), (
                f"{path.name} differs from the committed image on a machine "
                "whose recorded environment matches; the images and the code "
                "that draws them have diverged")


@needs_matplotlib
def test_the_figure_scripts_accept_an_output_directory():
    """Enforced over the source as well as by use, because the isolation above
    is only as good as the flag existing in both scripts."""
    for script in ("make_figures.py", "make_architecture_figures.py"):
        source = (SCRIPTS / script).read_text(encoding="utf-8")
        assert '"--out"' in source, script
        assert "parser.parse_args(argv)" in source, (
            f"{script} parses an empty list, so --out on the command line is "
            "silently ignored")


@needs_matplotlib
@pytest.mark.parametrize("script,forbidden", [
    ("make_figures.py", "fig.text("),
    ("make_architecture_figures.py", "C against D changes retrieval mode"),
])
def test_no_figure_carries_an_explanatory_footer(script, forbidden):
    """Amendment 1.31.4. Three figures carried paragraphs restating what the
    captions and the chapter already said. ``make_architecture_figures.py`` had
    said in its own docstring since it was written that neither of its figures
    carried a footer, and one of them did.

    A figure that argues its own case duplicates the prose it sits beside and
    cannot be reused anywhere that prose does not follow. The argument belongs
    in the chapter, which is the thing a reader can hold to it.
    """
    source = (SCRIPTS / script).read_text(encoding="utf-8")
    assert forbidden not in source, (
        f"{script} draws an explanatory footer again")


@needs_matplotlib
def test_the_committed_figures_carry_no_footer_text():
    """Over the SVG, which is the one format that says what its text is."""
    svgs = sorted(FIGURES.glob("*.svg"))
    if not svgs:
        pytest.skip("figures are not present")
    banned = ("no difference between them can be attributed",
              "explicitly not as an ablation",
              "no confidence interval is computed anywhere")
    for path in svgs:
        text = path.read_text(encoding="utf-8")
        for phrase in banned:
            assert phrase not in text, (path.name, phrase)


# --- appendix E, the cross-architecture comparison ---------------------------


def test_the_cross_architecture_appendix_declares_itself_exploratory():
    """Amendment 1.32.2. It was raised after every hypothesis was decided, and a
    post-hoc measurement that does not say so reads as a registered one."""
    path = DISSERTATION / "appendix_index_architectures.md"
    if not path.exists():
        pytest.skip("appendix E is not present")
    text = " ".join(path.read_text(encoding="utf-8").lower().split())
    for phrase in ("post-hoc and exploratory",
                   "no threshold is applied, no verdict is reached and no "
                   "hypothesis is revisited",
                   "h1 to h4 are unaffected"):
        assert phrase in text, phrase


def test_the_cross_architecture_appendix_preserves_both_index_hashes():
    """Neither build is the wrong one. Both are legitimate, both are recorded,
    and a reader has to be able to tell which machine produced which."""
    path = DISSERTATION / "appendix_index_architectures.md"
    if not path.exists():
        pytest.skip("appendix E is not present")
    text = path.read_text(encoding="utf-8")
    laptop = json.loads(
        (ROOT / "results" / "runs" / "20260814_055018_D_test" / "manifest.json")
        .read_text(encoding="utf-8"))["provenance"]["index_file_sha256"]
    pi = json.loads(
        (ROOT / "results" / "runs" / "20260815_040341_D_test_perf_pi5" /
         "manifest.json").read_text(encoding="utf-8"))["provenance"]["index_file_sha256"]
    assert laptop != pi
    assert laptop[:16] in text, "the laptop index hash is not shown"
    assert pi[:16] in text, "the Raspberry Pi index hash is not shown"


def test_appendix_e_makes_no_claim_its_generator_cannot_reproduce():
    """Amendment 1.32.6. The first draft asserted that the chunk records were
    byte-identical, that the vectors differed in the third and fourth decimal
    place, and that nomic-embed-text is not bit-reproducible between x86-64 and
    ARM64. The generator reads run records and manifests. It opens neither index
    file and compares no vector, so none of those three was reproducible from
    the stated inputs, however true they may be."""
    path = DISSERTATION / "appendix_index_architectures.md"
    if not path.exists():
        pytest.skip("appendix E is not present")
    text = " ".join(path.read_text(encoding="utf-8").lower().split())
    # The claims, not the words. "byte-identical" appears legitimately in the
    # sentence denying it, and a check that cannot tell an assertion from its
    # negation would force the denial out of the document.
    for phrase in ("chunk records are byte-identical",
                   "records themselves are byte-identical",
                   "decimal place",
                   "bit-reproducible",
                   "isolates the index build"):
        assert phrase not in text, f"unsupported claim returned: {phrase!r}"
    assert "the two serialised index files differ" in text
    assert "does not establish that the serialised records are byte-identical" in text
    assert "not the stored index vectors alone" in text


def test_appendix_e_reports_only_the_independent_pair():
    """Amendment 1.32.6. Arm D reuses Arm B's retrieval verbatim, so an Arm D
    row would be a copy of Arm B's reported as a second observation."""
    path = DISSERTATION / "appendix_index_architectures.md"
    if not path.exists():
        pytest.skip("appendix E is not present")
    text = path.read_text(encoding="utf-8")
    rows = [l for l in text.splitlines()
            if l.startswith("| Arm ") and "|" in l[6:] and "Questions" not in l]
    assert len(rows) == 1, f"expected one comparison row, found {rows}"
    assert "Arm B, laptop against Raspberry Pi 5" in rows[0]
    assert "drafts_reused_from: B" in text


def test_appendix_e_records_the_configuration_difference_it_found():
    """The two runs are not from byte-identical configurations, and saying only
    the index differed would be false."""
    path = DISSERTATION / "appendix_index_architectures.md"
    if not path.exists():
        pytest.skip("appendix E is not present")
    text = path.read_text(encoding="utf-8")
    assert "legitimately differ and are" in text
    assert "config_sha256" in text
    assert "does not claim the index was the" in text


def test_the_cross_architecture_generator_refuses_a_broken_shared_condition():
    """Every condition the comparison depends on is checked, not described."""
    import compare_index_architectures as cmp_arch

    original = cmp_arch.load

    def wrong(name):
        records, manifest, digests = original(name)
        if name == cmp_arch.PI:
            manifest = json.loads(json.dumps(manifest))
            manifest["provenance"]["index_metadata"]["embedding_model"] = "other"
        return records, manifest, digests

    cmp_arch.load = wrong
    try:
        with pytest.raises(SystemExit, match="index embedding_model"):
            cmp_arch.compare()
    finally:
        cmp_arch.load = original


def test_the_cross_architecture_generator_refuses_if_arm_d_stops_reusing_arm_b():
    """The reason Arm D is excluded is asserted, so the exclusion cannot outlive
    the fact that justifies it."""
    import compare_index_architectures as cmp_arch

    original = cmp_arch.load

    def wrong(name):
        records, manifest, digests = original(name)
        if name == "20260814_055018_D_test":
            records = [dict(r) for r in records]
            records[0] = {**records[0], "retrieval": {"results": []}}
        return tuple(records), manifest, digests

    cmp_arch.load = wrong
    try:
        with pytest.raises(SystemExit, match="does not reuse"):
            cmp_arch.reuse_evidence()
    finally:
        cmp_arch.load = original
