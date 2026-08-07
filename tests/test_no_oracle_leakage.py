"""Structural guarantee that the inference pipeline cannot read gold data.

``gold/conflicts.json`` contains the expected answer for every planted
conflict, the assertions the system is forbidden to make, and the list of
topics that have no answer in the corpus. If any component of the pipeline
read that file, the system would be scoring against its own answer key and
every result in the dissertation would be worthless.

Reviewers cannot verify that by reading code, and a future edit could
introduce the dependency by accident. So it is enforced here instead:

- No module in the inference packages may import the conflicts registry
- No module in the inference packages may reference the gold directory
- Gold data is addressed under ``config.evaluation``, never ``config.paths``,
  so a pipeline component using only ``paths`` cannot reach it

The separation is also visible in the layout: the registry lives in
``sme_assistant.evaluation``, not ``sme_assistant.kb``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sme_assistant.common.config import load_config

SRC = Path(__file__).resolve().parents[1] / "src" / "sme_assistant"

# Packages that execute during inference. Anything here runs while the system
# is answering a question, so anything here reading gold data is leakage.
INFERENCE_PACKAGES = ("kb", "ingest", "retrieve", "generate", "verify", "agent", "common")

FORBIDDEN_IMPORT_TOKENS = ("conflicts", "gold")
FORBIDDEN_TEXT_TOKENS = ("gold/", "gold\\\\", "conflicts.json")


def inference_modules() -> list[Path]:
    modules = []
    for package in INFERENCE_PACKAGES:
        modules.extend(sorted((SRC / package).rglob("*.py")))
    return modules


def test_inference_packages_exist():
    assert inference_modules(), "no modules found; the path assumption is wrong"


@pytest.mark.parametrize("module", inference_modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_module_does_not_import_gold_data(module: Path):
    """Walk the import graph rather than grepping, so aliases are caught too."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    imported: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.append(base)
            imported.extend(f"{base}.{alias.name}" for alias in node.names)

    for name in imported:
        lowered = name.lower()
        for token in FORBIDDEN_IMPORT_TOKENS:
            assert token not in lowered, (
                f"{module.relative_to(SRC)} imports {name!r}. Inference code must not "
                "reach evaluation gold data."
            )
        assert "evaluation" not in lowered, (
            f"{module.relative_to(SRC)} imports from the evaluation package ({name!r}). "
            "Evaluation may depend on inference, never the reverse."
        )


@pytest.mark.parametrize("module", inference_modules(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_module_does_not_reference_gold_paths(module: Path):
    """Catch a hard-coded path that bypasses the import graph entirely."""
    text = module.read_text(encoding="utf-8")
    # The docstring in this test file's subject modules may legitimately mention
    # the concept; only string literals are checked.
    tree = ast.parse(text, filename=str(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            for token in ("gold/", "conflicts.json"):
                assert token not in lowered, (
                    f"{module.relative_to(SRC)} contains the literal {node.value!r}. "
                    "Inference code must not address gold data by path."
                )


def test_gold_data_is_not_reachable_from_runtime_paths():
    """A component using only config.paths cannot find the gold data."""
    config = load_config()
    runtime_paths = config.require("paths")
    serialised = " ".join(str(v).lower() for v in runtime_paths.values())
    assert "gold" not in serialised
    assert "conflicts" not in serialised
    assert "test_set" not in serialised, (
        "The test set is gold data and must live under config.evaluation, "
        "not config.paths"
    )


def test_gold_data_is_reachable_from_evaluation_config():
    config = load_config()
    evaluation = config.require("evaluation")
    assert "conflicts" in evaluation
    assert config.path("evaluation.conflicts").exists()


def test_registry_lives_in_the_evaluation_package():
    """Layout should make the boundary obvious without reading any code."""
    assert (SRC / "evaluation" / "conflicts.py").exists()
    assert not (SRC / "kb" / "conflicts.py").exists(), (
        "The conflict registry must not sit in the kb package, which inference imports"
    )
