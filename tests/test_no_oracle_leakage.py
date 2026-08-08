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
- Gold data locations live in a **separate configuration file** that the
  runtime ``Config`` object never loads, so there is no config lookup at all,
  correct or otherwise, that reaches the answer key

The separation is also visible in the layout: the registry lives in
``sme_assistant.evaluation``, not ``sme_assistant.kb``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from sme_assistant.common.config import load_config

SRC = Path(__file__).resolve().parents[1] / "src" / "sme_assistant"

# Packages that execute during inference. Anything here runs while the system
# is answering a question, so anything here reading gold data is leakage.
INFERENCE_PACKAGES = ("kb", "ingest", "retrieve", "generate", "verify", "agent", "common")

FORBIDDEN_IMPORT_TOKENS = ("conflicts", "gold")
FORBIDDEN_CONFIG_LOOKUPS = (
    "evaluation.conflicts", "evaluation.test_set", "evaluation.question_set", "evaluation",
)
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
            for token in ("gold/", "gold\\", "conflicts.json", "evaluation.json"):
                assert token not in lowered, (
                    f"{module.relative_to(SRC)} contains the literal {node.value!r}. "
                    "Inference code must not address gold data by path."
                )
            for token in FORBIDDEN_CONFIG_LOOKUPS:
                assert lowered != token, (
                    f"{module.relative_to(SRC)} performs a config lookup for "
                    f"{node.value!r}. Even though that key no longer exists, an "
                    "inference module must never attempt to reach evaluation data."
                )


def test_runtime_config_contains_no_route_to_gold_data():
    """The structural guarantee, not a behavioural observation.

    An earlier design put evaluation paths under ``config.evaluation``. That
    proved no *current* leakage while leaving the route open: an inference
    module could have called ``config.path("evaluation.conflicts")`` and read
    the answer key, importing nothing and containing no literal path, so the
    other tests here would have passed.

    The files are now separate. This test asserts there is no key anywhere in
    the runtime configuration that leads to gold data.
    """
    config = load_config()
    serialised = json.dumps(config.as_dict()).lower()
    for token in ("gold", "conflicts", "test_set", "question_set", "evaluation"):
        assert token not in serialised, (
            f"Runtime config mentions {token!r}. Gold data must not be "
            "addressable from config.json at all."
        )


def test_evaluation_config_is_a_separate_file():
    from sme_assistant.evaluation.config import (
        DEFAULT_EVALUATION_CONFIG,
        load_evaluation_config,
    )

    runtime = load_config()
    evaluation = load_evaluation_config()
    assert evaluation.source != runtime.source
    assert DEFAULT_EVALUATION_CONFIG.name == "evaluation.json"
    assert evaluation.path("conflicts").exists()


def test_evaluation_protocol_is_declared():
    """The design and its aggregation are decided once, in one place.

    ``cross_validation: leave_one_family_out`` was asserted here until 8 August
    2026. Nothing is trained on the remaining folds, so there was no model being
    validated and the name claimed a resampling procedure the study does not
    perform. The design is a fixed held-out evaluation; leave-one-family-out is
    retained as a sensitivity analysis and is declared separately so the two
    cannot be conflated again.
    """
    from sme_assistant.evaluation.config import load_evaluation_config

    protocol = load_evaluation_config().protocol
    assert protocol.get("evaluation_design") == "fixed_held_out"
    assert protocol.get("sensitivity_analysis") == "leave_one_family_out"
    assert "cross_validation" not in protocol, (
        "the protocol still calls this cross-validation, which it is not"
    )
    assert protocol.get("aggregation") == "macro_average_by_family"
    assert protocol.get("rationale")
    assert protocol.get("confirmatory_contrast", "").startswith("B versus D")


def test_the_protocol_names_every_tuning_family():
    """A tuning rule that lists only some of them is worse than none.

    It reads as complete, so a reader checking whether CONF-01 is excluded from
    the reported set would conclude it is not.
    """
    from sme_assistant.evaluation.config import load_evaluation_config
    from sme_assistant.evaluation.conflicts import load_conflicts

    evaluation = load_evaluation_config()
    rule = evaluation.protocol.get("tuning_rule", "")
    for family in load_conflicts(evaluation.path("conflicts")).tuning_families:
        assert family.family_id in rule, (
            f"{family.family_id} is a tuning family but the declared rule does "
            "not name it"
        )


def test_registry_lives_in_the_evaluation_package():
    """Layout should make the boundary obvious without reading any code."""
    assert (SRC / "evaluation" / "conflicts.py").exists()
    assert not (SRC / "kb" / "conflicts.py").exists(), (
        "The conflict registry must not sit in the kb package, which inference imports"
    )
