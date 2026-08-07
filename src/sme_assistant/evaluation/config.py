"""Evaluation configuration. Gold data, deliberately unreachable from runtime.

``config.json`` describes how the system runs. This file describes how the
system is *judged*: where the conflict registry lives, where the test set
lives, and how results are aggregated.

The two are separate files, not separate keys in one file, and that is the
whole point. An earlier design placed evaluation paths under
``config.evaluation``. A reviewer observed that this proves no *current*
leakage rather than structural inaccessibility: an inference module could
still call ``config.path("evaluation.conflicts")`` and read the answer key.
The test would not have caught it, because nothing was imported and no literal
path appeared in the source.

Splitting the files removes the possibility rather than testing for its
absence. The runtime ``Config`` object has no key that leads here.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVALUATION_CONFIG = PROJECT_ROOT / "gold" / "evaluation.json"


class EvaluationConfigError(RuntimeError):
    """Raised when the evaluation configuration is missing or malformed."""


class EvaluationConfig:
    """Read-only access to evaluation settings and gold data locations."""

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source

    def path(self, name: str) -> Path:
        paths = self._data.get("paths", {})
        if name not in paths:
            raise EvaluationConfigError(
                f"No evaluation path {name!r} in {self.source}; have {sorted(paths)}"
            )
        candidate = Path(paths[name])
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)

    @property
    def protocol(self) -> dict[str, Any]:
        """How results are cross-validated and aggregated."""
        return self._data.get("protocol", {})

    def fingerprint(self) -> str:
        import hashlib

        canonical = json.dumps(self._data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return f"EvaluationConfig(source={self.source})"


@lru_cache(maxsize=None)
def load_evaluation_config(path: str | Path | None = None) -> EvaluationConfig:
    source = Path(path) if path else DEFAULT_EVALUATION_CONFIG
    if not source.exists():
        raise EvaluationConfigError(f"Evaluation config not found: {source}")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationConfigError(f"{source} is not valid JSON: {exc}") from exc
    if "paths" not in data:
        raise EvaluationConfigError(f"{source}: missing 'paths'")
    return EvaluationConfig(data, source)
