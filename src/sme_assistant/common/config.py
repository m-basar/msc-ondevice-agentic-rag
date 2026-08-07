"""Configuration loading.

Every tunable value in this project lives in config.json, never in code.
This is a deliberate research decision, not tidiness: Chapter 4 reports
thresholds and parameters, and each reported number must be traceable to
exactly one line in one file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# This file sits at src/sme_assistant/common/config.py
# parents[0] = common, [1] = sme_assistant, [2] = src, [3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


class Config:
    """Read-only access to the project configuration.

    Values are addressed with dotted keys, so a caller writes
    ``config.get("retrieval.top_k")`` rather than
    ``config["retrieval"]["top_k"]``. The dotted form fails with a useful
    message instead of a bare KeyError, which matters when the same code
    runs unattended on the Pi during a long evaluation run.
    """

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return a value, or ``default`` if any part of the path is missing."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        """Return a value, or raise if it is absent.

        Use this for anything the pipeline cannot run without. Failing at
        startup with a clear message beats failing halfway through a
        two-hour evaluation run.
        """
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            raise ConfigError(
                f"Missing required config key '{dotted_key}' in {self.source}"
            )
        return value

    def path(self, dotted_key: str) -> Path:
        """Resolve a configured path to an absolute path.

        Paths in config.json are relative to the project root. Resolving
        them here means the code behaves identically whether it is run from
        the project root, from scripts/, or from a scheduled job on the Pi
        with an unpredictable working directory.
        """
        raw = self.require(dotted_key)
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)

    def as_dict(self) -> dict[str, Any]:
        """Return a deep copy, for writing into a run manifest."""
        return json.loads(json.dumps(self._data))

    def fingerprint(self) -> str:
        """Stable hash of the configuration.

        Written into every results file so a table in Chapter 4 can be tied
        back to the exact parameters that produced it.
        """
        import hashlib

        canonical = json.dumps(self._data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def short_fingerprint(self) -> str:
        """First 12 characters of the configuration hash, for terminal output."""
        return self.fingerprint()[:12]

    def __repr__(self) -> str:
        return f"Config(source={self.source}, fingerprint={self.short_fingerprint()})"


@lru_cache(maxsize=None)
def load_config(config_path: str | Path | None = None) -> Config:
    """Load and cache the project configuration.

    Cached because the config is read on nearly every call path, and
    re-reading a file thousands of times during an evaluation run would
    pollute the latency measurements this project exists to report.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object")

    return Config(data, path)
