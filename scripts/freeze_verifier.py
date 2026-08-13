"""Freeze the verifier: what it was, exactly, when the test run began.

Run **after** development is finished and the prompt has stopped changing, and
**before** any test-split arm. It writes ``results/verifier_frozen.json``,
which is the record that lets someone establish, later, that the verifier
scored on the test split is the one that was developed and not a quietly
improved successor.

    python scripts/freeze_verifier.py

Four things are recorded, because any one of them can change without the others
noticing:

``git``               the commit, and whether the tree was dirty when frozen
``prompt_sha256``     the verifier system prompt, hashed verbatim
``model``             the digest Ollama reports, not just the tag. ``llama3.2:3b``
                      is a moving label; the digest is the model
``generation``        seed, temperature, num_predict, num_ctx

A dirty tree is recorded rather than refused, but it is recorded loudly. The
point is an honest record, and a freeze that silently tolerated uncommitted
changes would be worth less than none.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.hostinfo import git_commit, ollama_info  # noqa: E402
from sme_assistant.verify.verifier import VERIFIER_SYSTEM  # noqa: E402


def working_tree_is_clean() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
        return out.returncode == 0 and not out.stdout.strip()
    except Exception:
        return False


def model_digest(config, model: str) -> str | None:
    """The digest Ollama reports for a tag, which is what actually ran."""
    import urllib.error
    import urllib.request

    url = config.require("llm.ollama_url").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            for entry in json.load(response).get("models", []):
                if entry.get("name") == model or entry.get("model") == model:
                    return entry.get("digest")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None
    return None


def main() -> int:
    config = load_config()
    generation_model = config.require("llm.generation_model")
    verification_model = config.get("llm.verification_model") or generation_model
    clean = working_tree_is_clean()

    options = dict(config.require("verification"))
    confidence = options.pop("confidence", {})

    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git": git_commit(),
        "working_tree_clean": clean,
        "prompt_sha256": hashlib.sha256(VERIFIER_SYSTEM.encode("utf-8")).hexdigest(),
        "prompt_characters": len(VERIFIER_SYSTEM),
        "models": {
            "generation": {"tag": generation_model,
                           "digest": model_digest(config, generation_model)},
            "verification": {"tag": verification_model,
                             "digest": model_digest(config, verification_model)},
        },
        "generation_options": config.require("generation"),
        "verification_options": options,
        "confidence_policy": confidence,
        "retrieval": {"top_k": config.require("retrieval.top_k"),
                      "min_similarity": config.require("retrieval.min_similarity")},
        "config_sha256": config.fingerprint(),
        "backend": ollama_info(config.require("llm.ollama_url")),
        "declaration": (
            "The verifier frozen here is the one developed against the eight "
            "development families, six of them conflicts and two of them "
            "compatible negative controls. No test-split arm had been run when "
            "this record was written. Any change to the prompt, the models or "
            "these options after this point invalidates the freeze and must be "
            "recorded as a pre-registration amendment."
        ),
    }

    output = Path(config.path("paths.results")).parent / "verifier_frozen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"Frozen at        {record['frozen_at']}")
    print(f"Commit           {(record['git'] or {}).get('commit', '?')}")
    print(f"Working tree     {'clean' if clean else 'DIRTY - commit before freezing'}")
    print(f"Prompt sha256    {record['prompt_sha256'][:16]}...  "
          f"({record['prompt_characters']} chars)")
    for role, entry in record["models"].items():
        digest = entry["digest"]
        print(f"{role:<16} {entry['tag']}  {digest[:19] + '...' if digest else 'digest unavailable'}")
    print(f"Written to       {output}")

    if not clean:
        print("\nThe working tree was dirty. The freeze is recorded but is not "
              "reproducible from the commit alone. Commit and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
