"""Build the searchable index.

    python scripts/build_index.py --mock          # no model needed, ~1 second
    python scripts/build_index.py                 # real embeddings
    python scripts/build_index.py --out data/index_pi.json

The index records the corpus fingerprint it was built from and refuses to load
against a different corpus, so a stale index cannot silently produce answers
citing text that no longer exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import LLMError, build_client  # noqa: E402
from sme_assistant.ingest.index import Index, build_index  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true",
                        help="use the deterministic mock backend, no model required")
    parser.add_argument("--out", help="output path, defaults to config paths.index")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    client = build_client(config, mock=args.mock)

    endpoint = client.describe_endpoint()
    if not endpoint["reachable"]:
        print(f"Backend not reachable: {endpoint}", file=sys.stderr)
        return 1

    print(f"Corpus        {len(kb)} documents, sha256 {kb.short_fingerprint()}...")
    print(f"Backend       {endpoint['backend']} {endpoint.get('version')} "
          f"at {endpoint.get('base_url')}")
    print(f"Model store   {endpoint.get('model_store_fingerprint')}")
    print(f"Embedding     {getattr(client, 'embedding_model', 'unknown')}")
    print()

    try:
        index = build_index(kb, client, config, progress=not args.quiet)
    except LLMError as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else config.path("paths.index")
    index.save(out)

    print()
    print(json.dumps(index.summary(), indent=2))
    print(f"\nSaved to {out}  ({out.stat().st_size / 1024:.0f} KB)")

    # Prove it round-trips and that the staleness guard is live.
    reloaded = Index.load(out, kb=kb)
    assert len(reloaded) == len(index)
    print("Reloaded and verified against the corpus fingerprint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
