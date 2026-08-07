"""Print knowledge base statistics.

Run this to produce the corpus description for Chapter 3. The numbers in the
methodology chapter should come from this script rather than being counted by
hand, so that they stay correct if a document is added or edited.

    python scripts/kb_summary.py
    python scripts/kb_summary.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    config = load_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    summary = kb.summary()

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Knowledge base: {kb.root}")
    print(f"Config fingerprint: {config.fingerprint()}")
    print()
    print(f"  Documents        {summary['document_count']}")
    print(f"  Current          {summary['current_count']}")
    print(f"  Superseded       {summary['superseded_count']}")
    print(f"  Total words      {summary['total_words']:,}")
    print(f"  Words per doc    {summary['mean_words']} mean, "
          f"{summary['min_words']} min, {summary['max_words']} max")
    print()
    print("  Category   Documents")
    for category, count in summary["categories"].items():
        print(f"  {category:<10} {count}")
    print()
    print("  ID       Ver   Status       Words  Title")
    for doc in sorted(kb, key=lambda d: d.doc_id):
        print(f"  {doc.doc_id:<8} {doc.version:<5} {doc.status:<12} "
              f"{doc.word_count:>5}  {doc.title}")

    superseded = kb.superseded()
    if superseded:
        print()
        print("  Superseded chains (used by the conflicting-evidence test cases):")
        for doc in superseded:
            replacement = kb.by_id(doc.superseded_by)
            print(f"    {doc.doc_id} v{doc.version} ({doc.effective_date}) "
                  f"-> {replacement.doc_id} v{replacement.version} "
                  f"({replacement.effective_date})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
