"""Print knowledge base statistics and validate the corpus against the conflict registry.

Run this to produce the corpus description for Chapter 3. The numbers in the
methodology chapter should come from this script rather than being counted by
hand, so that they stay correct if a document is added or edited.

    python scripts/kb_summary.py
    python scripts/kb_summary.py --json
    python scripts/kb_summary.py --manifest > results/corpus_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.kb.conflicts import (  # noqa: E402
    ConflictRegistryError,
    load_conflicts,
    validate_against_corpus,
)
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="summary as JSON")
    group.add_argument("--manifest", action="store_true", help="full per-document manifest as JSON")
    args = parser.parse_args()

    config = load_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(config.path("paths.conflicts"))

    if args.manifest:
        manifest = kb.manifest()
        manifest["config_fingerprint"] = config.fingerprint()
        manifest["conflicts"] = registry.summary()
        print(json.dumps(manifest, indent=2))
        return 0

    summary = kb.summary()
    if args.json:
        print(json.dumps({"corpus": summary, "conflicts": registry.summary()}, indent=2))
        return 0

    print(f"Knowledge base:     {kb.root}")
    print(f"Config fingerprint: {config.fingerprint()}")
    print(f"Corpus fingerprint: {summary['corpus_fingerprint']}")
    print()
    print(f"  Documents        {summary['document_count']}")
    print(f"  Current          {summary['current_count']}")
    print(f"  Superseded       {summary['superseded_count']} "
          f"({summary['superseded_share']:.1%} of corpus)")
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

    print()
    print("Conflict registry")
    print(f"  Families         {len(registry)}")
    print(f"  Conflicting facts {sum(len(f.conflicting_facts) for f in registry.families)}")
    print(f"  Policy           {registry.expected_answer_policy['answer_with']}, "
          f"flag {registry.expected_answer_policy['flag']}, "
          f"confidence capped at {registry.expected_answer_policy['confidence_ceiling']}")
    print()
    print("  ID        Risk    Domain               Superseded -> Current   Facts")
    for family in registry.families:
        pair = f"{family.superseded_document} -> {family.current_document}"
        print(f"  {family.family_id:<9} {family.risk_level:<7} {family.domain:<20} "
              f"{pair:<23} {len(family.conflicting_facts)}")
        print(f"            {family.name}")

    print()
    print(f"  Deliberate gaps, fully absent:  {len(registry.fully_absent_topics)}")
    for topic in registry.fully_absent_topics:
        print(f"    - {topic}")
    print(f"  Deliberate gaps, partial:       {len(registry.partial_topics)}")
    for entry in registry.partial_topics:
        print(f"    - {entry['topic']} (mentioned in {entry['mentioned_in']})")

    print()
    try:
        validate_against_corpus(registry, kb)
    except ConflictRegistryError as exc:
        print(f"VALIDATION FAILED: {exc}")
        return 1
    print("Validation passed: registry and corpus agree, declared gaps are genuinely absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
