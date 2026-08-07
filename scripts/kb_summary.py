"""Print knowledge base statistics and validate the corpus against the conflict registry.

Run this to produce the corpus description for Chapter 3. The numbers in the
methodology chapter should come from this script rather than being counted by
hand, so that they stay correct if a document is added or edited.

    python scripts/kb_summary.py
    python scripts/kb_summary.py --json
    python scripts/kb_summary.py --manifest > results/corpus_manifest.json

Exit code is non-zero if the corpus and the registry disagree, so this doubles
as a continuous integration check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.hostinfo import git_commit  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import (  # noqa: E402
    ConflictRegistryError,
    load_conflicts,
    validate_against_corpus,
)
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402

LEGAL_REVIEW_DATE = "2026-08-07"


def build_manifest(config, kb, registry) -> dict:
    """Complete provenance record for the inputs to an experiment."""
    manifest = kb.manifest()
    manifest.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_versions": {
            "conflict_registry": registry.schema_version,
            "corpus_manifest": "1.0",
        },
        "config_sha256": config.fingerprint(),
        "conflict_registry_sha256": registry.fingerprint(),
        "git": git_commit(),
        "legal_review_date": LEGAL_REVIEW_DATE,
        "conflicts": registry.summary(),
    })
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="summary as JSON")
    group.add_argument("--manifest", action="store_true", help="full provenance manifest as JSON")
    args = parser.parse_args()

    config = load_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(load_evaluation_config().path("conflicts"))

    if args.manifest:
        print(json.dumps(build_manifest(config, kb, registry), indent=2))
        return 0

    summary = kb.summary()
    if args.json:
        print(json.dumps({"corpus": summary, "conflicts": registry.summary()}, indent=2))
        return 0

    git = git_commit()
    print(f"Knowledge base     {kb.root}")
    print(f"Config sha256      {config.short_fingerprint()}...")
    print(f"Corpus sha256      {kb.short_fingerprint()}...")
    print(f"Registry sha256    {registry.fingerprint()[:12]}...")
    if git.get("available"):
        dirty = "  (WORKING TREE DIRTY)" if git["dirty"] else ""
        print(f"Git commit         {git['commit_short']} on {git['branch']}{dirty}")
    print(f"Legal review       {LEGAL_REVIEW_DATE}")
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

    conflicts = registry.summary()
    print()
    print("Conflict registry")
    print(f"  Schema           {conflicts['schema_version']}")
    print(f"  Families         {conflicts['family_count']} "
          f"({conflicts['fact_count']} conflicting facts)")
    print(f"  Filter-solvable  {conflicts['filter_resolvable']}  "
          f"(a status filter alone resolves these)")
    print(f"  Needs reasoning  {conflicts['requires_reasoning']}  "
          f"(no metadata field distinguishes the documents)")
    print()
    print("  ID        Type                  Risk    Domain                Documents          Facts")
    for family in registry.families:
        docs = " / ".join(family.documents)
        print(f"  {family.family_id:<9} {family.conflict_type:<21} {family.risk_level:<7} "
              f"{family.domain:<21} {docs:<18} {len(family.conflicting_facts)}")
        print(f"            {family.name}")

    print()
    print("Confidence policy")
    for outcome, rule in registry.confidence_policy.items():
        print(f"  {outcome:<24} {rule['confidence']}, flag {rule['flag']}")

    print()
    print(f"  Deliberate gaps, fully absent:  {len(registry.fully_absent)}")
    for gap in registry.fully_absent:
        print(f"    - {gap.topic}")
        if gap.note:
            print(f"        note: {gap.note[:100]}...")
    print(f"  Deliberate gaps, partial:       {len(registry.partially_present)}")
    for partial in registry.partially_present:
        print(f"    - {partial.topic} (near miss in {partial.mentioned_in})")

    print()
    try:
        validate_against_corpus(registry, kb)
    except ConflictRegistryError as exc:
        print(f"VALIDATION FAILED: {exc}")
        return 1
    print("Validation passed:")
    print("  registry and corpus agree")
    print("  every declared conflicting value is anchored in document text")
    print("  declared gaps are genuinely absent")
    print("  partial gaps still have their near-miss evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
