"""Inspect the chunker's output.

Produces the chunking statistics for Chapter 3, and estimates the prompt cost
those chunks imply on each machine. Prompt length is the dominant latency term
on edge hardware, so chunk size is a cost decision as much as a retrieval one.

    python scripts/chunk_summary.py
    python scripts/chunk_summary.py --doc HR-13
    python scripts/chunk_summary.py --show HR-13#001
    python scripts/chunk_summary.py --sweep
    python scripts/chunk_summary.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.ingest.chunker import chunk_corpus, summarise_chunks  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402

# Median prompt-processing rates from the VALID benchmark runs, llama3.2:3b.
# See docs/BENCHMARKS.md.
PROMPT_RATES = {"laptop GPU": 4434.80, "laptop CPU": 183.91, "Pi 5": 24.54}
WORDS_TO_TOKENS = 1.33  # observed on this corpus: 903 tokens from ~680 words


def build(config):
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    return kb, chunk_corpus(
        kb,
        config.require("chunking.max_words"),
        config.require("chunking.overlap_sentences"),
        config.require("chunking.min_words"),
    )


def print_cost(mean_words: float, top_k: int) -> None:
    tokens = mean_words * top_k * WORDS_TO_TOKENS
    print(f"\nEstimated prompt cost at top_k={top_k}")
    print(f"  Evidence     {mean_words * top_k:.0f} words, about {tokens:.0f} tokens")
    print("  Machine        Prefill time")
    for machine, rate in PROMPT_RATES.items():
        print(f"  {machine:<14} {tokens / rate:>6.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", help="list chunks for one document")
    parser.add_argument("--show", help="print one chunk exactly as it will be embedded")
    parser.add_argument("--sweep", action="store_true", help="compare max_words settings")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    args = parser.parse_args()

    config = load_config()
    kb, chunks = build(config)

    if args.show:
        match = [c for c in chunks if c.chunk_id == args.show]
        if not match:
            print(f"No chunk {args.show!r}", file=sys.stderr)
            return 1
        chunk = match[0]
        print(f"chunk_id      {chunk.chunk_id}")
        print(f"citation      {chunk.citation}")
        print(f"status        {chunk.status}"
              + (f"  -> superseded by {chunk.superseded_by}" if chunk.superseded_by else ""))
        print(f"effective     {chunk.effective_date}")
        print(f"words         {chunk.word_count}   contains_table={chunk.contains_table}")
        print("\n--- text sent to the embedding model and the generator ---")
        print(chunk.embedding_text)
        return 0

    if args.doc:
        selected = [c for c in chunks if c.doc_id == args.doc]
        if not selected:
            print(f"No chunks for document {args.doc!r}", file=sys.stderr)
            return 1
        print(f"{args.doc}  {selected[0].doc_title}  ({selected[0].status})")
        for chunk in selected:
            path = "; ".join(chunk.sections) or "-"
            flag = " [table]" if chunk.contains_table else ""
            print(f"  {chunk.chunk_id}  {chunk.word_count:>4}w  {path}{flag}")
            print(f"      {chunk.text[:110].replace(chr(10), ' ')}...")
        return 0

    if args.sweep:
        print("max_words sweep, showing the retrieval and cost trade-off\n")
        print(f"{'max_words':>10} {'chunks':>7} {'mean w':>7} {'max w':>6} {'oversized':>10} "
              f"{'Pi prefill @k=4':>16}")
        for limit in (100, 140, 180, 220, 300):
            produced = chunk_corpus(kb, limit, config.require("chunking.overlap_sentences"),
                                    config.require("chunking.min_words"))
            s = summarise_chunks(produced)
            oversized = sum(1 for c in produced if c.word_count > limit)
            tokens = s["mean_words"] * 4 * WORDS_TO_TOKENS
            marker = "  <- current" if limit == config.require("chunking.max_words") else ""
            print(f"{limit:>10} {s['chunk_count']:>7} {s['mean_words']:>7} {s['max_words']:>6} "
                  f"{oversized:>10} {tokens / PROMPT_RATES['Pi 5']:>14.2f}s{marker}")
        print("\nOversized chunks are intact tables that exceed the limit by design.")
        return 0

    summary = summarise_chunks(chunks)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Corpus         {kb.root}")
    print(f"Corpus sha256  {kb.short_fingerprint()}...")
    print(f"Chunking       max_words={config.require('chunking.max_words')} "
          f"overlap={config.require('chunking.overlap_sentences')} "
          f"min_words={config.require('chunking.min_words')}")
    print()
    print(f"  Chunks            {summary['chunk_count']}")
    print(f"  Per document      {summary['chunks_per_document_mean']} mean, "
          f"{summary['chunks_per_document_max']} max")
    print(f"  Words per chunk   {summary['mean_words']} mean, {summary['median_words']} median, "
          f"{summary['min_words']} min, {summary['max_words']} max")
    print(f"  Total words       {summary['total_words']:,}")
    print(f"  Containing tables {summary['chunks_containing_tables']}")
    print(f"  Current           {summary['current_chunks']}")
    print(f"  Superseded        {summary['superseded_chunks']}")

    print_cost(summary["mean_words"], config.require("retrieval.top_k"))

    missing_context = [c.chunk_id for c in chunks if not c.sections]
    print()
    if missing_context:
        print(f"WARNING: {len(missing_context)} chunks have no heading path: {missing_context[:5]}")
        return 1
    print("Every chunk carries document title, heading path and full provenance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
