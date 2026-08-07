"""Ask the index a question and see what comes back.

Retrieval only. No generation yet, so what you are looking at is exactly the
evidence a generator would be given.

    python scripts/query.py --mock "what is the mileage rate?"
    python scripts/query.py "what is the mileage rate?"
    python scripts/query.py --mode current_only "what is the mileage rate?"
    python scripts/query.py --full "how do I report a lost laptop?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import build_client  # noqa: E402
from sme_assistant.ingest.index import Index  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402
from sme_assistant.retrieve.retriever import RetrievalMode, Retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mode", choices=[m.value for m in RetrievalMode], default="all")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--full", action="store_true", help="print the full evidence block")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    question = " ".join(args.question)
    config = load_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    index = Index.load(config.path("paths.index"), kb=kb)
    client = build_client(config, mock=args.mock)
    retriever = Retriever(index, client, config)

    result = retriever.retrieve(question, top_k=args.top_k, mode=RetrievalMode(args.mode))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"Question   {question}")
    print(f"Mode       {result.mode.value}   top_k={result.top_k}   "
          f"threshold={result.min_similarity}")
    print(f"Considered {result.candidates_considered} of {len(index)} chunks")
    print(f"Timing     embed {result.embed_seconds * 1000:.0f} ms, "
          f"search {result.search_seconds * 1000:.2f} ms")
    print()
    for scored in result:
        chunk = scored.chunk
        flag = "" if chunk.is_current else f"  SUPERSEDED -> {chunk.superseded_by}"
        print(f"  {scored.rank}. {scored.score:.4f}  {chunk.chunk_id}{flag}")
        print(f"     {chunk.citation}")
        print(f"     {chunk.text[:130].replace(chr(10), ' ')}...")
        print()

    if result.conflicts:
        print("Structural conflicts detected from metadata alone:")
        for conflict in result.conflicts:
            print(f"  {conflict['type']}: {conflict['superseded']} vs {conflict['current']}")
        print()

    if result.should_refuse:
        print(f"REFUSE: best score {result.best_score:.4f} is below the "
              f"{result.min_similarity} threshold. Nothing retrieved is relevant enough.")
    else:
        print(f"Best score {result.best_score:.4f}, above the "
              f"{result.min_similarity} threshold.")

    if args.full:
        print("\n--- evidence block as the generator would receive it ---")
        print(result.evidence_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
