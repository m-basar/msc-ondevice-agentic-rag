"""Ask a question end to end: retrieve, then generate a grounded answer.

This is the BASELINE arm. No verification, no conflict handling. It is what a
competent standard RAG implementation would produce, and it is what the
verification layer has to beat.

    python scripts/ask.py --mock "what is the mileage rate?"
    python scripts/ask.py "what is the mileage rate for business travel?"
    python scripts/ask.py --model llama3.2:1b "what is the mileage rate?"
    python scripts/ask.py --mode current_only "what is the mileage rate?"
    python scripts/ask.py --show-prompt "what is the company pension scheme?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import build_client  # noqa: E402
from sme_assistant.generate.generator import Generator  # noqa: E402
from sme_assistant.ingest.index import Index, index_path_for  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402
from sme_assistant.retrieve.retriever import RetrievalMode, Retriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=[m.value for m in RetrievalMode], default="all")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    question = " ".join(args.question)
    config = load_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    index = Index.load(index_path_for(config, mock=args.mock), kb=kb, config=config)
    client = build_client(config, mock=args.mock)

    retrieval = Retriever(index, client, config).retrieve(
        question, top_k=args.top_k, mode=RetrievalMode(args.mode)
    )
    result = Generator(client, config).answer(question, retrieval, model=args.model)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"Question    {question}")
    print(f"Model       {result.generation.model}   mode={retrieval.mode.value}   "
          f"top_k={retrieval.top_k}")
    print()
    print("Retrieved:")
    for scored in retrieval:
        flag = "" if scored.chunk.is_current else "  SUPERSEDED"
        print(f"  {scored.rank}. {scored.score:.4f}  {scored.chunk_id}{flag}")
    if retrieval.should_refuse:
        print(f"  (best score {retrieval.best_score:.4f} is below the "
              f"{retrieval.min_similarity} threshold: no evidence was given to the model)")
    print()

    if args.show_prompt:
        print("--- prompt ---")
        print(result.prompt)
        print("--- end prompt ---\n")

    print("ANSWER")
    print(result.answer)
    print()

    print("Citation audit")
    print(f"  Cited chunks        {list(result.citations) or 'none'}")
    if result.document_citations:
        print(f"  Cited documents     {list(result.document_citations)}")
    print(f"  Hallucinated        {list(result.hallucinated_citations) or 'none'}")
    print(f"  Retrieved, uncited  {list(result.uncited_chunks) or 'none'}")
    print(f"  Valid citation ids  {result.has_valid_citation_ids}")
    print(f"  Refusal heuristic   {result.refusal_heuristic}  (diagnostic only)")
    if result.cited_superseded:
        print()
        print(f"  *** CITED A WITHDRAWN DOCUMENT: {list(result.cited_superseded)}")
        print("      The baseline quoted policy that is no longer in force.")

    generation = result.generation
    print()
    print(f"Timing      retrieve+embed {retrieval.embed_seconds * 1000:.0f} ms, "
          f"generate {generation.wall_seconds:.2f} s")
    print(f"Tokens      {generation.prompt_tokens} prompt "
          f"({generation.prompt_tokens_per_second} tok/s), "
          f"{generation.eval_tokens} generated "
          f"({generation.eval_tokens_per_second} tok/s)")
    if generation.cpu_temp_c:
        note = " THROTTLED" if generation.throttled else ""
        freq = f" @{generation.arm_frequency_hz / 1e9:.2f}GHz" if generation.arm_frequency_hz else ""
        print(f"Device      {generation.cpu_temp_c:.1f}C{freq}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
