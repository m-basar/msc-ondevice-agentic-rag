"""Retrieval evaluation on the development split.

``top_k: 4`` and ``min_similarity: 0.32`` were chosen from four pilot questions
and have been assumptions ever since. This measures them.

The measurement matters more than usual here. Stage 5 cannot resolve a conflict
it never sees: if retrieval returns only one side of a disagreement, the
verifier has nothing to compare and Arm D fails for a reason that has nothing to
do with verification. **Conflict-pair recall is therefore the metric that
decides whether the experiment can work at all**, and it is reported before any
arm is run rather than discovered afterwards as an explanation for a null
result.

Development split only. Running this on the test split would tune retrieval
against the questions that produce the reported numbers.

Run:

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --mock     # no Ollama, structure only

What is reported
----------------
``recall@k``            questions whose expected chunks appear in the top k,
                        strict (all of them) and lenient (at least one)
``MRR``                 mean reciprocal rank of the first expected chunk
``conflict-pair recall`` the chunks carrying **both disputed claims** present in
                        the top k, not merely a chunk from each document. A
                        conflict the retriever never assembles cannot be
                        detected by any verifier
``threshold sweep``     for each candidate ``min_similarity``, how many
                        answerable questions would be wrongly refused against
                        how many unanswerable ones would be correctly refused
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import MockClient, build_client  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.ingest.index import Index, build_index, index_path_for  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402
from sme_assistant.retrieve.retriever import Retriever  # noqa: E402

K_VALUES = (1, 2, 3, 4, 6, 8, 10)
THRESHOLDS = (0.0, 0.20, 0.25, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.60)


def anchor_chunks(family, chunk_texts) -> set[str]:
    """The chunks carrying this family's disputed claims, one per document."""
    found: set[str] = set()
    for fact in family.conflicting_facts:
        for doc_id, anchor in fact.anchors.items():
            for chunk_id, text in chunk_texts.items():
                if chunk_id.startswith(doc_id + "#") and anchor in text:
                    found.add(chunk_id)
    return found


def evaluate(retriever, questions, registry, chunk_texts, k_values=K_VALUES) -> dict:
    max_k = max(k_values)
    ranked: dict[str, list] = {}
    best: dict[str, float] = {}

    for question in questions:
        result = retriever.retrieve(question.text, top_k=max_k, min_similarity=0.0)
        ranked[question.question_id] = [s.chunk_id for s in result]
        best[question.question_id] = result.results[0].score if len(result) else 0.0

    # --- recall and MRR, over questions that name expected chunks ------------
    with_expected = [q for q in questions if q.expected_chunks]
    recall: dict[str, dict[str, float]] = {}
    for k in k_values:
        strict = lenient = 0
        for question in with_expected:
            top = set(ranked[question.question_id][:k])
            wanted = set(question.expected_chunks)
            if wanted <= top:
                strict += 1
            if wanted & top:
                lenient += 1
        n = len(with_expected) or 1
        recall[f"@{k}"] = {
            "strict": round(strict / n, 4),
            "lenient": round(lenient / n, 4),
        }

    reciprocal = []
    for question in with_expected:
        order = ranked[question.question_id]
        positions = [order.index(c) + 1 for c in question.expected_chunks if c in order]
        reciprocal.append(1 / min(positions) if positions else 0.0)
    mrr = round(sum(reciprocal) / len(reciprocal), 4) if reciprocal else None

    # --- conflict-pair recall ------------------------------------------------
    # The metric that decides whether the experiment can work. A verifier cannot
    # resolve a disagreement it was never shown both sides of.
    pair: dict[str, dict] = {}
    by_type: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    by_family: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for question in questions:
        if not question.family_id:
            continue
        family = registry.by_id(question.family_id)
        order = ranked[question.question_id]
        # The chunks that actually carry the disputed claims, not merely any
        # chunk from the right documents. Checking documents let a family score
        # 1.00 while the passages stating the disagreement never reached the
        # model, which is the only thing a verifier could reason over.
        wanted = anchor_chunks(family, chunk_texts)
        for k in k_values:
            both = wanted <= set(order[:k]) if wanted else False
            by_type[family.conflict_type][k].append(both)
            by_family[family.family_id][k].append(both)

    for conflict_type, per_k in sorted(by_type.items()):
        pair[conflict_type] = {
            f"@{k}": round(sum(v) / len(v), 4) for k, v in sorted(per_k.items())
        }

    # Per family as well as per type. A type-level 0.67 over three families can
    # mean "all three are middling" or "two are perfect and one is broken", and
    # those call for entirely different responses. The type-level figure alone
    # cannot tell them apart.
    per_family = {
        family_id: {
            "type": registry.by_id(family_id).conflict_type,
            **{f"@{k}": round(sum(v) / len(v), 4) for k, v in sorted(per_k.items())},
        }
        for family_id, per_k in sorted(by_family.items())
    }

    # --- threshold sweep -----------------------------------------------------
    answerable = [q for q in questions if q.answerability == "answerable"]
    unanswerable = [q for q in questions if q.answerability == "unanswerable"]
    sweep = []
    for threshold in THRESHOLDS:
        wrongly_refused = sum(1 for q in answerable if best[q.question_id] < threshold)
        correctly_refused = sum(1 for q in unanswerable if best[q.question_id] < threshold)
        sweep.append({
            "min_similarity": threshold,
            "answerable_wrongly_refused": wrongly_refused,
            "answerable_total": len(answerable),
            "unanswerable_correctly_refused": correctly_refused,
            "unanswerable_total": len(unanswerable),
            "youden_j": round(
                (correctly_refused / len(unanswerable) if unanswerable else 0)
                - (wrongly_refused / len(answerable) if answerable else 0),
                4,
            ),
        })

    scores = {
        "answerable": sorted(round(best[q.question_id], 4) for q in answerable),
        "unanswerable": sorted(round(best[q.question_id], 4) for q in unanswerable),
    }

    return {
        "questions": len(questions),
        "questions_with_expected_chunks": len(with_expected),
        "recall": recall,
        "mrr": mrr,
        "conflict_pair_recall": pair,
        "conflict_pair_recall_by_family": per_family,
        "threshold_sweep": sweep,
        "top_score_distribution": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="use the deterministic mock client, for structure only")
    parser.add_argument("--split", default="dev",
                        help="only 'dev' is permitted without --force-test")
    parser.add_argument("--force-test", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.split != "dev" and not args.force_test:
        raise SystemExit(
            "Retrieval evaluation runs on the development split. Tuning retrieval "
            "against the test questions would tune it against the numbers the "
            "dissertation reports."
        )

    config = load_config()
    evaluation = load_evaluation_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(evaluation.path("conflicts"))
    question_set = load_question_set(evaluation.path("question_set"))
    questions = list(question_set.split(args.split))

    client = MockClient(config) if args.mock else build_client(config)
    if args.mock:
        index = build_index(kb, client, config)
    else:
        index = Index.load(index_path_for(config), kb=kb, config=config)

    from sme_assistant.ingest.chunker import chunk_corpus

    chunk_texts = {
        c.chunk_id: c.text
        for c in chunk_corpus(
            kb,
            config.require("chunking.max_words"),
            config.require("chunking.overlap_sentences"),
            config.require("chunking.min_words"),
        )
    }
    report = evaluate(
        Retriever(index, client, config), questions, registry, chunk_texts
    )
    report["split"] = args.split
    report["mock"] = args.mock
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["provenance"] = {
        "corpus_sha256": kb.fingerprint(),
        "chunk_set_sha256": index.chunk_set_sha256,
        "registry_sha256": registry.fingerprint(),
        "current_settings": {
            "top_k": config.require("retrieval.top_k"),
            "min_similarity": config.require("retrieval.min_similarity"),
        },
    }

    print(f"Split           {args.split}, {report['questions']} questions"
          f"{' (MOCK)' if args.mock else ''}")
    print(f"Corpus          {kb.short_fingerprint()}")
    print(f"Current setting top_k={config.require('retrieval.top_k')} "
          f"min_similarity={config.require('retrieval.min_similarity')}")
    print()
    print("Recall over questions naming expected chunks "
          f"({report['questions_with_expected_chunks']}):")
    print("   k    strict  lenient")
    for k, values in report["recall"].items():
        print(f"  {k:>4}   {values['strict']:.3f}   {values['lenient']:.3f}")
    print(f"  MRR    {report['mrr']}")
    print()
    print("Conflict-pair recall, both documents present in the top k:")
    for conflict_type, per_k in report["conflict_pair_recall"].items():
        row = "  ".join(f"{k} {v:.2f}" for k, v in per_k.items())
        print(f"  {conflict_type:22} {row}")
    print()
    print("Conflict-pair recall by family, which the type average can hide:")
    for family_id, row in report["conflict_pair_recall_by_family"].items():
        marks = "  ".join(f"{k} {v:.2f}" for k, v in row.items() if k.startswith("@"))
        print(f"  {family_id:9} {row['type']:22} {marks}")
    print()
    print("Threshold sweep (a question is refused when its best score is below):")
    print("  min_sim   answerable wrongly refused   unanswerable correctly refused   J")
    for row in report["threshold_sweep"]:
        print(f"  {row['min_similarity']:>6.2f}   "
              f"{row['answerable_wrongly_refused']:>3}/{row['answerable_total']:<3}"
              f"                    "
              f"{row['unanswerable_correctly_refused']:>3}/{row['unanswerable_total']:<3}"
              f"               {row['youden_j']:+.3f}")

    out = Path(config.path("paths.results")).parent / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "_mock" if args.mock else ""
    path = out / f"{stamp}_{args.split}{tag}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {path}")

    best_j = max(report["threshold_sweep"], key=lambda r: r["youden_j"])
    print(f"\nHighest Youden J at min_similarity={best_j['min_similarity']}. "
          "Treat as a candidate, not a decision: J weights the two error types "
          "equally, and wrongly refusing an answerable question is not equally "
          "bad as answering an unanswerable one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
