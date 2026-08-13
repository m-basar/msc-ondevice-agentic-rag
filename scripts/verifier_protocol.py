"""Execute the diagnostic protocol in docs/VERIFIER_PROTOCOL.md.

96 verifier calls: 8 development families x 3 paraphrases x 2 models x 2
evidence conditions. Nothing here is reported as a result, and nothing here
touches the test split.

    python scripts/verifier_protocol.py
    python scripts/verifier_protocol.py --dry-run    # design and counts only

The protocol document must be committed before this runs. The check is
enforced rather than trusted: a pre-registered decision rule written after the
numbers are known is not a pre-registered decision rule, and the easiest way to
end up with one is to write the document and the results in the same sitting.

The two conditions
------------------
``full``        the deployed retrieval, top_k 6 and min_similarity 0.30
``oracle_pair`` only the chunks anchoring both sides of the disputed fact

The oracle condition answers one question: can the verifier do the reasoning
when the reasoning is all that is left to do? Selecting it requires already
knowing which passages carry the disputed claims, which is what the system is
supposed to work out, so it is an instrument and never a candidate design.

Both models audit the **same draft**, generated once per question and
condition. Regenerating per model would let generator sampling differences
appear as verifier differences, which is the same error the B and D arms share
a draft to avoid.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_retrieval import anchor_chunks  # noqa: E402
from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import build_client  # noqa: E402
from sme_assistant.evaluation.answer_scoring import chunk_text_map  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.evaluation.stopping_gate import (  # noqa: E402
    DECLARED_TO_INFERRED,
    MAJORITY,
)
from sme_assistant.generate.generator import Generator  # noqa: E402
from sme_assistant.ingest.index import Index, index_path_for  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402
from sme_assistant.retrieve.retriever import Retriever  # noqa: E402
from sme_assistant.verify import schema  # noqa: E402
from sme_assistant.verify.verifier import (  # noqa: E402
    VERIFIER_SYSTEM,
    build_verification_prompt,
)

PROTOCOL = ROOT / "docs" / "VERIFIER_PROTOCOL.md"
MODELS = ("llama3.2:3b", "qwen2.5:3b")
CONDITIONS = ("full", "oracle_pair")

# Everything that determines what the 96 calls do. All of it must be committed
# before the run, not merely the document describing the run.
RUNTIME_PATHS = (
    "src",                          # verifier, prompt, parsing, retrieval
    "scripts/verifier_protocol.py",  # this harness
    "scripts/evaluate_retrieval.py",  # anchor_chunks, which builds the oracle
    "gold",                         # families, questions, declared types
    "config.json",                  # models, top_k, min_similarity, options
    "docs/VERIFIER_PROTOCOL.md",    # the design and the decision rules
)


def git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(ROOT), *args],
                         capture_output=True, text=True)
    return out.stdout.strip()


def require_committed_protocol() -> dict[str, str]:
    """Refuse to run until the protocol **and everything that runs** is committed.

    Checking only the protocol file would have been theatre. The 96 calls are
    produced by the verifier prompt, the parsing rules, the retrieval settings
    and the gold data, none of which live in ``docs/``. An uncommitted change
    to any of them would influence every result while the protocol sat frozen
    in git looking authoritative.

    ``results/`` is deliberately not checked: it is where this run writes.
    """
    if not PROTOCOL.exists():
        raise SystemExit(f"{PROTOCOL} does not exist. Write the protocol first.")

    dirty = git("status", "--porcelain", "--", *RUNTIME_PATHS)
    if dirty:
        raise SystemExit(
            "The runtime state is not committed:\n\n"
            + "\n".join(f"  {line}" for line in dirty.splitlines())
            + "\n\nThese paths decide what the 96 calls do. A pre-registered "
            "protocol sitting above uncommitted verifier code, prompts, "
            "configuration or gold data is not a frozen experiment, however "
            "carefully the protocol itself was written.\n\n"
            "  git add -A && git commit"
        )
    return {
        "commit": git("rev-parse", "HEAD"),
        "protocol_committed_at": git("log", "-1", "--format=%cI", "--", str(PROTOCOL)),
        "runtime_paths_checked": list(RUNTIME_PATHS),
        # Recorded, not enforced. Untracked output under results/ does not
        # affect the run, and refusing on it would only teach the operator to
        # reach for a bypass flag.
        "other_paths_dirty": bool(git("status", "--porcelain")),
    }


def pair_is_present(anchors, given) -> bool:
    """Were both sides of the disputed fact actually in the evidence?

    Both sides, from two different documents. One side present is not the
    pair, and counting it as such would make the confound check pass exactly
    when it most needs to fail: a verifier shown only one position has nothing
    to detect, and its silence would be read as a reasoning failure.
    """
    given = set(given)
    return len({a.split("#")[0] for a in anchors if a in given}) >= 2


def oracle_retrieval(retriever, question, family, chunk_texts, total_chunks: int):
    """A retrieval containing only the chunks anchoring the disputed fact.

    Built by ranking every chunk and then keeping the anchors, so each kept
    chunk carries its real similarity score and the object is a genuine
    ``RetrievalResult`` rather than a hand-assembled imitation.
    """
    wanted = anchor_chunks(family, chunk_texts, question.expected_chunks)
    everything = retriever.retrieve(
        question.text, top_k=total_chunks, min_similarity=0.0
    )
    kept = tuple(s for s in everything.results if s.chunk_id in wanted)
    return dataclasses.replace(
        everything, results=kept, top_k=len(kept), below_threshold=not kept,
    ), sorted(wanted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the design and the call count, run nothing")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    config = load_config()
    evaluation = load_evaluation_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(evaluation.path("conflicts"))
    question_set = load_question_set(evaluation.path("question_set"))

    questions = [q for q in question_set.split("dev") if q.family_id]
    families = {q.family_id for q in questions}
    calls = len(questions) * len(args.models) * len(CONDITIONS)

    print(f"Families    {len(families)}  ({', '.join(sorted(families))})")
    print(f"Questions   {len(questions)}   all paraphrases, no selection")
    print(f"Models      {', '.join(args.models)}")
    print(f"Conditions  {', '.join(CONDITIONS)}")
    print(f"Calls       {calls}\n")

    if args.dry_run:
        print("Dry run. Nothing executed.")
        return 0

    provenance = require_committed_protocol()
    print(f"Protocol committed at {provenance['protocol_committed_at']}")
    print(f"Running at commit     {provenance['commit'][:12]}\n")

    client = build_client(config)
    index = Index.load(index_path_for(config), kb=kb, config=config)
    retriever = Retriever(index, client, config)
    generator = Generator(client, config)
    chunk_texts = {e.chunk_id: e.chunk.text for e in index}

    rows: list[dict] = []
    print(f"{'question':<14}{'declared':<22}{'cond':<12}{'model':<14}"
          f"{'pair':<6}inferred")
    print("-" * 96)

    for question in questions:
        family = registry.by_id(question.family_id)
        expected = DECLARED_TO_INFERRED.get(family.conflict_type)

        for condition in CONDITIONS:
            if condition == "full":
                retrieval = retriever.retrieve(question.text)
                wanted = sorted(anchor_chunks(
                    family, chunk_texts, question.expected_chunks
                ))
            else:
                retrieval, wanted = oracle_retrieval(
                    retriever, question, family, chunk_texts, len(index)
                )

            present = {s.chunk_id for s in retrieval.results}
            pair_present = pair_is_present(wanted, present)

            # One draft per question and condition. Both models audit it.
            draft = generator.answer(question.text, retrieval)
            prompt = build_verification_prompt(
                question.text, draft.answer, retrieval.evidence_text()
            )
            options = {k: v for k, v in config.require("verification").items()
                       if k != "confidence" and not k.startswith("_")}

            for model in args.models:
                generation = client.generate(prompt, model=model, options=options)
                result = schema.parse(
                    generation.text, present, {}, draft.answer
                )
                detected = result.conflict_detected
                classified = result.relationship == expected
                print(f"{question.question_id:<14}{family.conflict_type:<22}"
                      f"{condition:<12}{model:<14}"
                      f"{'yes' if pair_present else 'NO':<6}{result.relationship}"
                      + ("  <-- detected" if detected else ""))

                rows.append({
                    "question_id": question.question_id,
                    "family_id": question.family_id,
                    "declared": family.conflict_type,
                    "is_conflict": bool(family.is_conflict),
                    "condition": condition,
                    "model": model,
                    "chunks_given": sorted(present),
                    "anchor_chunks": wanted,
                    "pair_present": pair_present,
                    # The seven pre-committed outputs, recorded separately.
                    "detected": detected,
                    "classified": classified,
                    "expected_relationship": expected,
                    "inferred_relationship": result.relationship,
                    "parse_failed": result.parse_failed,
                    "invented_ids": list(result.invented_ids),
                    "evidence_ids_valid": not result.invented_ids,
                    "validation_failures": list(result.validation_failures),
                    "escalate": result.escalate,
                    "rationale": result.rationale,
                    "raw": generation.text,
                    "seconds": generation.wall_seconds,
                    "options": generation.options,
                })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(config.path("paths.results")).parent / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stamp}_verifier_protocol.json"
    path.write_text(json.dumps({
        "protocol": "docs/VERIFIER_PROTOCOL.md",
        "provenance": provenance,
        "prompt_sha256": __import__("hashlib").sha256(
            VERIFIER_SYSTEM.encode("utf-8")
        ).hexdigest(),
        # The state the calls actually ran against, so the record does not rely
        # on the commit alone being read correctly later.
        "config_sha256": config.fingerprint(),
        "corpus_sha256": kb.fingerprint(),
        "chunk_set_sha256": getattr(index, "chunk_set_sha256", None),
        "registry_sha256": registry.fingerprint(),
        "models": list(args.models),
        "conditions": list(CONDITIONS),
        "rows": rows,
    }, indent=2), encoding="utf-8")

    print()
    summarise(rows, args.models)
    print(f"\nWritten to {path}")
    print("\nApply the decision rules in docs/VERIFIER_PROTOCOL.md section 4, "
          "in order. Do not read past the first rule that matches.")
    return 0


def summarise(rows: list[dict], models) -> None:
    """Family-level majorities, per model and condition, controls separate."""
    by_family = defaultdict(list)
    for row in rows:
        by_family[(row["model"], row["condition"], row["family_id"])].append(row)

    print("Family-level majority, detection / classification "
          f"({MAJORITY} of 3 paraphrases):")
    print(f"  {'model':<14}{'condition':<12}{'genuine det':>12}"
          f"{'genuine cls':>12}{'CONTROL fp':>12}{'pair absent':>12}")
    for model in models:
        for condition in CONDITIONS:
            genuine_d = genuine_c = controls_fp = genuine_n = controls_n = 0
            absent = 0
            for (m, c, _fid), group in by_family.items():
                if (m, c) != (model, condition):
                    continue
                detected = sum(r["detected"] for r in group) >= MAJORITY
                classified = sum(r["classified"] for r in group) >= MAJORITY
                if not any(r["pair_present"] for r in group):
                    absent += 1
                if group[0]["is_conflict"]:
                    genuine_n += 1
                    genuine_d += detected
                    genuine_c += classified
                else:
                    controls_n += 1
                    controls_fp += detected
            print(f"  {model:<14}{condition:<12}"
                  f"{f'{genuine_d}/{genuine_n}':>12}{f'{genuine_c}/{genuine_n}':>12}"
                  f"{f'{controls_fp}/{controls_n}':>12}{absent:>12}")

    failures = sum(r["parse_failed"] for r in rows)
    invented = sum(not r["evidence_ids_valid"] for r in rows)
    print(f"\n  parse failures {failures}/{len(rows)}   "
          f"invalid evidence ids {invented}/{len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
