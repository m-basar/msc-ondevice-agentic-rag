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

# Blocks, not repeats, and one until the evidence says otherwise.
#
# A block is a complete 96-call pass in protocol order. The decision rules are
# applied **independently to each block** and the block outcomes reported with
# their mean and range. Three blocks are three results, not 288 independent
# observations: the calls within a block are not independent of each other, and
# pooling them would inflate every denominator.
#
# This is 1 because the evidence for raising it does not yet exist. The first
# determinism check repeated each prompt back to back and found 4 of 12 raw
# outputs changing, and I read that as the verifier being unreproducible. It
# was not: every first call matched the recorded run, and the changes appeared
# only on immediate repetition, which no real run performs. Whether any
# reported outcome moves is what matters and was not measured.
#
# scripts/check_determinism.py now measures it in protocol order and parses
# every response. Raise this to 3 only if that check shows a reported outcome
# moving, and record the reason in the pre-registration.
REPEATS = 1

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


def require_clean_pilot(config, *, allow_override: bool = False) -> None:
    """The protocol runs only after a pilot serves no invalid revision.

    This was written into the protocol document and enforced nowhere, which
    makes it a wish. Diagnosing a verifier while a known defect sits in the
    path that serves its output attributes the defect's effects to the model,
    and both defects found so far were in that path.
    """
    from sme_assistant.evaluation.run_writer import read_run
    from sme_assistant.evaluation.stopping_gate import evaluate_gate

    runs = sorted(p for p in Path(config.path("paths.results")).glob("*_D_dev_*")
                  if p.is_dir())
    if not runs:
        raise SystemExit(
            "No development Arm D run to check the precondition against.\n"
            "Run scripts/run_arms.py --split dev first."
        )
    latest = runs[-1]
    _, records = read_run(latest)
    invalid = evaluate_gate(records, None).invalid_revisions_served
    if invalid and not allow_override:
        raise SystemExit(
            f"{latest.name} served {invalid} invalid revision(s).\n\n"
            "The precondition in docs/VERIFIER_PROTOCOL.md section 7 is that a "
            "pilot serves none. Running the diagnostic over a pipeline with a "
            "known defect in the serving path would attribute its effects to "
            "the model.\n\nFix it, re-run the pilot, then run this."
        )
    print(f"Precondition  {latest.name}: {invalid} invalid revisions served"
          + ("   (OVERRIDDEN)" if invalid and allow_override else ""))


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
    parser.add_argument("--repeats", type=int, default=REPEATS, metavar="BLOCKS",
                        help="complete passes; the rules are applied per block")
    parser.add_argument("--i-accept-a-defective-pipeline", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    config = load_config()
    evaluation = load_evaluation_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(evaluation.path("conflicts"))
    question_set = load_question_set(evaluation.path("question_set"))

    questions = [q for q in question_set.split("dev") if q.family_id]
    families = {q.family_id for q in questions}
    calls = len(questions) * len(args.models) * len(CONDITIONS) * args.repeats

    print(f"Families    {len(families)}  ({', '.join(sorted(families))})")
    print(f"Questions   {len(questions)}   all paraphrases, no selection")
    print(f"Models      {', '.join(args.models)}")
    print(f"Conditions  {', '.join(CONDITIONS)}")
    print(f"Blocks      {args.repeats}  (complete passes; rules applied per block)")
    print(f"Calls       {calls}\n")

    if args.dry_run:
        print("Dry run. Nothing executed.")
        return 0

    provenance = require_committed_protocol()
    require_clean_pilot(config, allow_override=args.i_accept_a_defective_pipeline)
    print(f"Protocol committed at {provenance['protocol_committed_at']}")
    print(f"Running at commit     {provenance['commit'][:12]}\n")

    client = build_client(config)
    index = Index.load(index_path_for(config), kb=kb, config=config)
    retriever = Retriever(index, client, config)
    generator = Generator(client, config)
    chunk_texts = {e.chunk_id: e.chunk.text for e in index}

    # --- build every input once ---------------------------------------------
    # The draft is held fixed across repeats and models. Regenerating it would
    # add generator variability to verifier variability, and afterwards the two
    # could not be told apart. Same reason B and D share a draft.
    options = {k: v for k, v in config.require("verification").items()
               if k != "confidence" and not k.startswith("_")}
    sweep = []
    for question in questions:
        family = registry.by_id(question.family_id)
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
            draft = generator.answer(question.text, retrieval)
            sweep.append({
                "question": question,
                "family": family,
                "expected": DECLARED_TO_INFERRED.get(family.conflict_type),
                "condition": condition,
                "present": present,
                "anchors": wanted,
                "pair_present": pair_is_present(wanted, present),
                "draft": draft.answer,
                "prompt": build_verification_prompt(
                    question.text, draft.answer, retrieval.evidence_text()
                ),
            })
    print(f"{len(sweep)} prompts built, drafts fixed across repeats\n")

    # --- sweep, then sweep again ---------------------------------------------
    rows: list[dict] = []
    print(f"{'rep':<5}{'question':<14}{'declared':<22}{'cond':<12}{'model':<14}"
          f"{'pair':<6}inferred")
    print("-" * 101)
    for repeat in range(1, args.repeats + 1):
        for item in sweep:
            for model in args.models:
                generation = client.generate(
                    item["prompt"], model=model, options=options
                )
                result = schema.parse(
                    generation.text, item["present"], {}, item["draft"]
                )
                detected = result.conflict_detected
                print(f"{repeat:<5}{item['question'].question_id:<14}"
                      f"{item['family'].conflict_type:<22}{item['condition']:<12}"
                      f"{model:<14}{'yes' if item['pair_present'] else 'NO':<6}"
                      f"{result.relationship}"
                      + ("  <-- detected" if detected else ""))
                rows.append({
                    "repeat": repeat,
                    "question_id": item["question"].question_id,
                    "family_id": item["question"].family_id,
                    "declared": item["family"].conflict_type,
                    "is_conflict": bool(item["family"].is_conflict),
                    "condition": item["condition"],
                    "model": model,
                    "chunks_given": sorted(item["present"]),
                    "anchor_chunks": item["anchors"],
                    "pair_present": item["pair_present"],
                    # The pre-committed outputs, recorded separately.
                    "detected": detected,
                    "classified": result.relationship == item["expected"],
                    "expected_relationship": item["expected"],
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
        print(f"  --- sweep {repeat} of {args.repeats} complete ---\n")

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


def stability(rows: list[dict]) -> dict[str, float]:
    """How often the blocks of one prompt agreed with each other.

    Reported per outcome, because they are not equally stable, and per prompt,
    never from totals. Between pilots 02 and 03 the aggregate relationship
    counts were identical, 36/4/1 both times, while four questions moved: two
    one way, two the other, cancelling. Read from totals that is perfect
    stability with a tenth of the questions having changed answer.
    """
    cells = defaultdict(list)
    for row in rows:
        cells[(row["model"], row["condition"], row["question_id"])].append(row)

    measures = {}
    for field in ("detected", "classified", "parse_failed",
                  "inferred_relationship", "raw"):
        agreed = sum(len({r[field] for r in group}) == 1 for group in cells.values())
        measures["raw_text" if field == "raw" else field] = (
            agreed / len(cells) if cells else 1.0
        )
    return measures


def block_result(rows: list[dict], model: str, condition: str) -> dict:
    """Family-level majorities for one block of one model and condition.

    The majority is of the family's three paraphrases, within the block. Across
    blocks the results are reported side by side rather than pooled, because a
    block is one run of the protocol and three runs are three results.
    """
    by_family = defaultdict(list)
    for row in rows:
        if row["model"] == model and row["condition"] == condition:
            by_family[row["family_id"]].append(row)

    counts = dict(genuine_detected=0, genuine_classified=0, genuine=0,
                  controls_false=0, controls=0, pair_absent=0)
    for group in by_family.values():
        needed = len(group) / 2
        detected = sum(r["detected"] for r in group) > needed
        classified = sum(r["classified"] for r in group) > needed
        if not any(r["pair_present"] for r in group):
            counts["pair_absent"] += 1
        if group[0]["is_conflict"]:
            counts["genuine"] += 1
            counts["genuine_detected"] += detected
            counts["genuine_classified"] += classified
        else:
            counts["controls"] += 1
            counts["controls_false"] += detected
    return counts


def summarise(rows: list[dict], models) -> None:
    blocks = sorted({r["repeat"] for r in rows})

    if len(blocks) > 1:
        measures = stability(rows)
        print(f"Block agreement over {len(blocks)} blocks "
              "(1.00 means every block of a prompt agreed):")
        for field in ("raw_text", "inferred_relationship", "parse_failed",
                      "classified", "detected"):
            flag = "" if measures[field] == 1.0 else "   <-- varies between blocks"
            print(f"  {field:<24}{measures[field]:>6.2f}{flag}")
        print("  Below 1.00 is a distribution, not a value.\n")

    print("Family-level majority per block, detection / classification "
          "(majority of 3 paraphrases):")
    print(f"  {'model':<14}{'condition':<12}{'block':>6}{'genuine det':>13}"
          f"{'genuine cls':>13}{'CONTROL fp':>12}{'pair absent':>12}")
    for model in models:
        for condition in CONDITIONS:
            seen = []
            for block in blocks:
                c = block_result([r for r in rows if r["repeat"] == block],
                                 model, condition)
                seen.append(c)
                det = f"{c['genuine_detected']}/{c['genuine']}"
                cls = f"{c['genuine_classified']}/{c['genuine']}"
                fp = f"{c['controls_false']}/{c['controls']}"
                print(f"  {model:<14}{condition:<12}{block:>6}"
                      f"{det:>13}{cls:>13}{fp:>12}{c['pair_absent']:>12}")
            if len(seen) > 1:
                detected = [c["genuine_detected"] for c in seen]
                print(f"  {'':<26}{'mean':>6}{sum(detected) / len(detected):>13.1f}"
                      f"{'':>13}{'':>12}   range {min(detected)}-{max(detected)}")

    failures = sum(r["parse_failed"] for r in rows)
    invented = sum(not r["evidence_ids_valid"] for r in rows)
    print(f"\n  parse failures {failures}/{len(rows)}   "
          f"invalid evidence ids {invented}/{len(rows)}")
    if len(blocks) > 1:
        print(f"  {len(blocks)} blocks are {len(blocks)} results, not "
              f"{len(rows)} independent observations.")


if __name__ == "__main__":
    raise SystemExit(main())
