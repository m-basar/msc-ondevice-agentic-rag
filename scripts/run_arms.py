"""Run the experimental arms over a split and record everything.

Four arms, one question set, one index. The arms differ in exactly the three
respects the pre-registration names, and nothing else: retrieval mode, evidence
format, and whether the verification layer runs.

    A   ALL          PLAIN         no verification   naive retrieval-augmented generation
    B   ALL          WITH_STATUS   no verification   metadata shown in the prompt
    C   CURRENT_ONLY WITH_STATUS   no verification   the cheap metadata filter
    D   ALL          WITH_STATUS   verification      the contribution

Run:

    python scripts/run_arms.py --split dev            # all four arms
    python scripts/run_arms.py --split dev --arms B D # a subset
    python scripts/run_arms.py --split dev --mock     # structure only, no Ollama

The test split is refused without ``--i-have-frozen-everything``. That flag is
deliberately awkward to type. A test run is the one that produces reported
numbers, and it should not be possible to start one by editing a default.

Ordering
--------
Questions are answered arm by arm rather than question by question. On the Pi
this matters: a model stays loaded across a whole arm instead of being swapped
per question, and the thermal profile of an arm is at least internally
consistent rather than interleaved with three others.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import MockClient, build_client  # noqa: E402
from sme_assistant.evaluation.answer_scoring import chunk_text_map, score_answer  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.evaluation.run_writer import ArmDefinition, RunWriter  # noqa: E402
from sme_assistant.generate.generator import Generator  # noqa: E402
from sme_assistant.ingest.index import Index, build_index, index_path_for  # noqa: E402
from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402
from sme_assistant.retrieve.retriever import EvidenceFormat, RetrievalMode, Retriever  # noqa: E402
from sme_assistant.verify import Verifier  # noqa: E402

ARMS = {
    "A": dict(description="Naive RAG: identifier and text only, no status metadata",
              retrieval_mode=RetrievalMode.ALL, evidence_format=EvidenceFormat.PLAIN,
              verification=False),
    "B": dict(description="Status metadata shown in the prompt",
              retrieval_mode=RetrievalMode.ALL, evidence_format=EvidenceFormat.WITH_STATUS,
              verification=False),
    "C": dict(description="Metadata filter: superseded documents dropped before ranking",
              retrieval_mode=RetrievalMode.CURRENT_ONLY,
              evidence_format=EvidenceFormat.WITH_STATUS, verification=False),
    "D": dict(description="Verification layer over the same evidence as B",
              retrieval_mode=RetrievalMode.ALL, evidence_format=EvidenceFormat.WITH_STATUS,
              verification=True),
}


def arm_definition(name: str, config) -> ArmDefinition:
    spec = ARMS[name]
    return ArmDefinition(
        arm=name,
        description=spec["description"],
        retrieval_mode=spec["retrieval_mode"].value,
        evidence_format=spec["evidence_format"].value,
        verification=spec["verification"],
        generation_model=config.require("llm.generation_model"),
        verification_model=(
            config.get("llm.verification_model") if spec["verification"] else None
        ),
    )


def run_arm(name, questions, *, retriever, generator, verifier, config, kb, index,
            question_set, registry, split, tag, root=None) -> Path:
    spec = ARMS[name]
    writer = RunWriter(
        arm_definition(name, config), split=split, tag=tag, config=config,
        kb=kb, index=index, question_set=question_set, registry=registry, root=root,
    )

    for position, question in enumerate(questions, start=1):
        retrieval = retriever.retrieve(question.text, mode=spec["retrieval_mode"])
        answer = generator.answer(
            question.text, retrieval, evidence_format=spec["evidence_format"]
        )
        record = answer
        if spec["verification"]:
            record = verifier.verify(answer)

        served = getattr(record, "final_answer", answer.answer)
        writer.record(
            question_id=question.question_id,
            question=question.text,
            answer=record,
            group_id=question.group_id,
            family_id=question.family_id,
            category=question.category,
            scoring=score_answer(served, chunk_text_map(retrieval)).to_dict(),
            extra={"expected_behaviour": question.expected_behaviour,
                   "risk_level": question.risk_level},
        )
        if position % 10 == 0 or position == len(questions):
            print(f"    {position}/{len(questions)}")

    return writer.finish({"questions": len(questions)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--i-have-frozen-everything", action="store_true",
                        help="required to run the test split")
    args = parser.parse_args()

    if args.split == "test" and not args.i_have_frozen_everything:
        raise SystemExit(
            "A test-split run produces the reported numbers. Pass "
            "--i-have-frozen-everything only when the verifier is frozen, the "
            "gold data is frozen, and the pre-registration is committed."
        )

    config = load_config()
    evaluation = load_evaluation_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    registry = load_conflicts(evaluation.path("conflicts"))
    question_set = load_question_set(evaluation.path("question_set"))
    questions = list(question_set.split(args.split))

    client = MockClient(config) if args.mock else build_client(config)
    index = (build_index(kb, client, config) if args.mock
             else Index.load(index_path_for(config), kb=kb, config=config))

    retriever = Retriever(index, client, config)
    generator = Generator(client, config)
    verifier = Verifier(client, config)

    print(f"Split      {args.split}, {len(questions)} questions"
          f"{'  (MOCK)' if args.mock else ''}")
    print(f"Corpus     {kb.short_fingerprint()}   chunks {index.chunk_set_sha256[:12]}")
    print(f"Models     {config.require('llm.generation_model')} / "
          f"{config.get('llm.verification_model')}")
    print(f"Retrieval  top_k={config.require('retrieval.top_k')} "
          f"min_similarity={config.require('retrieval.min_similarity')}\n")

    directories = {}
    for name in args.arms:
        print(f"  Arm {name}: {ARMS[name]['description']}")
        directories[name] = run_arm(
            name, questions, retriever=retriever, generator=generator,
            verifier=verifier, config=config, kb=kb, index=index,
            question_set=question_set, registry=registry, split=args.split,
            tag=args.tag or ("mock" if args.mock else ""),
        )
        print(f"    -> {directories[name].name}\n")

    manifest = Path(config.path("paths.results")) / (
        f"latest_{args.split}{'_mock' if args.mock else ''}.json"
    )
    manifest.write_text(json.dumps(
        {name: str(path) for name, path in directories.items()}, indent=2
    ), encoding="utf-8")
    print(f"Run index written to {manifest}")
    print("\nNext: python scripts/summarise_arms.py --split " + args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
