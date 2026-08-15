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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.common.config import load_config  # noqa: E402
from sme_assistant.common.llm_client import MockClient, build_client  # noqa: E402
from sme_assistant.evaluation.answer_scoring import chunk_text_map, score_answer  # noqa: E402
from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.conflicts import load_conflicts  # noqa: E402
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.evaluation.replay import load_drafts  # noqa: E402
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


def write_rejection(config, *, split, condition, stage, reason, **detail) -> Path:
    """Retain a machine-readable record of why a run must not be reported.

    Amendment 1.19. Uniquely named, so a second failure never overwrites the
    first, and never instructing anyone to delete anything: the run directory
    stays exactly as written and is excluded by this record, not by removal.
    Deleting the evidence of a failed condition is how a failed condition
    becomes an unrecorded one.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    target = Path(config.path("paths.results")) / (
        f"REJECTED_{split}_performance_{condition}_{stage}_{stamp}.json"
    )
    target.write_text(json.dumps({
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "reason": reason,
        "hardware_condition": condition,
        "split": split,
        **detail,
        "retention": (
            "Every run directory written by this invocation is retained "
            "unchanged. Nothing is deleted or re-tagged. This record excludes "
            "the affected run from analysis; the files remain as evidence."
        ),
    }, indent=2), encoding="utf-8", newline="\n")
    return target


def run_arm(name, questions, *, retriever, generator, verifier, config, kb, index,
            question_set, registry, split, tag, root=None, drafts=None,
            drafts_source=None, performance_only=False,
            hardware_condition=None, placement=None) -> Path:
    """Run one arm. ``drafts`` lets D reuse B's exact answers.

    B versus D is the confirmatory contrast, and it only isolates verification
    if the two share a draft. Regenerating independently would let sampling
    noise in the first pass masquerade as a verification effect: D could look
    better or worse than B for reasons that have nothing to do with the layer
    being evaluated.
    """
    spec = ARMS[name]
    produced: dict[str, object] = {}
    # Amendment 1.15 forbids a hardware run producing any quality figure, and
    # amendment 1.16 makes that structural rather than a rule to remember:
    # score_answer is not called at all, so there is no citation metric in the
    # record to be picked up by accident later.
    writer = RunWriter(
        arm_definition(name, config), split=split, tag=tag, config=config,
        kb=kb, index=index, question_set=question_set, registry=registry, root=root,
        purpose="performance" if performance_only else "quality",
        hardware_condition=hardware_condition,
        placement=placement,
    )

    for position, question in enumerate(questions, start=1):
        reused = (drafts or {}).get(question.question_id)
        if reused is not None:
            answer = reused
            retrieval = answer.retrieval
        else:
            retrieval = retriever.retrieve(question.text, mode=spec["retrieval_mode"])
            answer = generator.answer(
                question.text, retrieval, evidence_format=spec["evidence_format"]
            )
        produced[question.question_id] = answer

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
            scoring=(
                None if performance_only
                else score_answer(served, chunk_text_map(retrieval)).to_dict()
            ),
            extra={"expected_behaviour": question.expected_behaviour,
                   "risk_level": question.risk_level},
        )
        if position % 10 == 0 or position == len(questions):
            print(f"    {position}/{len(questions)}")

    directory = writer.finish({
        "questions": len(questions),
        "drafts_reused_from": "B" if drafts else None,
        # Which run the drafts came from, when they were replayed rather than
        # generated in this process. Without it a controlled re-run and an
        # ordinary one leave identical-looking directories.
        "drafts_replayed_from": drafts_source,
        # Stated in the run rather than inferred from it later. A D run that
        # did not reuse B's drafts is exploratory, and six months on nobody
        # will remember which of two identical-looking directories that was.
        "isolates_verification": bool(drafts) if spec["verification"] else None,
    })
    return directory, produced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--performance-only", action="store_true",
        help="Timing run under amendment 1.15. No answer scoring, arms B and D "
             "only, a tag and an explicit placement are required, and the run "
             "index is written separately so it cannot be mistaken for the "
             "frozen quality run.",
    )
    parser.add_argument(
        "--placement", choices=["gpu", "cpu"],
        help="Where generation runs. Required with --performance-only, because "
             "a latency figure whose device is unrecorded measures nothing.",
    )
    parser.add_argument(
        "--hardware-condition", choices=["laptop_gpu", "laptop_cpu", "pi5_cpu"],
        help="Named condition from config.json. Required with --performance-only.",
    )
    parser.add_argument("--i-have-frozen-everything", action="store_true",
                        help="required to run the test split")
    parser.add_argument("--d-without-b", action="store_true",
                        help="allow an exploratory Arm D that cannot support H2")
    parser.add_argument("--reuse-drafts-from", metavar="RUN_DIR",
                        help="replay a recorded Arm B run's drafts instead of "
                             "regenerating them, for a controlled re-run")
    args = parser.parse_args()

    # Amendment 1.16. Every one of these is a way a performance run could end
    # up indistinguishable from the frozen quality run, so each is refused up
    # front rather than checked later.
    if args.performance_only:
        problems = []
        if not args.tag:
            problems.append(
                "--tag is required: an untagged run on the test split is named "
                "exactly like a quality run"
            )
        if args.split != "test":
            problems.append(
                f"--split must be 'test', not {args.split!r}. H5 and RQ4 are "
                "stated over the test questions, and --split defaults to dev, "
                "so a timing run would silently measure the wrong set"
            )
        if args.mock:
            problems.append(
                "--mock cannot be combined with a performance run: mock "
                "timings measure the harness, not the model"
            )
        if set(args.arms) != {"B", "D"}:
            problems.append(
                "arms must be exactly B and D. H5 is a ratio of the two, and a "
                f"run of one of them cannot produce it; got {sorted(args.arms)}"
            )
        if not args.placement:
            problems.append("--placement is required (gpu or cpu)")
        if not args.hardware_condition:
            problems.append("--hardware-condition is required")

        # A condition names a machine and a device together. Accepting
        # pi5_cpu with GPU placement would record a Pi CPU-only result
        # produced on a GPU, which is a false provenance block rather than a
        # mistake in the numbers.
        expected = {"laptop_gpu": "gpu", "laptop_cpu": "cpu", "pi5_cpu": "cpu"}
        if args.hardware_condition and args.placement:
            wanted = expected.get(args.hardware_condition)
            if wanted and wanted != args.placement:
                problems.append(
                    f"--hardware-condition {args.hardware_condition} requires "
                    f"--placement {wanted}, not {args.placement}"
                )
        if problems:
            parser.error(
                "performance-only run refused:\n  - " + "\n  - ".join(problems)
            )
    elif args.placement or args.hardware_condition:
        parser.error(
            "--placement and --hardware-condition apply only to "
            "--performance-only runs"
        )

    config_overrides: dict[str, object] = {}
    if args.placement:
        # num_gpu 0 keeps every layer on the CPU. Recorded as well as applied,
        # because a latency number whose device was assumed rather than set is
        # not a measurement of that device.
        config_overrides["num_gpu"] = 0 if args.placement == "cpu" else -1

    if args.split == "test" and args.d_without_b:
        raise SystemExit(
            "--d-without-b marks a run exploratory. The test split produces "
            "the reported numbers, so it cannot be exploratory."
        )

    if args.split == "test" and not args.i_have_frozen_everything:
        raise SystemExit(
            "A test-split run produces the reported numbers. Pass "
            "--i-have-frozen-everything only when the verifier is frozen, the "
            "gold data is frozen, and the pre-registration is committed."
        )

    config = load_config()
    if config_overrides:
        # Applied to the generation block, which OllamaClient starts from and
        # which RunWriter records verbatim in the manifest, so the device the
        # run actually used is in its own provenance.
        config._data.setdefault("generation", {}).update(config_overrides)

    observed_placement = None
    placement_observations: dict[str, dict] = {}
    if args.performance_only and not args.mock:
        # Ollama keeps a model resident, and a resident model keeps the
        # placement it was loaded with. Without an eviction a cpu run started
        # after a gpu run measures the gpu. Amendment 1.17.
        from sme_assistant.common.llm_client import OllamaClient  # noqa: E402

        probe = OllamaClient(config)
        embedding_model = config.require("llm.embedding_model")
        # An embedding model does not serve /api/generate, so it must be
        # evicted through the embedding endpoint or it stays resident with its
        # previous placement. Amendment 1.18.
        to_evict = [
            (config.require("llm.generation_model"), False),
            (config.require("llm.verification_model"), False),
            (embedding_model, True),
        ]
        for model, is_embedding in to_evict:
            try:
                probe.unload(model, embedding=is_embedding)
            except Exception as exc:
                # Amendment 1.19. Printing and continuing then announced
                # "evicted loaded models", which was a claim the run had just
                # failed to establish. A model that was not evicted keeps its
                # previous placement, so the condition is not the one named.
                record = write_rejection(
                    config, split=args.split,
                    condition=args.hardware_condition, stage="eviction",
                    reason="eviction_failed", model=model,
                    embedding_endpoint=is_embedding, error=str(exc),
                    note=("The requested placement cannot be established while "
                          "a model remains resident from a previous run."),
                )
                print(f"\n  EVICTION FAILED for {model}: {exc}")
                print(f"  Rejection record written to {record.name}")
                return 1

        try:
            preflight = probe.preflight(args.placement)
        except Exception as exc:
            record = write_rejection(
                config, split=args.split, condition=args.hardware_condition,
                stage="preflight", reason="preflight_failed", error=str(exc),
                note=("A synthetic evict-load-observe cycle failed, so eviction "
                      "or residency reporting cannot be relied on for this run."),
            )
            print(f"\n  PREFLIGHT FAILED: {exc}")
            print(f"  Rejection record written to {record.name}")
            return 1

        print(f"  evicted and preflighted: {args.placement} placement confirmed "
              "on a synthetic prompt")
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

    # B before D, so D can reuse B's drafts and the contrast isolates
    # verification rather than two independent first passes.
    ordered = sorted(args.arms, key=lambda a: (a == "D", a))
    directories, drafts = {}, {}

    if args.reuse_drafts_from:
        # A controlled re-run: hold the drafts fixed and change only what is
        # under test. Regenerating B would mix the change with a second
        # sample from the generator, and the two could not be told apart.
        source = Path(args.reuse_drafts_from)
        if not source.is_absolute():
            source = ROOT / source
        drafts["B"] = load_drafts(source, index, expect_arm="B")
        print(f"  Replaying {len(drafts['B'])} Arm B drafts from {source.name}")
        print("    evidence hashes verified against the recorded run\n")
        if "B" in ordered:
            raise SystemExit(
                "--reuse-drafts-from replays Arm B, so Arm B must not also be "
                "run. Use '--arms D' for a verification-only re-run."
            )
    for name in ordered:
        print(f"  Arm {name}: {ARMS[name]['description']}")
        reuse = drafts.get("B") if name == "D" else None
        if name == "D" and reuse:
            print("    reusing Arm B drafts, so B vs D isolates verification")
        elif name == "D" and not args.d_without_b:
            # Previously a warning, which meant a run whose confirmatory
            # contrast was invalid still produced a full results directory
            # that looked exactly like a valid one. A warning scrolls past;
            # by the time the numbers are being read, nobody remembers it.
            raise SystemExit(
                "Arm D was requested without Arm B. D would generate its own "
                "draft, so B vs D would compare two independent first passes "
                "and sampling noise would be indistinguishable from a "
                "verification effect. That is the confirmatory contrast for "
                "H2.\n\n"
                "Run '--arms B D', or pass --d-without-b if you genuinely want "
                "an exploratory D-only run."
            )
        elif name == "D":
            print("    EXPLORATORY: Arm B was not run. D generated its own "
                  "draft, so B vs D does not isolate verification and this "
                  "run cannot support H2.")
        directories[name], drafts[name] = run_arm(
            name, questions, retriever=retriever, generator=generator,
            verifier=verifier, config=config, kb=kb, index=index,
            question_set=question_set, registry=registry, split=args.split,
            tag=args.tag or ("mock" if args.mock else ""), drafts=reuse,
            performance_only=args.performance_only,
            hardware_condition=args.hardware_condition,
            placement=args.placement,
            # Repository-relative with forward slashes. Recorded verbatim,
            # a Windows path is unreadable on the Pi - the same portability
            # bug that once broke the run index across the two machines.
            drafts_source=(
                Path(args.reuse_drafts_from).as_posix()
                if name == "D" and reuse and args.reuse_drafts_from else None
            ),
        )
        print(f"    -> {directories[name].name}\n")
        if args.performance_only and not args.mock:
            # After **every** arm. Checking only the first would miss a device
            # change between B and D, which is exactly when it could happen:
            # D loads the verifier model that B never touched. Amendment 1.18.
            from sme_assistant.common.llm_client import OllamaClient  # noqa: E402

            try:
                seen = OllamaClient(config).observed_placement()
            except Exception as exc:
                # /api/ps unreachable is not evidence of CPU placement.
                record = write_rejection(
                    config, split=args.split,
                    condition=args.hardware_condition, stage=f"residency_{name}",
                    reason="residency_unreadable", arm=name, error=str(exc),
                    run_directory=str(directories[name].relative_to(ROOT).as_posix()),
                    note=("Residency could not be read, so the device this arm "
                          "ran on is unknown."),
                )
                print(f"\n  RESIDENCY UNREADABLE after arm {name}: {exc}")
                print(f"  Rejection record written to {record.name}")
                return 1

            placement_observations[name] = seen
            observed_placement = seen
            wanted_gpu = args.placement == "gpu"
            if not seen.get("complete", False) or seen["any_on_gpu"] != wanted_gpu:
                record = write_rejection(
                    config, split=args.split,
                    condition=args.hardware_condition, stage=f"placement_{name}",
                    reason=("residency_incomplete" if not seen.get("complete", False)
                            else "placement_mismatch"),
                    arm=name,
                    requested_placement=args.placement,
                    observed_placement=("gpu" if seen["any_on_gpu"] else "cpu"),
                    models_loaded=seen["models_loaded"],
                    vram_bytes=seen["vram_bytes"],
                    sizes=seen.get("sizes"),
                    residency_complete=seen.get("complete"),
                    run_directory=str(directories[name].relative_to(ROOT).as_posix()),
                    observations_so_far=placement_observations,
                    note=("The run is retained unchanged and excluded from "
                          "analysis by this record."),
                )
                print("\n  PLACEMENT NOT CONFIRMED\n"
                      f"    arm      : {name}\n"
                      f"    requested: {args.placement}\n"
                      f"    observed : {'gpu' if seen['any_on_gpu'] else 'cpu'}\n"
                      f"    complete : {seen.get('complete')}\n"
                      f"    loaded   : {seen['models_loaded']}\n"
                      f"  Rejection record written to {record.name}\n"
                      "  The run is retained and must not be reported.")
                return 1
            print(f"    placement confirmed for arm {name}: {args.placement}")

    # A performance run writes its own index. Overwriting latest_test.json
    # would repoint the name that every earlier note and script uses for the
    # frozen quality run, which is how a timing execution silently becomes the
    # reported one. Amendment 1.16.
    if args.performance_only:
        manifest = Path(config.path("paths.results")) / (
            f"latest_{args.split}_performance_{args.hardware_condition}.json"
        )
    else:
        manifest = Path(config.path("paths.results")) / (
            f"latest_{args.split}{'_mock' if args.mock else ''}.json"
        )
    # Repository-relative, so a run index written on Windows can be read on the
    # Pi. Absolute paths made the two machines unable to share a run index.
    index_payload: dict[str, object] = {
        name: str(Path(path).relative_to(ROOT).as_posix())
        for name, path in directories.items()
    }
    if args.performance_only:
        index_payload["_performance"] = {
            "hardware_condition": args.hardware_condition,
            "requested_placement": args.placement,
            "observed_placement": observed_placement,
            "observed_placement_by_arm": placement_observations,
            "split": args.split,
        }
    manifest.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(f"Run index written to {manifest}")
    if args.performance_only:
        print("\nPerformance-only run. No answers were scored and none may be.")
        print("Next: python scripts/analyse_performance.py "
              f"--index {manifest.name}")
    else:
        print("\nNext: python scripts/summarise_arms.py --split " + args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
