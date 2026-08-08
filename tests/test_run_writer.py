"""Tests for run persistence and blinded review.

Stage 4 printed answers to a terminal and saved nothing. The pilot's four
outputs survive only because they were pasted into a document by hand, and the
chunk identifiers they cite no longer refer to the same text. These tests exist
so that cannot happen again.
"""

from __future__ import annotations

import json

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.common.llm_client import MockClient
from sme_assistant.evaluation.answer_scoring import chunk_text_map, score_answer
from sme_assistant.evaluation.run_writer import (
    ArmDefinition,
    RunWriter,
    read_run,
    write_review_sheet,
)
from sme_assistant.generate.generator import Generator
from sme_assistant.ingest.index import build_index
from sme_assistant.kb.loader import load_knowledge_base
from sme_assistant.retrieve.retriever import EvidenceFormat, Retriever


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def kb(config):
    return load_knowledge_base(config.path("paths.kb_docs"))


@pytest.fixture(scope="module")
def index(kb, config):
    return build_index(kb, MockClient(config), config)


def arm(name="B", evidence="with_status"):
    return ArmDefinition(
        arm=name,
        description="test arm",
        retrieval_mode="all",
        evidence_format=evidence,
        verification=False,
        generation_model="mock-generate",
    )


def make_run(tmp_path, kb, index, config, name="B", evidence=EvidenceFormat.WITH_STATUS,
             questions=("what is the mileage rate?", "how much annual leave?")):
    client = MockClient(config)
    retriever = Retriever(index, client, config)
    generator = Generator(client, config)
    writer = RunWriter(
        arm(name, evidence.value), split="dev", config=config, kb=kb, index=index,
        root=tmp_path,
    )
    for position, question in enumerate(questions, start=1):
        retrieval = retriever.retrieve(question, min_similarity=0.0)
        answer = generator.answer(question, retrieval, evidence_format=evidence)
        writer.record(
            question_id=f"DEV-{position:03d}",
            question=question,
            answer=answer,
            family_id="CONF-01" if position == 1 else None,
            category="conflict" if position == 1 else "factual",
            scoring=score_answer(answer.answer, chunk_text_map(retrieval)).to_dict(),
        )
    return writer.finish({"note": "test"})


# --- the manifest -----------------------------------------------------------


def test_manifest_records_everything_needed_to_reproduce(tmp_path, kb, index, config):
    directory = make_run(tmp_path, kb, index, config)
    manifest, _ = read_run(directory)

    provenance = manifest["provenance"]
    for key in ("config_sha256", "corpus_sha256", "chunk_set_sha256",
                "index_file_sha256", "git", "seed", "generation_options",
                "retrieval", "chunking"):
        assert key in provenance, f"manifest is missing {key!r}"

    assert provenance["corpus_sha256"] == kb.fingerprint()
    assert provenance["chunk_set_sha256"] == index.chunk_set_sha256
    assert provenance["seed"] == config.get("generation.seed")
    assert manifest["arm"]["arm"] == "B"
    assert manifest["split"] == "dev"


def test_the_seed_actually_reaches_the_manifest(tmp_path, kb, index, config):
    """It was previously recorded while never being sent to the model.

    A manifest that records a seed the run did not use is worse than one that
    records nothing, because it asserts a control that was not applied.
    """
    directory = make_run(tmp_path, kb, index, config)
    manifest, _ = read_run(directory)
    assert manifest["provenance"]["generation_options"]["seed"] is not None
    assert "seed" in config.require("generation")


# --- the records ------------------------------------------------------------


def test_the_exact_prompt_and_evidence_are_stored(tmp_path, kb, index, config):
    """A hash proves nothing changed. The text lets the run be understood
    without re-executing it, which matters when it took hours on a Pi."""
    directory = make_run(tmp_path, kb, index, config)
    _, answers = read_run(directory)
    for record in answers:
        assert record["prompt"], "the prompt was not stored"
        assert record["evidence"], "the evidence snapshot was not stored"
        assert len(record["prompt_sha256"]) == 64
        assert record["question"] in record["prompt"]


def test_each_record_carries_its_question_id_split_and_arm(tmp_path, kb, index, config):
    directory = make_run(tmp_path, kb, index, config)
    _, answers = read_run(directory)
    assert [r["question_id"] for r in answers] == ["DEV-001", "DEV-002"]
    assert all(r["split"] == "dev" for r in answers)
    assert all(r["arm"] == "B" for r in answers)


def test_scoring_travels_with_the_answer(tmp_path, kb, index, config):
    directory = make_run(tmp_path, kb, index, config)
    _, answers = read_run(directory)
    for record in answers:
        assert "scoring" in record
        assert "citation_support" in record["scoring"]
        assert record["scoring"]["is_lower_bound"] is True


def test_records_are_written_as_they_arrive(tmp_path, kb, index, config):
    """A run that dies at question 80 of 100 must leave 79 usable answers.

    On a Pi at several minutes per question, buffering until the end is the
    difference between losing a log line and losing an evening.
    """
    client = MockClient(config)
    retriever = Retriever(index, client, config)
    writer = RunWriter(arm(), split="dev", config=config, kb=kb, index=index, root=tmp_path)
    retrieval = retriever.retrieve("annual leave", min_similarity=0.0)
    answer = Generator(client, config).answer("q", retrieval)
    writer.record(question_id="DEV-001", question="q", answer=answer)

    # Not finished, yet the record is already on disk.
    lines = writer.answers_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["question_id"] == "DEV-001"


def test_every_record_carries_its_unit_of_independence(tmp_path, kb, index, config):
    """Aggregation happens later, often from the results file alone.

    A record that does not say which family it belongs to cannot be
    macro-averaged without going back to the question set to find out, and the
    person doing the aggregating is the one most likely to skip that step.
    """
    directory = make_run(tmp_path, kb, index, config)
    _, answers = read_run(directory)

    assert all(record.get("group_id") for record in answers)
    conflict = answers[0]
    assert conflict["group_id"] == "CONF-01", "a conflict answer groups by family"
    assert conflict["family_id"] == "CONF-01"

    standalone = answers[1]
    assert standalone["group_id"] == standalone["question_id"], (
        "a question belonging to no family is its own group, not pooled with others"
    )


def test_grouping_survives_blinding(tmp_path, kb, index, config):
    """Manual scores must be macro-averagable too.

    The group identifier says nothing about which arm produced the answer, so
    carrying it through costs no blinding.
    """
    directory = make_run(tmp_path, kb, index, config)
    sheet = tmp_path / "review.jsonl"
    write_review_sheet([directory], sheet)

    items = [json.loads(line) for line in sheet.read_text(encoding="utf-8").splitlines()]
    assert all("group_id" in item for item in items)
    assert all("arm" not in item for item in items)


def test_summary_records_the_environment_at_both_ends(tmp_path, kb, index, config):
    """A Pi that started at 60 degrees and ended throttled at 85 was not the
    same machine throughout."""
    directory = make_run(tmp_path, kb, index, config)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert "environment_at_end" in summary
    assert summary["records"] == 2
    assert summary["elapsed_seconds"] >= 0


# --- blinded review ---------------------------------------------------------


def test_review_sheet_hides_which_arm_produced_each_answer(tmp_path, kb, index, config):
    """The person scoring built the system and wants one arm to win.

    That is not a criticism of them; it is why blinding exists as a method.
    """
    run_b = make_run(tmp_path / "b", kb, index, config, name="B",
                     evidence=EvidenceFormat.WITH_STATUS)
    run_a = make_run(tmp_path / "a", kb, index, config, name="A",
                     evidence=EvidenceFormat.PLAIN)

    sheet = tmp_path / "review.jsonl"
    key = write_review_sheet([run_a, run_b], sheet)

    items = [json.loads(line) for line in sheet.read_text(encoding="utf-8").splitlines()]
    assert len(items) == 4
    for item in items:
        assert item["system"].startswith("system_")
        for leak in ("arm", "model", "prompt", "wall_seconds", "citations",
                     "generation", "retrieval", "scoring"):
            assert leak not in item, f"the review sheet leaks {leak!r}"

    assert set(key) == {"A", "B"}
    assert len(set(key.values())) == 2


def test_the_key_is_written_separately_and_warns(tmp_path, kb, index, config):
    run = make_run(tmp_path, kb, index, config)
    sheet = tmp_path / "review.jsonl"
    write_review_sheet([run], sheet)
    key_file = tmp_path / "review_key.json"
    assert key_file.exists(), "the key must be a separate file from the sheet"
    payload = json.loads(key_file.read_text(encoding="utf-8"))
    assert "Do not open" in payload["warning"]
    assert payload["seed"] == 42


def test_review_order_is_shuffled_but_reproducible(tmp_path, kb, index, config):
    run = make_run(tmp_path, kb, index, config,
                   questions=("a?", "b?", "c?", "d?", "e?", "f?"))
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"
    write_review_sheet([run], first, seed=7)
    write_review_sheet([run], second, seed=7)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")

    third = tmp_path / "three.jsonl"
    write_review_sheet([run], third, seed=8)
    order_a = [json.loads(l)["question_id"] for l in first.read_text().splitlines()]
    order_c = [json.loads(l)["question_id"] for l in third.read_text().splitlines()]
    assert order_a != order_c, "a different seed should give a different order"


# --- what the model actually saw --------------------------------------------


def test_arm_a_records_the_plain_evidence_it_was_actually_shown(tmp_path, kb, index, config):
    """The saved evidence must match the arm, not the default.

    record() called evidence_text() with no argument, which defaults to
    WITH_STATUS. Arm A is defined by receiving evidence with no status markers,
    so its records carried supersession markers the model never saw, and the
    saved prompt and the saved evidence disagreed with each other.
    """
    directory = make_run(tmp_path, kb, index, config, name="A", evidence=EvidenceFormat.PLAIN)
    _, answers = read_run(directory)

    for record in answers:
        assert "SUPERSEDED" not in record["evidence"], (
            "an Arm A record carries a supersession marker, so it is not the "
            "evidence the model was shown"
        )
        assert record["evidence"] in record["prompt"], (
            "the saved evidence does not appear in the saved prompt, so one of "
            "the two is a reconstruction rather than a record"
        )


def test_arm_b_records_the_status_markers_it_was_shown(tmp_path, kb, index, config):
    """The other half of the same control.

    Retrieval is widened deliberately so a superseded chunk is certainly in the
    evidence. Without that this test could pass by retrieving nothing withdrawn,
    and would be asserting nothing.
    """
    client = MockClient(config)
    retrieval = Retriever(index, client, config).retrieve(
        "expenses mileage pence per mile claim", top_k=40, min_similarity=0.0
    )
    assert any(not s.chunk.is_current for s in retrieval), (
        "no superseded chunk retrieved, so this test cannot exercise the marker"
    )

    answer = Generator(client, config).answer(
        "What is the mileage rate?", retrieval, evidence_format=EvidenceFormat.WITH_STATUS
    )
    writer = RunWriter(
        arm("B", "with_status"), split="dev", config=config, kb=kb, index=index,
        root=tmp_path,
    )
    writer.record(question_id="DEV-001", question="q", answer=answer)
    _, answers = read_run(writer.finish())
    assert "SUPERSEDED" in answers[0]["evidence"]


def test_an_unknown_evidence_format_is_refused(tmp_path, kb, index, config):
    """Silently recording the wrong format is the failure being prevented."""
    from sme_assistant.evaluation.run_writer import ArmDefinition, RunWriter
    from sme_assistant.generate.generator import Generator
    from sme_assistant.retrieve.retriever import Retriever

    bad = ArmDefinition(
        arm="X", description="malformed", retrieval_mode="all",
        evidence_format="not_a_format", verification=False,
        generation_model=config.require("llm.generation_model"),
    )
    client = MockClient(config)
    writer = RunWriter(bad, split="dev", config=config, kb=kb, index=index, root=tmp_path)
    retrieval = Retriever(index, client, config).retrieve("annual leave", min_similarity=0.0)
    answer = Generator(client, config).answer("q", retrieval)

    with pytest.raises(ValueError, match="not a known format"):
        writer.record(question_id="DEV-001", question="q", answer=answer)


# --- blinding ---------------------------------------------------------------


def test_the_review_sheet_does_not_leak_the_arm_through_the_evidence(tmp_path, kb, index, config):
    """Arm A's evidence has no status markers, which identified it on sight.

    A reviewer who noticed the pattern once had deanonymised the whole sheet.
    Blinding a careful reader can defeat is worse than none, because it is
    reported as a control.
    """
    run_a = make_run(tmp_path, kb, index, config, name="A", evidence=EvidenceFormat.PLAIN)
    run_b = make_run(tmp_path, kb, index, config, name="B", evidence=EvidenceFormat.WITH_STATUS)

    sheet = tmp_path / "review.jsonl"
    write_review_sheet([run_a, run_b], sheet)
    items = [json.loads(line) for line in sheet.read_text(encoding="utf-8").splitlines()]

    assert items, "no items to review"
    for item in items:
        assert "evidence" not in item
        assert "prompt" not in item
        assert "arm" not in item
        assert "SUPERSEDED" not in json.dumps(item)


def test_the_run_still_stores_the_evidence_verbatim(tmp_path, kb, index, config):
    """Removing it from the blinded sheet must not remove it from the record."""
    directory = make_run(tmp_path, kb, index, config)
    _, answers = read_run(directory)
    assert all(r["evidence"] for r in answers)
    assert all(r["evidence_sha256"] for r in answers)


# --- what the run was scored against ----------------------------------------


def test_the_manifest_records_the_gold_data_it_was_scored_against(tmp_path, kb, index, config):
    """A run recording the model and the corpus but not the question set or the
    rubric cannot be re-scored, and cannot be shown to have used the gold data
    it claims."""
    from sme_assistant.evaluation.config import load_evaluation_config
    from sme_assistant.evaluation.conflicts import load_conflicts
    from sme_assistant.evaluation.question_set import load_question_set
    from sme_assistant.evaluation.run_writer import RunWriter

    evaluation = load_evaluation_config()
    question_set = load_question_set(evaluation.path("question_set"))
    registry = load_conflicts(evaluation.path("conflicts"))

    writer = RunWriter(
        arm(), split="test", config=config, kb=kb, index=index,
        question_set=question_set, registry=registry, root=tmp_path,
    )
    provenance = writer.manifest["provenance"]
    assert provenance["question_set_sha256"]
    assert provenance["registry_sha256"] == registry.fingerprint()
    assert provenance["preregistration_sha256"], "the pre-registration was not hashed"
    assert provenance["question_set_metadata"]["chunk_set_sha256"]
