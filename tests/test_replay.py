"""Replaying a recorded run must reproduce it, and must refuse when it cannot.

A rebuild that quietly differs from the original is worse than no rebuild:
everything downstream still looks right, and the comparison the re-run exists
to make is silently against a different input.
"""

from __future__ import annotations

import json

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.common.llm_client import MockClient
from sme_assistant.evaluation.replay import (
    ReplayMismatch,
    load_drafts,
    rebuild_answer,
    rebuild_retrieval,
)
from sme_assistant.evaluation.run_writer import ArmDefinition, RunWriter
from sme_assistant.generate.generator import Generator
from sme_assistant.ingest.index import build_index
from sme_assistant.kb.loader import load_knowledge_base
from sme_assistant.retrieve.retriever import Retriever


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def pieces(config):
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    client = MockClient(config)
    index = build_index(kb, client, config)
    return kb, client, index, Retriever(index, client, config)


@pytest.fixture()
def recorded_run(tmp_path, config, pieces):
    """A real Arm B run, written and read back the way the harness does."""
    kb, client, index, retriever = pieces
    generator = Generator(client, config)
    writer = RunWriter(
        ArmDefinition(
            arm="B", description="test", retrieval_mode="all",
            evidence_format="with_status", verification=False,
            generation_model="mock-generate", verification_model=None,
        ),
        split="dev", tag="replaytest", config=config, kb=kb, index=index,
        question_set=None, registry=None, root=tmp_path,
    )
    answers = {}
    for question in ["How much annual leave?", "What is the mileage rate?"]:
        retrieval = retriever.retrieve(question, min_similarity=0.0)
        answer = generator.answer(question, retrieval)
        answers[question] = answer
        writer.record(question_id=question[:12], question=question, answer=answer,
                      group_id="G", family_id=None, category="factual")
    return writer.finish({}), answers


# --- the reconstruction is exact ---------------------------------------------


def test_the_rebuilt_evidence_matches_what_was_recorded(recorded_run, pieces):
    directory, original = recorded_run
    _, _, index, _ = pieces

    drafts = load_drafts(directory, index, expect_arm="B")

    assert len(drafts) == 2
    for answer in original.values():
        rebuilt = drafts[answer.question[:12]]
        assert rebuilt.answer == answer.answer
        assert rebuilt.prompt == answer.prompt
        assert rebuilt.retrieval.evidence_text() == answer.retrieval.evidence_text()
        assert ([s.chunk_id for s in rebuilt.retrieval]
                == [s.chunk_id for s in answer.retrieval])
        assert rebuilt.citations == answer.citations


def test_the_rebuilt_retrieval_keeps_scores_and_order(recorded_run, pieces):
    directory, original = recorded_run
    _, _, index, _ = pieces
    record = json.loads(
        (directory / "answers.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    retrieval = rebuild_retrieval(record, index)
    source = original["How much annual leave?"].retrieval

    assert [s.rank for s in retrieval] == [s.rank for s in source]
    assert [round(s.score, 4) for s in retrieval] == [
        round(s.score, 4) for s in source
    ]
    assert retrieval.top_k == source.top_k
    assert retrieval.mode == source.mode


# --- and it refuses when it cannot be exact ----------------------------------


def test_a_changed_corpus_is_refused_not_absorbed(recorded_run, pieces):
    """The failure this check exists for.

    If the chunk text has moved on since the recorded run, replaying against
    it would compare two different experiments while every downstream number
    still looked plausible.
    """
    directory, _ = recorded_run
    _, _, index, _ = pieces
    record = json.loads(
        (directory / "answers.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    record["evidence_sha256"] = "0" * 64

    with pytest.raises(ReplayMismatch) as excinfo:
        rebuild_answer(record, index)
    assert "does not match" in str(excinfo.value)
    assert "two different experiments" in str(excinfo.value)


def test_replaying_the_wrong_arm_is_refused(recorded_run, pieces):
    """Arm A's drafts would change what verification is measured against.

    The arms differ in evidence format, so the numbers would still look
    reasonable while measuring something else entirely.
    """
    directory, _ = recorded_run
    _, _, index, _ = pieces

    with pytest.raises(ReplayMismatch) as excinfo:
        load_drafts(directory, index, expect_arm="A")
    assert "is Arm B, not Arm A" in str(excinfo.value)


def test_an_empty_run_is_refused(tmp_path, pieces):
    _, _, index, _ = pieces
    (tmp_path / "manifest.json").write_text(
        json.dumps({"arm": {"arm": "B"}}), encoding="utf-8"
    )
    (tmp_path / "answers.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ReplayMismatch):
        load_drafts(tmp_path, index, expect_arm="B")


def test_replay_does_not_call_the_model(recorded_run, pieces):
    """A replay that regenerated anything would defeat its own purpose."""
    directory, _ = recorded_run
    _, client, index, _ = pieces
    before = client.call_count

    load_drafts(directory, index, expect_arm="B")

    assert client.call_count == before
