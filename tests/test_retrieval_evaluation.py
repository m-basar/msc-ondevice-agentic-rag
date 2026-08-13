"""Tests for the retrieval evaluation.

The metric that matters here is conflict-pair recall. Stage 5 cannot resolve a
disagreement it was never shown both sides of, so a low figure means Arm D
would fail for a reason that has nothing to do with verification. Measuring it
before running any arm is the difference between a finding and an excuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_retrieval import evaluate  # noqa: E402

from sme_assistant.common.config import load_config
from sme_assistant.common.llm_client import MockClient
from sme_assistant.evaluation.config import load_evaluation_config
from sme_assistant.evaluation.conflicts import load_conflicts
from sme_assistant.evaluation.question_set import load_question_set
from sme_assistant.ingest.index import build_index
from sme_assistant.kb.loader import load_knowledge_base
from sme_assistant.retrieve.retriever import Retriever


@pytest.fixture(scope="module")
def report():
    config = load_config()
    evaluation = load_evaluation_config()
    kb = load_knowledge_base(config.path("paths.kb_docs"))
    client = MockClient(config)
    retriever = Retriever(build_index(kb, client, config), client, config)
    questions = list(load_question_set(evaluation.path("question_set")).split("dev"))

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
    return evaluate(
        retriever, questions, load_conflicts(evaluation.path("conflicts")), chunk_texts
    )


def test_conflict_pair_recall_is_reported_per_type(report):
    """Pooling the types would hide the one that decides the experiment."""
    pairs = report["conflict_pair_recall"]
    assert "mutually_exclusive" in pairs
    assert "stricter_looser" in pairs
    for per_k in pairs.values():
        assert all(0.0 <= v <= 1.0 for v in per_k.values())


def test_recall_is_monotonic_in_k(report):
    """A larger k cannot retrieve less. If it does, the ranking is unstable."""
    for measure in ("strict", "lenient"):
        values = [report["recall"][k][measure] for k in report["recall"]]
        assert values == sorted(values), f"{measure} recall falls as k grows"


def test_strict_recall_never_exceeds_lenient(report):
    for k, values in report["recall"].items():
        assert values["strict"] <= values["lenient"], k


def test_the_threshold_sweep_spans_the_configured_value(report):
    """The sweep must include the setting in force, or it cannot judge it."""
    configured = load_config().require("retrieval.min_similarity")
    assert any(row["min_similarity"] == configured for row in report["threshold_sweep"])


def test_raising_the_threshold_can_only_refuse_more(report):
    answerable = [r["answerable_wrongly_refused"] for r in report["threshold_sweep"]]
    unanswerable = [r["unanswerable_correctly_refused"] for r in report["threshold_sweep"]]
    assert answerable == sorted(answerable)
    assert unanswerable == sorted(unanswerable)


def test_the_evaluation_refuses_the_test_split_from_the_command_line():
    """Tuning retrieval on the test questions would tune it against the
    numbers the dissertation reports."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_retrieval.py", "--split", "test", "--mock"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "development split" in result.stderr


def test_pair_recall_counts_the_disputed_chunks_not_the_documents():
    """A family scored 1.00 while the passages stating the disagreement never
    reached the model, because the check looked at document identifiers."""
    import inspect

    from evaluate_retrieval import anchor_chunks, evaluate

    source = inspect.getsource(evaluate)
    assert "anchor_chunks(family" in source
    assert 'c.split("#")[0] for c in order' not in source, (
        "pair recall is still counting documents rather than the chunks carrying "
        "the disputed claims"
    )
