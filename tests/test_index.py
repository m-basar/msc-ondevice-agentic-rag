"""Tests for model access and the searchable index.

The index is where a specific silent failure lives: an index built from an
earlier version of the corpus still answers questions, still cites document
identifiers, and produces text that is no longer in the knowledge base. Nothing
about the output looks wrong. Most of these tests exist to make that
impossible rather than unlikely.
"""

from __future__ import annotations

import json
import math

import pytest

from sme_assistant.common.config import load_config
from sme_assistant.common.llm_client import (
    Generation,
    LLMError,
    MockClient,
    OllamaClient,
    _hashing_vector,
    build_client,
)
from sme_assistant.ingest.index import Index, IndexError_, build_index
from sme_assistant.kb.loader import load_knowledge_base


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def kb(config):
    return load_knowledge_base(config.path("paths.kb_docs"))


@pytest.fixture(scope="module")
def index(kb, config):
    return build_index(kb, MockClient(config), config)


# --- the mock backend -------------------------------------------------------


def test_mock_embeddings_are_deterministic():
    """Across runs, machines and Python versions.

    ``hash`` is salted per process in Python, so a mock built on it would give
    different vectors on every run and make failures irreproducible.
    """
    assert _hashing_vector("annual leave policy") == _hashing_vector("annual leave policy")


def test_mock_embeddings_are_normalised():
    vector = _hashing_vector("the fire assembly point is the staff car park")
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-9


def test_mock_embeddings_track_lexical_overlap():
    """Random vectors would make every retrieval test meaningless.

    A hashing vectoriser is not semantic, but it does rank a related passage
    above an unrelated one, which is enough to exercise retrieval logic
    without a model.
    """
    query = _hashing_vector("how many days of annual leave do I get")
    related = _hashing_vector("employees receive 25 days of annual leave per leave year")
    unrelated = _hashing_vector("forklift trucks require a thorough examination under LOLER")
    assert cosine(query, related) > cosine(query, unrelated)


def test_mock_generation_is_deterministic_and_records_prompts():
    client = MockClient(load_config())
    first = client.generate("What is the mileage rate?")
    second = client.generate("What is the mileage rate?")
    assert first.text == second.text
    assert client.prompts == ["What is the mileage rate?"] * 2
    assert client.call_count == 2


def test_mock_can_return_canned_responses():
    client = MockClient(load_config(), responses={"mileage": "55 pence per mile"})
    assert client.generate("about mileage rates").text == "55 pence per mile"
    assert "MOCK ANSWER" in client.generate("something else").text


def test_build_client_never_falls_back_to_mock(config):
    """A failed run is recoverable. A run silently scored against fabricated
    output is not, so mock selection is explicit and never automatic."""
    assert isinstance(build_client(config, mock=True), MockClient)
    assert isinstance(build_client(config, mock=False), OllamaClient)


def test_unknown_backend_is_rejected(config, monkeypatch):
    monkeypatch.setattr(config, "_data", {**config.as_dict(), "llm": {**config.require("llm"), "backend": "nonsense"}})
    with pytest.raises(LLMError, match="Unknown llm.backend"):
        build_client(config, mock=False)


# --- generation accounting --------------------------------------------------


def test_generation_reports_rates():
    generation = Generation(
        text="ok", model="m", prompt_tokens=900, eval_tokens=250,
        prompt_seconds=36.0, eval_seconds=68.0, wall_seconds=104.0,
    )
    assert generation.prompt_tokens_per_second == 25.0
    assert generation.eval_tokens_per_second == pytest.approx(3.68, abs=0.01)
    assert set(generation.to_dict()) >= {
        "prompt_tokens", "eval_tokens", "prompt_seconds", "eval_seconds",
        "cpu_temp_c", "arm_frequency_hz", "throttled",
    }


def test_generation_rates_are_none_when_untimed():
    generation = Generation(text="ok", model="m", prompt_tokens=0, eval_tokens=0,
                            prompt_seconds=0.0, eval_seconds=0.0, wall_seconds=0.0)
    assert generation.prompt_tokens_per_second is None
    assert generation.eval_tokens_per_second is None


# --- the index --------------------------------------------------------------


def test_index_covers_every_chunk(index, kb):
    assert len(index) > len(kb), "there should be more chunks than documents"
    assert index.dimensions == 256
    assert index.corpus_sha256 == kb.fingerprint()


def test_index_embeds_the_contextual_text_not_the_bare_body(index):
    """The contextual header is the reason a bare figure is findable at all."""
    entry = next(e for e in index if "55 pence per mile" in e.chunk.text)
    from sme_assistant.common.llm_client import _hashing_vector as vec

    with_context = cosine(entry.vector, vec(entry.chunk.embedding_text))
    without_context = cosine(entry.vector, vec(entry.chunk.text))
    assert with_context > without_context, (
        "the index appears to have embedded the bare chunk text, discarding the "
        "document title and heading path"
    )


def test_index_records_full_provenance(index, config):
    metadata = index.metadata
    for key in ("built_at", "backend", "embedding_model", "corpus_sha256",
                "chunking", "config_sha256", "seed", "dimensions"):
        assert key in metadata, f"index metadata is missing {key!r}"
    assert metadata["config_sha256"] == config.fingerprint()
    assert metadata["chunking"]["max_words"] == config.require("chunking.max_words")


def test_index_round_trips(index, kb, tmp_path):
    path = index.save(tmp_path / "index.json")
    reloaded = Index.load(path, kb=kb)
    assert len(reloaded) == len(index)
    assert reloaded.corpus_sha256 == index.corpus_sha256
    for original, restored in zip(index, reloaded):
        assert restored.chunk_id == original.chunk_id
        assert restored.chunk.text == original.chunk.text
        assert restored.chunk.status == original.chunk.status
        assert restored.chunk.effective_date == original.chunk.effective_date
        assert restored.chunk.sections == original.chunk.sections
        assert restored.chunk.overlap_source == original.chunk.overlap_source
        assert restored.vector == original.vector


def test_stale_index_is_rejected(index, kb, tmp_path):
    """The failure this module exists to prevent."""
    path = index.save(tmp_path / "index.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["corpus_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(IndexError_, match="Rebuild the index"):
        Index.load(path, kb=kb)

    # Explicitly overridable, because sometimes you really do want the old one.
    assert len(Index.load(path, kb=kb, allow_stale=True)) == len(index)


def test_index_without_a_corpus_check_still_loads(index, tmp_path):
    path = index.save(tmp_path / "index.json")
    assert len(Index.load(path)) == len(index)


def test_schema_version_mismatch_is_rejected(index, tmp_path):
    path = index.save(tmp_path / "index.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "0.1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IndexError_, match="schema"):
        Index.load(path)


def test_missing_index_gives_actionable_error(tmp_path):
    with pytest.raises(IndexError_, match="build_index"):
        Index.load(tmp_path / "nothing.json")


def test_malformed_index_is_rejected(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexError_, match="not valid JSON"):
        Index.load(path)


def test_superseded_chunks_survive_a_save_and_load(index, kb, tmp_path):
    """The conflict must still be detectable after a round trip."""
    path = index.save(tmp_path / "index.json")
    reloaded = Index.load(path, kb=kb)
    old = [e for e in reloaded if e.chunk.doc_id == "HR-03"]
    assert old, "superseded chunks vanished from the reloaded index"
    assert all(e.chunk.status == "superseded" for e in old)
    assert all(e.chunk.superseded_by == "HR-13" for e in old)


def test_summary_reports_what_is_needed_to_trust_the_index(index):
    summary = index.summary()
    for key in ("chunk_count", "dimensions", "embedding_model", "backend",
                "corpus_sha256", "built_at", "current_chunks", "superseded_chunks"):
        assert key in summary
    assert summary["current_chunks"] + summary["superseded_chunks"] == summary["chunk_count"]
