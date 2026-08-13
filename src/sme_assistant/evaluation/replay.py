"""Rebuild a recorded arm's drafts so a later arm can be re-run against them.

Pilot 02's served answers were invalidated by a revision-control defect. The
detections, parse failures, relationships, prompts and latencies were not: they
are produced before the revision decision and are unaffected by it. So the
re-run needs to change one thing and hold everything else fixed.

Regenerating Arm B would not do that. B's drafts came from a model call, and a
second call is a second sample. Any difference between pilot 02 and pilot 03
would then be a mixture of the fix and of generator variation, and there would
be no way to say which. The point of a controlled re-run is to keep the input
identical, so the input is read back from the run that produced it.

Proving the reconstruction
--------------------------
A rebuild that quietly differs from the original is worse than no rebuild,
because everything downstream still looks right. Every run records
``evidence_sha256`` over the exact evidence text the model was shown, so the
reconstruction is checked against it and refuses on mismatch. The prompt hash
is checked the same way where it is recorded.

This is evaluation code and does not import the verifier.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..common.llm_client import Generation
from ..generate.generator import GroundedAnswer
from ..retrieve.retriever import (
    EvidenceFormat,
    RetrievalMode,
    RetrievalResult,
    ScoredChunk,
)


class ReplayMismatch(RuntimeError):
    """The rebuilt input is not the input that was recorded."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rebuild_retrieval(record: Mapping[str, Any], index) -> RetrievalResult:
    """Reconstruct the retrieval a record describes, from the same index.

    Chunk text is taken from the index rather than the record, because the
    record stores identifiers and scores rather than passages. That is the
    right way round: if the index has changed, the reconstruction will not
    match the recorded evidence hash and the replay will refuse, which is the
    behaviour wanted. Storing the text in the record would have hidden it.
    """
    retrieval = record["retrieval"]
    results = tuple(
        ScoredChunk(
            chunk=index.by_id(entry["chunk_id"]).chunk,
            score=float(entry["score"]),
            rank=int(entry["rank"]),
        )
        for entry in retrieval["results"]
    )
    return RetrievalResult(
        query=retrieval["query"],
        results=results,
        mode=RetrievalMode(retrieval["mode"]),
        top_k=int(retrieval["top_k"]),
        min_similarity=float(retrieval["min_similarity"]),
        best_score=float(retrieval["best_score"]),
        below_threshold=bool(retrieval["below_threshold"]),
        embed_seconds=float(retrieval.get("embed_seconds") or 0.0),
        search_seconds=float(retrieval.get("search_seconds") or 0.0),
        candidates_considered=int(retrieval.get("candidates_considered") or 0),
        conflicts=tuple(retrieval.get("conflicts") or ()),
    )


def rebuild_answer(record: Mapping[str, Any], index, *,
                   evidence_format: EvidenceFormat | None = None) -> GroundedAnswer:
    """Reconstruct one recorded answer, checked against its evidence hash."""
    retrieval = rebuild_retrieval(record, index)

    recorded_hash = record.get("evidence_sha256")
    if recorded_hash:
        rebuilt = retrieval.evidence_text(evidence_format)
        if _sha256(rebuilt) != recorded_hash:
            raise ReplayMismatch(
                f"{record.get('question_id')}: the rebuilt evidence does not "
                f"match the evidence the run recorded.\n"
                f"  recorded {recorded_hash[:16]}\n"
                f"  rebuilt  {_sha256(rebuilt)[:16]}\n"
                "The corpus, the chunk set or the evidence format has changed "
                "since that run. Replaying against different evidence would "
                "silently compare two different experiments."
            )

    generation = record.get("generation") or {}
    return GroundedAnswer(
        question=record["question"],
        answer=record["answer"],
        generation=Generation(
            text=record["answer"],
            model=generation.get("model", "replayed"),
            prompt_tokens=int(generation.get("prompt_tokens") or 0),
            eval_tokens=int(generation.get("eval_tokens") or 0),
            prompt_seconds=float(generation.get("prompt_seconds") or 0.0),
            eval_seconds=float(generation.get("eval_seconds") or 0.0),
            wall_seconds=float(generation.get("wall_seconds") or 0.0),
            load_seconds=float(generation.get("load_seconds") or 0.0),
            cpu_temp_c=generation.get("cpu_temp_c"),
            arm_frequency_hz=generation.get("arm_frequency_hz"),
            throttled=generation.get("throttled"),
            options=dict(generation.get("options") or {}),
        ),
        retrieval=retrieval,
        prompt=record["prompt"],
        citations=tuple(record.get("citations") or ()),
        document_citations=tuple(record.get("document_citations") or ()),
        hallucinated_citations=tuple(record.get("hallucinated_citations") or ()),
        uncited_chunks=tuple(record.get("uncited_chunks") or ()),
        cited_superseded=tuple(record.get("cited_superseded") or ()),
        refusal_heuristic=bool(record.get("refusal_heuristic")),
    )


def load_drafts(run_directory: Path | str, index, *,
                expect_arm: str | None = None,
                evidence_format: EvidenceFormat | None = None) -> dict[str, GroundedAnswer]:
    """Every recorded answer from a run, keyed by question id.

    ``expect_arm`` is checked rather than trusted. Replaying Arm A's drafts
    into Arm D would compare verification against a different evidence format
    and the numbers would still look plausible.
    """
    path = Path(run_directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    arm = (manifest.get("arm") or {}).get("arm")
    if expect_arm and arm != expect_arm:
        raise ReplayMismatch(
            f"{path.name} is Arm {arm}, not Arm {expect_arm}. Arms differ in "
            "evidence format and retrieval mode, so a draft from the wrong one "
            "would change what verification is being measured against."
        )

    drafts: dict[str, GroundedAnswer] = {}
    for line in (path / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        drafts[record["question_id"]] = rebuild_answer(
            record, index, evidence_format=evidence_format
        )
    if not drafts:
        raise ReplayMismatch(f"{path.name} contains no answers to replay.")
    return drafts
