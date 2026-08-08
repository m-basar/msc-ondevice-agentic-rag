"""Persisting experimental runs so a result can be traced to what produced it.

Stage 4 printed answers to a terminal. Nothing was saved, so the pilot's four
outputs survive only because they were pasted into a document by hand, and the
chunk identifiers they cite no longer refer to the same text. That is the
failure this module prevents.

Every run writes three things:

``manifest.json``   what produced the run: every hash, the git state, the
                    machine, the model, the options, and the arm definition
``answers.jsonl``   one record per question, including the exact prompt and the
                    exact evidence the model was shown
``summary.json``    aggregate metrics

Recording the prompt verbatim matters more than it looks. A run is only
reproducible if the input can be reconstructed, and the prompt is the product
of the corpus, the chunker, the index, the retriever, the arm's evidence format
and the question. Storing a hash of it proves nothing was silently different;
storing the text itself means the run can be understood without re-executing
anything.

Blinded review
--------------
``write_review_sheet`` emits answers in randomised order with arm labels
replaced by opaque codes. Manual scoring is done against that file. The author
of a system scoring their own system's arms, knowing which is which, is not a
credible measurement, and the fix costs one function.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..common.config import Config, load_config
from ..common.hostinfo import environment, git_commit
from ..kb.loader import KnowledgeBase

SCHEMA_VERSION = "1.0"


def sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArmDefinition:
    """One experimental condition, described completely enough to rerun it."""

    arm: str
    description: str
    retrieval_mode: str
    evidence_format: str
    verification: bool
    generation_model: str
    verification_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "description": self.description,
            "retrieval_mode": self.retrieval_mode,
            "evidence_format": self.evidence_format,
            "verification": self.verification,
            "generation_model": self.generation_model,
            "verification_model": self.verification_model,
        }


class RunWriter:
    """Creates a run directory and appends records as questions are answered.

    Records are written as they arrive rather than at the end, so a run that
    dies at question 80 of 100 still leaves 79 usable answers. On a Raspberry
    Pi at several minutes per question that is the difference between losing an
    evening and losing a line of a log file.
    """

    def __init__(
        self,
        arm: ArmDefinition,
        *,
        split: str,
        tag: str = "",
        config: Config | None = None,
        kb: KnowledgeBase | None = None,
        index: Any = None,
        question_set: Any = None,
        registry: Any = None,
        root: Path | None = None,
    ) -> None:
        self.config = config or load_config()
        self.question_set = question_set
        self.registry = registry
        self.arm = arm
        self.split = split
        self.started = datetime.now(timezone.utc)

        stamp = self.started.strftime("%Y%m%d_%H%M%S")
        label = f"_{tag}" if tag else ""
        base = root or self.config.path("paths.results")
        self.directory = Path(base) / f"{stamp}_{arm.arm}_{split}{label}"
        self.directory.mkdir(parents=True, exist_ok=True)

        self.answers_path = self.directory / "answers.jsonl"
        self.manifest_path = self.directory / "manifest.json"
        self.summary_path = self.directory / "summary.json"
        self.record_count = 0

        self.manifest = self._build_manifest(kb, index)
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )

    def _build_manifest(self, kb, index) -> dict[str, Any]:
        base_url = self.config.require("llm.ollama_url")
        index_path = self.config.path("paths.index")
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.directory.name,
            "started_at": self.started.isoformat(),
            "split": self.split,
            "arm": self.arm.to_dict(),
            "provenance": {
                "config_sha256": self.config.fingerprint(),
                "corpus_sha256": kb.fingerprint() if kb else None,
                "chunk_set_sha256": getattr(index, "chunk_set_sha256", None),
                "index_file_sha256": sha256_of(index_path),
                "index_metadata": getattr(index, "metadata", None),
                "git": git_commit(),
                # What the run was scored against, not only what produced it.
                # A results directory that records the model and the corpus but
                # not the question set or the rubric cannot be re-scored, and
                # cannot be shown to have used the gold data it claims.
                "question_set_sha256": (
                    sha256_of_text(
                        json.dumps(
                            [q.to_dict() for q in self.question_set], sort_keys=True
                        )
                    )
                    if self.question_set is not None
                    else None
                ),
                "question_set_metadata": (
                    dict(self.question_set.metadata)
                    if self.question_set is not None
                    else None
                ),
                "registry_sha256": (
                    self.registry.fingerprint() if self.registry is not None else None
                ),
                "preregistration_sha256": sha256_of(
                    Path(__file__).resolve().parents[3] / "docs" / "PREREGISTRATION.md"
                ),
                "seed": self.config.get("generation.seed"),
                "generation_options": self.config.require("generation"),
                "retrieval": {
                    "top_k": self.config.require("retrieval.top_k"),
                    "min_similarity": self.config.require("retrieval.min_similarity"),
                },
                "chunking": self.config.require("chunking"),
            },
            "environment": environment(base_url),
            "host": platform.node(),
        }

    def record(
        self,
        *,
        question_id: str,
        question: str,
        answer: Any,
        group_id: str | None = None,
        family_id: str | None = None,
        category: str | None = None,
        scoring: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one answered question.

        The prompt and the evidence are stored verbatim, not only hashed. A
        hash proves nothing changed; the text lets the run be understood
        without re-executing it, which matters when the run took four hours on
        a Raspberry Pi.

        ``group_id`` is the unit of independence: the conflict family for a
        conflict question, the gap topic for an unanswerable one, otherwise the
        question itself. It is written into every record because aggregation
        happens later and often elsewhere, and a results file that does not
        say which answers belong to the same family cannot be macro-averaged
        without going back to the question set to find out. Defaulting it to
        the family, then to the question id, keeps single questions working
        without silently pooling paraphrases.
        """
        payload = answer.to_dict() if hasattr(answer, "to_dict") else dict(answer)
        prompt = getattr(answer, "prompt", "") or payload.get("prompt", "")
        evidence = ""
        retrieval = getattr(answer, "retrieval", None)
        if retrieval is not None:
            # Rendered in the arm's own format. This called evidence_text() with
            # no argument, which defaults to WITH_STATUS, so an Arm A record
            # stored evidence carrying supersession markers that the model was
            # never shown. The saved prompt and the saved evidence disagreed,
            # and the arm whose whole definition is "no status metadata" was the
            # one recorded wrongly.
            from ..retrieve.retriever import EvidenceFormat

            try:
                fmt = EvidenceFormat(self.arm.evidence_format)
            except ValueError as exc:
                raise ValueError(
                    f"Arm {self.arm.arm} declares evidence_format "
                    f"{self.arm.evidence_format!r}, which is not a known format. "
                    "Recording evidence in the wrong format silently misreports "
                    "what the model saw."
                ) from exc
            evidence = retrieval.evidence_text(fmt)

        record = {
            "question_id": question_id,
            "group_id": group_id or family_id or question_id,
            "family_id": family_id,
            "category": category,
            "split": self.split,
            "arm": self.arm.arm,
            "question": question,
            "prompt": prompt,
            "prompt_sha256": sha256_of_text(prompt),
            "evidence": evidence,
            "evidence_sha256": sha256_of_text(evidence),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        if scoring is not None:
            record["scoring"] = dict(scoring)
        if extra:
            record["extra"] = dict(extra)

        with self.answers_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.record_count += 1

    def finish(self, summary: Mapping[str, Any] | None = None) -> Path:
        """Close the run, recording the environment again.

        The environment is captured at the end as well as the start because a
        Raspberry Pi that began a run at 60 degrees and ended it throttled at
        85 was not running the same machine throughout.
        """
        finished = datetime.now(timezone.utc)
        payload = {
            "run_id": self.directory.name,
            "arm": self.arm.arm,
            "split": self.split,
            "records": self.record_count,
            "started_at": self.started.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_seconds": round((finished - self.started).total_seconds(), 2),
            "environment_at_end": environment(self.config.require("llm.ollama_url")),
            **(dict(summary) if summary else {}),
        }
        self.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.directory


def read_run(directory: Path | str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a run's manifest and answers."""
    path = Path(directory)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    answers = [
        json.loads(line)
        for line in (path / "answers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return manifest, answers


def write_review_sheet(
    runs: Iterable[Path | str],
    output: Path | str,
    *,
    seed: int = 42,
) -> dict[str, str]:
    """Write a blinded file for manual scoring, and return the key.

    Answers from every arm are pooled, shuffled and stripped of anything that
    identifies which system produced them. Arm labels become opaque codes.

    This exists because the person scoring these answers built the system and
    wants one arm to win. That is not a criticism of them; it is the reason
    blinding exists as a method. The key is written to a separate file so that
    scoring can be completed before the mapping is looked at.

    The evidence block is **not** included, though an earlier version included
    it. Arm A is defined by receiving evidence with no status markers, so the
    presence or absence of "[SUPERSEDED, replaced by X]" in the evidence
    identified the arm on sight, and a reviewer who noticed the pattern once had
    deanonymised the whole sheet. Blinding that a careful reader can defeat is
    worse than none, because it is reported as a control.

    Conflict handling is scored from the answer against the question's rubric,
    which is what the reviewer is judging in any case. Citation support is
    scored automatically from the full record, where the evidence is still
    stored verbatim.
    """
    pooled: list[dict[str, Any]] = []
    arms: list[str] = []
    for directory in runs:
        _, answers = read_run(directory)
        for record in answers:
            if record["arm"] not in arms:
                arms.append(record["arm"])
            pooled.append(record)

    rng = random.Random(seed)
    codes = [f"system_{chr(ord('A') + i)}" for i in range(len(arms))]
    rng.shuffle(codes)
    key = dict(zip(arms, codes))

    rng.shuffle(pooled)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for position, record in enumerate(pooled, start=1):
            handle.write(json.dumps({
                "item": position,
                "question_id": record["question_id"],
                # Carried through so manual scores can be macro-averaged by
                # family later. It reveals nothing about which arm produced the
                # answer, so blinding is unaffected.
                "group_id": record.get("group_id", record["question_id"]),
                "question": record["question"],
                "system": key[record["arm"]],
                "answer": record["answer"],
                # Deliberately omitted: arm, model, prompt, timings, the
                # evidence block, and every automatic metric. A reviewer who can
                # see the arm, how fast it was, or whether the evidence carried
                # status markers, is no longer blind.
            }) + "\n")

    key_path = output_path.with_name(output_path.stem + "_key.json")
    key_path.write_text(json.dumps({
        "warning": "Do not open until manual scoring is complete.",
        "seed": seed,
        "mapping": key,
    }, indent=2), encoding="utf-8")
    return key
