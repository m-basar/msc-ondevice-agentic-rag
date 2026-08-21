"""Run Arm D live, for demonstration only.

Amendment 1.27. Nothing this module produces is scored, and nothing it produces
is written into a frozen run directory. It builds the same pipeline the
experiment used, so what is demonstrated is the artefact rather than a
reimplementation of it, but it calls ``score_answer`` nowhere: there is no
quality metric in the output to be mistaken for a result later.

Only Arm D runs here. A live four-arm comparison would produce something that
looks exactly like the reported experiment and is not it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..common.config import load_config
from ..common.hostinfo import ollama_info
from ..common.llm_client import build_client, canonical_model_name
from ..generate.generator import Generator
from ..ingest.index import Index, index_path_for
from ..retrieve.retriever import EvidenceFormat, RetrievalMode, Retriever
from ..verify import Verifier
from .replay import ArmAnswer, _arm_answer

#: Arm D, restated here rather than imported from ``scripts/run_arms.py``.
#: The dashboard must not import the experiment runner: that dependency would
#: run the wrong way and would put a demonstrator on the experiment's path.
LIVE_ARM = "D"
LIVE_RETRIEVAL = RetrievalMode.ALL
LIVE_EVIDENCE = EvidenceFormat.WITH_STATUS


def _file_sha256(path: Path) -> str | None:
    """Restated here rather than imported from ``evaluation.run_writer``.

    Two lines of hashlib against a dependency from the demonstrator onto the
    experiment runner, which is the direction this module exists to avoid.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


class LiveUnavailable(RuntimeError):
    """The models or the index needed to answer are not present."""


@dataclass
class LiveAssistant:
    """Arm D over the frozen index, for questions typed at the dashboard."""

    config: Any
    client: Any
    retriever: Retriever
    generator: Generator
    verifier: Verifier

    @classmethod
    def build(cls, root: Path | str | None = None) -> "LiveAssistant":
        config = load_config() if root is None else load_config(Path(root) / "config.json")
        index = Index.load(index_path_for(config))
        client = build_client(config)
        return cls(
            config=config, client=client,
            retriever=Retriever(index, client, config),
            generator=Generator(client, config),
            verifier=Verifier(client, config),
        )

    def model_status(self) -> dict[str, Any]:
        """Check before accepting a question, not after failing to answer one.

        A dashboard that takes a question and then dies on a connection error
        has wasted the user's attention and told them nothing useful. This
        reports what is missing while the input is still disabled.
        """
        generation = self.config.require("llm.generation_model")
        verification = self.config.get("llm.verification_model")
        base_url = self.config.get("llm.ollama_url") or ""
        status: dict[str, Any] = {
            "generation": generation,
            "verification": verification,
            "base_url": base_url,
            "ready": False,
            "detail": "",
        }
        try:
            info = ollama_info(base_url)
        except Exception as exc:  # noqa: BLE001 - reported, never raised onward
            status["detail"] = f"could not reach Ollama at {base_url}: {exc}"
            return status
        if not info.get("reachable"):
            status["detail"] = (
                f"Ollama is not reachable at {base_url}. Start it with "
                "'ollama serve', or use Frozen Study Replay, which needs no "
                "models at all."
            )
            return status
        present = {canonical_model_name(m.get("name"))
                   for m in info.get("models") or []}
        missing = [m for m in (generation, verification)
                   if m and canonical_model_name(m) not in present]
        if missing:
            status["detail"] = (
                "Ollama is running but these models are not pulled: "
                + ", ".join(missing)
                + ". Pull them with 'ollama pull <model>'."
            )
            return status
        status["ready"] = True
        status["version"] = info.get("version")
        return status

    def answer(self, question: str) -> ArmAnswer:
        """Retrieve, draft, verify. One question, Arm D, nothing recorded."""
        text = (question or "").strip()
        if not text:
            raise LiveUnavailable("no question was given")
        retrieval = self.retriever.retrieve(text, mode=LIVE_RETRIEVAL)
        draft = self.generator.answer(text, retrieval,
                                      evidence_format=LIVE_EVIDENCE)
        verified = self.verifier.verify(draft)
        payload = (verified.to_dict() if hasattr(verified, "to_dict")
                   else dict(verified))
        payload.update({
            "arm": LIVE_ARM,
            "question_id": "live",
            "question": text,
            "category": "live demonstration",
            "family_id": None,
        })
        payload.setdefault("answer", getattr(verified, "final_answer", "") or "")
        return _arm_answer(payload)

    def frozen_arm_d_agreement(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Compare this live pipeline with the frozen Arm D manifest, field by field.

        Amendment 1.31.3. The earlier check compared two model names and the
        two mode constants and called that agreement. Everything that decides
        what the models are *given* went unchecked: the configuration
        fingerprint, the sampling options, the retrieval parameters and the
        index the evidence is drawn from. A live answer produced at
        ``top_k = 3`` against a rebuilt index, beside a frozen record produced
        at ``top_k = 6``, would have been displayed as the same arm.

        Returns the comparison rather than raising. The dashboard runs on
        machines where the index has legitimately been rebuilt, and a
        demonstrator that refuses to start is less honest than one that says
        which fields differ. What must not happen is silence.
        """
        arm = dict(manifest.get("arm") or {})
        provenance = dict(manifest.get("provenance") or {})
        # The same key the run writer records: config["generation"], not a
        # nested "options" block, which does not exist and quietly compared
        # an empty dict against four real values the first time this ran.
        live_options = {k: v for k, v in (self.config.get("generation") or {}).items()
                        if not k.startswith("_")}
        frozen_options = {k: v for k, v in
                          (provenance.get("generation_options") or {}).items()
                          if not k.startswith("_")}
        live_retrieval = {
            "top_k": self.config.get("retrieval.top_k"),
            "min_similarity": self.config.get("retrieval.min_similarity"),
        }
        frozen_retrieval = {k: v for k, v in
                            (provenance.get("retrieval") or {}).items()
                            if not k.startswith("_")}
        index_metadata = getattr(self.retriever.index, "metadata", {}) or {}

        fields = {
            "arm": (LIVE_ARM, arm.get("arm")),
            "retrieval_mode": (LIVE_RETRIEVAL.value, arm.get("retrieval_mode")),
            "evidence_format": (LIVE_EVIDENCE.value, arm.get("evidence_format")),
            "verification": (True, arm.get("verification")),
            "generation_model": (
                canonical_model_name(self.config.require("llm.generation_model")),
                canonical_model_name(arm.get("generation_model") or "")),
            "verification_model": (
                canonical_model_name(self.config.get("llm.verification_model") or ""),
                canonical_model_name(arm.get("verification_model") or "")),
            "config_sha256": (self.config.fingerprint(),
                              provenance.get("config_sha256")),
            "generation_options": (live_options, frozen_options),
            "retrieval": (live_retrieval, frozen_retrieval),
            "corpus_sha256": (index_metadata.get("corpus_sha256"),
                              provenance.get("corpus_sha256")),
            "chunk_set_sha256": (index_metadata.get("chunk_set_sha256"),
                                 provenance.get("chunk_set_sha256")),
            "index_file_sha256": (_file_sha256(index_path_for(self.config)),
                                  provenance.get("index_file_sha256")),
        }
        differs = sorted(k for k, (live, frozen) in fields.items() if live != frozen)
        return {
            "matches": not differs,
            "differs": differs,
            "fields": {k: {"live": live, "frozen": frozen}
                       for k, (live, frozen) in fields.items()},
        }
