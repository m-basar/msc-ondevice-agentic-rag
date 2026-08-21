"""Source identity: can a fabricated run get in.

Amendment 1.31.2. Every check the analysis and the replay made was internal.
The manifest agreed with the directory name, the arms agreed with one another,
the record count matched. Four fabricated runs that agreed among themselves
passed all of it, and a manifest declaring ``purpose: "unexpected-purpose"``
passed a blocklist that had never heard of it.

These tests build the attack and assert it now fails. Each was confirmed to
pass against the previous implementation before being accepted here; a negative
test that has never seen the bug it describes is a claim about code, not a check
on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sme_assistant.demo.replay import ReplayUnavailable, load_replay_library
from sme_assistant.evaluation.analysis import (AnalysisError,
                                               FROZEN_QUALITY_RUNS,
                                               DIAGNOSTIC_RUN,
                                               load_diagnostic_source)
from sme_assistant.evaluation.authenticity import (FROZEN_RUN_DIGESTS,
                                                   AuthenticityError,
                                                   RunDigest,
                                                   authenticate,
                                                   canonical,
                                                   check_question_identity,
                                                   digest_of,
                                                   read_run_content)
from sme_assistant.evaluation.question_set import load_question_set

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
GOLD = ROOT / "gold" / "question_set.json"

needs_runs = pytest.mark.skipif(
    not (RUNS / DIAGNOSTIC_RUN).exists(),
    reason="the frozen quality runs are not present")


def copy_runs(destination: Path, edit=None, *, reseal=None) -> Path:
    """Copy the four frozen runs, optionally editing one on the way.

    ``reseal`` re-records the content digests for what was actually written.
    That models the one thing a digest cannot defend against: somebody who
    finds the check failing and updates the recorded value to match the file
    rather than restoring the file. Tests of the layers *behind* the digest
    have to reseal, or the digest catches everything first and those layers are
    never exercised - which would make them the sort of rule this project has
    now had to correct five times.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for name in FROZEN_QUALITY_RUNS:
        source = RUNS / name
        target = destination / name
        target.mkdir(exist_ok=True)
        records, manifest = read_run_content(source)
        records = [dict(r) for r in records]
        manifest = json.loads(json.dumps(manifest))
        if edit is not None:
            records, manifest = edit(name, records, manifest)
        (target / "answers.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        (target / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        if reseal is not None:
            written, written_manifest = read_run_content(target)
            reseal.setitem(FROZEN_RUN_DIGESTS, name, RunDigest(
                answers=digest_of(list(written)),
                manifest=digest_of(written_manifest)))
    return destination


# --- the digest itself -------------------------------------------------------


@needs_runs
def test_every_frozen_run_authenticates_as_committed():
    for name in FROZEN_QUALITY_RUNS:
        records, manifest = read_run_content(RUNS / name)
        assert authenticate(name, records, manifest)


@needs_runs
def test_the_digest_does_not_depend_on_line_endings():
    """The authoring checkout holds CRLF and the object store holds LF, so a
    byte digest would pass on the machine it was computed on and fail for
    anyone who cloned the repository fresh. A false alarm about the integrity
    of frozen evidence is the worst kind to raise."""
    path = RUNS / DIAGNOSTIC_RUN / "answers.jsonl"
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    both = []
    for variant in (lf, crlf):
        both.append(digest_of([json.loads(line) for line
                               in variant.decode("utf-8").splitlines()
                               if line.strip()]))
    assert both[0] == both[1] == FROZEN_RUN_DIGESTS[DIAGNOSTIC_RUN].answers


@needs_runs
def test_a_single_altered_score_changes_the_digest():
    records, manifest = read_run_content(RUNS / DIAGNOSTIC_RUN)
    tampered = [dict(r) for r in records]
    tampered[0] = {**tampered[0], "answer": tampered[0]["answer"] + " "}
    with pytest.raises(AuthenticityError, match="do not match"):
        authenticate(DIAGNOSTIC_RUN, tampered, manifest)


def test_canonical_form_is_order_independent():
    assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})


def test_a_run_outside_the_closed_list_is_refused():
    with pytest.raises(AuthenticityError, match="not one of the four"):
        authenticate("20260901_120000_D_test", [], {})


# --- fabricated runs ---------------------------------------------------------


@needs_runs
def test_replay_refuses_four_mutually_consistent_inventions(tmp_path):
    """The attack the previous implementation could not see.

    Four runs whose names, manifests, hashes and records agree with one another
    perfectly, and are wholly invented. Every earlier check compared a run with
    itself or with its siblings, so all of them passed.
    """
    root = tmp_path / "runs"
    root.mkdir()
    names = [f"20260901_10000{i}_{arm}_test"
             for i, arm in enumerate("ABCD")]
    for name, arm in zip(names, "ABCD"):
        directory = root / name
        directory.mkdir()
        manifest = {
            "run_id": name, "split": "test", "started_at": "2026-09-01T10:00:00Z",
            "arm": {"arm": arm, "description": "invented",
                    "retrieval_mode": "all", "evidence_format": "with_status",
                    "verification": arm == "D",
                    "generation_model": "llama3.2:3b",
                    "verification_model": "qwen2.5:3b"},
            "provenance": {
                "corpus_sha256": "0" * 64, "chunk_set_sha256": "1" * 64,
                "config_sha256": "2" * 64, "question_set_sha256": "3" * 64,
                "registry_sha256": "4" * 64, "index_file_sha256": "5" * 64,
                "question_set_metadata": {"summary": {"by_split": {"test": 2}}},
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest),
                                                 encoding="utf-8")
        records = [{"arm": arm, "question_id": f"INVENTED-Q{n}",
                    "question": f"invented question {n}", "category": "conflict",
                    "family_id": "CONF-99", "answer": "invented",
                    "citations": [], "cited_superseded": [],
                    "hallucinated_citations": [], "has_valid_citation_ids": True}
                   for n in (1, 2)]
        (directory / "answers.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(ReplayUnavailable, match="not one of the four"):
        load_replay_library(root, names, expected_questions=2)


@needs_runs
def test_replay_refuses_any_declared_purpose(tmp_path, monkeypatch):
    """A blocklist admits everything it has not heard of. The four quality runs
    declare no purpose, so anything that declares one is something else."""
    def edit(name, records, manifest):
        if name == FROZEN_QUALITY_RUNS[0]:
            manifest["purpose"] = "unexpected-purpose"
        return records, manifest

    root = copy_runs(tmp_path / "runs", edit, reseal=monkeypatch)
    with pytest.raises(ReplayUnavailable, match="unexpected-purpose"):
        load_replay_library(root, FROZEN_QUALITY_RUNS, expected_questions=68)


@needs_runs
def test_replay_refuses_an_altered_frozen_record(tmp_path):
    def edit(name, records, manifest):
        if name == FROZEN_QUALITY_RUNS[3]:
            records[0] = {**records[0], "answer": "something else entirely"}
        return records, manifest

    root = copy_runs(tmp_path / "runs", edit)
    with pytest.raises(ReplayUnavailable, match="do not match"):
        load_replay_library(root, FROZEN_QUALITY_RUNS, expected_questions=68)


# --- question identity -------------------------------------------------------


@needs_runs
def test_the_diagnostic_refuses_a_swapped_question_to_family_assignment(
        tmp_path, monkeypatch):
    """The reviewer's attack, reproduced.

    ``CONF-02-Q1`` and ``CONF-06-Q1`` swapped between families. The record
    count is unchanged, every family is still present, all 45 registered
    questions are still there, and the swap turns a misclassification into an
    exact match. Every check before amendment 1.31.2 passed.
    """
    def edit(name, records, manifest):
        by_id = {r["question_id"]: r for r in records}
        a, b = by_id.get("CONF-02-Q1"), by_id.get("CONF-06-Q1")
        if a is not None and b is not None:
            a["family_id"], b["family_id"] = b["family_id"], a["family_id"]
        return records, manifest

    root = copy_runs(tmp_path / "runs", edit, reseal=monkeypatch)
    with pytest.raises(AnalysisError, match="contradicts the frozen question set"):
        load_diagnostic_source(root, DIAGNOSTIC_RUN,
                               question_set=load_question_set(GOLD))


@needs_runs
def test_the_diagnostic_refuses_a_substituted_question_text(tmp_path, monkeypatch):
    def edit(name, records, manifest):
        for record in records:
            if record["question_id"] == "CONF-02-Q1":
                record["question"] = "When does annual leave accrue?"
        return records, manifest

    root = copy_runs(tmp_path / "runs", edit, reseal=monkeypatch)
    with pytest.raises(AnalysisError, match="contradicts the frozen question set"):
        load_diagnostic_source(root, DIAGNOSTIC_RUN,
                               question_set=load_question_set(GOLD))


@needs_runs
def test_the_swap_is_caught_by_the_digest_when_it_has_not_been_resealed(tmp_path):
    """The outer layer, stated separately so the two are not confused.

    A swap against the committed digests fails on content before identity is
    ever consulted. The test above removes that layer deliberately, to show
    that the inner one holds on its own."""
    def edit(name, records, manifest):
        by_id = {r["question_id"]: r for r in records}
        a, b = by_id.get("CONF-02-Q1"), by_id.get("CONF-06-Q1")
        if a is not None and b is not None:
            a["family_id"], b["family_id"] = b["family_id"], a["family_id"]
        return records, manifest

    root = copy_runs(tmp_path / "runs", edit)
    with pytest.raises(AnalysisError, match="do not match"):
        load_diagnostic_source(root, DIAGNOSTIC_RUN,
                               question_set=load_question_set(GOLD))


@needs_runs
def test_question_identity_refuses_an_unknown_identifier():
    records, _ = read_run_content(RUNS / DIAGNOSTIC_RUN)
    edited = [dict(r) for r in records]
    edited[0] = {**edited[0], "question_id": "NOT-A-QUESTION"}
    with pytest.raises(AuthenticityError, match="does not answer the frozen"):
        check_question_identity(edited, load_question_set(GOLD))


@needs_runs
def test_question_identity_refuses_a_duplicate():
    records, _ = read_run_content(RUNS / DIAGNOSTIC_RUN)
    edited = [dict(r) for r in records]
    edited[1] = dict(edited[0])
    with pytest.raises(AuthenticityError, match="more than once"):
        check_question_identity(edited, load_question_set(GOLD))


# --- the mapping -------------------------------------------------------------


@needs_runs
def test_the_diagnostic_refuses_a_mapping_with_substituted_values():
    """Amendment 1.31.1. The check compared keys. A mapping with every key
    intact and a value re-pointed passed it, and re-pointing a value is exactly
    how you turn a wrong classification into an exact one."""
    from sme_assistant.evaluation.analysis import verifier_relationship_diagnostic
    from sme_assistant.evaluation.analysis import FROZEN_DIAGNOSTIC_SHAPE
    from sme_assistant.evaluation.stopping_gate import DECLARED_TO_INFERRED

    swapped = dict(DECLARED_TO_INFERRED)
    swapped["version_supersession"] = "mutually_exclusive"
    assert set(swapped) == set(DECLARED_TO_INFERRED)  # the old check passed
    with pytest.raises(AnalysisError, match="substituted"):
        verifier_relationship_diagnostic(
            records=[], declared_type={}, mapping=swapped,
            pair_present={}, shape=FROZEN_DIAGNOSTIC_SHAPE)


# --- live Arm D against the frozen Arm D manifest ----------------------------


@needs_runs
def test_live_arm_d_matches_the_frozen_manifest_in_every_compared_field():
    """Amendment 1.31.3. The earlier check compared two model names and two
    mode constants. Everything deciding what those models are *given* went
    unchecked: the configuration fingerprint, the sampling options, the
    retrieval parameters and the index the evidence comes from.

    This does not need Ollama. It builds the pipeline, which loads the index
    and the configuration, and compares them; it asks no model anything.
    """
    from sme_assistant.demo.live import LiveAssistant

    _, manifest = read_run_content(RUNS / DIAGNOSTIC_RUN)
    try:
        assistant = LiveAssistant.build(ROOT)
    except Exception as exc:  # noqa: BLE001 - the index is what may be absent
        pytest.skip(f"the live pipeline cannot be built here: {exc}")
    agreement = assistant.frozen_arm_d_agreement(manifest)
    assert agreement["matches"], agreement["differs"]
    for field in ("config_sha256", "generation_options", "retrieval",
                  "index_file_sha256", "corpus_sha256", "chunk_set_sha256",
                  "generation_model", "verification_model", "retrieval_mode",
                  "evidence_format", "verification", "arm"):
        assert field in agreement["fields"], field
        pair = agreement["fields"][field]
        assert pair["live"] == pair["frozen"], (field, pair)


@needs_runs
def test_the_live_comparison_notices_a_changed_retrieval_parameter():
    """A live answer produced at top_k = 3, beside a frozen record produced at
    top_k = 6, was displayed as the same arm."""
    from sme_assistant.demo.live import LiveAssistant

    _, manifest = read_run_content(RUNS / DIAGNOSTIC_RUN)
    manifest = json.loads(json.dumps(manifest))
    manifest["provenance"]["retrieval"]["top_k"] = 3
    try:
        assistant = LiveAssistant.build(ROOT)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the live pipeline cannot be built here: {exc}")
    agreement = assistant.frozen_arm_d_agreement(manifest)
    assert not agreement["matches"]
    assert "retrieval" in agreement["differs"]


@needs_runs
def test_the_live_comparison_notices_a_changed_config_fingerprint():
    from sme_assistant.demo.live import LiveAssistant

    _, manifest = read_run_content(RUNS / DIAGNOSTIC_RUN)
    manifest = json.loads(json.dumps(manifest))
    manifest["provenance"]["config_sha256"] = "0" * 64
    manifest["provenance"]["index_file_sha256"] = "1" * 64
    try:
        assistant = LiveAssistant.build(ROOT)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the live pipeline cannot be built here: {exc}")
    agreement = assistant.frozen_arm_d_agreement(manifest)
    assert not agreement["matches"]
    assert {"config_sha256", "index_file_sha256"} <= set(agreement["differs"])
