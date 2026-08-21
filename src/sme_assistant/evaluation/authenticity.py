"""Content authentication for the frozen quality runs.

Amendment 1.31.2. Every check the analysis and the replay made about a run was
a check of *internal consistency*: the manifest called itself what the directory
called it, the arms agreed on a corpus hash, the record count matched. A set of
four fabricated runs whose names, hashes and records agreed with one another
passed all of it, because nothing compared them with anything outside
themselves. Consistency is not identity.

What is recorded here is a content digest of each frozen run: the SHA-256 of
the canonical JSON of the parsed records, and of the parsed manifest.

**Why content and not the file's bytes.** ``.gitattributes`` normalises line
endings to LF in the object store while the authoring checkout holds CRLF, so
the same committed file has two different byte digests depending on the platform
it is read on. A byte digest would therefore pass on the two machines it was
computed on and fail for anyone who cloned the repository fresh on Linux, which
is a false alarm about the integrity of frozen evidence - the worst kind of
false alarm to raise. Digesting the parsed structure authenticates every value
in every record and is independent of how the lines end.

**What these digests do and do not establish.** They were computed on 21 August
from the files as committed, not sealed when the runs were executed. They cannot
prove that the files were not altered between those dates; what shows that is
the commit history. Each frozen set enters it at a different point: the four
quality runs at ``be55077``, the laptop timing runs at ``eede351`` and the Pi
timing runs at ``d4e9a90``. Naming one commit for all of them, as an earlier
version of this docstring did, was accurate only while the table held the
quality runs alone. What they do is make any alteration
*from now on* fail loudly at the point of use, in the analysis and in the
demonstrator, rather than being caught only by someone who thought to run
``git diff``. That is a narrower claim than "these are the original files" and
it is the one this module is entitled to.

**This module names no path to gold data and must keep it that way.**
``demo/replay.py`` imports it and ``demo/live.py`` imports ``demo/replay``, so a
reference here would put the live assistant one import away from a file holding
the gold answer to every question. ``check_question_identity`` is given a
question set; it never goes and finds one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence


class AuthenticityError(RuntimeError):
    """A run's content does not match its recorded digest, or is unreadable."""


class RunDigest(NamedTuple):
    """The digests that identify one frozen run.

    ``summary`` is optional only so that a caller building a digest for a run
    without one is not forced to invent a value. Amendment 1.32.3 records why it
    exists at all: the first version covered ``answers.jsonl`` and
    ``manifest.json``, and the performance analysis also reads ``summary.json``,
    so a third of what it consumed was outside the guarantee the module claimed
    to give.
    """

    answers: str
    manifest: str
    summary: str | None = None


#: The four frozen quality runs, by content. Amendment 1.31.2.
#:
#: Computed from the committed files, which ``git show be55077`` reproduces for
#: these four. Any change to a single scored value, citation or provenance field
#: in any of them changes the digest and stops the analysis.
FROZEN_RUN_DIGESTS: dict[str, RunDigest] = {
    "20260814_054606_A_test": RunDigest(
        answers="b4de4ce7366ca849d82a2118f1631fe23cb2c5305db2f5f9363b0e32c9dce939",
        manifest="16579fb6a11a00ccfb12fd28f24da5477cbda4cd9b789d3369d1804b0d3f57d8",
        summary="21109c69e0d631b6b932c0eb746478b5980ab8c382e9f39893db525f6a655c72"),
    "20260814_054754_B_test": RunDigest(
        answers="4b5504ee598893de07ba3a70ceb255a37cced18283c0b65eb403237e654e3ccc",
        manifest="27fe5bb2a3c0877c72b20cd5b444de4ca6a35696c0a870d8f90551e726e7ea7b",
        summary="826d95cc29c1919823a0e009adfe8ee32facc898721e016f00ddc6a5b57fb3af"),
    "20260814_054908_C_test": RunDigest(
        answers="76926cb35fb795e22d4d951dd691fe70175591b08a68e5ea389bc91943a5224b",
        manifest="82535e5c8a026d104e36dab47dc818240968f4ec83539f02695be733d240174e",
        summary="9721a4dee4943a6b9039f9da2bc88b34df681ffd4f9a5676e50e47f457c2b8e9"),
    "20260814_055018_D_test": RunDigest(
        answers="c09d626d4f99f932a7a509ac9369604859bed671a69c4ecdac6b3cfd89ecd813",
        manifest="de1452c0bc91ec9afd3c9428505aaf12327b5b0c787727b60b4a3541ebcfc7f4",
        summary="c5c760b92e545463f3b01fb572a167b6d80ce2798d93feebae371fa93c318158"),
}

#: The six frozen performance runs. Amendment 1.32.3.
#:
#: Held separately because they are separate evidence with a separate rule:
#: 1.15 declared them performance-only before any was executed, and no quality
#: figure may come from them. Keeping the two tables apart means a caller has to
#: name which kind of run it expects, and ``load_replay_library`` asking for a
#: quality run cannot be satisfied by a timing run under any circumstances.
#:
#: **These are not in be55077.** The laptop timing runs enter the history at
#: ``eede351`` and the Pi runs at ``d4e9a90``, both after the quality analysis
#: was signed off. An earlier version of this module said be55077 carried all
#: authenticated content, which was true when it covered only the four quality
#: runs and stopped being true the moment these were added.
FROZEN_PERFORMANCE_DIGESTS: dict[str, RunDigest] = {
    "20260815_030104_B_test_perf_laptop_gpu": RunDigest(
        answers="e72b831611e510a5d7670e2f1d51afd5c89fda6f2c95b4198dc8d16261096585",
        manifest="af8dba6fd64ff1a1740fd6934e864bcbdab4b7db82d8bb67fd9605cd932a60c6",
        summary="a9c8168fed6dbd6be06042c800e016e9ee3ecc63a96ca2a7b8492ed1c35229bc"),
    "20260815_030219_D_test_perf_laptop_gpu": RunDigest(
        answers="13a1521ae9a5ced8a57d2cd7a8a674e30913ab773acbc7ea7d7e03dab60ed10a",
        manifest="54ae435d287fc0e3b76b4011b5bb4ed74b30662b6139abe42eaca5d7dd8c44ce",
        summary="d4ef9c54edadfda0245cc3b520b0dbf7def6fe53ee457be97aeba2a18837a173"),
    "20260815_030713_B_test_perf_laptop_cpu": RunDigest(
        answers="dc1c85062d78cc91a9b2e6cb9425d2380dfba826f0b66c4b79956250f749116f",
        manifest="0796e9839797d8b9f1f70fe69c29930a0ee9eac3b20dd9dd4fa8b296f14ea9db",
        summary="7c1971fb2dfc119565d07ac80bc1be12024e87042723766d8da53c2a31249c6b"),
    "20260815_031446_D_test_perf_laptop_cpu": RunDigest(
        answers="8684974a293c27f8008413dbf5a2eac5d0866638e2da2d6c922fd7b5497e2f4b",
        manifest="a962aa8b2678c76c2bc6570505090bdfd4097ddeabcee8947d5e89b34b309f37",
        summary="9eb27c56eb74ddedccb39062d749b01e0732132aa299f737b3c7afc9c0396070"),
    "20260815_030131_B_test_perf_pi5": RunDigest(
        answers="704ce41dca051dba7ba77d1206215ffc79f737f50e384a36a1915a865b6567c9",
        manifest="b46f5a2c3ab41bba238050189e6792c6583ead27df60ea80ab70a4a9b2632e77",
        summary="f1b532aa0d5132bc3c9d2b960371707237681520d9aad0fb8d2da73be9cb0ace"),
    "20260815_040341_D_test_perf_pi5": RunDigest(
        answers="858b0ceb1317c7ab79f33c4205413bb20c5ae5ed174cd2410ccd385ba14a4fd1",
        manifest="70a382ebf8afc54bc87330b387e42e6787f6d4c5927161cd87373a7cfbe61ecc",
        summary="f994026df5e6ef66a9103826eb218fe818745cdcf899073b70ac41a90bc5dc46"),
}


#: The three performance run-index files, which name which directories each
#: hardware condition used. Amendment 1.32.3: the performance analysis is
#: pointed at one of these, so a swapped path here redirects the whole analysis
#: while every run it then reads authenticates perfectly.
FROZEN_PERFORMANCE_INDEX_DIGESTS: dict[str, str] = {
    "latest_test_performance_laptop_cpu.json":
        "4efadde2a3b687c163ebb2cddc6dea893b418dc0a9a0eb9326c0a85e3521195c",
    "latest_test_performance_laptop_gpu.json":
        "be9e8b4252cb8fc91e901547fb5129b74aa67cf1da97b52a425dba1348b04d06",
    "latest_test_performance_pi5_cpu.json":
        "a36c860c7d28eebe16609f0f480dbab6d9c4a7147ca95eb96112b1c52917bfe7",
}


def canonical(payload: Any) -> str:
    """A stable text form of a parsed JSON value.

    Sorted keys and no insignificant whitespace, so that two files carrying the
    same values digest the same however they were written out.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def digest_of(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def read_run_content(directory: Path) -> tuple[tuple[Mapping[str, Any], ...],
                                       Mapping[str, Any]]:
    """Parse one run directory's records and manifest, or raise."""
    answers_path = directory / "answers.jsonl"
    manifest_path = directory / "manifest.json"
    if not answers_path.exists() or not manifest_path.exists():
        raise AuthenticityError(
            f"{directory.name} is missing answers.jsonl or manifest.json")
    try:
        records = tuple(
            json.loads(line)
            for line in answers_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuthenticityError(f"{directory.name} is not readable: {exc}") from exc
    return records, manifest


def read_summary(directory: Path) -> Mapping[str, Any] | None:
    """Parse a run's ``summary.json``, or ``None`` where there is none."""
    path = directory / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuthenticityError(
            f"{directory.name}/summary.json is not readable: {exc}") from exc


def authenticate_index_file(path: Path) -> str:
    """Check a performance run-index file against its recorded digest.

    Amendment 1.32.3. These name which directories a hardware condition used.
    Swapping a path here redirects the entire performance analysis while every
    run it subsequently reads authenticates perfectly, which is the one gap a
    per-run digest cannot close.
    """
    expected = FROZEN_PERFORMANCE_INDEX_DIGESTS.get(path.name)
    if expected is None:
        raise AuthenticityError(
            f"{path.name} is not one of the frozen performance index files "
            f"({sorted(FROZEN_PERFORMANCE_INDEX_DIGESTS)})")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthenticityError(f"{path.name} is not readable: {exc}") from exc
    found = digest_of(payload)
    if found != expected:
        raise AuthenticityError(
            f"{path.name} does not match its recorded content digest.\n"
            f"  expected {expected}\n  found    {found}\n"
            "This file decides which run directories the performance analysis "
            "reads, so a change here redirects every figure it produces."
        )
    return found


def authenticate(name: str,
                 records: Sequence[Mapping[str, Any]],
                 manifest: Mapping[str, Any],
                 summary: Mapping[str, Any] | None = None,
                 *,
                 table: Mapping[str, RunDigest] | None = None,
                 kind: str = "quality") -> dict[str, str]:
    """Check one run's content against its recorded digest.

    Raises rather than returning a flag. A caller that receives a false and
    carries on is the failure mode amendment 1.16.1 is about, and there is no
    legitimate reason to display an unauthenticated frozen run beside an
    authenticated one.

    ``table`` names which frozen set the run must belong to, defaulting to the
    four quality runs. A caller wanting a timing run has to say so, which is
    what keeps 1.15's separation from depending on the caller remembering it.
    """
    digests = FROZEN_RUN_DIGESTS if table is None else table
    expected = digests.get(name)
    if expected is None:
        raise AuthenticityError(
            f"{name} is not one of the frozen {kind} runs "
            f"({sorted(digests)}). A run created later cannot enter "
            "the analysis or the replay, whatever it calls itself."
        )
    found = RunDigest(answers=digest_of(list(records)),
                      manifest=digest_of(manifest),
                      summary=None if summary is None else digest_of(summary))
    if found.answers != expected.answers:
        raise AuthenticityError(
            f"{name}: the answers do not match the recorded content digest.\n"
            f"  expected {expected.answers}\n  found    {found.answers}\n"
            "This is the frozen evidence for H1 to H5. Restore it from the "
            "commit rather than re-recording the digest."
        )
    if found.manifest != expected.manifest:
        raise AuthenticityError(
            f"{name}: the manifest does not match the recorded content "
            f"digest.\n  expected {expected.manifest}\n  found    "
            f"{found.manifest}"
        )
    # The summary is read by the performance analysis, so it is covered.
    # Amendment 1.32.3.
    if expected.summary is not None:
        if found.summary is None:
            raise AuthenticityError(
                f"{name}: summary.json is missing, and the recorded digest "
                "says it should be present. The performance analysis reads it."
            )
        if found.summary != expected.summary:
            raise AuthenticityError(
                f"{name}: the summary does not match the recorded content "
                f"digest.\n  expected {expected.summary}\n  found    "
                f"{found.summary}"
            )
    out = {"answers_sha256": found.answers, "manifest_sha256": found.manifest}
    if found.summary is not None:
        out["summary_sha256"] = found.summary
    return out


def authenticated_run(runs_root: Path | str, name: str, *,
                      table: Mapping[str, RunDigest] | None = None,
                      kind: str = "quality") -> tuple[
        tuple[Mapping[str, Any], ...], Mapping[str, Any], dict[str, str]]:
    """Read and authenticate one frozen run in a single call."""
    directory = Path(runs_root) / name
    records, manifest = read_run_content(directory)
    summary = read_summary(directory)
    return records, manifest, authenticate(name, records, manifest, summary,
                                           table=table, kind=kind)


#: Fields on an answer record that restate the question. Every one of them is
#: compared with the frozen question set. Amendment 1.31.2: swapping two
#: questions between families changed a wrong classification into an exact one
#: and every existing check still passed, because the family identifier a record
#: carried was never compared with the family the question set assigns it.
IDENTITY_FIELDS = (("question", "text"), ("category", "category"),
                   ("family_id", "family_id"))


def check_question_identity(records: Sequence[Mapping[str, Any]],
                            question_set: Any,
                            *,
                            split: str = "test",
                            where: str = "run") -> dict[str, Any]:
    """Compare a run's questions with the frozen question set, or raise.

    Not "are the identifiers plausible" but "is this question, under this
    identifier, in this family, the one the frozen set defines". A record that
    answers ``CONF-02-Q1`` with ``CONF-06-Q1``'s text, or files it under a
    family the question set does not put it in, is refused here.
    """
    expected = {q.question_id: q for q in question_set.questions
                if q.split == split}
    if not expected:
        raise AuthenticityError(
            f"the frozen question set defines no {split} split to check against")
    found = [r.get("question_id") for r in records]
    duplicates = sorted({q for q in found if found.count(q) > 1})
    if duplicates:
        raise AuthenticityError(f"{where} answers {duplicates} more than once")
    if set(found) != set(expected):
        extra = sorted(set(found) - set(expected))
        missing = sorted(set(expected) - set(found))
        raise AuthenticityError(
            f"{where} does not answer the frozen {split} split. "
            f"Unknown: {extra[:5]}. Missing: {missing[:5]}."
        )
    mismatches: list[str] = []
    for record in records:
        question = expected[record["question_id"]]
        for record_field, question_field in IDENTITY_FIELDS:
            mine = record.get(record_field)
            theirs = getattr(question, question_field)
            if record_field == "category" and not mine:
                # Older records may omit the category; absence is not a
                # contradiction, a different value is.
                continue
            if mine != theirs:
                mismatches.append(
                    f"{record['question_id']}.{record_field}: run has {mine!r}, "
                    f"the frozen question set has {theirs!r}")
    if mismatches:
        raise AuthenticityError(
            f"{where} contradicts the frozen question set on "
            f"{len(mismatches)} field(s):\n  " + "\n  ".join(mismatches[:8])
        )
    return {"split": split, "questions": len(expected),
            "identity_fields_checked": [f for f, _ in IDENTITY_FIELDS]}
