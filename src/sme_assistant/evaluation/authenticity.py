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
from the files as committed, not sealed when the runs were executed on 14
August. They cannot prove that the files were not altered between those dates;
what shows that is the commit history, where ``be55077`` carries the same
content and predates every analysis commit. What they do is make any alteration
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
    """The two digests that identify one frozen run."""

    answers: str
    manifest: str


#: The four frozen quality runs, by content. Amendment 1.31.2.
#:
#: Computed from the committed files, which ``git show be55077`` reproduces. Any
#: change to a single scored value, timing, citation or provenance field in any
#: of these files changes the digest and stops the analysis.
FROZEN_RUN_DIGESTS: dict[str, RunDigest] = {
    "20260814_054606_A_test": RunDigest(
        answers="b4de4ce7366ca849d82a2118f1631fe23cb2c5305db2f5f9363b0e32c9dce939",
        manifest="16579fb6a11a00ccfb12fd28f24da5477cbda4cd9b789d3369d1804b0d3f57d8"),
    "20260814_054754_B_test": RunDigest(
        answers="4b5504ee598893de07ba3a70ceb255a37cced18283c0b65eb403237e654e3ccc",
        manifest="27fe5bb2a3c0877c72b20cd5b444de4ca6a35696c0a870d8f90551e726e7ea7b"),
    "20260814_054908_C_test": RunDigest(
        answers="76926cb35fb795e22d4d951dd691fe70175591b08a68e5ea389bc91943a5224b",
        manifest="82535e5c8a026d104e36dab47dc818240968f4ec83539f02695be733d240174e"),
    "20260814_055018_D_test": RunDigest(
        answers="c09d626d4f99f932a7a509ac9369604859bed671a69c4ecdac6b3cfd89ecd813",
        manifest="de1452c0bc91ec9afd3c9428505aaf12327b5b0c787727b60b4a3541ebcfc7f4"),
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


def authenticate(name: str,
                 records: Sequence[Mapping[str, Any]],
                 manifest: Mapping[str, Any]) -> dict[str, str]:
    """Check one run's content against its recorded digest.

    Raises rather than returning a flag. A caller that receives a false and
    carries on is the failure mode amendment 1.16.1 is about, and there is no
    legitimate reason to display an unauthenticated frozen run beside an
    authenticated one.
    """
    expected = FROZEN_RUN_DIGESTS.get(name)
    if expected is None:
        raise AuthenticityError(
            f"{name} is not one of the four frozen quality runs "
            f"({sorted(FROZEN_RUN_DIGESTS)}). A run created later cannot enter "
            "the analysis or the replay, whatever it calls itself."
        )
    found = RunDigest(answers=digest_of(list(records)),
                      manifest=digest_of(manifest))
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
    return {"answers_sha256": found.answers, "manifest_sha256": found.manifest}


def authenticated_run(runs_root: Path | str, name: str) -> tuple[
        tuple[Mapping[str, Any], ...], Mapping[str, Any], dict[str, str]]:
    """Read and authenticate one frozen run in a single call."""
    directory = Path(runs_root) / name
    records, manifest = read_run_content(directory)
    return records, manifest, authenticate(name, records, manifest)


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
