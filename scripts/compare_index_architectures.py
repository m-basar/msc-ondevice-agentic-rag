"""What two device-local retrieval pipelines return for the same questions.

    python scripts/compare_index_architectures.py

Amendment 1.32.2, corrected by 1.32.6. ``data/index.json`` is a build artefact
and is not committed: each machine builds its own from the committed corpus.
Pulling this repository onto the Raspberry Pi 5 showed that the two builds are
not the same file, and this measures what that is observed to cost.

**What this compares, stated at the width the evidence supports.** Two runs of
the same arm over the same questions, one on the authoring laptop and one on the
Pi. Each device embedded the corpus locally *and* embedded each query locally, so
what differs between the two runs is the whole device-local embedding and
retrieval pipeline, not the stored index alone. An earlier version of this script
claimed it "isolates the index build and nothing else". It does not, and cannot:
nothing here reads either index file or compares a single vector.

**Only Arm B is compared.** Arm D reuses Arm B's retrieval block and Arm B's
draft verbatim on both devices, which is the design that makes B against D a
single-variable contrast. Its retrieval rows are therefore copies of B's, and
reporting them as a second observation would double-count one measurement. That
reuse is asserted below rather than assumed, and the count is printed so a reader
can see it is total.

**This is post-hoc and exploratory.** The question was not registered; it was
raised after every hypothesis was decided. No threshold is applied, no verdict is
reached, and no hypothesis is revisited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.evaluation.authenticity import (  # noqa: E402
    FROZEN_PERFORMANCE_DIGESTS, authenticated_run)

RUNS = ROOT / "results" / "runs"

#: The independent pair: same arm, same split, two devices.
LAPTOP = "20260814_054754_B_test"
PI = "20260815_030131_B_test_perf_pi5"

#: Arm D on each device, read only to assert that it reused Arm B's retrieval.
DERIVED = (("laptop", "20260814_055018_D_test", LAPTOP),
           ("Raspberry Pi 5", "20260815_040341_D_test_perf_pi5", PI))

#: Conditions that must hold for the comparison to mean anything, checked rather
#: than assumed. Amendment 1.32.6: the previous version checked two of these and
#: described the rest as held.
SHARED_PROVENANCE = ("question_set_sha256", "registry_sha256", "corpus_sha256",
                     "chunk_set_sha256")
SHARED_INDEX_RECIPE = ("embedding_model", "dimensions", "chunk_count",
                       "backend", "seed")


def retrieved(record: dict) -> list[str]:
    return [r["chunk_id"] for r in
            ((record.get("retrieval") or {}).get("results") or [])]


def by_question(records) -> dict[str, dict]:
    return {r["question_id"]: r for r in records}


def load(name: str):
    performance = name in FROZEN_PERFORMANCE_DIGESTS
    return authenticated_run(
        RUNS, name,
        table=FROZEN_PERFORMANCE_DIGESTS if performance else None,
        kind="performance" if performance else "quality")


def shared_conditions(laptop_manifest, pi_manifest):
    """Every condition held across the pair, and every one that is not."""
    lp = laptop_manifest["provenance"]
    pp = pi_manifest["provenance"]
    held, differs = [], []
    for field in SHARED_PROVENANCE:
        (held if lp.get(field) == pp.get(field) else differs).append(
            (field, lp.get(field), pp.get(field)))
    same = (json.dumps(lp.get("retrieval"), sort_keys=True)
            == json.dumps(pp.get("retrieval"), sort_keys=True))
    (held if same else differs).append(
        ("retrieval parameters", lp.get("retrieval"), pp.get("retrieval")))
    li = lp.get("index_metadata") or {}
    pi_meta = pp.get("index_metadata") or {}
    for field in SHARED_INDEX_RECIPE:
        (held if li.get(field) == pi_meta.get(field) else differs).append(
            (f"index {field}", li.get(field), pi_meta.get(field)))
    for label, a, b in (
        ("Ollama version",
         (li.get("endpoint") or {}).get("version"),
         (pi_meta.get("endpoint") or {}).get("version")),
        ("model store fingerprint",
         (li.get("endpoint") or {}).get("model_store_fingerprint"),
         (pi_meta.get("endpoint") or {}).get("model_store_fingerprint")),
        ("config_sha256", lp.get("config_sha256"), pp.get("config_sha256")),
    ):
        (held if a == b else differs).append((label, a, b))
    return held, differs


def reuse_evidence() -> list[dict]:
    """Assert, per device, that Arm D reused Arm B's retrieval and draft."""
    out = []
    for device, derived_name, source_name in DERIVED:
        derived, _, _ = load(derived_name)
        source, _, _ = load(source_name)
        d, s = by_question(derived), by_question(source)
        shared = sorted(set(d) & set(s))
        identical = sum(
            1 for q in shared
            if json.dumps(d[q].get("retrieval"), sort_keys=True)
            == json.dumps(s[q].get("retrieval"), sort_keys=True))
        drafts = sum(1 for q in shared
                     if (d[q].get("draft_answer") or "") == (s[q].get("answer") or ""))
        summary = json.loads(
            (RUNS / derived_name / "summary.json").read_text(encoding="utf-8"))
        declared = summary.get("drafts_reused_from")
        if declared != "B":
            raise SystemExit(
                f"{derived_name} does not declare drafts_reused_from='B' "
                f"(it says {declared!r}); the reuse this appendix relies on is "
                "not what the run recorded")
        if identical != len(shared) or drafts != len(shared):
            raise SystemExit(
                f"{derived_name} does not reuse {source_name} on every question "
                f"({identical} of {len(shared)} retrieval, {drafts} drafts). "
                "Arm D would then be an independent observation and this "
                "appendix's reason for excluding it would not hold")
        out.append({"device": device, "derived": derived_name,
                    "questions": len(shared), "identical_retrieval": identical,
                    "identical_drafts": drafts, "declared": declared})
    return out


def compare() -> dict:
    laptop_records, laptop_manifest, _ = load(LAPTOP)
    pi_records, pi_manifest, _ = load(PI)
    laptop, pi = by_question(laptop_records), by_question(pi_records)
    shared = sorted(set(laptop) & set(pi))
    if not shared:
        raise SystemExit(f"{LAPTOP} and {PI} share no questions")

    held, differs = shared_conditions(laptop_manifest, pi_manifest)
    blocking = [d for d in differs
                if d[0] in SHARED_PROVENANCE or d[0] == "retrieval parameters"
                or d[0].startswith("index ")]
    if blocking:
        raise SystemExit(
            "the two runs differ on a condition the comparison depends on: "
            + ", ".join(f"{f} ({a!r} against {b!r})" for f, a, b in blocking))

    same_set = same_order = 0
    differences = []
    for question_id in shared:
        a, b = retrieved(laptop[question_id]), retrieved(pi[question_id])
        prefix = 0
        for x, y in zip(a, b):
            if x != y:
                break
            prefix += 1
        if set(a) == set(b):
            same_set += 1
            if a == b:
                same_order += 1
                continue
            differences.append({"question_id": question_id, "kind": "order",
                                "prefix": prefix, "laptop": a, "pi": b})
        else:
            differences.append({"question_id": question_id, "kind": "set",
                                "prefix": prefix, "laptop": a, "pi": b,
                                "only_laptop": sorted(set(a) - set(b)),
                                "only_pi": sorted(set(b) - set(a))})
    lp, pp = laptop_manifest["provenance"], pi_manifest["provenance"]
    return {
        "questions": len(shared),
        "identical_set": same_set,
        "identical_set_and_order": same_order,
        "top_k": (lp.get("retrieval") or {}).get("top_k"),
        "laptop_index_sha256": lp["index_file_sha256"],
        "pi_index_sha256": pp["index_file_sha256"],
        "held": held,
        "differs": differs,
        "differences": differences,
    }


def short(value) -> str:
    text = str(value)
    return f"`{text[:16]}...`" if len(text) > 24 else f"`{text}`"


def main() -> int:
    r = compare()
    reuse = reuse_evidence()
    n = r["questions"]
    differences = r["differences"]
    out: list[str] = []
    add = out.append

    add("# Appendix E: Retrieval on two devices")
    add("")
    add("**Post-hoc and exploratory**, governed by pre-registration amendment")
    add("1.32.2 as corrected by 1.32.6. The question was not registered; it was")
    add("raised by pulling the repository onto the Raspberry Pi 5 after every")
    add("hypothesis was already decided. **No threshold is applied, no verdict")
    add("is reached and no hypothesis is revisited.**")
    add("")
    add("Generated by `scripts/compare_index_architectures.py` from committed,")
    add("authenticated runs. Every count below is computed by that script and")
    add("none is typed. No measurement was taken for this appendix: it compares")
    add("evidence that already existed.")
    add("")
    add("## E.1 What is compared, and what is not")
    add("")
    add("`data/index.json` is a build artefact and is **not** in the repository.")
    add("Each machine builds its own from the committed corpus. The two device")
    add("builds are not the same file:")
    add("")
    add("| | Serialised index file |")
    add("|---|---|")
    add(f"| Authoring laptop | `{r['laptop_index_sha256']}` |")
    add(f"| Raspberry Pi 5 | `{r['pi_index_sha256']}` |")
    add("")
    add("**That is the whole of what those hashes establish: the two serialised")
    add("index files differ.** Nothing in this appendix reads either index file")
    add("or compares a single vector, so no claim is made here about how they")
    add("differ, by how much, or why. The corpus and chunk-set fingerprints")
    add("agree, which establishes equivalent chunk identifiers, sections and")
    add("text under those fingerprints' definitions; it does not establish that")
    add("the serialised records are byte-identical.")
    add("")
    add("Each device also embedded every **query** locally. What is compared")
    add("below is therefore the observed retrieval output of two device-local")
    add("embedding and retrieval pipelines, not the stored index vectors alone.")
    add("")
    add("### Conditions checked before comparing")
    add("")
    add("The comparison means nothing unless the two runs agree on everything")
    add("but the device. Each condition below is checked by the generator, which")
    add("refuses to emit this appendix if a blocking one differs.")
    add("")
    add("| Condition | Held |")
    add("|---|---|")
    for field, value, _ in r["held"]:
        add(f"| {field} | {short(value)} |")
    add("")
    if r["differs"]:
        add(f"{len(r['differs'])} condition(s) legitimately differ and are")
        add("recorded rather than passed over:")
        add("")
        add("| Condition | Laptop | Raspberry Pi 5 |")
        add("|---|---|---|")
        for field, a, b in r["differs"]:
            add(f"| {field} | {short(a)} | {short(b)} |")
        add("")
        add("The two runs were executed from configurations that are not")
        add("byte-identical, and this appendix does not claim the index was the")
        add("only difference between them. Every field the comparison depends")
        add("on - the question set, the conflict registry, the corpus, the chunk")
        add("set, the retrieval parameters and the index build recipe - is held,")
        add("and the generator refuses if any of them is not.")
        add("")
    add("### Why only Arm B")
    add("")
    add("Arm D reuses Arm B's retrieval block and Arm B's draft verbatim, which")
    add("is the design that makes B against D a single-variable contrast. Its")
    add("retrieval rows are copies, so reporting them beside B's would")
    add("double-count one measurement. The reuse is checked, not assumed:")
    add("")
    add("| Device | Arm D run | Retrieval identical to Arm B | Drafts identical | Declared |")
    add("|---|---|---:|---:|---|")
    for e in reuse:
        add(f"| {e['device']} | `{e['derived']}` | "
            f"{e['identical_retrieval']} / {e['questions']} | "
            f"{e['identical_drafts']} / {e['questions']} | "
            f"`drafts_reused_from: {e['declared']}` |")
    add("")
    add("## E.2 What the two devices retrieved")
    add("")
    add(f"Arm B answered the same {n} test questions on both devices under the")
    add("conditions checked above.")
    add("")
    add("| | Questions | Identical chunk set | Identical set and order |")
    add("|---|---:|---:|---:|")
    add(f"| Arm B, laptop against Raspberry Pi 5 | {n} | {r['identical_set']} | "
        f"{r['identical_set_and_order']} |")
    add("")
    add(f"## E.3 Every question that differed, all {len(differences)} of them")
    add("")
    add("Listed individually rather than as a percentage, because this many")
    add("cases can be read.")
    add("")
    add("| Question | Difference | Ranks identical before it |")
    add("|---|---|---:|")
    for d in differences:
        if d["kind"] == "set":
            what = (f"rank-{d['prefix'] + 1} chunk `{d['only_laptop'][0]}` "
                    f"against `{d['only_pi'][0]}`")
        else:
            what = (f"`{d['laptop'][d['prefix']]}` and `{d['pi'][d['prefix']]}` "
                    f"swap ranks {d['prefix'] + 1} and {d['prefix'] + 2}")
        add(f"| `{d['question_id']}` | {what} | {d['prefix']} |")
    add("")
    prefixes = [d["prefix"] for d in differences] or [r["top_k"]]
    add(f"The two devices agree on at least the first **{min(prefixes)}** ranks")
    add(f"of every question, and on all {r['top_k']} ranks of the "
        f"{n - len(differences)} questions not listed above.")
    add("")
    add("## E.4 What this does and does not affect")
    add("")
    add("**H1 to H4 are unaffected.** They are scored entirely on the four")
    add("frozen quality runs, all executed on the authoring laptop over one")
    add("index. The Pi runs are performance-only under amendment 1.15 and")
    add("contribute no quality figure to any hypothesis.")
    add("")
    add("**H5 is a ratio measured within the Pi**, where both arms used the same")
    add("Pi-built index, so the comparison is internally consistent. The caveat")
    add(f"is stated rather than omitted: on {len(differences)} of {n} questions")
    add("the Pi generated from marginally different evidence than the laptop")
    add("did, so its absolute latencies are not strictly a measurement of the")
    add("laptop's workload.")
    add("")
    add("**What it supports about deployment**, and no more: the artefact")
    add("reproduces its retrieval substrate on the target device from the corpus")
    add("alone rather than shipping a prebuilt index, and the substrate it")
    add(f"reproduced returned the same evidence on {r['identical_set']} of {n}")
    add("questions in this one comparison. That is an observation over one")
    add("corpus, one device pair and one question set.")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
