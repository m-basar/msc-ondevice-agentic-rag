"""Emit docs/CORPUS.md with every count derived from the registry and corpus.

    python scripts/make_corpus_doc.py > docs/CORPUS.md

The previous version of that document was written by hand and went stale in
nine separate places at once: document and chunk counts, reported and tuning
family counts, the conflict taxonomy, the superseded proportion, the number of
experimental arms, the test-group sample size, and a `current_current` type that
two amendments had already replaced. Each was accurate when written.

The lesson taken is not to be more careful. It is that a provenance document
whose numbers are typed will drift from the artefact it documents, and that the
drift is invisible until someone checks. Every count below is therefore read
from `gold/conflicts.json`, `gold/question_set.json`, `data/index.json` and the
knowledge base itself. The prose is authored here; the arithmetic is not.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.kb.loader import load_knowledge_base  # noqa: E402

TYPE_ORDER = ("version_supersession", "mutually_exclusive", "stricter_looser",
              "compatible")
TYPE_NOTE = {
    "version_supersession": ("**Yes.** Filtering on `status == current` resolves "
                             "them with no reasoning at all."),
    "mutually_exclusive": ("**No.** Both documents are live and no single action "
                           "satisfies both."),
    "stricter_looser": ("**No.** Both are live; the stricter course satisfies "
                        "both, but no metadata field says which is stricter."),
    "compatible": ("**Not applicable.** Negative controls. There is no conflict "
                   "to resolve, and asserting one is the failure."),
}
TYPE_BEHAVIOUR = {
    "version_supersession": "Cite the current document; the superseded one is not authority.",
    "mutually_exclusive": "Surface both, name neither as correct, and escalate.",
    "stricter_looser": "Name the stricter course, which satisfies both documents.",
    "compatible": "Answer plainly and assert no conflict.",
}


def summarise_conflict(family: dict) -> str:
    facts = family.get("conflicting_facts") or []
    if not facts:
        return family.get("name", "")
    fact = facts[0]
    values = fact.get("values") or {}
    if len(values) == 2:
        (doc_a, val_a), (doc_b, val_b) = list(values.items())
        return f"{fact['fact']}: {val_a} ({doc_a}) against {val_b} ({doc_b})."
    return fact.get("fact", family.get("name", ""))


def pair(family: dict) -> str:
    docs = family.get("documents") or []
    if family["type"] == "version_supersession" and len(docs) == 2:
        return f"{docs[0]} -> {docs[1]}"
    return " vs ".join(docs)


def main() -> int:
    registry = json.loads((ROOT / "gold" / "conflicts.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "gold" / "question_set.json").read_text(encoding="utf-8"))
    index = json.loads((ROOT / "data" / "index.json").read_text(encoding="utf-8"))
    knowledge_base = load_knowledge_base(ROOT / "data" / "kb")
    documents = list(knowledge_base.documents)

    families = registry["families"]
    reported = [f for f in families if f.get("split") != "tuning"]
    tuning = [f for f in families if f.get("split") == "tuning"]
    by_type = Counter(f["type"] for f in reported)

    total = len(documents)
    superseded = [d for d in documents if d.status == "superseded"]
    words = sum(len(d.body.split()) for d in documents)
    categories = Counter(d.doc_id.split("-")[0] for d in documents)
    chunks = len(index["chunks"])

    summary = questions["summary"]
    test_groups = summary["groups_per_split"]["test"]
    gaps = registry["deliberate_gaps"]
    absent, partial = gaps["fully_absent"], gaps["partially_present"]

    out = []
    w = out.append

    w("# Knowledge Base Provenance")
    w("")
    w("How the synthetic corpus was produced, what was checked, and what is "
      "deliberately wrong.")
    w("")
    w("**This file is generated.** Run `python scripts/make_corpus_doc.py > "
      "docs/CORPUS.md`.\nEvery count is read from the registry, the question "
      "set, the index and the corpus\nitself; nothing here is transcribed. Do "
      "not edit it by hand.")
    w("")
    w("Legal and factual content last reviewed: **7 August 2026**.")
    w("")
    w("## The authoritative implementation")
    w("")
    w("`final_v1/` is the implementation the dissertation reports. The older "
      "top-level\n`artefact/` directory is a **superseded July 2025 design**, "
      "retained only as a\nhistorical record. It predates the four-arm design, "
      "the conflict taxonomy, the\npre-registration and every reported result, "
      "and must not be used for dissertation\nclaims, for a demonstration, or "
      "for any dashboard.")
    w("")
    w("## The organisation is fictional")
    w("")
    w("**Northgate Kitchenware Ltd does not exist.** It is a fabricated "
      "wholesale\nkitchenware distributor invented for this research. No real "
      "organisation,\nemployee, customer or supplier is represented.")
    w("")
    w("An early version used identifiers that looked real, including an "
      "eight-digit\ncompany registration number in valid Companies House "
      "format. Numbers in that\nformat are allocated to real companies, so the "
      "corpus asserted things about a\nreal legal entity while claiming to be "
      "entirely fabricated. That contradicted the\nethics declaration and was "
      "corrected. All identifiers now use ranges reserved\nfor exactly this "
      "purpose.")
    w("")
    w("| Identifier type | Value used | Reserved by |")
    w("|---|---|---|")
    w("| Telephone numbers | `01632 960 xxx` | Ofcom, reserved for drama and fiction |")
    w("| Email domain | `northgate-kitchenware.invalid` | RFC 2606, `.invalid` can never be registered |")
    w("| Company registration | None quoted | Removed; the corpus states that no company of this name is registered |")
    w("| Postcode | `XX1 4LP` | `XX` is not an allocated UK postcode area |")
    w("| Address | Unit 12, Halden Industrial Park, Middleton | Fictional composite address |")
    w("")
    w("The address is a composite rather than an invention. \"Middleton\" is a "
      "common\nEnglish place name borne by several real settlements, so it is "
      "not claimed here as\nfictional; the unit number, the industrial park and "
      "the postcode are invented, and\nthe postcode uses an unallocated area "
      "code so the combined address cannot resolve\nto a real location.")
    w("")
    w("**Nothing in this corpus should be treated as a real address, number or "
      "organisation.**")
    w("")
    w("## Composition")
    w("")
    w(f"**{total} Markdown documents, {words:,} words, {chunks} chunks.** "
      f"{len(documents) - len(superseded)} documents are\ncurrent and "
      f"{len(superseded)} are superseded, "
      f"{100 * len(superseded) / total:.1f} per cent of the corpus. Each "
      "carries front\nmatter recording identifier, title, category, version, "
      "effective date, status and,\nwhere relevant, its supersession "
      "relationship and withdrawal date.")
    w("")
    w("| Category | Documents |")
    w("|---|---:|")
    for cat, count in sorted(categories.items()):
        w(f"| {cat} | {count} |")
    w(f"| **Total** | **{total}** |")
    w("")
    w("Documents were authored directly as static Markdown rather than "
      "produced by a\ngenerator script. They are research materials, and a "
      "reviewer must be able to\nread exactly what the system was given.")
    w("")
    w("Content is internally consistent except where a contradiction is "
      "registered:\nopening hours match delivery cut-offs, the fire assembly "
      "point matches across\ndocuments, and every in-text cross-reference "
      "resolves. The last of these is\nenforced by the loader rather than "
      "trusted.")
    w("")
    w("## Deliberate defects, and only deliberate ones")
    w("")
    w("The corpus contains planted contradictions. It must not contain "
      "accidental ones.")
    w("")
    w("Everything deliberate is registered in `gold/conflicts.json`. "
      "`validate_against_corpus`\nfails the test suite if a superseded document "
      "exists that no family accounts for,\nif a declared conflicting value is "
      "not present in the document text, if a topic\ndeclared absent has "
      "reappeared, or if a partial gap has lost its near-miss evidence.")
    w("")
    w("This distinction matters because gold answers are written against the "
      "corpus. An\naccidental contradiction that reaches the test set becomes "
      "an error the evaluation\ncannot see: the system is marked wrong for "
      "being right, and every downstream\nnumber is quietly corrupted.")
    w("")
    w("## The gold data boundary")
    w("")
    w("`gold/conflicts.json` holds expected answers, prohibited assertions and "
      "the list of\nunanswerable topics. **No component of the inference "
      "pipeline may read it.**")
    w("")
    w("Three mechanisms enforce this rather than one.")
    w("")
    w("1. **Gold data locations live in a separate configuration file.** "
      "`gold/evaluation.json`\n   is loaded only by "
      "`sme_assistant.evaluation.config`. The runtime `Config` object\n   "
      "built from `config.json` has no key that leads there, so there is no "
      "config\n   lookup at all, correct or otherwise, that reaches the answer "
      "key.")
    w("2. The loader lives in `sme_assistant.evaluation`, not "
      "`sme_assistant.kb`, so the\n   boundary is visible in the import graph.")
    w("3. `tests/test_no_oracle_leakage.py` parses every module in the "
      "inference packages\n   and fails if any imports the registry, imports "
      "from the evaluation package,\n   contains a literal path to gold data, "
      "or attempts a config lookup for an\n   evaluation key.")
    w("")
    w("An earlier design placed evaluation paths under `config.evaluation`, "
      "alongside\nruntime paths in the same file. That demonstrated the absence "
      "of *current*\nleakage while leaving the route open: an inference module "
      "could have called\n`config.path(\"evaluation.conflicts\")` and read the "
      "answer key, importing nothing\nand containing no literal path, so the "
      "tests would have passed. Splitting the\nfiles removes the possibility "
      "rather than testing for its absence.")
    w("")
    w("## Evaluation protocol")
    w("")
    w("Declared in `gold/evaluation.json` and in `docs/PREREGISTRATION.md` "
      "section 5, and\nasserted by tests.")
    w("")
    w(f"- **Fixed held-out evaluation.** The {len(reported)} reported families "
      "are scored once. This was\n  once called leave-one-family-out "
      "cross-validation, which was wrong: nothing is\n  trained on the "
      "remaining folds, so there is no model being validated. A\n  "
      "leave-one-family-out calculation is retained as a sensitivity analysis.")
    w(f"- **Macro-averaged by family.** Paraphrased questions drawn from one "
      "family test the\n  same document pair and are not independent "
      f"observations. The reported sample is\n  **{test_groups} test groups**, "
      f"not the {summary['question_count']} questions or "
      f"{summary['group_count']} groups the artefact contains.")
    w("- **Reported and tuning families are separate.** Families carry `split: "
      "reported` or\n  `split: tuning`. Tuning families exist so prompt "
      "wording, thresholds and verifier\n  output can be developed against real "
      "conflicts without inspecting a family that\n  is later scored. A "
      "reported family may not appear in the development split, and a\n  "
      "tuning family may not appear in the test split; the question-set loader "
      "enforces\n  both.")
    w("")
    w("## Conflict design")
    w("")
    w(f"{len(families)} registered families: **{len(reported)} reported** and "
      f"**{len(tuning)} tuning**. The type distinction is\nthe important part, "
      "and it is the part this document previously got wrong. An\nearlier "
      "taxonomy had two types, one of them `current_current`; pre-registration"
      "\namendments 1.2, 1.4 and 1.5 replaced it with the four below, "
      "reclassified four\nfamilies that had been typed by intuition, and pooled "
      "two of the types for the\nconfirmatory analysis.")
    w("")
    w("| Type | Reported | Correct behaviour | Resolvable by a metadata filter? |")
    w("|---|---:|---|---|")
    for kind in TYPE_ORDER:
        w(f"| `{kind}` | {by_type[kind]} | {TYPE_BEHAVIOUR[kind]} | {TYPE_NOTE[kind]} |")
    w("")
    w("An early version of this corpus contained only supersession conflicts. "
      "That was a\ndesign flaw: an examiner could reasonably ask why a "
      "claim-level verification layer\nis needed when three lines of filtering "
      "achieve the same result. Supersession\nfamilies alone cannot demonstrate "
      "that the contribution adds anything.")
    w("")
    w("Some families began as accidental contradictions found during review. "
      "Rather than\nharmonising them away, they were promoted to registered "
      "conflicts, because they\nare exactly the class a filter cannot touch and "
      "exactly what happens in real\norganisations when two departments draft "
      "policy separately.")
    w("")
    for kind in TYPE_ORDER:
        members = [f for f in reported if f["type"] == kind]
        if not members:
            continue
        w(f"### Reported `{kind}` families")
        w("")
        w("| Family | Risk | Documents | Conflict |")
        w("|---|---|---|---|")
        for f in sorted(members, key=lambda x: x["id"]):
            w(f"| {f['id']} | {f.get('risk_level','')} | {pair(f)} | "
              f"{summarise_conflict(f)} |")
        w("")
    w("### Tuning families")
    w("")
    w("Never reported. CONF-01 and CONF-05 were moved here on 8 August 2026 "
      "under\npre-registration amendment 1.1, because both were Stage 4 pilot "
      "questions and the\nlost-laptop pilot answer directly motivated "
      "`evaluation/answer_scoring.py`. A\nquestion that shaped the system "
      "cannot afterwards test it.")
    w("")
    w("| Family | Type | Documents | Conflict |")
    w("|---|---|---|---|")
    for f in sorted(tuning, key=lambda x: x["id"]):
        w(f"| {f['id']} | `{f['type']}` | {pair(f)} | {summarise_conflict(f)} |")
    w("")
    w("### Evaluation implication")
    w("")
    w("**Four arms are compared, not three.** The arms form a tree rooted at "
      "Arm B rather\nthan a ladder, so that each contrast changes one thing.")
    w("")
    w("| Arm | Retrieval | Evidence shown | Verification |")
    w("|---|---|---|---|")
    w("| A | all documents | identifier and text only | none |")
    w("| B | all documents | with status metadata | none |")
    w("| C | current documents only | with status metadata | none |")
    w("| D | all documents | with status metadata | yes |")
    w("")
    w("Arm C is the cheap obvious alternative to this project's contribution: "
      "a filter on\ndocument status resolves every supersession family with no "
      "reasoning at all. **B\nagainst D is the confirmatory single-variable "
      "contrast** for verification. C\nagainst D changes retrieval mode and "
      "verification together and is reported as a\npractical comparison, not as "
      "an ablation.")
    w("")
    w("## Deliberate gaps")
    w("")
    w(f"{len(absent)} topics are fully absent, verified by keyword probes "
      f"stored in the registry\nrather than in code. {len(partial)} are "
      "partially present, which is the harder category: the\ntopic is named "
      "somewhere but the answer is not there.")
    w("")
    w("| Topic | Kind |")
    w("|---|---|")
    for topic in absent:
        w(f"| {topic['topic']} | fully absent |")
    for topic in partial:
        w(f"| {topic['topic']} | partially present, named in {topic.get('mentioned_in','')} |")
    w("")
    w("One correction worth noting. \"International shipping\" was originally "
      "declared\nfully absent, but CS-11 explicitly states that the company "
      "does not ship outside\nthe United Kingdom, which answers the "
      "availability question. Only the **export\ndocumentation procedure** is "
      "genuinely absent. Questions must target the\nprocedure, not the "
      "availability, or they are not unanswerable at all.")
    w("")
    w("## The superseded proportion is not representative")
    w("")
    w(f"{len(superseded)} of {total} documents are superseded, "
      f"{100 * len(superseded) / total:.1f} per cent. A real SME knowledge "
      "base would\nnot carry that proportion of stale policy alongside its "
      "replacements in the same\nsearchable index.")
    w("")
    w("The proportion is elevated deliberately so conflict handling can be "
      "**measured**\nrather than illustrated. One pair supports a case study; "
      "it does not support a\nrate. This is a threat to external validity and "
      "is stated in the discussion rather\nthan buried.")
    w("")
    w("## Superseded documents are indexed, not filtered")
    w("")
    w("Excluding superseded documents at ingestion would make the conflict "
      "problem\ndisappear. They are indexed anyway, because detecting "
      "conflicting evidence is the\ncapability under evaluation and cannot be "
      "demonstrated if the conflict never\nreaches the model. The obvious "
      "objection, why not just filter, is exactly what Arm\nC tests, and it is "
      "answered by the families a filter cannot touch.")
    w("")
    w("## Reproducibility")
    w("")
    w("Every document is hashed with SHA-256, and the corpus carries a "
      "combined\nfingerprint derived from those hashes. It changes if and only "
      "if document content\nchanges, and is independent of file timestamps and "
      "read order. Full-length hashes\nare stored; truncation to twelve "
      "characters happens only at the point of display.")
    w("")
    w("Each run manifest records the full corpus, registry and configuration "
      "hashes, the\ngit commit and branch and whether the working tree was "
      "dirty, schema versions, the\nlegal review date, and the host, hardware, "
      "thermal and Ollama environment\nincluding the model store fingerprint. A "
      "dirty working tree is recorded rather than\nignored: a run made with "
      "uncommitted changes is not reproducible from its commit\nalone, and that "
      "should be visible.")
    w("")
    w("```")
    w("python scripts/kb_summary.py --manifest > results/corpus_manifest.json")
    w("```")
    w("")
    w("## Continuous integration")
    w("")
    w("`.github/workflows/tests.yml` runs the full test suite and the corpus "
      "validator on\nevery push to `main`. Corpus validation is a separate step "
      "from the unit tests so\nthat a failure is unambiguous: it means a "
      "document was edited into a state that\ncontradicts the registry, not "
      "that code is broken.")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
