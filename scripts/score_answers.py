"""Score the blinded answers by hand, one at a time, without seeing the arm.

The primary metric of this study is manual. Conflict handling and answer
correctness are three-point judgements made by a person, because no automatic
check decides whether an answer adequately discloses that two live documents
disagree, and the pre-registration says so rather than pretending otherwise.
Every figure produced automatically so far excludes both.

    python scripts/score_answers.py build        # write the blinded sheet
    python scripts/score_answers.py score        # judge, resumably
    python scripts/score_answers.py status       # how far through
    python scripts/score_answers.py consistency  # identical answers, different scores
    python scripts/score_answers.py unseal --i-have-finished-scoring

Ordering is enforced rather than requested. ``unseal`` is the only command that
opens the key, and it refuses to run until every item carries a judgement, so
the mapping from opaque code to arm cannot be consulted part way through a pass
and then influence the remainder. The flag is deliberately awkward to type, for
the same reason ``--i-have-frozen-everything`` is in ``run_arms.py``: the
irreversible step should not be reachable by editing a default.

No model is called. No run directory is read after ``build``. Nothing here
computes a score by arm, an aggregate or a comparison; joining these judgements
to the arms and analysing them is a later, separate step, and keeping it
separate is what makes the blinding a control rather than a claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sme_assistant.evaluation.config import load_evaluation_config  # noqa: E402
from sme_assistant.evaluation.manual_scoring import (  # noqa: E402
    HELP,
    InputError,
    Judgement,
    ScoringError,
    append_judgement,
    consistency_report,
    describe_flags,
    load_judgements,
    load_sheet,
    next_unscored,
    open_session,
    parse_score_input,
    progress,
    render_item,
)
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.evaluation.run_writer import read_run, write_review_sheet  # noqa: E402

DEFAULT_DIR = ROOT / "results" / "manual"
DEFAULT_SHEET = DEFAULT_DIR / "review_sheet.jsonl"
DEFAULT_JUDGEMENTS = DEFAULT_DIR / "judgements.jsonl"
DEFAULT_SESSION = DEFAULT_DIR / "session.json"


def key_path_for(sheet: Path) -> Path:
    return sheet.with_name(sheet.stem + "_key.json")


# --- build -------------------------------------------------------------------


def discover_test_runs(runs_root: Path) -> list[Path]:
    """The four frozen test runs, one per arm, checked rather than assumed.

    A sheet built from the wrong directories is not detectable later: it is a
    well-formed blinded file containing the wrong answers. The manifests say
    which split and which arm each directory holds, so the check is available
    and there is no reason not to make it.
    """
    candidates = sorted(p for p in runs_root.glob("*_test") if p.is_dir())
    by_arm: dict[str, list[Path]] = {}
    for directory in candidates:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("split") != "test":
            continue
        by_arm.setdefault(manifest["arm"]["arm"], []).append(directory)

    problems = [f"{arm}: {[str(p.name) for p in dirs]}"
                for arm, dirs in sorted(by_arm.items()) if len(dirs) != 1]
    if problems:
        raise ScoringError(
            "Expected exactly one test run per arm; found "
            + "; ".join(problems)
            + ". Pass the run directories explicitly with --runs if this is intended."
        )
    if not by_arm:
        raise ScoringError(f"No test runs found under {runs_root}")
    return [by_arm[arm][0] for arm in sorted(by_arm)]


def command_build(args: argparse.Namespace) -> int:
    sheet = Path(args.sheet)
    judgements_path = Path(args.judgements)

    if judgements_path.exists() and not args.force:
        print(
            f"Refusing to rebuild: {judgements_path} already holds judgements.\n"
            "Item numbers are positions in a shuffled file, so rebuilding would "
            "leave existing scores pointing at different answers. Use --force "
            "only if you intend to discard them.",
            file=sys.stderr,
        )
        return 1

    runs = [Path(r) for r in args.runs] if args.runs else discover_test_runs(
        ROOT / "results" / "runs"
    )
    question_set = load_question_set(load_evaluation_config().path("question_set"))

    total = 0
    for directory in runs:
        _, answers = read_run(directory)
        total += len(answers)

    # The return value is the arm-to-code mapping. It is deliberately discarded
    # rather than printed: this command is run by the same person who will do
    # the scoring, and a mapping echoed to a terminal is a mapping they have
    # seen. write_review_sheet has already written it to the sealed key file.
    write_review_sheet(runs, sheet, seed=args.seed, question_set=question_set)

    print(f"Built {sheet}")
    print(f"  {total} answers from {len(runs)} runs, pooled and shuffled")
    print(f"  arm labels replaced by opaque codes; key sealed at {key_path_for(sheet)}")
    print(f"  evidence, prompt, timings and automatic metrics omitted")
    print()
    print("Do not open the key file. `unseal` will print it once scoring is complete.")
    print("Start scoring:  python scripts/score_answers.py score")
    return 0


# --- score -------------------------------------------------------------------


def command_score(args: argparse.Namespace) -> int:
    sheet = Path(args.sheet)
    items = load_sheet(sheet)
    judgements_path = Path(args.judgements)
    open_session(Path(args.session), sheet, item_count=len(items))
    judgements = load_judgements(judgements_path)

    total = len(items)
    position = next_unscored(items, judgements) or 1
    if args.start:
        position = max(1, min(args.start, total))

    state = progress(items, judgements)
    print(f"\n{state['scored']} of {total} scored. Resuming at item {position}.")
    print("? for help, q to save and quit. Every judgement is written immediately.\n")

    skipped: set[int] = set()

    while True:
        item = items[position - 1]
        state = progress(items, judgements)
        print(
            render_item(
                item,
                position=position,
                total=total,
                scored=state["scored"],
                existing=judgements.get(item.item),
            )
        )
        try:
            raw = input("score> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped. Everything scored so far is on disk.")
            break

        lowered = raw.lower()

        if lowered in {"q", "quit", "exit"}:
            break
        if lowered in {"?", "h", "help"}:
            print(HELP)
            continue
        if lowered in {"p", ""}:
            continue
        if lowered in {"s", "skip"}:
            skipped.add(position)
            position = position + 1 if position < total else 1
            continue
        if lowered in {"b", "back"}:
            position = position - 1 if position > 1 else total
            continue
        if lowered.startswith("g"):
            try:
                target = int(lowered[1:].strip())
            except ValueError:
                print("  Use g followed by an item position, for example g 42.")
                continue
            if not 1 <= target <= total:
                print(f"  Item position must be between 1 and {total}.")
                continue
            position = target
            continue

        try:
            parsed = parse_score_input(raw)
        except InputError as exc:
            print(f"  {exc} Type ? for the input format.")
            continue

        note = ""
        if parsed.wants_note:
            try:
                note = input("  note> ").strip()
            except (EOFError, KeyboardInterrupt):
                note = ""

        previous = judgements.get(item.item)
        judgement = append_judgement(
            judgements_path,
            Judgement(
                item=item.item,
                question_id=item.question_id,
                score=parsed.score,
                asserts_conflict=parsed.asserts_conflict,
                abstained=parsed.abstained,
                uncertain=parsed.uncertain,
                note=note,
                revision=(previous.revision + 1) if previous else 0,
            ),
        )
        judgements[item.item] = judgement
        skipped.discard(position)

        flags = describe_flags(judgement)
        print(f"  recorded {judgement.score}{(' - ' + flags) if flags else ''}")

        following = next_unscored(items, judgements, after=position)
        if following is None:
            print("\nEvery item has a judgement.")
            break
        position = following

    final = progress(items, judgements)
    print()
    print(f"  scored     {final['scored']} of {final['total']}")
    print(f"  remaining  {final['remaining']}")
    print(f"  uncertain  {final['uncertain']}")
    if skipped:
        print(f"  skipped this session: {sorted(skipped)}")
    print(f"  log        {judgements_path}")
    if final["complete"]:
        print("\nScoring is complete. Run `consistency` before unsealing the key.")
    return 0


# --- status and consistency --------------------------------------------------


def command_status(args: argparse.Namespace) -> int:
    items = load_sheet(Path(args.sheet))
    judgements = load_judgements(Path(args.judgements))
    state = progress(items, judgements)
    for name in ("total", "scored", "remaining", "uncertain", "revised", "with_notes"):
        print(f"  {name:<11}{state[name]}")
    if state["uncertain"]:
        flagged = sorted(j.item for j in judgements.values() if j.uncertain)
        print(f"\n  flagged uncertain: {flagged}")
    return 0


def command_consistency(args: argparse.Namespace) -> int:
    items = load_sheet(Path(args.sheet))
    judgements = load_judgements(Path(args.judgements))
    report = consistency_report(items, judgements)

    print(f"  duplicate groups         {report['duplicate_groups']}")
    print(f"  fully scored             {report['fully_scored_groups']}")
    print(f"  not yet fully scored     {report['not_yet_fully_scored']}")
    print(f"  scored consistently      {report['consistent']}")
    print(f"  scored differently       {len(report['divergent'])}")

    for entry in report["divergent"]:
        print(f"\n  {entry['question_id']}  items {entry['items']}")
        for j in entry["judgements"]:
            marks = [k for k in ("asserts_conflict", "abstained") if j[k]]
            print(f"    item {j['item']:>4}  score {j['score']}"
                  + (f"  {', '.join(marks)}" if marks else ""))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  written to {args.output}")
    return 0


# --- unseal ------------------------------------------------------------------


def command_unseal(args: argparse.Namespace) -> int:
    items = load_sheet(Path(args.sheet))
    judgements = load_judgements(Path(args.judgements))
    state = progress(items, judgements)

    if not state["complete"]:
        print(
            f"Refusing to unseal: {state['remaining']} of {state['total']} items "
            "are unscored.\nThe key exists to be opened after scoring, not during "
            "it. Finish the pass first.",
            file=sys.stderr,
        )
        return 1
    if not args.i_have_finished_scoring:
        print(
            "Refusing to unseal without --i-have-finished-scoring.",
            file=sys.stderr,
        )
        return 1

    report = consistency_report(items, judgements)
    if report["divergent"]:
        print(
            f"Note: {len(report['divergent'])} groups of identical answers were "
            "scored differently.\nReconcile them before analysing, or record that "
            "you chose not to. `consistency` lists them.\n"
        )

    key = json.loads(key_path_for(Path(args.sheet)).read_text(encoding="utf-8"))
    print(json.dumps(key["mapping"], indent=2))
    return 0


# --- entry point -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sheet", default=str(DEFAULT_SHEET))
    parser.add_argument("--judgements", default=str(DEFAULT_JUDGEMENTS))
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="write the blinded review sheet")
    build.add_argument("--runs", nargs="+", metavar="RUN_DIR")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=command_build)

    score = sub.add_parser("score", help="judge the answers, one at a time")
    score.add_argument("--session", default=str(DEFAULT_SESSION))
    score.add_argument("--start", type=int, default=0, metavar="POSITION")
    score.set_defaults(func=command_score)

    status = sub.add_parser("status", help="how far through the pass you are")
    status.set_defaults(func=command_status)

    consistency = sub.add_parser(
        "consistency", help="identical answers that received different judgements"
    )
    consistency.add_argument("--output", metavar="PATH")
    consistency.set_defaults(func=command_consistency)

    unseal = sub.add_parser("unseal", help="print the arm mapping, once scoring is done")
    unseal.add_argument("--i-have-finished-scoring", action="store_true")
    unseal.set_defaults(func=command_unseal)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ScoringError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
