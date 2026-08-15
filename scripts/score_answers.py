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
    ABSTENTION_HELP,
    ABSTENTION_ORDER_SEED,
    HELP,
    AbstentionJudgement,
    InputError,
    Judgement,
    ScoringError,
    abstention_agreement,
    abstention_order,
    append_abstention,
    append_judgement,
    arm_signature_audit,
    consistency_report,
    describe_flags,
    load_abstention,
    load_judgements,
    load_sheet,
    next_unscored,
    open_session,
    parse_abstention_input,
    parse_score_input,
    positional_drift_report,
    progress,
    render_abstention_item,
    render_item,
)
from sme_assistant.evaluation.question_set import load_question_set  # noqa: E402
from sme_assistant.evaluation.run_writer import read_run, write_review_sheet  # noqa: E402

DEFAULT_DIR = ROOT / "results" / "manual"
DEFAULT_SHEET = DEFAULT_DIR / "review_sheet.jsonl"
DEFAULT_JUDGEMENTS = DEFAULT_DIR / "judgements.jsonl"
DEFAULT_ABSTENTION = DEFAULT_DIR / "abstention_pass.jsonl"
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

    # The blinding check that amendment 1.13 says was missing. It runs on the
    # sheet just written, before it can be scored, and it tests for text the
    # system serves verbatim rather than for the give-aways already known. The
    # existing sheet cannot be repaired; this stops the next one repeating it.
    audit = arm_signature_audit(load_sheet(sheet))
    if not audit["blind"]:
        templates = ", ".join(f["template"] for f in audit["findings"])
        print(
            f"\nThis sheet is not blind. It carries {templates} verbatim on "
            f"{sum(len(f['items']) for f in audit['findings'])} items, which "
            f"exposes {audit['items_exposed']} of {audit['total_items']} through "
            "the opaque codes those items share.\n"
            "Only the verified arm serves that text, so recognising it once "
            "identifies that arm across the whole sheet. See amendment 1.13.\n"
            "The sheet has been written but must not be scored as blind.",
            file=sys.stderr,
        )
        return 1

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
                arm_identified=parsed.arm_identified,
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


def command_abstention(args: argparse.Namespace) -> int:
    """The focused re-pass: one property, no rubric, a different order.

    Recorded to its own log. The first pass is not edited and not shown, so the
    two remain independently auditable and the second is not anchored to the
    first.
    """
    sheet = Path(args.sheet)
    items = load_sheet(sheet)
    open_session(Path(args.session), sheet, item_count=len(items))

    ordered = abstention_order(items, seed=args.order_seed)
    log = Path(args.abstention_log)
    decided = load_abstention(log)
    total = len(ordered)

    position = 1
    for index, item in enumerate(ordered, start=1):
        if item.item not in decided:
            position = index
            break
    else:
        print(f"\nAll {total} items already have an abstention judgement.")
        return 0

    print(f"\n{len(decided)} of {total} decided. Resuming at {position}.")
    print("One question per item: did the answer decline? ? for help, q to quit.\n")

    while True:
        item = ordered[position - 1]
        print(
            render_abstention_item(
                item, position=position, total=total, decided=len(decided)
            )
        )
        try:
            raw = input("declined? ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped. Everything decided so far is on disk.")
            break

        lowered = raw.lower()
        if lowered in {"q", "quit", "exit"}:
            break
        if lowered in {"?", "h", "help"}:
            print(ABSTENTION_HELP)
            continue
        if lowered in {"p", ""}:
            continue
        if lowered == "s":
            position = position + 1 if position < total else 1
            continue
        if lowered == "b":
            position = position - 1 if position > 1 else total
            continue

        try:
            abstained, wants_note = parse_abstention_input(raw)
        except InputError as exc:
            print(f"  {exc} Type ? for the input format.")
            continue

        note = ""
        if wants_note:
            try:
                note = input("  note> ").strip()
            except (EOFError, KeyboardInterrupt):
                note = ""

        previous = decided.get(item.item)
        decided[item.item] = append_abstention(
            log,
            AbstentionJudgement(
                item=item.item,
                question_id=item.question_id,
                abstained=abstained,
                note=note,
                revision=(previous.revision + 1) if previous else 0,
            ),
        )
        print(f"  recorded {'declined' if abstained else 'answered'}")

        following = next(
            (
                index
                for index, candidate in enumerate(ordered, start=1)
                if index > position and candidate.item not in decided
            ),
            None,
        ) or next(
            (
                index
                for index, candidate in enumerate(ordered, start=1)
                if candidate.item not in decided
            ),
            None,
        )
        if following is None:
            print(f"\nAll {total} items decided.")
            break
        position = following

    print(f"\n  decided  {len(decided)} of {total}")
    print(f"  log      {log}")
    if len(decided) == total:
        print("\nRun `agreement` to compare the two passes.")
    return 0


def command_agreement(args: argparse.Namespace) -> int:
    first = load_judgements(Path(args.judgements))
    second = load_abstention(Path(args.abstention_log))
    if not second:
        print("No abstention re-pass recorded yet. Run `abstention` first.",
              file=sys.stderr)
        return 1

    report = abstention_agreement(first, second)
    print(f"  items compared        {report['compared']}")
    print(f"  agreed                {report['agreed']}")
    print(f"  disagreed             {report['disagreed']}")
    rate = report["agreement_rate"]
    print(f"  agreement rate        " + ("n/a" if rate is None else f"{rate:.3f}"))
    print(f"  missed by first pass  {report['missed_by_first_pass']}")
    print(f"  missed by second pass {report['missed_by_second_pass']}")
    if report["first_pass_only"]:
        print(f"  not yet re-passed     {len(report['first_pass_only'])}")

    for entry in report["disagreements"]:
        print(f"    item {entry['item']:>4}  {entry['question_id']:<32}"
              f"  first {entry['first_pass']}  second {entry['second_pass']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  written to {args.output}")
    return 0


def command_drift(args: argparse.Namespace) -> int:
    """Was a disagreement caused by position in one pass, or by criterion?"""
    items = load_sheet(Path(args.sheet))
    first = load_judgements(Path(args.judgements))
    second = load_abstention(Path(args.abstention_log))
    if not second:
        print("No abstention re-pass recorded yet.", file=sys.stderr)
        return 1

    report = positional_drift_report(items, first, second, order_seed=args.order_seed)
    for label, key in (("second pass says declined, first did not", "second_pass_only"),
                       ("first pass says declined, second did not", "first_pass_only")):
        block = report[key]
        print(f"\n  {label}: {block['n']}")
        if not block["n"]:
            continue
        print(f"    positions in pass 1   {block['first_pass_span'][0]} to "
              f"{block['first_pass_span'][1]}"
              + ("   CONFINED TO THE TAIL" if block["confined_to_first_pass_tail"] else ""))
        print(f"    positions in pass 2   {block['second_pass_span'][0]} to "
              f"{block['second_pass_span'][1]}"
              + ("   CONFINED TO THE TAIL" if block["confined_to_second_pass_tail"] else ""))

    print("\n  internal consistency on repeated answers")
    for name, block in report["internal_consistency"].items():
        print(f"    {name:<12} {block['consistent']}/{block['groups']} consistent")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  written to {args.output}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    items = load_sheet(Path(args.sheet))
    judgements = load_judgements(Path(args.judgements))
    state = progress(items, judgements)
    for name in ("total", "scored", "remaining", "uncertain", "revised", "with_notes"):
        print(f"  {name:<11}{state[name]}")
    if state["uncertain"]:
        flagged = sorted(j.item for j in judgements.values() if j.uncertain)
        print(f"\n  flagged uncertain: {flagged}")

    # Reported every time, whatever it says. A blinding control that is only
    # examined when it looks good is not a control.
    print("\n  blinding (amendment 1.13)")
    print(f"    arm identifiable      {state['arm_identified']}")
    print(f"    of items asked        {state['identification_asked']}")
    print(f"    scored before flag    {state['identification_not_asked']}")
    rate = state["unblinding_rate"]
    print(f"    self-reported rate    "
          + ("not yet measurable" if rate is None else f"{rate:.3f}"))
    print("    this rate is a self-report and bounds nothing in either direction")

    audit = arm_signature_audit(items)
    if not audit["blind"]:
        for finding in audit["findings"]:
            print(f"\n    known leak: {finding['template']} verbatim on "
                  f"{len(finding['items'])} items across {finding['questions']} "
                  f"questions, exposing {finding['items_exposed']} of "
                  f"{audit['total_items']} items")
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
    # Amendment 1.14 makes the abstention re-pass part of scoring, so the key
    # stays sealed until it is finished too. Unsealing between the passes would
    # let the arm mapping reach the second one, which is the whole thing the
    # ordering exists to prevent.
    second = load_abstention(Path(args.abstention_log))
    outstanding = [i.item for i in items if i.item not in second]
    if outstanding:
        print(
            f"Refusing to unseal: the abstention re-pass has {len(outstanding)} of "
            f"{len(items)} items outstanding.\nAmendment 1.14 makes it part of "
            "scoring. Run `abstention` first.",
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
    parser.add_argument("--abstention-log", default=str(DEFAULT_ABSTENTION))
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

    abstention = sub.add_parser(
        "abstention", help="focused re-pass: did the answer decline, yes or no"
    )
    abstention.add_argument("--session", default=str(DEFAULT_SESSION))
    abstention.add_argument("--order-seed", type=int, default=ABSTENTION_ORDER_SEED)
    abstention.set_defaults(func=command_abstention)

    agreement = sub.add_parser(
        "agreement", help="compare the two passes on the abstention field"
    )
    agreement.add_argument("--output", metavar="PATH")
    agreement.set_defaults(func=command_agreement)

    drift = sub.add_parser(
        "drift", help="was a disagreement caused by position, or by criterion"
    )
    drift.add_argument("--order-seed", type=int, default=ABSTENTION_ORDER_SEED)
    drift.add_argument("--output", metavar="PATH")
    drift.set_defaults(func=command_drift)

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
