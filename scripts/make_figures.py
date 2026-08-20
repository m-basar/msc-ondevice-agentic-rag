"""Generate every figure in Chapter 4 from the committed analysis outputs.

    python scripts/make_figures.py

Writes PNG files to ``docs/dissertation/figures/``. Requires matplotlib, which
is the ``plots`` optional dependency in ``pyproject.toml``.

Amendment 1.26.4 governs this script and it holds to three rules.

**No value is typed in.** Every number plotted is read from
``results/analysis/hypotheses.json`` or from the three performance reports. A
figure that carries a transcribed number can drift from the result it depicts,
and nothing would catch it.

**Nothing is aggregated here.** The script arranges and draws. Group-level
means come from the ``family_level`` and ``group_level`` fields the analysis
writes, not from a mean taken at plot time, because a number computed in the
plotting layer has no test behind it. Where a distribution is shown it is the
observed points, read through ``analyse_performance.performance_run`` so that
the 47 validation checks run before anything is plotted.

**The two parameters the figures need are parsed from the reports, not typed.**
Sample sizes come from the arm blocks and the H5 range is read out of the
hypothesis statement, with a hard failure if the statement stops matching.

**No inferential error bars.** Section 5 of the pre-registration computes no
confidence interval anywhere in this study, and a figure implying one would
claim a precision the design does not have. Spread appears as the observed
points and quartiles, labelled as such.

Colours are the Okabe-Ito set, which is designed for colour-vision deficiency.
Identity is never carried by colour alone: every bar is directly labelled and
every figure has a companion table in the chapter.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analyse_performance  # noqa: E402
import figure_provenance  # noqa: E402

ANALYSIS = ROOT / "results" / "analysis"
FIGURES = ROOT / "docs" / "dissertation" / "figures"

#: Okabe-Ito, validated for colour-vision deficiency with
#: ``scripts/validate_palette.js`` from the data-visualisation guidance.
#:
#: The assignment is not the palette's default order. B and D are the
#: confirmatory contrast and appear alone as a two-series comparison in two
#: figures, so they take the pair that separates most widely: blue against
#: vermillion, dE 31.2 for normal vision and 21.9 under protanopia. The
#: default order would have put orange against vermillion there, which passes
#: at dE 15.6 but only just clears the floor. Worst all-pairs separation across
#: the four is dE 11.0 under deuteranopia.
#:
#: Colour follows the arm, never its rank, so an arm keeps its hue in every
#: figure regardless of how it scores.
ARM_COLOUR = {"A": "#009E73", "B": "#0072B2", "C": "#E69F00", "D": "#D55E00"}
ARMS = ("A", "B", "C", "D")
INK = "#1a1a1a"
MUTED = "#666666"
GRID = "#d9d9d9"
CONDITIONS = (
    ("laptop_gpu", "Laptop, GPU"),
    ("laptop_cpu", "Laptop, CPU"),
    ("pi5_cpu", "Raspberry Pi 5, CPU"),
)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.size": 8.5,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def load_hypotheses() -> dict:
    return json.loads((ANALYSIS / "hypotheses.json").read_text(encoding="utf-8"))


def load_performance() -> dict:
    return {
        key: json.loads(
            (ANALYSIS / f"performance_latest_test_performance_{key}.json")
            .read_text(encoding="utf-8")
        )
        for key, _ in CONDITIONS
    }


def predicted_range(statement: str) -> tuple[float, float]:
    """The H5 range, read out of the hypothesis statement rather than typed.

    Hard-coding 1.5 and 2.5 would let the figure keep drawing the old band if
    the pre-registered range were ever amended. Failing loudly is the point.
    """
    found = re.findall(r"([0-9]+(?:\.[0-9]+)?)x", statement)
    if len(found) != 2:
        raise SystemExit(
            f"cannot read the predicted range from the H5 statement: {statement!r}"
        )
    return float(found[0]), float(found[1])


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    # Constant metadata. Matplotlib's default writes its own version into the
    # PNG's Software field, so an otherwise identical regeneration under a
    # different matplotlib produced different bytes while amendment 1.26 called
    # the figures reproducible. Amendment 1.30.7.
    fig.savefig(path, metadata=dict(figure_provenance.PNG_METADATA))
    plt.close(fig)
    return path


def bars(ax, values: dict[str, float], *, fmt: str, ceiling: float) -> None:
    """Thin bars, one colour per arm, every bar directly labelled.

    The labels are not decoration. The orange in the Okabe-Ito set sits below
    3:1 contrast against white, so identity has to be carried by something
    other than the fill.

    The axis is given headroom above the scale ceiling so that a bar at the
    ceiling still has room for its label. Without it a value of 1.000 collides
    with the panel title.
    """
    top = ceiling * 1.16
    for index, arm in enumerate(ARMS):
        ax.bar(index, values[arm], width=0.62, color=ARM_COLOUR[arm],
               edgecolor="white", linewidth=1.2)
        ax.text(index, values[arm] + ceiling * 0.035, fmt.format(values[arm]),
                ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([f"Arm {a}" for a in ARMS])
    ax.set_ylim(0, top)


# --- Figure 4.1 --------------------------------------------------------------


def figure_arm_comparison(report: dict) -> Path:
    """The five section 4 primary metrics, by arm.

    Conflict handling is one metric, not two, but the hypotheses partition its
    families into supersession and live disagreement and score them separately,
    so it is shown on both sets. An earlier version of this figure called that
    "all five metrics" while omitting false-conflict rate entirely, which is the
    kind of caption error a reader cannot check.

    Two rows because the metrics use two scales, and the level of aggregation is
    stated on every panel because it is not the same one throughout: rubric
    metrics are family means, the rate metrics are per question except
    abstention, which is per gap topic.
    """
    hyp = report["hypotheses"]
    primary = report["primary_metrics"]
    superseded = primary["superseded_citation_rate"]["by_arm"]
    controls = hyp["H2c"]["false_conflict_rate"]
    abstention = hyp["H4"]["appropriate_abstention"]

    rubric = [
        ("Conflict handling, supersession", "higher is better",
         hyp["H1"]["levels"]["family_level"],
         f"family level\n{hyp['H1']['levels']['families']} families"),
        ("Conflict handling, live disagreement", "higher is better",
         hyp["H2"]["levels"]["family_level"],
         f"family level\n{hyp['H2']['levels']['families']} families"),
        ("Answer correctness", "higher is better",
         primary["answer_correctness"]["group_level"],
         f"group level\n{primary['answer_correctness']['groups']} groups"),
    ]
    rates = [
        ("Superseded citation rate", "LOWER is better",
         {a: superseded[a]["rate"] for a in ARMS},
         f"question level\n{superseded['A']['questions']} questions "
         f"in {primary['superseded_citation_rate']['families']} families"),
        ("False-conflict rate", "LOWER is better",
         {a: controls[a]["rate"] for a in ARMS},
         f"question level\n{controls['A']['questions']} questions "
         f"in {controls['A']['groups']} controls"),
        ("Appropriate abstention", "higher is better",
         {a: abstention[a]["groups_all_hit"] / abstention[a]["groups"] for a in ARMS},
         f"gap-topic level\n{hyp['H4']['gap_topics']} topics"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 6.2))
    for ax, (title, merit, values, n) in zip(axes[0], rubric):
        bars(ax, values, fmt="{:.2f}", ceiling=2.0)
        ax.set_title(f"{title}\n{merit}\n({n})", color=INK)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    axes[0][0].set_ylabel("Mean rubric score (0 to 2)")

    for ax, (title, merit, values, n) in zip(axes[1], rates):
        bars(ax, values, fmt="{:.3f}", ceiling=1.0)
        ax.set_title(f"{title}\n{merit}\n({n})", color=INK)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1][0].set_ylabel("Proportion (0 to 1)")
    fig.tight_layout()
    return save(fig, "fig_4_1_arm_comparison.png")


# --- Figure 4.2 --------------------------------------------------------------


def figure_paired_differences(report: dict) -> Path:
    """Per-family D minus B, against each hypothesis's own decision rule.

    B against D is the confirmatory contrast for both, per section 2, but the
    two hypotheses do not test the same thing and must not be drawn as though
    they did. H1's second leg asks whether the arms sit **within** a symmetric
    0.25 margin, so the band is two-sided. H2 predicts D **above** B by more
    than 0.25, so the criterion is a one-sided region and a symmetric band
    would imply H2 could be satisfied by a difference in either direction.
    """
    hyp = report["hypotheses"]
    threshold = hyp["H1"]["superiority_leg"]["decisions"]["B_vs_A"]["effect_threshold"]
    panels = [
        ("H1: supersession families",
         "within-margin leg, symmetric +/-{:.2f} margin".format(threshold),
         "symmetric",
         hyp["H1"]["within_margin_leg"]["contrasts"]["D_vs_B"],
         hyp["H1"]["within_margin_leg"]["decisions"]["D_vs_B"]["verdict"],
         hyp["H1"]["verdict"]),
        ("H2: pooled live-disagreement families",
         "superiority, D must exceed B by more than {:.2f}".format(threshold),
         "one_sided",
         hyp["H2"]["contrasts"]["D_vs_B"],
         hyp["H2"]["decisions"]["D_vs_B"]["verdict"],
         hyp["H2"]["verdict"]),
    ]

    heights = [len(p[3]["per_family"]) for p in panels]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.6, 6.0), gridspec_kw={"height_ratios": heights},
    )
    for ax, (title, rule, kind, contrast, contrast_verdict, verdict) in zip(axes, panels):
        families = list(contrast["per_family"])
        values = [contrast["per_family"][f] for f in families]
        positions = range(len(families))

        if kind == "symmetric":
            ax.axvspan(-threshold, threshold, color=GRID, alpha=0.6, zorder=0)
        else:
            ax.axvspan(threshold, 0.85, color=ARM_COLOUR["C"], alpha=0.20, zorder=0)
            ax.axvline(threshold, color=ARM_COLOUR["C"], linewidth=1.4, zorder=1)
        ax.axvline(0, color=MUTED, linewidth=1.0, zorder=1)
        for y, value in zip(positions, values):
            ax.plot([0, value], [y, y], color=MUTED, linewidth=1.2, zorder=2)
            ax.plot(value, y, "o", markersize=8, color=ARM_COLOUR["D"],
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            ax.text(value + (0.022 if value >= 0 else -0.022), y,
                    f"{value:+.3f}", va="center", fontsize=7.5, color=INK,
                    ha="left" if value >= 0 else "right")
        difference = contrast["paired_mean_difference"]
        ax.axvline(difference, color=INK, linewidth=1.8, zorder=4)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(families)
        ax.set_ylim(len(families) - 0.45, -0.55)
        ax.set_xlim(-0.85, 0.85)
        ax.set_title(
            f"{title}\n{rule}\n"
            f"{contrast['better']} better, {contrast['tied']} tied, "
            f"{contrast['worse']} worse. Paired mean difference "
            f"{difference:+.4f}, {contrast_verdict}. Hypothesis: {verdict}.",
            color=INK,
        )
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel(
        "Arm D minus Arm B, mean rubric score per family (points on a 0 to 2 scale).\n"
        "Positive favours the verification layer.")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GRID, alpha=0.6, edgecolor="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor=ARM_COLOUR["C"], alpha=0.20,
                      edgecolor="none"),
        plt.Line2D([0], [0], color=INK, linewidth=1.8),
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                   color=ARM_COLOUR["D"], markeredgecolor="white"),
    ]
    labels = [
        f"H1: within the symmetric {threshold} margin",
        f"H2: the one-sided region above +{threshold}",
        "Paired mean difference",
        "One conflict family",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.tight_layout()
    return save(fig, "fig_4_2_paired_family_differences.png")


# --- Figure 4.3 --------------------------------------------------------------


def latency_series() -> dict[str, dict[str, list[float]]]:
    """Per-question wall clock, read through the validated loader.

    The performance reports carry summary statistics only, so a distribution
    has to come from the run records. Going through ``performance_run`` and
    ``validate`` means the same 47 checks that gate the reported ratio also gate
    these points; reading the JSONL directly would skip them.
    """
    series: dict[str, dict[str, list[float]]] = {}
    for key, _ in CONDITIONS:
        index = json.loads(
            (ROOT / "results" / "runs" / f"latest_test_performance_{key}.json")
            .read_text(encoding="utf-8")
        )
        index_meta = index.pop("_performance", {})
        runs = {
            arm: analyse_performance.performance_run(ROOT / rel)
            for arm, rel in sorted(index.items())
        }
        outcome = analyse_performance.validate(runs, index_meta)
        if not outcome["valid"]:
            raise SystemExit(
                f"{key} failed validation, refusing to plot: "
                f"{outcome['checks_failed']}"
            )
        series[key] = {
            arm: [record["wall_seconds"] for record in answers]
            for arm, (_, answers, _) in runs.items()
        }
    return series


def figure_latency_distributions(series: dict) -> Path:
    """Observed spread, not an inferential interval.

    One panel per platform with its own axis. A shared axis would compress the
    GPU condition into the baseline, and a log axis would make a threefold
    difference look like a small one.
    """
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6))
    for ax, (key, label) in zip(axes, CONDITIONS):
        data = [series[key]["B"], series[key]["D"]]
        box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                         medianprops={"color": INK, "linewidth": 1.4},
                         whiskerprops={"color": MUTED},
                         capprops={"color": MUTED},
                         boxprops={"edgecolor": "white", "linewidth": 1.2})
        for patch, arm in zip(box["boxes"], ("B", "D")):
            patch.set_facecolor(ARM_COLOUR[arm])
            patch.set_alpha(0.85)
        for index, (values, arm) in enumerate(zip(data, ("B", "D")), start=1):
            jitter = [index + (i % 11 - 5) * 0.018 for i in range(len(values))]
            ax.plot(jitter, values, "o", markersize=2.2, color=INK, alpha=0.4,
                    markeredgewidth=0, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Arm B\nn={len(data[0])}", f"Arm D\nn={len(data[1])}"])
        ax.set_title(label, color=INK)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("End-to-end latency per answer (seconds)")
    fig.text(0.5, -0.06,
             "Box is the interquartile range with the median; whiskers reach "
             f"1.5 x IQR. Points are the {len(series['pi5_cpu']['B'])} "
             "individual questions.\n"
             "Spread is observed, not an inferential interval: no confidence "
             "interval is computed anywhere in this study.",
             ha="center", fontsize=7.5, color=MUTED, linespacing=1.5)
    fig.tight_layout()
    return save(fig, "fig_4_3_latency_distributions.png")


# --- Figure 4.4 --------------------------------------------------------------


def figure_latency_overhead(performance: dict) -> Path:
    """Mean cost per platform, and the ratio against the H5 prediction."""
    fig = plt.figure(figsize=(8.4, 5.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.42)

    for column, (key, label) in enumerate(CONDITIONS):
        ax = fig.add_subplot(grid[0, column])
        arms = performance[key]["arms"]
        baseline = arms["B"]["wall_seconds"]["mean"]
        verification = arms["D"]["verification_seconds"]["mean"]
        draft = arms["D"]["wall_seconds"]["mean"] - verification
        top = arms["D"]["wall_seconds"]["mean"] * 1.55

        ax.bar(0, baseline, width=0.6, color=ARM_COLOUR["B"],
               edgecolor="white", linewidth=1.2)
        ax.bar(1, draft, width=0.6, color=ARM_COLOUR["B"], alpha=0.45,
               edgecolor="white", linewidth=1.2,
               label="Draft (replayed)")
        ax.bar(1, verification, width=0.6, bottom=draft, color=ARM_COLOUR["D"],
               edgecolor="white", linewidth=1.2, label="Verification")
        ax.text(0, baseline + top * 0.03, f"{baseline:.2f} s", ha="center",
                va="bottom", fontsize=7.5, color=INK)
        ax.text(1, draft + verification + top * 0.03,
                f"{draft + verification:.2f} s", ha="center", va="bottom",
                fontsize=7.5, color=INK)
        ax.text(1, draft + verification / 2, f"{verification:.2f} s",
                ha="center", va="center", fontsize=7, color="white")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Arm B", "Arm D"])
        ax.set_ylim(0, top)
        ax.set_title(
            f"{label}\n(n={arms['B']['questions']} questions per arm)", color=INK)
        if column == 0:
            ax.set_ylabel("Mean latency per answer (seconds)")
            ax.legend(loc="upper left")

    ax = fig.add_subplot(grid[1, :])
    lower, upper = predicted_range(performance["pi5_cpu"]["H5"]["statement"])
    ratios = [performance[key]["H5"]["ratio"] for key, _ in CONDITIONS]
    labels = [label for _, label in CONDITIONS]
    ax.axhspan(lower, upper, color=ARM_COLOUR["C"], alpha=0.18, zorder=0,
               label=f"H5 predicted range, {lower:g}x to {upper:g}x")
    for index, (ratio, key) in enumerate(zip(ratios, [k for k, _ in CONDITIONS])):
        verdict = performance[key]["H5"]["verdict"]
        ax.bar(index, ratio, width=0.5, color=ARM_COLOUR["D"],
               edgecolor="white", linewidth=1.2, zorder=2)
        ax.text(index, ratio + 0.09, f"{ratio:.2f}x", ha="center", va="bottom",
                fontsize=8, color=INK)
        ax.text(index, 0.16, verdict.upper(), ha="center", va="bottom",
                fontsize=7.5, color="white", zorder=3, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Arm D / Arm B, mean latency")
    ax.set_ylim(0, max(ratios) * 1.42)
    ax.set_title("Latency ratio against the H5 prediction", color=INK)
    ax.legend(loc="upper right")
    fig.text(0.5, -0.02,
             "H5 is stated over the Raspberry Pi 5 and is scored only there. "
             "The two laptop ratios are descriptive RQ4 figures and carry no "
             "verdict.\nArm D replays Arm B's drafts, so the draft segment is "
             "the same generation work in both arms and the difference is the "
             "verification pass alone.",
             ha="center", fontsize=7.5, color=MUTED, linespacing=1.5)
    return save(fig, "fig_4_4_latency_overhead.png")


def main() -> int:
    style()
    report = load_hypotheses()
    performance = load_performance()
    written = [
        figure_arm_comparison(report),
        figure_paired_differences(report),
        figure_latency_distributions(latency_series()),
        figure_latency_overhead(performance),
    ]
    for path in written:
        if figure_provenance.carries_a_version(path):
            raise SystemExit(
                f"{path.name} still names a matplotlib version. Constant "
                "metadata is what makes two regenerations comparable; "
                "amendment 1.30.7."
            )
        print(f"  wrote {path.relative_to(ROOT)}")
    print(f"  wrote {figure_provenance.write_environment(FIGURES).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
