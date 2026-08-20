"""Draw Figures 3.1 and 3.2 for the methodology chapter.

    python scripts/make_architecture_figures.py

Writes an editable SVG and a 300 dpi PNG for each figure into
``docs/dissertation/figures/``.

These are schematics of a design, not plots of data, so unlike
``scripts/make_figures.py`` they contain literal text. What they must not
contain is any component the artefact does not have. The diagram they replace
showed a vector database, an embedding model, a query-analysis stage, a
claim-extraction agent and a next-action agent that were never built, and
described the confidence mechanism as calibrated. Every label below names
something present in ``final_v1``, and the parameters are the ones in
``config.json``.

The dashboard is deliberately absent. It is not built, and a figure is not the
place to promise one.

Neither figure carries an explanatory footer. What a diagram omits, and why, is
an argument, and an argument belongs in the chapter that can be held to it. A
figure that states its own case duplicates the prose around it and cannot be
reused anywhere the prose does not follow.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "dissertation" / "figures"

#: Okabe-Ito, the same assignment the results figures use, so an arm keeps its
#: colour across the whole dissertation.
ARM = {"A": "#009E73", "B": "#0072B2", "C": "#E69F00", "D": "#D55E00"}
INK = "#1a1a1a"
MUTED = "#666666"
EDGE = "#9a9a9a"
STAGE = "#eef3f8"
STORE = "#f3f0e8"
OUTPUT = "#eef7f3"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 8.5, "savefig.bbox": "tight", "svg.fonttype": "none",
        # Deterministic element ids, so two runs of this script produce the
        # same SVG rather than one that differs only in generated identifiers.
        "svg.hashsalt": "sme-assistant-figures",
    })


def box(ax, x, y, w, h, label, *, fill=STAGE, edge=EDGE, size=8.0,
        weight="normal", colour=INK, lw=1.1, radius=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=size, color=colour, zorder=3, linespacing=1.45,
            fontweight=weight)


def arrow(ax, start, end, *, colour=MUTED, lw=1.3, style_="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style_, mutation_scale=11, color=colour,
        linewidth=lw, zorder=4, shrinkA=1, shrinkB=1,
        linestyle=(0, (4, 3)) if dashed else "solid"))


def route(ax, points, *, colour=MUTED, lw=1.3):
    """A connector routed through explicit waypoints, arrowhead at the end.

    Diagonals that cut across a diagram are hard to follow and collide with
    whatever label sits near their midpoint. Matplotlib's angled connection
    styles manage one bend; these routes need two, so the waypoints are given
    rather than inferred.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs[:-1] + [xs[-1]], ys[:-1] + [ys[-1]], color=colour,
            linewidth=lw, zorder=4, solid_capstyle="round",
            solid_joinstyle="round")
    arrow(ax, points[-2], points[-1], colour=colour, lw=lw)


def save(fig, stem: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, dpi in ((".svg", None), (".png", 300)):
        path = FIGURES / f"{stem}{suffix}"
        # Matplotlib stamps a creation date into SVG metadata by default, which
        # makes an otherwise identical regeneration produce a different file.
        # A committed artefact that changes on every run cannot be checked
        # against its source, so the stamp is suppressed.
        options = {"metadata": {"Date": None}} if suffix == ".svg" else {"dpi": dpi}
        fig.savefig(path, **options)
        written.append(path)
    plt.close(fig)
    return written


def figure_architecture(config: dict, chunks: int, documents: int) -> list[Path]:
    """Figure 3.1, the implemented pipeline inside its deployment boundary."""
    generation, verification = config["generation"], config["verification"]
    llm, chunking, retrieval = config["llm"], config["chunking"], config["retrieval"]

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 68); ax.axis("off")

    # Deployment boundary.
    ax.add_patch(FancyBboxPatch(
        (1.5, 2.0), 97, 58.5, boxstyle="round,pad=0,rounding_size=0.012",
        facecolor="#fbfbfc", edgecolor=INK, linewidth=1.5,
        linestyle=(0, (6, 4)), zorder=1))
    ax.text(3.5, 57.6, "Deployment boundary: Raspberry Pi 5, 16 GB RAM, CPU only",
            fontsize=9.5, color=INK, fontweight="bold", va="center")
    ax.text(3.5, 54.4, f"Local {llm['backend'].capitalize()} runtime. All inference "
            "on device. No cloud inference and no network egress.",
            fontsize=8.2, color=MUTED, va="center")

    # --- Lane 1: index build, offline -------------------------------------
    ax.text(3.5, 48.6, "INDEX BUILD, ONCE", fontsize=7.6, color=MUTED,
            fontweight="bold", va="center")
    lane1_y, h1 = 38.0, 8.2
    stages1 = [
        (3.5, 21.0, f"Synthetic SME corpus\n{documents} Markdown documents", STORE),
        (27.5, 18.5, f"Chunker\nmax {chunking['max_words']} words, min "
                     f"{chunking['min_words']}\n{chunking['overlap_sentences']}-sentence overlap", STAGE),
        (49.0, 14.0, f"{chunks} chunks", STORE),
        (66.0, 15.5, f"Embeddings\n{llm['embedding_model']}", STAGE),
        (84.0, 13.0, "Local vector\nindex", STORE),
    ]
    for x, w, label, fill in stages1:
        box(ax, x, lane1_y, w, h1, label, fill=fill)
    for i in range(len(stages1) - 1):
        x, w, _, _ = stages1[i]
        arrow(ax, (x + w, lane1_y + h1 / 2), (stages1[i + 1][0], lane1_y + h1 / 2))

    # --- Lane 2: per question ---------------------------------------------
    ax.text(3.5, 32.0, "PER QUESTION", fontsize=7.6, color=MUTED,
            fontweight="bold", va="center")
    lane2_y, h2 = 19.5, 9.4
    box(ax, 3.5, lane2_y, 15.0, h2, "Question", fill=OUTPUT)
    box(ax, 22.5, lane2_y, 19.5, h2,
        f"Retriever\ntop_k = {retrieval['top_k']}\n"
        f"min_similarity = {retrieval['min_similarity']:.2f}", fill=STAGE)
    box(ax, 46.0, lane2_y, 16.5, h2, "Evidence block\nchunk id + text\n+ status metadata",
        fill=STORE)
    box(ax, 66.5, lane2_y, 15.5, h2,
        f"Grounded draft\n{llm['generation_model']}\ntemp {generation['temperature']}",
        fill="#dceaf5", edge=ARM["B"], lw=1.6)
    box(ax, 86.0, lane2_y, 11.0, h2,
        f"Verifier\n{llm['verification_model']}\ntemp {verification['temperature']}",
        fill="#fbe0d2", edge=ARM["D"], lw=1.6)
    ax.text(91.5, lane2_y + h2 + 1.6, "Arm D only", ha="center", fontsize=7.4,
            color=ARM["D"], fontweight="bold")

    for a, b in ((18.5, 22.5), (42.0, 46.0), (62.5, 66.5), (82.0, 86.0)):
        arrow(ax, (a, lane2_y + h2 / 2), (b, lane2_y + h2 / 2))
    # Index feeds the retriever, routed as a right angle rather than a diagonal.
    route(ax, [(90.5, lane1_y), (90.5, 33.2), (32.25, 33.2),
               (32.25, lane2_y + h2)], colour=EDGE, lw=1.1)
    ax.text(61.5, 34.6, "index consulted at query time", fontsize=7.2,
            color=MUTED, ha="center", style="italic")

    # --- Outputs -----------------------------------------------------------
    out_y = 4.0
    out_h = 9.6
    box(ax, 3.5, out_y, 27.0, out_h,
        "Served answer\nwith [chunk id] citations", fill=OUTPUT, edge="#7fbfa6")
    box(ax, 34.5, out_y, 27.0, out_h,
        "Conflict relationship or\ninsufficient-evidence status", fill=OUTPUT,
        edge="#7fbfa6")
    box(ax, 65.5, out_y, 31.5, out_h,
        "Rule-based categorical confidence\n(declared mapping, not calibrated)\n"
        "+ per-run provenance and timing record", fill=OUTPUT, edge="#7fbfa6")
    # Arms A, B and C serve the draft; Arm D serves whatever the verifier
    # returns, which is the draft unchanged whenever it has no complaint.
    route(ax, [(70.0, lane2_y), (70.0, 17.9), (17.0, 17.9),
               (17.0, out_y + out_h)], colour=ARM["B"], lw=1.3)
    route(ax, [(91.5, lane2_y), (91.5, 15.4), (48.0, 15.4),
               (48.0, out_y + out_h)], colour=ARM["D"], lw=1.3)
    route(ax, [(91.5, lane2_y), (91.5, 15.4), (81.0, 15.4),
               (81.0, out_y + out_h)], colour=ARM["D"], lw=1.3)
    ax.text(43.0, 18.5, "arms A, B and C serve the draft", fontsize=7.2,
            color=ARM["B"], ha="center")
    ax.text(66.0, 16.1, "arm D serves what the verifier returns",
            fontsize=7.2, color=ARM["D"], ha="center")

    return save(fig, "fig_3_1_system_architecture")


def figure_arms() -> list[Path]:
    """Figure 3.2, the arms as a tree rooted at B.

    The three leaves sit on one row so that the only thing distinguishing the
    confirmatory edge is what it changes, not where it happens to be drawn.
    """
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")

    ax.text(50, 61.0,
            "Shared by all four arms: corpus, chunk set, index, embedding "
            "model, generation model, seed and retrieval parameters.",
            ha="center", fontsize=8.2, color=MUTED)

    root_x, root_y, root_w, root_h = 30.0, 42.0, 40.0, 12.0
    box(ax, root_x, root_y, root_w, root_h,
        "Arm B  (root)\nall documents retrieved\n"
        "evidence shown with status metadata\nno verification",
        fill="#dceaf5", edge=ARM["B"], lw=2.2, size=8.4)

    leaf_y, leaf_h = 14.0, 16.0
    leaves = {
        "A": (1.0, 27.0, "Arm A\nall documents\nidentifier and text only\n"
                         "no status metadata\nno verification"),
        "D": (36.5, 27.0, "Arm D\nall documents\nwith status metadata\n"
                          "B's draft replayed,\nthen verified"),
        "C": (72.0, 27.0, "Arm C\ncurrent documents only\nwith status metadata\n"
                          "no verification"),
    }
    for arm, (x, w, label) in leaves.items():
        highlight = arm == "D"
        box(ax, x, leaf_y, w, leaf_h, label, size=8.2,
            fill="#fbe0d2" if highlight else "white",
            edge=ARM[arm], lw=2.4 if highlight else 1.5)

    # Edges, one variable each.
    arrow(ax, (root_x + 4.0, root_y), (16.0, leaf_y + leaf_h),
          colour=ARM["A"], lw=1.8)
    arrow(ax, (50.0, root_y), (50.0, leaf_y + leaf_h), colour=ARM["D"], lw=3.0)
    arrow(ax, (root_x + root_w - 4.0, root_y), (85.0, leaf_y + leaf_h),
          colour=ARM["C"], lw=1.8)

    ax.text(13.0, 36.4, "changes\nevidence format", ha="center", fontsize=8.0,
            color=ARM["A"], linespacing=1.5)
    ax.text(88.0, 36.4, "changes\nretrieval mode", ha="center", fontsize=8.0,
            color=ARM["C"], linespacing=1.5)
    # The two diagonal edges fan outwards, so the clear space beside the
    # vertical edge is low down rather than high up. Placed higher, these
    # labels collided with the Arm A edge on one side and Arm C's on the other.
    ax.text(52.0, 33.6, "changes verification only", ha="left", fontsize=8.6,
            color=ARM["D"], fontweight="bold")
    ax.text(52.0, 31.0, "THE CONFIRMATORY CONTRAST", ha="left", fontsize=7.8,
            color=ARM["D"], fontweight="bold")

    # C against D: two variables at once, so not an ablation.
    arrow(ax, (63.5, 22.0), (72.0, 22.0), colour=MUTED, lw=1.4, dashed=True,
          style_="<|-|>")
    ax.text(50.0, 6.4,
            "C against D changes retrieval mode *and* verification, so no "
            "difference between them can be attributed to either alone.\n"
            "It is reported as the practical comparison a practitioner would "
            "make, explicitly not as an ablation isolating verification.",
            ha="center", fontsize=8.0, color=MUTED, linespacing=1.6)

    return save(fig, "fig_3_2_experimental_arms")


def main() -> int:
    style()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    index = json.loads((ROOT / "data" / "index.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "src"))
    from sme_assistant.kb.loader import load_knowledge_base
    documents = len(list(load_knowledge_base(ROOT / "data" / "kb").documents))

    written = figure_architecture(config, len(index["chunks"]), documents)
    written += figure_arms()
    for path in written:
        print(f"  wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
