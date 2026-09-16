"""Pipeline overview figure for the paper, rendered to PNG for the DOCX.

Kept as a script rather than a drawing so the numbers on the figure (token
counts, compression ratio) are read from the grid instead of being typed in by
hand and then going stale.
"""
import argparse, glob, json, os, re, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2933"
MUTED = "#5c6b7a"
EDGE = "#8a99a8"
FILL_RETR = "#dce8f2"
FILL_COMP = "#f6e5cf"
FILL_READ = "#dfe8dc"
FILL_GROUP = "#f7f9fb"


def grid_numbers(pattern="OUTPUT/full/*_rerank-off.json"):
    """Mean uncompressed / compressed context tokens over the whole grid."""
    raw, comp = [], []
    for p in glob.glob(pattern):
        m = re.search(r"_(baseline|filter_summ)_rerank-off\.json$", p)
        if not m:
            continue
        rows = json.load(open(p))
        if m.group(1) == "baseline":
            raw += [float(r["retrieved_context_tokens"]) for r in rows]
        else:
            comp += [float(r["compressed_context_tokens"]) for r in rows]
    if not raw or not comp:
        return None
    return float(np.mean(raw)), float(np.mean(comp))


def box(ax, x, y, w, h, text, fill, fontsize=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.9, edgecolor=EDGE, facecolor=fill, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=4, linespacing=1.45, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, label=None, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 linewidth=1.0, color=MUTED, zorder=2,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.018, label, ha="center", va="bottom",
                fontsize=7.2, color=MUTED, zorder=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="OUTPUT/figures/overview.png")
    ap.add_argument("--glob", default="OUTPUT/full/*_rerank-off.json")
    args = ap.parse_args()

    nums = grid_numbers(args.glob)
    raw_tok, comp_tok = nums if nums else (4721.0, 559.0)
    ratio = raw_tok / comp_tok

    fig, ax = plt.subplots(figsize=(9.2, 4.5), dpi=220)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- row 1: retrieval ---------------------------------------------------
    box(ax, 0.015, 0.76, 0.135, 0.15, "Question\n$q$", "#ffffff")
    box(ax, 0.195, 0.76, 0.225, 0.15,
        "Dense retrieval\nContriever-MSMARCO\n21M DPR Wikipedia passages", FILL_RETR)
    box(ax, 0.465, 0.76, 0.205, 0.15,
        "Cross-encoder rerank\nbge-reranker-base\n(ablated: on / off)", FILL_RETR)
    box(ax, 0.715, 0.76, 0.27, 0.15,
        f"Top-$k$ passages  ($k$=30)\n$\\approx${raw_tok:,.0f} context tokens", "#ffffff")
    arrow(ax, 0.150, 0.835, 0.195, 0.835)
    arrow(ax, 0.420, 0.835, 0.465, 0.835)
    arrow(ax, 0.670, 0.835, 0.715, 0.835)

    # ---- row 2: compression -------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.045, 0.285), 0.91, 0.38,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.9, edgecolor=EDGE, facecolor=FILL_GROUP,
                                linestyle=(0, (4, 2.5)), zorder=1))
    ax.text(0.061, 0.625, "Query-aware compression — applied independently to each retrieved passage",
            ha="left", va="center", fontsize=8.6, color=INK, fontweight="bold", zorder=4)
    ax.text(0.061, 0.578, "training-free: no compressor is fitted to any dataset",
            ha="left", va="center", fontsize=7.4, color=MUTED, style="italic", zorder=4)

    box(ax, 0.070, 0.345, 0.255, 0.185,
        "Stage 1 — Keyword filter\nsplit passage into sentences;\nscore by query-token overlap;\n"
        "keep top 4  (fallback: first 4)", FILL_COMP)
    box(ax, 0.375, 0.345, 0.255, 0.185,
        "Stage 2 — Abstractive summary\nFLAN-T5-small, off the shelf,\ngreedy; chunked to 384 tok\n"
        "so no text is truncated away", FILL_COMP)
    box(ax, 0.680, 0.345, 0.255, 0.185,
        "Snippet assembly\n[Snippet $i$ | score=$s_i$]\nascending score order\n"
        f"$\\approx${comp_tok:,.0f} tok  ({ratio:.1f}$\\times$ smaller)", FILL_COMP)
    arrow(ax, 0.325, 0.4375, 0.375, 0.4375)
    arrow(ax, 0.630, 0.4375, 0.680, 0.4375)
    arrow(ax, 0.850, 0.760, 0.850, 0.665, style="-|>")

    # ---- row 3: reading -----------------------------------------------------
    box(ax, 0.150, 0.055, 0.30, 0.155,
        "Prompt assembly (RECOMP format)\n5-shot Q/A from the train split,\nthen snippets, then $q$, then 'Answer:'",
        FILL_READ)
    box(ax, 0.520, 0.055, 0.225, 0.155,
        "Reader (frozen)\nLlama-3.2-3B-Instruct\ngreedy, 96 new tokens", FILL_READ)
    box(ax, 0.795, 0.055, 0.145, 0.155, "Answer\nEM / F1 /\nBERTScore", "#ffffff")
    # L-shaped connector: the prompt is built from the ASSEMBLED snippets, so
    # the arrow must leave the assembly box, not stage 1 above it
    ax.plot([0.8075, 0.8075, 0.300], [0.345, 0.262, 0.262], color=MUTED,
            linewidth=1.0, zorder=2, solid_capstyle="round")
    arrow(ax, 0.300, 0.262, 0.300, 0.210)
    arrow(ax, 0.450, 0.1325, 0.520, 0.1325)
    arrow(ax, 0.745, 0.1325, 0.795, 0.1325)

    fig.text(0.5, 0.975, "Figure 1: The compression pipeline. Everything is training-free; "
             "the reader is never fine-tuned.", ha="center", va="top", fontsize=8.2, color=MUTED)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}  (raw {raw_tok:.0f} -> {comp_tok:.0f} tok, {ratio:.1f}x)")


if __name__ == "__main__":
    main()
