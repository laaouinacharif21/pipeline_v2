# -*- coding: utf-8 -*-
"""
Layer-wise semantic separation across words and model families.

Three panels, one per family, each showing seven curves. Every curve is the
mean Sep(l) over the models in that family for one target word, plotted
against relative depth so that families with different layer counts are
comparable.

A second figure overlays the controlled words only, with the mean and spread
across them, and shows bank separately. Bank predates the controlled design
and reaches perfect context-only classification, so it is drawn apart rather
than averaged in.

Usage
    python -m src.analysis.cross_word_separation
    python -m src.analysis.cross_word_separation --words bank,bat,crane
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import FAMILIES, load_sep
from src.utils.paths import get_results_root, ensure_dir

WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]
UNCONTROLLED = "bank"

TITLES = {"llama": "LLaMA family", "qwen": "Qwen family",
          "bert": "BERT-family encoders"}

COLOURS = {
    "bank": "#2C2C2A", "bat": "#1f77b4", "crane": "#d62728",
    "seal": "#2ca02c", "plant": "#9467bd", "pupil": "#ff7f0e",
    "club": "#17becf",
}

GRID = np.linspace(0.0, 1.0, 41)

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def family_curve(family, word):
    """Mean Sep(l) over the models of one family, interpolated onto a common grid."""
    curves = []
    for m in FAMILIES[family]:
        s = load_sep(m, word)
        if s is None or len(s) < 5:
            continue
        depth = s.layer.values / s.layer.values.max()
        curves.append(np.interp(GRID, depth, s.separation.values))
    if not curves:
        return None, None
    a = np.vstack(curves)
    return a.mean(axis=0), a.std(axis=0)


def fig_by_family(words, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), sharey=True)
    for ax, family in zip(axes, ["llama", "qwen", "bert"]):
        for w in words:
            mean, _ = family_curve(family, w)
            if mean is None:
                continue
            ax.plot(GRID, mean, linewidth=2.2, color=COLOURS.get(w),
                    linestyle="--" if w == UNCONTROLLED else "-",
                    label=w + (" (high-context)" if w == UNCONTROLLED else ""))
        ax.set_title(TITLES[family])
        ax.set_xlabel("Relative depth")
        ax.axhline(0, color="grey", linewidth=1, linestyle=":", alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Semantic separation  Sep(l)")
    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("Layer-wise semantic separation across model architectures",
                 fontsize=19, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_separation_by_family.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"  Saved -> {out}")


def fig_controlled_vs_bank(words, out_dir):
    controlled = [w for w in words if w != UNCONTROLLED]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), sharey=True)

    for ax, family in zip(axes, ["llama", "qwen", "bert"]):
        stack = []
        for w in controlled:
            mean, _ = family_curve(family, w)
            if mean is not None:
                stack.append(mean)
        if stack:
            a = np.vstack(stack)
            m, sd = a.mean(axis=0), a.std(axis=0)
            ax.fill_between(GRID, m - sd, m + sd, color="#1f77b4", alpha=0.18)
            ax.plot(GRID, m, linewidth=2.6, color="#1f77b4",
                    label=f"six controlled datasets (mean)")
            j = int(np.argmax(m))
            ax.plot(GRID[j], m[j], marker="o", markersize=8,
                    color="#1f77b4", zorder=5)
            ax.annotate(f"peak {GRID[j]:.2f}", xy=(GRID[j], m[j]),
                        xytext=(6, -16), textcoords="offset points",
                        fontsize=11, color="#1f77b4")
        bm, _ = family_curve(family, UNCONTROLLED)
        if bm is not None:
            ax.plot(GRID, bm, linewidth=2.4, color="#2C2C2A", linestyle="--",
                    label=f"{UNCONTROLLED} dataset (high context)")
            j = int(np.argmax(bm))
            ax.plot(GRID[j], bm[j], marker="s", markersize=8, color="#2C2C2A", zorder=5)
            ax.annotate(f"peak {GRID[j]:.2f}", xy=(GRID[j], bm[j]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=11, color="#2C2C2A")
        ax.set_title(TITLES[family])
        ax.set_xlabel("Relative depth")
        ax.axhline(0, color="grey", linewidth=1, linestyle=":", alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Semantic separation  Sep(l)")
    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("Semantic separation peaks at different depths by dataset construction",
                 fontsize=19, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_separation_controlled_vs_bank.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"  Saved -> {out}")


def peak_summary(words):
    print(f"\n  {'family':<8}{'word':<8}{'peak Sep':>10}{'at depth':>10}{'final':>9}")
    print("  " + "-" * 45)
    for family in ["llama", "qwen", "bert"]:
        for w in words:
            mean, _ = family_curve(family, w)
            if mean is None:
                continue
            i = int(np.argmax(mean))
            print(f"  {family:<8}{w:<8}{mean[i]:>10.4f}{GRID[i]:>10.2f}{mean[-1]:>9.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default=",".join(WORDS))
    a = ap.parse_args()
    words = [w.strip() for w in a.words.split(",") if w.strip()]

    out_dir = ensure_dir(get_results_root() / "analysis" / "_cross_word" / "figures")
    print(f"\nSeparation curves   words: {len(words)}\n")
    fig_by_family(words, out_dir)
    fig_controlled_vs_bank(words, out_dir)

    # A shared y-axis makes the contrast with bank readable but compresses the
    # controlled words. This version drops bank and scales each panel to its
    # own range, so the shape of the controlled curves is visible.
    controlled = [w for w in words if w != UNCONTROLLED]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, family in zip(axes, ["llama", "qwen", "bert"]):
        for w in controlled:
            mean, _ = family_curve(family, w)
            if mean is not None:
                ax.plot(GRID, mean, linewidth=2.4, color=COLOURS.get(w), label=w)
        ax.set_title(TITLES[family])
        ax.set_xlabel("Relative depth")
        ax.axhline(0, color="grey", linewidth=1, linestyle=":", alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Semantic separation  Sep(l)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=12, frameon=False)
    fig.suptitle("Controlled datasets only, scaled independently per family",
                 fontsize=19, y=1.02)
    plt.tight_layout()
    out = out_dir / "fig_separation_controlled_only.png"
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close()
    print(f"  Saved -> {out}")

    peak_summary(words)


if __name__ == "__main__":
    main()
