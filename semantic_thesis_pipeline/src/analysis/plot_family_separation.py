# -*- coding: utf-8 -*-
"""
Layer-wise semantic separation curves by model family.

Produces one figure per family plus a combined three-panel comparison.
Per-family panels use absolute layer index; the combined panel uses
relative depth, since families differ in layer count.

Usage:
    python -m src.analysis.plot_family_separation --word bank
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import FAMILIES, load_sep, analysis_dir

TITLES = {"llama": "LLaMA family", "qwen": "Qwen family",
          "bert": "BERT-family encoders"}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
})


def plot_family(fam, models, word, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    n = 0
    for i, m in enumerate(models):
        df = load_sep(m, word)
        if df is None:
            print(f"  [SKIP] {m}")
            continue
        ax.plot(df["layer"], df["separation"], marker="o", markersize=5,
                linewidth=2.2, label=m, color=COLORS[i % len(COLORS)])
        n += 1
    if n == 0:
        plt.close()
        return
    ax.set_xlabel("Layer")
    ax.set_ylabel("Semantic separation  Sep(l)")
    ax.set_title(f"{TITLES[fam]}  --  \"{word}\"")
    ax.axhline(0, color="grey", linewidth=1, linestyle="--", alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    out = out_dir / f"{fam}_family_separation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_combined(word, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    for ax, (fam, models) in zip(axes, FAMILIES.items()):
        for i, m in enumerate(models):
            df = load_sep(m, word)
            if df is None:
                continue
            rel = df["layer"] / df["layer"].max()
            ax.plot(rel, df["separation"], marker="o", markersize=4,
                    linewidth=2.2, label=m, color=COLORS[i % len(COLORS)])
        ax.set_title(TITLES[fam])
        ax.set_xlabel("Relative depth")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--", alpha=0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=12)
    axes[0].set_ylabel("Semantic separation  Sep(l)")
    plt.tight_layout()
    out = out_dir / "all_families_separation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def peak_table(word, out_dir):
    import pandas as pd
    rows = []
    print(f"\n{'Model':<22}{'peak Sep':>10}{'at layer':>10}{'rel depth':>11}{'final':>9}")
    print("-" * 62)
    for fam, models in FAMILIES.items():
        for m in models:
            df = load_sep(m, word)
            if df is None:
                continue
            i = df["separation"].idxmax()
            rel = df["layer"][i] / df["layer"].max()
            rows.append({"family": fam, "model": m, "peak_sep": df["separation"][i],
                         "peak_layer": int(df["layer"][i]), "peak_rel_depth": rel,
                         "final_sep": df["separation"].iloc[-1],
                         "n_layers": len(df)})
            print(f"{m:<22}{df['separation'][i]:>10.4f}{int(df['layer'][i]):>10}"
                  f"{rel:>11.2f}{df['separation'].iloc[-1]:>9.4f}")
    path = out_dir.parent.parent / "separation_peaks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="bank")
    a = ap.parse_args()

    out_dir = analysis_dir(a.word, "plots/separation")
    print(f"word: {a.word}\noutput: {out_dir}\n")

    for fam, models in FAMILIES.items():
        plot_family(fam, models, a.word, out_dir)
    plot_combined(a.word, out_dir)
    peak_table(a.word, out_dir)


if __name__ == "__main__":
    main()
