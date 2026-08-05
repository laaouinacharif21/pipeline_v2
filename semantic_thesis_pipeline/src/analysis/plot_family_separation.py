"""
Layer-wise semantic separation curves by model family.

Produces one figure per family plus a combined three-panel comparison.
Layer depth is shown both as absolute index (per family) and as relative
depth (combined panel), since families differ in layer count.

Usage:
    python -m src.analysis.plot_family_separation
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import get_results_root, infer_family

RESULTS_ROOT = get_results_root()
OUT_DIR = RESULTS_ROOT / "plots" / "separation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "llama": ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"],
    "qwen": ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"],
    "bert": ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"],
}

TITLES = {"llama": "LLaMA family", "qwen": "Qwen family",
          "bert": "BERT-family encoders"}

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
})


def load(model):
    fam = infer_family(model)
    p = RESULTS_ROOT / fam / model / "metrics" / "semantic_separation.csv"
    if not p.exists():
        print(f"  [SKIP] {model}")
        return None
    return pd.read_csv(p).sort_values("layer").reset_index(drop=True)


def plot_family(fam, models):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for i, m in enumerate(models):
        df = load(m)
        if df is None:
            continue
        ax.plot(df["layer"], df["separation"], marker="o", markersize=5,
                linewidth=2.2, label=m, color=COLORS[i % len(COLORS)])
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    ax.set_xlabel("Layer")
    ax.set_ylabel("Semantic separation  Sep(l)")
    ax.set_title(TITLES[fam])
    ax.axhline(0, color="grey", linewidth=1, linestyle="--", alpha=0.6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / f"{fam}_family_separation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_combined():
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    for ax, (fam, models) in zip(axes, FAMILIES.items()):
        for i, m in enumerate(models):
            df = load(m)
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
        ax.legend(fontsize=12)
    axes[0].set_ylabel("Semantic separation  Sep(l)")
    plt.tight_layout()
    out = OUT_DIR / "all_families_separation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def peaks():
    print(f"\n{'Model':<22}{'peak Sep':>10}{'at layer':>10}{'rel depth':>11}{'final':>9}")
    print("-" * 62)
    for fam, models in FAMILIES.items():
        for m in models:
            df = load(m)
            if df is None:
                continue
            i = df["separation"].idxmax()
            print(f"{m:<22}{df['separation'][i]:>10.4f}"
                  f"{int(df['layer'][i]):>10}"
                  f"{df['layer'][i]/df['layer'].max():>11.2f}"
                  f"{df['separation'].iloc[-1]:>9.4f}")


if __name__ == "__main__":
    print(f"results root: {RESULTS_ROOT}\n")
    for fam, models in FAMILIES.items():
        plot_family(fam, models)
    plot_combined()
    peaks()
