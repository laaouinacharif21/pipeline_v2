# -*- coding: utf-8 -*-
"""
pub_figs_134.py
---------------
Generates publication-quality versions of:
  Fig 1 -- Semantic separation across layers (3 families, 3 panels)
  Fig 3 -- Effective rank of up_proj across layers (3 families, 3 panels)
  Fig 4 -- Erank vs separation scatter with regression + r annotation (3 families)

Output: results/plots/publication/
Usage:  python -m src.analysis.pub_figs_134
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import get_results_root, infer_family

RESULTS_ROOT = get_results_root()
PLOT_DIR = RESULTS_ROOT / "plots" / "publication"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -- Style ------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.15,
})

FAMILIES = {
    "LLaMA": {
        "models": ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"],
        "key": "llama",
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    },
    "Qwen": {
        "models": ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"],
        "key": "qwen",
        "colors": ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"],
    },
    "BERT": {
        "models": ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"],
        "key": "bert",
        "colors": ["#17becf", "#aec7e8", "#ffbb78", "#98df8a"],
    },
}


def load_separation(model):
    family = infer_family(model)
    p = RESULTS_ROOT / family / model / "metrics" / "semantic_separation.csv"
    return pd.read_csv(p) if p.exists() else None


def load_erank(model):
    family = infer_family(model)
    p = RESULTS_ROOT / family / model / "parameters" / "effective_rank.csv"
    return pd.read_csv(p) if p.exists() else None


# -- Fig 1: Semantic separation across layers -------------------------------

def fig1_semantic_separation():
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)

    for ax, (fname, finfo) in zip(axes, FAMILIES.items()):
        for model, color in zip(finfo["models"], finfo["colors"]):
            df = load_separation(model)
            if df is None:
                continue
            ax.plot(df["layer"], df["separation"],
                    label=model, color=color, linewidth=2, alpha=0.9)

        ax.set_title(f"{fname} Family", fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Semantic Separation", fontsize=11)
        ax.legend(fontsize=8.5, framealpha=0.6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle("Semantic Separation Across Transformer Layers",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = PLOT_DIR / f"fig1_semantic_separation.{ext}"
        fig.savefig(path)
        print(f"  Saved -> {path}")
    plt.close()


# -- Fig 3: Effective rank (up_proj) across layers -------------------------

def fig3_erank_layers():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, (fname, finfo) in zip(axes, FAMILIES.items()):
        for model, color in zip(finfo["models"], finfo["colors"]):
            df = load_erank(model)
            if df is None or "up_proj_erank" not in df.columns:
                continue
            ax.plot(df["layer"], df["up_proj_erank"],
                    label=model, color=color, linewidth=2,
                    marker="o", markersize=3, alpha=0.9)

        ax.set_title(f"{fname} Family", fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Effective Rank (up_proj)", fontsize=11)
        ax.legend(fontsize=8.5, framealpha=0.6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle("Effective Rank of up_proj Across Transformer Layers",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = PLOT_DIR / f"fig3_erank_layers.{ext}"
        fig.savefig(path)
        print(f"  Saved -> {path}")
    plt.close()


# -- Fig 4: Erank vs separation scatter (per family, with regression) -------

def fig4_erank_vs_separation():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (fname, finfo) in zip(axes, FAMILIES.items()):
        all_x, all_y = [], []

        for model, color in zip(finfo["models"], finfo["colors"]):
            edf = load_erank(model)
            sdf = load_separation(model)
            if edf is None or sdf is None:
                continue
            if "up_proj_erank" not in edf.columns:
                continue
            merged = edf.merge(sdf, on="layer").dropna(
                subset=["up_proj_erank", "separation"])
            x = merged["up_proj_erank"].values
            y = merged["separation"].values
            ax.scatter(x, y, color=color, alpha=0.55, s=22,
                       label=model, zorder=3)
            all_x.extend(x)
            all_y.extend(y)

        if len(all_x) > 3:
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            m, b = np.polyfit(all_x, all_y, 1)
            xline = np.linspace(all_x.min(), all_x.max(), 200)
            ax.plot(xline, m * xline + b, "k--", linewidth=1.8,
                    alpha=0.7, zorder=4)
            r, p = pearsonr(all_x, all_y)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else \
                  "*" if p < 0.05 else "n.s."
            ax.set_title(f"{fname} Family\nr = {r:+.3f}  {sig}  "
                         f"(n = {len(all_x)} layers)",
                         fontweight="bold", fontsize=12)
        else:
            ax.set_title(f"{fname} Family", fontweight="bold")

        ax.set_xlabel("Effective Rank (up_proj)", fontsize=11)
        ax.set_ylabel("Semantic Separation", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle("up_proj Effective Rank vs Semantic Separation (per Layer)",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = PLOT_DIR / f"fig4_erank_vs_separation.{ext}"
        fig.savefig(path)
        print(f"  Saved -> {path}")
    plt.close()


# -- Main -------------------------------------------------------------------

def main():
    print(f"\nGenerating publication figures -> {PLOT_DIR}\n")
    print("Fig 1: Semantic separation across layers ...")
    fig1_semantic_separation()
    print("Fig 3: Effective rank across layers ...")
    fig3_erank_layers()
    print("Fig 4: Erank vs separation scatter ...")
    fig4_erank_vs_separation()
    print("\nAll done.")


if __name__ == "__main__":
    main()