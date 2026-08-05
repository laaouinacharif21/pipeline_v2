# -*- coding: utf-8 -*-
"""
phase2_layer_level.py
---------------------
Correlates per-layer effective rank of up_proj with per-layer semantic
separation within each model. This gives 32 data points per model instead
of 1, providing much stronger statistical power than model-level analysis.

Output:
    results/phase2_layer_level.csv
    results/plots/phase2/layer_level_*.png

Usage:
    python -m src.analysis.phase2_layer_level
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import get_results_root, infer_family

RESULTS_ROOT = get_results_root()
PLOT_DIR = RESULTS_ROOT / "plots" / "phase2"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FAMILY_COLORS = {"llama": "#1f77b4", "qwen": "#d62728", "bert": "#2ca02c"}

ALL_MODELS = [
    "llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b",
    "qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b",
    "bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base",
]


def load_data(model_name):
    family = infer_family(model_name)
    erank_path = RESULTS_ROOT / family / model_name / "parameters" / "effective_rank.csv"
    sep_path   = RESULTS_ROOT / family / model_name / "metrics" / "semantic_separation.csv"
    if not erank_path.exists() or not sep_path.exists():
        return None
    edf = pd.read_csv(erank_path)
    sdf = pd.read_csv(sep_path)
    merged = edf.merge(sdf, on="layer")
    if "up_proj_erank" not in merged.columns or "separation" not in merged.columns:
        return None
    return merged.dropna(subset=["up_proj_erank", "separation"])


def run():
    summary_rows = []
    all_data = []

    print(f"\n{'='*65}")
    print("PHASE 2 - Layer-level: up_proj Erank vs Semantic Separation")
    print(f"{'='*65}")
    print(f"{'Model':<22} {'Family':<8} {'n':>4} {'Pearson r':>10} {'p':>8} {'Spearman r':>12} {'Sig'}")
    print("-"*65)

    for model in ALL_MODELS:
        family = infer_family(model)
        df = load_data(model)
        if df is None or len(df) < 5:
            continue

        x = df["up_proj_erank"].values
        y = df["separation"].values

        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."

        print(f"{model:<22} {family:<8} {len(df):>4} {pr:>+10.3f} {pp:>8.4f} {sr:>+12.3f}  {sig}")

        summary_rows.append({
            "model": model, "family": family, "n_layers": len(df),
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
            "sig": sig
        })

        for _, row in df.iterrows():
            all_data.append({
                "model": model, "family": family,
                "layer": row["layer"],
                "up_proj_erank": row["up_proj_erank"],
                "separation": row["separation"]
            })

    summary_df = pd.DataFrame(summary_rows)
    all_df = pd.DataFrame(all_data)

    # Save
    summary_df.to_csv(RESULTS_ROOT / "phase2_layer_level.csv", index=False)
    print(f"\nSaved -> results/phase2_layer_level.csv")

    # Family-level pooled correlations
    print(f"\n{'='*65}")
    print("POOLED within-family layer-level correlations")
    print(f"{'='*65}")
    for family in ["llama", "qwen", "bert"]:
        fdf = all_df[all_df["family"] == family].dropna()
        if len(fdf) < 5:
            continue
        pr, pp = pearsonr(fdf["up_proj_erank"], fdf["separation"])
        sr, sp = spearmanr(fdf["up_proj_erank"], fdf["separation"])
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."
        print(f"  {family.upper():<8} n={len(fdf):>4}  Pearson r={pr:+.3f} p={pp:.6f} {sig}  Spearman r={sr:+.3f}")

    # Plots
    plot_scatter(all_df)
    plot_per_model(summary_df)
    print("\nDone.")


def plot_scatter(all_df):
    """3-panel scatter: per-layer erank vs separation, one panel per family."""
    families = [("LLaMA", "llama"), ("Qwen", "qwen"), ("BERT", "bert")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (fname, fkey) in zip(axes, families):
        fdf = all_df[all_df["family"] == fkey].dropna()
        if fdf.empty:
            ax.set_title(fname)
            continue

        models = fdf["model"].unique()
        cmap = plt.cm.tab10.colors
        for i, model in enumerate(models):
            mdf = fdf[fdf["model"] == model]
            ax.scatter(mdf["up_proj_erank"], mdf["separation"],
                       alpha=0.5, s=20, color=cmap[i % len(cmap)], label=model)

        # Regression line
        x, y = fdf["up_proj_erank"].values, fdf["separation"].values
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, m * xline + b, "k--", linewidth=1.5, alpha=0.7)

        pr, pp = pearsonr(x, y)
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."
        ax.set_title(f"{fname}  r={pr:+.3f} {sig}\n(n={len(fdf)} layer observations)",
                     fontweight="bold", fontsize=11)
        ax.set_xlabel("up_proj Effective Rank", fontsize=10)
        ax.set_ylabel("Semantic Separation", fontsize=10)
        ax.legend(fontsize=7, framealpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Layer-level: up_proj Effective Rank vs Semantic Separation",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = PLOT_DIR / "layer_level_scatter.png"
    fig.savefig(path)
    plt.close()
    print(f"  Saved -> {path}")


def plot_per_model(summary_df):
    """Bar chart of per-model Pearson r, colored by family."""
    decoder = summary_df[summary_df["family"].isin(["llama", "qwen"])].copy()
    encoder = summary_df[summary_df["family"] == "bert"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [9, 4]})

    for ax, data, title in zip(axes,
                                [decoder, encoder],
                                ["Decoder Models", "Encoder Models"]):
        if data.empty:
            continue
        colors = [FAMILY_COLORS[f] for f in data["family"]]
        ax.bar(range(len(data)), data["pearson_r"],
               color=colors, alpha=0.85, edgecolor="white")

        for i, (_, row) in enumerate(data.iterrows()):
            if row["sig"] != "n.s.":
                y = row["pearson_r"]
                ax.text(i, y + (0.02 if y >= 0 else -0.07),
                        row["sig"], ha="center", fontsize=9, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data["model"].tolist(), rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Pearson r (layer-level)", fontsize=10)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-1.05, 0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle("Per-model layer-level: up_proj Erank vs Semantic Separation",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = PLOT_DIR / "layer_level_per_model.png"
    fig.savefig(path)
    plt.close()
    print(f"  Saved -> {path}")


if __name__ == "__main__":
    run()