# -*- coding: utf-8 -*-
"""
publication_plots.py
--------------------
Generates all publication-quality figures for the journal paper.

Figures produced:
    Fig 1 -- Partial correlation heatmap (all models x projections)
    Fig 2 -- Effective rank vs semantic separation scatter (per family)
    Fig 3 -- Effective rank across layers line plot (per family x projection)
    Fig 4 -- Cross-family up_proj erank correlation bar chart
    Fig 5 -- TruthfulQA bleu_acc vs mean up_proj erank scatter

All figures saved to: results/plots/publication/

Usage:
    python -m src.analysis.publication_plots
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

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.1,
})

FAMILY_COLORS = {
    "llama": "#1f77b4",
    "qwen":  "#d62728",
    "bert":  "#2ca02c",
}
FAMILY_MARKERS = {
    "llama": "o",
    "qwen":  "s",
    "bert":  "^",
}
PROJ_COLORS = {
    "q_proj":  "#378ADD",
    "v_proj":  "#1D9E75",
    "up_proj": "#D4537E",
}

LLAMA_MODELS = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"]
QWEN_MODELS  = ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]
BERT_MODELS  = ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"]
ALL_MODELS   = LLAMA_MODELS + QWEN_MODELS + BERT_MODELS
PROJECTIONS  = ["q_proj", "v_proj", "up_proj"]

TRUTHFULQA = {
    "llama-7b": 0.36, "llama-2-7b": 0.37, "llama-3-8b": 0.36, "llama-3.1-8b": 0.38,
    "qwen-7b": 0.39, "qwen1.5-7b": 0.40, "qwen2-7b": 0.39, "qwen2.5-7b": 0.41, "qwen3-8b": 0.42,
}

# ── Data loaders ───────────────────────────────────────────────────────────

def load_erank(model_name):
    family = infer_family(model_name)
    p = RESULTS_ROOT / family / model_name / "parameters" / "effective_rank.csv"
    return pd.read_csv(p) if p.exists() else None

def load_separation(model_name):
    family = infer_family(model_name)
    p = RESULTS_ROOT / family / model_name / "metrics" / "semantic_separation.csv"
    return pd.read_csv(p) if p.exists() else None

def load_partial_corr(model_name):
    family = infer_family(model_name)
    p = RESULTS_ROOT / "partial_correlations" / family / f"{model_name}_partial_correlations.csv"
    if p.exists():
        return pd.read_csv(p)
    # fallback: compute from raw data
    return None

# ── Fig 1: Partial correlation heatmap ────────────────────────────────────

def fig1_partial_corr_heatmap():
    """Heatmap of partial correlations (controlling depth) for all models."""
    # Build matrix from erank-separation correlations as proxy
    # (Use direct Pearson r from effective rank analysis summary)
    summary_path = RESULTS_ROOT / "effective_rank" / "effective_rank_summary.csv"
    if not summary_path.exists():
        print("  [SKIP] Fig 1 -- effective_rank_summary.csv not found")
        return

    df = pd.read_csv(summary_path)
    models = df["model"].unique().tolist()
    
    # Build r matrix: rows=models, cols=projections
    matrix = np.full((len(models), len(PROJECTIONS)), np.nan)
    sig_matrix = np.full((len(models), len(PROJECTIONS)), "", dtype=object)

    for i, model in enumerate(models):
        for j, proj in enumerate(PROJECTIONS):
            row = df[(df["model"] == model) & (df["proj"] == proj)]
            if not row.empty:
                r = float(row["erank_sep_r"].values[0])
                p = float(row["erank_sep_p"].values[0])
                matrix[i, j] = r
                sig_matrix[i, j] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    fig, ax = plt.subplots(figsize=(6, max(5, len(models) * 0.45)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(PROJECTIONS)))
    ax.set_xticklabels(PROJECTIONS, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(PROJECTIONS)):
            if not np.isnan(matrix[i, j]):
                val = f"{matrix[i,j]:+.2f}"
                sig = sig_matrix[i, j]
                txt = f"{val}\n{sig}" if sig else val
                color = "white" if abs(matrix[i, j]) > 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold" if sig else "normal")

    # Family separators
    llama_end = len(LLAMA_MODELS) - 0.5
    qwen_end  = len(LLAMA_MODELS) + len(QWEN_MODELS) - 0.5
    ax.axhline(llama_end, color="white", linewidth=2)
    ax.axhline(qwen_end,  color="white", linewidth=2)

    # Family labels on right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([
        len(LLAMA_MODELS) / 2 - 0.5,
        len(LLAMA_MODELS) + len(QWEN_MODELS) / 2 - 0.5,
        len(LLAMA_MODELS) + len(QWEN_MODELS) + len(BERT_MODELS) / 2 - 0.5,
    ])
    ax2.set_yticklabels(["LLaMA", "Qwen", "BERT"], fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.tick_params(length=0)

    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.6, pad=0.12)
    ax.set_title("Effective Rank vs Semantic Separation\nCorrelations across All Models",
                 fontweight="bold", pad=10)
    ax.spines[:].set_visible(False)

    path = PLOT_DIR / "fig1_correlation_heatmap.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved -> {path}")


# ── Fig 2: Erank vs separation scatter per family ─────────────────────────

def fig2_erank_separation_scatter():
    """3-panel scatter: erank vs separation for each family."""
    families = [("LLaMA", LLAMA_MODELS, "llama"),
                ("Qwen",  QWEN_MODELS,  "qwen"),
                ("BERT",  BERT_MODELS,  "bert")]

    proj = "up_proj"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, (fname, models, fkey) in zip(axes, families):
        color = FAMILY_COLORS[fkey]
        markers = ["o", "s", "^", "D", "v"]
        all_x, all_y = [], []

        for mi, model in enumerate(models):
            edf = load_erank(model)
            sdf = load_separation(model)
            if edf is None or sdf is None:
                continue
            col = f"{proj}_erank"
            if col not in edf.columns:
                continue
            merged = edf.merge(sdf, on="layer")
            x = merged[col].values
            y = merged["separation"].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[mask], y[mask]
            ax.scatter(x, y, alpha=0.5, s=25, color=color,
                       marker=markers[mi % len(markers)], label=model, zorder=3)
            all_x.extend(x)
            all_y.extend(y)

        if len(all_x) > 2:
            m, b = np.polyfit(all_x, all_y, 1)
            xline = np.linspace(min(all_x), max(all_x), 100)
            ax.plot(xline, m * xline + b, "k--", linewidth=1.5, alpha=0.6)
            r, p = pearsonr(all_x, all_y)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.set_title(f"{fname}\nr = {r:+.3f} {sig}", fontweight="bold")
        else:
            ax.set_title(fname, fontweight="bold")

        ax.set_xlabel(f"Effective Rank ({proj})", fontsize=10)
        ax.set_ylabel("Semantic Separation", fontsize=10)
        ax.legend(fontsize=7, ncol=1, framealpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Effective Rank of up_proj vs Semantic Separation (per Layer)",
                 fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    path = PLOT_DIR / "fig2_erank_separation_scatter.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved -> {path}")


# ── Fig 3: Erank across layers ─────────────────────────────────────────────

def fig3_erank_layers():
    """Line plots of effective rank across layers for each family."""
    families = [("LLaMA", LLAMA_MODELS, "llama"),
                ("Qwen",  QWEN_MODELS,  "qwen"),
                ("BERT",  BERT_MODELS,  "bert")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    model_colors = plt.cm.tab10.colors

    for ax, (fname, models, fkey) in zip(axes, families):
        for mi, model in enumerate(models):
            edf = load_erank(model)
            if edf is None:
                continue
            col = "up_proj_erank"
            if col not in edf.columns:
                continue
            ax.plot(edf["layer"], edf[col],
                    label=model, color=model_colors[mi],
                    linewidth=2, marker="o", markersize=3)

        ax.set_title(f"{fname} Family\nup_proj Effective Rank per Layer",
                     fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Effective Rank", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / "fig3_erank_layers.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved -> {path}")


# ── Fig 4: Cross-family correlation bar chart ─────────────────────────────

def fig4_crossfamily_bars():
    """Grouped bar chart of up_proj correlations across all decoder models."""
    summary_path = RESULTS_ROOT / "effective_rank" / "effective_rank_summary.csv"
    if not summary_path.exists():
        print("  [SKIP] Fig 4 -- summary not found")
        return

    df = pd.read_csv(summary_path)
    up = df[df["proj"] == "up_proj"].copy()

    # Separate families
    up["family"] = up["model"].apply(infer_family)
    decoder = up[up["family"].isin(["llama", "qwen"])].copy()
    encoder = up[up["family"] == "bert"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [9, 4]})

    for ax, data, title in zip(axes,
                                [decoder, encoder],
                                ["Decoder Models (LLaMA + Qwen)",
                                 "Encoder Models (BERT)"]):
        colors = [FAMILY_COLORS[f] for f in data["family"]]
        bars = ax.bar(range(len(data)), data["erank_sep_r"],
                      color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Significance stars
        for i, (_, row) in enumerate(data.iterrows()):
            p = row["erank_sep_p"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                y = row["erank_sep_r"]
                offset = 0.02 if y >= 0 else -0.06
                ax.text(i, y + offset, sig, ha="center", fontsize=9, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data["model"].tolist(), rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Pearson r (up_proj erank vs separation)", fontsize=10)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-1.05, 0.3)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=FAMILY_COLORS["llama"], lw=6, label="LLaMA"),
        Line2D([0], [0], color=FAMILY_COLORS["qwen"],  lw=6, label="Qwen"),
        Line2D([0], [0], color=FAMILY_COLORS["bert"],  lw=6, label="BERT"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=9, loc="lower left")

    plt.suptitle("up_proj Effective Rank vs Semantic Separation — All Models",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    path = PLOT_DIR / "fig4_crossfamily_bars.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved -> {path}")


# ── Fig 5: TruthfulQA vs erank ────────────────────────────────────────────

def fig5_truthfulqa_erank():
    """Scatter: mean up_proj erank vs TruthfulQA bleu_acc per model."""
    rows = []
    for model, score in TRUTHFULQA.items():
        edf = load_erank(model)
        if edf is None:
            continue
        col = "up_proj_erank"
        if col not in edf.columns:
            continue
        mean_er = float(edf[col].mean(skipna=True))
        rows.append({"model": model, "family": infer_family(model),
                     "bleu_acc": score, "mean_up_erank": mean_er})

    if not rows:
        print("  [SKIP] Fig 5 -- no data")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for family, grp in df.groupby("family"):
        ax.scatter(grp["mean_up_erank"], grp["bleu_acc"],
                   color=FAMILY_COLORS[family],
                   marker=FAMILY_MARKERS[family],
                   s=90, zorder=3, label=family.upper())
        for _, row in grp.iterrows():
            ax.annotate(row["model"],
                        (row["mean_up_erank"], row["bleu_acc"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=FAMILY_COLORS[family])

    # Regression
    x, y = df["mean_up_erank"].values, df["bleu_acc"].values
    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m * xline + b, "k--", linewidth=1.2, alpha=0.5)

    r, p = pearsonr(x, y)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.set_title(f"Mean up_proj Effective Rank vs TruthfulQA Accuracy\n"
                 f"Pearson r = {r:+.3f}  {sig}  (n={len(df)})",
                 fontweight="bold")
    ax.set_xlabel("Mean Effective Rank (up_proj)", fontsize=11)
    ax.set_ylabel("TruthfulQA bleu_acc", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    path = PLOT_DIR / "fig5_truthfulqa_erank.pdf"
    fig.savefig(path)
    fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"\nGenerating publication figures -> {PLOT_DIR}\n")
    print("Fig 1: Correlation heatmap ...")
    fig1_partial_corr_heatmap()
    print("Fig 2: Erank vs separation scatter ...")
    fig2_erank_separation_scatter()
    print("Fig 3: Erank across layers ...")
    fig3_erank_layers()
    print("Fig 4: Cross-family bar chart ...")
    fig4_crossfamily_bars()
    print("Fig 5: TruthfulQA vs erank ...")
    fig5_truthfulqa_erank()
    print(f"\nAll figures saved to {PLOT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
