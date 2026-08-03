# -*- coding: utf-8 -*-
"""
phase2_erank_hallucination.py
-----------------------------
Correlates mean effective rank of up_proj (per model) with TruthfulQA
bleu_acc scores to test whether geometric structure predicts hallucination
resistance.

Usage:
    python -m src.analysis.phase2_erank_hallucination
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

# TruthfulQA bleu_acc scores from opencompass summary
TRUTHFULQA_SCORES = {
    "llama-7b":     0.36,
    "llama-2-7b":   0.37,
    "llama-3-8b":   0.36,
    "llama-3.1-8b": 0.38,
    "qwen-7b":      0.39,
    "qwen1.5-7b":   0.40,
    "qwen2-7b":     0.39,
    "qwen2.5-7b":   0.41,
    "qwen3-8b":     0.42,
}

PROJECTIONS = ["q_proj", "v_proj", "up_proj"]


def load_erank(model_name: str) -> pd.DataFrame | None:
    family = infer_family(model_name)
    path = RESULTS_ROOT / family / model_name / "parameters" / "effective_rank.csv"
    if not path.exists():
        print(f"  [MISSING] {model_name} effective_rank.csv")
        return None
    return pd.read_csv(path)


def build_combined_table() -> pd.DataFrame:
    rows = []
    for model_name, bleu_acc in TRUTHFULQA_SCORES.items():
        df = load_erank(model_name)
        if df is None:
            continue
        family = infer_family(model_name)
        row = {
            "model":    model_name,
            "family":   family,
            "bleu_acc": bleu_acc,
        }
        for proj in PROJECTIONS:
            col = f"{proj}_erank"
            if col in df.columns:
                row[f"mean_{proj}_erank"] = float(df[col].mean(skipna=True))
                row[f"std_{proj}_erank"]  = float(df[col].std(skipna=True))
            else:
                row[f"mean_{proj}_erank"] = float("nan")
                row[f"std_{proj}_erank"]  = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def correlate(df: pd.DataFrame):
    print("\n" + "="*65)
    print("PHASE 2: Effective Rank vs TruthfulQA (bleu_acc) Correlations")
    print("="*65)
    print(f"{'Projection':<14} {'Pearson r':>10} {'p':>8} {'Spearman r':>12} {'p':>8}  {'Sig'}")
    print("-"*65)

    results = {}
    for proj in PROJECTIONS:
        col = f"mean_{proj}_erank"
        valid = df[[col, "bleu_acc"]].dropna()
        if len(valid) < 3:
            print(f"{proj:<14} insufficient data")
            continue
        pr, pp = pearsonr(valid[col], valid["bleu_acc"])
        sr, sp = spearmanr(valid[col], valid["bleu_acc"])
        sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else "n.s."
        print(f"{proj:<14} {pr:>+10.3f} {pp:>8.4f} {sr:>+12.3f} {sp:>8.4f}  {sig}")
        results[proj] = {"pearson_r": pr, "pearson_p": pp,
                         "spearman_r": sr, "spearman_p": sp}
    return results


def plot_scatter(df: pd.DataFrame, corr_results: dict):
    plot_dir = RESULTS_ROOT / "plots" / "phase2"
    plot_dir.mkdir(parents=True, exist_ok=True)

    colors = {"llama": "#1f77b4", "qwen": "#d62728"}
    markers = {"llama": "o", "qwen": "s"}

    for proj in PROJECTIONS:
        col = f"mean_{proj}_erank"
        valid = df[[col, "bleu_acc", "model", "family"]].dropna()
        if valid.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for family, grp in valid.groupby("family"):
            ax.scatter(
                grp[col], grp["bleu_acc"],
                label=family.upper(),
                color=colors.get(family, "#2ca02c"),
                marker=markers.get(family, "^"),
                s=80, zorder=3
            )
            for _, row in grp.iterrows():
                ax.annotate(
                    row["model"],
                    (row[col], row["bleu_acc"]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color=colors.get(family, "#2ca02c")
                )

        # regression line
        x = valid[col].values
        y = valid["bleu_acc"].values
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, m * xline + b, "k--", linewidth=1, alpha=0.5)

        r = corr_results.get(proj, {}).get("pearson_r", float("nan"))
        p = corr_results.get(proj, {}).get("pearson_p", float("nan"))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(
            f"Mean {proj} Effective Rank vs TruthfulQA Accuracy\n"
            f"Pearson r = {r:+.3f}  {sig}",
            fontweight="bold", fontsize=12
        )
        ax.set_xlabel(f"Mean Effective Rank ({proj})", fontsize=11)
        ax.set_ylabel("TruthfulQA bleu_acc", fontsize=11)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        fname = plot_dir / f"phase2_erank_vs_truthfulqa_{proj}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot -> {fname}")


def main():
    print("Building combined erank + TruthfulQA table ...")
    df = build_combined_table()

    print("\nPer-model summary:")
    print(f"{'Model':<18} {'Family':<8} {'bleu_acc':>9} "
          f"{'mean_up_erank':>14} {'mean_q_erank':>13} {'mean_v_erank':>13}")
    print("-"*80)
    for _, row in df.iterrows():
        print(f"{row['model']:<18} {row['family']:<8} {row['bleu_acc']:>9.3f} "
              f"{row['mean_up_proj_erank']:>14.1f} "
              f"{row['mean_q_proj_erank']:>13.1f} "
              f"{row['mean_v_proj_erank']:>13.1f}")

    corr_results = correlate(df)

    print("\nGenerating scatter plots ...")
    plot_scatter(df, corr_results)

    out_csv = RESULTS_ROOT / "phase2_erank_hallucination.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved table -> {out_csv}")
    print("\nDone.")


if __name__ == "__main__":
    main()