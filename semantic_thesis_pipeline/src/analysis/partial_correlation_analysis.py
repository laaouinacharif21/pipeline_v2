# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

RESULTS_ROOT = Path("/home/tianyu/semantic_thesis_pipeline/results")
OUT_DIR      = Path("/home/tianyu/semantic_thesis_pipeline/results/partial_correlations")
PLOT_DIR     = Path("/home/tianyu/semantic_thesis_pipeline/results/plots/partial_correlation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = {
    "llama": ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"],
    "qwen":  ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"],
    "bert":  ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"],
}

PROJECTIONS = ["q_proj", "v_proj", "up_proj"]


# -- Core functions -------------------------------------------------------------

def partial_corr_controlling_depth(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute partial correlation r(x, y | layer_index).
    Returns (r, p_value).
    """
    n = len(x)
    layer = np.arange(n, dtype=float)

    def residuals(var):
        slope, intercept, _, _, _ = stats.linregress(layer, var)
        return var - (slope * layer + intercept)

    ex = residuals(x)
    ey = residuals(y)

    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


def raw_corr(x: np.ndarray, y: np.ndarray) -> tuple:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def load_model_data(family: str, model: str) -> dict | None:
    """Load spectral norms and semantic separation for one model."""
    base = RESULTS_ROOT / family / model

    sep_path = base / "metrics" / "semantic_separation.csv"
    param_path = base / "parameters" / "parameter_stats.csv"

    if not sep_path.exists():
        print(f"  [SKIP] {family}/{model} -- semantic_separation.csv not found")
        return None
    if not param_path.exists():
        print(f"  [SKIP] {family}/{model} -- parameter_stats.csv not found")
        return None

    sep_df = pd.read_csv(sep_path)
    param_df = pd.read_csv(param_path)

    # Align on layer index
    sep_df = sep_df.sort_values("layer").reset_index(drop=True)
    param_df = param_df.sort_values("layer").reset_index(drop=True)

    # Keep only shared layers
    shared = set(sep_df["layer"]).intersection(set(param_df["layer"]))
    sep_df = sep_df[sep_df["layer"].isin(shared)].reset_index(drop=True)
    param_df = param_df[param_df["layer"].isin(shared)].reset_index(drop=True)

    return {"sep": sep_df, "param": param_df}


def analyse_model(family: str, model: str) -> dict | None:
    data = load_model_data(family, model)
    if data is None:
        return None

    sep = data["sep"]["separation"].values
    param = data["param"]

    results = {"family": family, "model": model}

    for proj in PROJECTIONS:
        col = f"{proj}_spectral_norm"
        if col not in param.columns:
            # try alternative column naming
            alt = f"{proj}_s_norm"
            if alt in param.columns:
                col = alt
            else:
                print(f"  [WARN] {model}: column '{col}' not found -- skipping {proj}")
                results[f"{proj}_raw_r"] = np.nan
                results[f"{proj}_raw_p"] = np.nan
                results[f"{proj}_partial_r"] = np.nan
                results[f"{proj}_partial_p"] = np.nan
                continue

        norm = param[col].values

        r_raw, p_raw = raw_corr(norm, sep)
        r_par, p_par = partial_corr_controlling_depth(norm, sep)

        results[f"{proj}_raw_r"]     = r_raw
        results[f"{proj}_raw_p"]     = p_raw
        results[f"{proj}_partial_r"] = r_par
        results[f"{proj}_partial_p"] = p_par

    return results


# -- Run analysis ---------------------------------------------------------------

all_results = []

for family, models in FAMILIES.items():
    print(f"\n{'='*60}")
    print(f"Family: {family.upper()}")
    print(f"{'='*60}")
    for model in models:
        print(f"\n  Model: {model}")
        res = analyse_model(family, model)
        if res is None:
            continue
        all_results.append(res)

        for proj in PROJECTIONS:
            r_raw = res.get(f"{proj}_raw_r", np.nan)
            r_par = res.get(f"{proj}_partial_r", np.nan)
            p_par = res.get(f"{proj}_partial_p", np.nan)
            sig = "***" if p_par < 0.001 else "**" if p_par < 0.01 else "*" if p_par < 0.05 else "n.s."
            print(f"    {proj:12s}  raw r={r_raw:+.3f}  ->  partial r={r_par:+.3f} {sig}")

# -- Save CSV -------------------------------------------------------------------

df = pd.DataFrame(all_results)
out_csv = OUT_DIR / "partial_corr_summary.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved CSV -> {out_csv}")

# -- Plot: raw vs partial correlation comparison --------------------------------

fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=False)

colors = {"raw": "#4A90D9", "partial": "#E05C5C"}

for ax_i, proj in enumerate(PROJECTIONS):
    ax = axes[ax_i]

    models_list  = [r["model"] for r in all_results]
    raw_vals     = [r.get(f"{proj}_raw_r", np.nan) for r in all_results]
    partial_vals = [r.get(f"{proj}_partial_r", np.nan) for r in all_results]
    p_vals       = [r.get(f"{proj}_partial_p", np.nan) for r in all_results]

    x = np.arange(len(models_list))
    width = 0.35

    bars1 = ax.bar(x - width/2, raw_vals,     width, label="Raw r",     color=colors["raw"],     alpha=0.85)
    bars2 = ax.bar(x + width/2, partial_vals, width, label="Partial r", color=colors["partial"], alpha=0.85)

    # significance markers on partial bars
    for xi, (pv, pval) in enumerate(zip(partial_vals, p_vals)):
        if np.isnan(pv):
            continue
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        if sig:
            ypos = pv + (0.04 if pv >= 0 else -0.08)
            ax.text(xi + width/2, ypos, sig, ha="center", va="bottom", fontsize=9, color="#333333")

    # Family separator lines
    family_sizes = [len(v) for v in FAMILIES.values()]
    boundaries = np.cumsum(family_sizes[:-1]) - 0.5
    for b in boundaries:
        ax.axvline(b, color="#cccccc", linewidth=1.0, linestyle="--")

    # Family labels
    family_names = list(FAMILIES.keys())
    starts = [0] + list(np.cumsum(family_sizes[:-1]))
    for fi, (fname, fsize) in enumerate(zip(family_names, family_sizes)):
        mid = starts[fi] + fsize / 2 - 0.5
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.05,
                fname.upper(), ha="center", va="bottom",
                fontsize=9, color="#555555", style="italic")

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"{proj}", fontweight="bold")
    ax.set_ylabel("Pearson r" if ax_i == 0 else "")
    ax.set_ylim(-1.0, 1.15)
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

fig.suptitle(
    "Raw vs Partial Correlation (controlling for layer depth)\n"
    "Spectral Norm of Projection Matrices vs Semantic Separation",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plot_path = PLOT_DIR / "partial_correlation_comparison.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved plot -> {plot_path}")

# -- Summary table --------------------------------------------------------------

print("\n" + "="*80)
print("SUMMARY: Does the correlation survive controlling for layer depth?")
print("="*80)
print(f"{'Model':<22} {'Proj':<12} {'Raw r':>8} {'Partial r':>10} {'p':>8} {'Significant?':>14}")
print("-"*80)

for res in all_results:
    model = res["model"]
    for proj in PROJECTIONS:
        r_raw = res.get(f"{proj}_raw_r", np.nan)
        r_par = res.get(f"{proj}_partial_r", np.nan)
        p_par = res.get(f"{proj}_partial_p", np.nan)
        sig = "YES ***" if p_par < 0.001 else "YES **" if p_par < 0.01 else "YES *" if p_par < 0.05 else "no"
        if not np.isnan(r_raw):
            print(f"{model:<22} {proj:<12} {r_raw:>+8.3f} {r_par:>+10.3f} {p_par:>8.4f} {sig:>14}")

print("\nDone.")