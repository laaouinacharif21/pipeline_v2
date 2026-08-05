"""
Phase 0: compare mean-pooled vs target-token extraction.

For every model, projection and geometric measure, computes both the raw
Pearson correlation with Sep(l) and the partial correlation controlling
for layer depth, under both extraction methods.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECTIONS = ["q_proj", "v_proj", "up_proj"]

MODELS = (
    [("llama", m) for m in ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"]]
    + [("qwen", m) for m in ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]]
    + [("bert", m) for m in ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"]]
)

TREES = {"mean_pooled": "results", "target_token": "results_targettoken"}


def star(p):
    if np.isnan(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def partial_corr(x, y, z):
    """Pearson correlation of x and y after removing linear dependence on z."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.pearsonr(rx, ry)


def find_col(df, proj, measure):
    candidates = {
        "spectral_norm": [f"{proj}_spectral_norm", f"{proj}_s_norm", f"{proj}_specnorm"],
        "erank": [f"{proj}_erank", f"{proj}_effective_rank"],
    }[measure]
    for c in candidates:
        if c in df.columns:
            return c
    return None


rows = []
for fam, model in MODELS:
    for tree_name, tree in TREES.items():
        sep_path = Path(tree) / fam / model / "metrics" / "semantic_separation.csv"
        if not sep_path.exists():
            continue
        sep_df = pd.read_csv(sep_path).sort_values("layer").reset_index(drop=True)

        for measure, fname in [("spectral_norm", "parameter_stats.csv"),
                               ("erank", "effective_rank.csv")]:
            par_path = Path("results") / fam / model / "parameters" / fname
            if not par_path.exists():
                continue
            par_df = pd.read_csv(par_path).sort_values("layer").reset_index(drop=True)

            shared = sorted(set(sep_df.layer) & set(par_df.layer))
            s = sep_df[sep_df.layer.isin(shared)].reset_index(drop=True)
            g = par_df[par_df.layer.isin(shared)].reset_index(drop=True)
            depth = np.asarray(shared, dtype=float)
            y = s["separation"].values

            for proj in PROJECTIONS:
                col = find_col(g, proj, measure)
                if col is None or g[col].isna().all():
                    continue
                x = g[col].values
                if np.std(x) == 0 or np.std(y) == 0:
                    continue
                r_raw, p_raw = stats.pearsonr(x, y)
                r_par, p_par = partial_corr(x, y, depth)
                rows.append({
                    "family": fam, "model": model, "extraction": tree_name,
                    "measure": measure, "projection": proj, "n_layers": len(shared),
                    "raw_r": round(r_raw, 3), "raw_p": p_raw, "raw_sig": star(p_raw),
                    "partial_r": round(r_par, 3), "partial_p": p_par, "partial_sig": star(p_par),
                })

df = pd.DataFrame(rows)
out_dir = Path("results_targettoken")
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "phase0_full_comparison.csv", index=False)

for measure in ["erank", "spectral_norm"]:
    for proj in PROJECTIONS:
        sub = df[(df.measure == measure) & (df.projection == proj)]
        if sub.empty:
            continue
        piv = sub.pivot_table(index=["family", "model"], columns="extraction",
                              values=["raw_r", "partial_r"], aggfunc="first")
        print(f"\n{'='*70}\n{measure.upper()}  --  {proj}\n{'='*70}")
        print(piv.to_string())

print(f"\nFull table -> {out_dir / 'phase0_full_comparison.csv'}")
