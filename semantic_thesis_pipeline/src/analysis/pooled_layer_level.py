# -*- coding: utf-8 -*-
"""
Pooled layer-level analysis.

Raw pooling across models mixes between-model scale differences with the
within-model layer relationship of interest, so both variables are z-scored
within each model before pooling. Layer depth co-varies with both, so the
depth-controlled partial correlation is the primary statistic. Spearman is
reported alongside Pearson as a monotonicity check.

All p-values are exploratory: layers within a model are not independent
observations and no multiple-comparison correction is applied.

Usage:
    python -m src.analysis.pooled_layer_level --word bank
    python -m src.analysis.pooled_layer_level --word bank --column q_proj_spectral_norm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import FAMILIES, merge_geometry_sep, analysis_dir, star
from src.utils.paths import infer_family


def z(v):
    v = np.asarray(v, dtype=float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def residualise(a, b):
    return a - np.polyval(np.polyfit(b, a, 1), b)


def report(label, x, y):
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    print(f"  {label:<18} n={len(x):>4}  "
          f"Pearson r={pr:+.3f} p={pp:.5f} {star(pp):<5} "
          f"Spearman r={sr:+.3f} p={sp:.5f} {star(sp)}")
    return {"label": label, "n": len(x), "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp}


def build(word, column):
    which = "erank" if column.endswith("_erank") else "params"
    rows = []
    for fam, models in FAMILIES.items():
        for model in models:
            m = merge_geometry_sep(model, word, which=which)
            if m is None or column not in m.columns:
                continue
            rows.append(pd.DataFrame({
                "model": model, "family": fam, "layer": m["layer"],
                "depth": m["layer"] / m["layer"].max(),
                "geo_z": z(m[column]), "sep_z": z(m["separation"]),
            }))
    if not rows:
        raise SystemExit(f"No data for word='{word}', column='{column}'")
    return pd.concat(rows, ignore_index=True)


def run(word, column):
    df = build(word, column)
    summary = []

    print(f"\n{'='*86}")
    print(f"POOLED LAYER-LEVEL  --  {column} vs Sep(l), within-model standardised")
    print(f"word: {word}   models: {df.model.nunique()}")
    print(f"{'='*86}")

    print("\n[1] Uncontrolled")
    for fam in ["llama", "qwen", "bert"]:
        d = df[df.family == fam]
        if len(d) >= 5:
            summary.append({"group": fam, "control": "none", **report(fam.upper(), d.geo_z, d.sep_z)})
    d = df[df.family != "bert"]
    summary.append({"group": "decoders", "control": "none", **report("ALL DECODERS", d.geo_z, d.sep_z)})

    print("\n[2] Depth-controlled (primary)")
    for fam in ["llama", "qwen", "bert"]:
        d = df[df.family == fam]
        if len(d) >= 5:
            summary.append({"group": fam, "control": "depth", **report(
                fam.upper(), residualise(d.geo_z.values, d.depth.values),
                residualise(d.sep_z.values, d.depth.values))})
    d = df[df.family != "bert"]
    summary.append({"group": "decoders", "control": "depth", **report(
        "ALL DECODERS", residualise(d.geo_z.values, d.depth.values),
        residualise(d.sep_z.values, d.depth.values))})

    print("\n[3] Depth co-variation (decoders)")
    print("  depth~geometry r=%+.3f p=%.5f" % pearsonr(d.depth, d.geo_z))
    print("  depth~sep      r=%+.3f p=%.5f" % pearsonr(d.depth, d.sep_z))

    print("\n[4] Leave-one-model-out (decoders, depth-controlled)")
    for model in sorted(d.model.unique()):
        s = d[d.model != model]
        report(f"w/o {model}",
               residualise(s.geo_z.values, s.depth.values),
               residualise(s.sep_z.values, s.depth.values))

    out = analysis_dir(word)
    pd.DataFrame(summary).to_csv(out / f"pooled_{column}.csv", index=False)
    df.to_csv(out / f"pooled_{column}_data.csv", index=False)
    print(f"\nSaved -> {out / f'pooled_{column}.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="bank")
    ap.add_argument("--column", default="up_proj_erank")
    a = ap.parse_args()
    run(a.word, a.column)
