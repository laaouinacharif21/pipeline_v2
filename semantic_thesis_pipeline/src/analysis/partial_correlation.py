# -*- coding: utf-8 -*-
"""
Partial correlation between projection-matrix geometry and semantic separation,
controlling for layer depth.

Raw layer-wise correlations are confounded because both geometry and Sep(l)
vary systematically with depth. The depth-controlled partial correlation is
reported as the primary statistic.

All p-values are exploratory: layers within a model are not independent
observations and no multiple-comparison correction is applied.

Usage:
    python -m src.analysis.partial_correlation --word bank
    python -m src.analysis.partial_correlation --word bank --measure erank
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import (
    FAMILIES, PROJECTIONS, merge_geometry_sep, analysis_dir, star,
)

COLUMN = {
    "spectral_norm": "{proj}_spectral_norm",
    "erank": "{proj}_erank",
}


def residualise(a, b):
    """Remove the linear component of b from a."""
    return a - np.polyval(np.polyfit(b, a, 1), b)


def analyse(model, word, measure):
    which = "erank" if measure == "erank" else "params"
    m = merge_geometry_sep(model, word, which=which)
    if m is None:
        return None

    depth = m["layer"].values.astype(float)
    sep = m["separation"].values
    rows = []

    for proj in PROJECTIONS:
        col = COLUMN[measure].format(proj=proj)
        if col not in m.columns or m[col].isna().all() or m[col].std() == 0:
            continue
        g = m[col].values
        r_raw, p_raw = stats.pearsonr(g, sep)
        rg, rs = residualise(g, depth), residualise(sep, depth)
        r_par, p_par = stats.pearsonr(rg, rs)
        s_par, ps_par = stats.spearmanr(rg, rs)
        rows.append({
            "model": model, "projection": proj, "measure": measure,
            "n_layers": len(m),
            "raw_r": r_raw, "raw_p": p_raw, "raw_sig": star(p_raw),
            "partial_r": r_par, "partial_p": p_par, "partial_sig": star(p_par),
            "partial_spearman_r": s_par, "partial_spearman_p": ps_par,
        })
    return rows


def run(word, measure):
    print(f"\n{'='*78}")
    print(f"PARTIAL CORRELATION  --  {measure} vs Sep(l), controlling for layer depth")
    print(f"word: {word}")
    print(f"{'='*78}")

    out = []
    for fam, models in FAMILIES.items():
        print(f"\n{fam.upper()}")
        print(f"  {'model':<22}{'proj':<10}{'raw r':>9}{'partial r':>12}{'p':>10}  sig")
        print("  " + "-" * 66)
        for model in models:
            rows = analyse(model, word, measure)
            if not rows:
                print(f"  {model:<22}[no data]")
                continue
            for r in rows:
                r["family"] = fam
                out.append(r)
                print(f"  {model:<22}{r['projection']:<10}"
                      f"{r['raw_r']:>+9.3f}{r['partial_r']:>+12.3f}"
                      f"{r['partial_p']:>10.4f}  {r['partial_sig']}")

    df = pd.DataFrame(out)
    d = analysis_dir(word)
    path = d / f"partial_correlation_{measure}.csv"
    df.to_csv(path, index=False)

    print(f"\n{'='*78}")
    print("SUMMARY: significant depth-controlled associations")
    print(f"{'='*78}")
    for proj in PROJECTIONS:
        s = df[df.projection == proj]
        if s.empty:
            continue
        dec = s[s.family != "bert"]
        enc = s[s.family == "bert"]
        nd = (dec.partial_p < .05).sum()
        ne = (enc.partial_p < .05).sum()
        sign = "negative" if dec.partial_r.mean() < 0 else "positive"
        print(f"  {proj:<10} decoders {nd}/{len(dec)} ({sign})   encoders {ne}/{len(enc)}")

    print(f"\nSaved -> {path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="bank")
    ap.add_argument("--measure", default="spectral_norm",
                    choices=["spectral_norm", "erank"])
    a = ap.parse_args()
    run(a.word, a.measure)
