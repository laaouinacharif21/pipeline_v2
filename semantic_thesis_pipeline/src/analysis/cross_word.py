# -*- coding: utf-8 -*-
"""
Cross-word generalisation analysis.

Reports the depth-controlled partial correlation between projection-matrix
geometry and semantic separation for every (model, word) pair, so that an
association observed for one target word can be checked against the others.

Words are discovered from the results tree; geometry is word-independent and
read once per model.

All p-values are exploratory: layers within a model are not independent
observations and no multiple-comparison correction is applied.

Usage:
    python -m src.analysis.cross_word --measure spectral_norm --projection q_proj
    python -m src.analysis.cross_word --measure erank --projection up_proj --words bank,crane
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import (
    FAMILIES, DECODERS, merge_geometry_sep, star,
)
from src.utils.paths import get_results_root, available_words, ensure_dir

COLUMN = {
    "spectral_norm": "{proj}_spectral_norm",
    "fro_norm": "{proj}_fro_norm",
    "erank": "{proj}_erank",
    "stable_rank": "{proj}_stable_rank",
}


def residualise(a, b, degree=1):
    """Remove the component of a explained by a polynomial in b.

    Depth is controlled linearly by default. Both geometry and separation vary
    smoothly with depth, so a linear control can leave shared curvature in the
    residuals; the quadratic control is reported alongside as a stress test.
    Higher degrees are not used: with 28 to 36 layers per model, a cubic fit
    absorbs most of the variance in both variables.
    """
    return a - np.polyval(np.polyfit(b, a, degree), b)


def cell(model, word, measure, proj):
    which = "erank" if measure == "erank" else "params"
    m = merge_geometry_sep(model, word, which=which)
    if m is None:
        return None
    col = COLUMN[measure].format(proj=proj)
    if col not in m.columns or m[col].isna().all() or m[col].std() == 0:
        return None
    depth = m["layer"].values.astype(float)
    g, s = m[col].values, m["separation"].values

    rg, rs = residualise(g, depth), residualise(s, depth)
    r, p = stats.pearsonr(rg, rs)
    sr, sp = stats.spearmanr(rg, rs)

    qg, qs = residualise(g, depth, 2), residualise(s, depth, 2)
    rq, pq = stats.pearsonr(qg, qs)

    return {"r": r, "p": p, "n": len(m),
            "spearman_r": sr, "spearman_p": sp,
            "r_quad": rq, "p_quad": pq}


def run(measure, proj, words):
    print(f"\n{'=' * 78}")
    print(f"CROSS-WORD  --  {proj} {measure} vs Sep(l), depth-controlled")
    print(f"words: {', '.join(words)}")
    print(f"{'=' * 78}\n")

    rows = []
    header = f"  {'model':<22}" + "".join(f"{w:>14}" for w in words)
    print(header)
    print("  " + "-" * (22 + 14 * len(words)))

    for fam, models in FAMILIES.items():
        for model in models:
            line = f"  {model:<22}"
            for w in words:
                c = cell(model, w, measure, proj)
                if c is None:
                    line += f"{'--':>14}"
                    continue
                line += f"{c['r']:>+9.3f}{star(c['p']):<5}"
                rows.append({"family": fam, "model": model, "word": w,
                             "projection": proj, "measure": measure,
                             "partial_r": c["r"], "partial_p": c["p"],
                             "partial_spearman_r": c["spearman_r"],
                             "partial_spearman_p": c["spearman_p"],
                             "partial_r_quad": c["r_quad"],
                             "partial_p_quad": c["p_quad"],
                             "n_layers": c["n"], "sig": star(c["p"])})
            print(line)

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo data.")
        return

    print(f"\n{'=' * 78}")
    print("CONSISTENCY ACROSS WORDS")
    print(f"{'=' * 78}")
    print(f"  {'word':<12}{'dec sig':>9}{'enc sig':>9}{'mean r':>9}"
          f"{'quad r':>9}{'quad sig':>10}")
    print("  " + "-" * 60)
    for w in words:
        d = df[(df.word == w) & (df.model.isin(DECODERS))]
        e = df[(df.word == w) & (~df.model.isin(DECODERS))]
        if d.empty:
            continue
        print(f"  {w:<12}{f'{(d.partial_p < .05).sum()}/{len(d)}':>9}"
              f"{f'{(e.partial_p < .05).sum()}/{len(e)}':>9}"
              f"{d.partial_r.mean():>+9.3f}{d.partial_r_quad.mean():>+9.3f}"
              f"{f'{(d.partial_p_quad < .05).sum()}/{len(d)}':>10}")

    print("\n  Quadratic depth control, decoders by family")
    print(f"    {'family':<8}{'linear':>9}{'quadratic':>11}{'retained':>10}")
    print("    " + "-" * 38)
    for fam in ["llama", "qwen"]:
        d = df[(df.family == fam) & (df.model.isin(DECODERS))]
        if d.empty:
            continue
        lin, quad = d.partial_r.mean(), d.partial_r_quad.mean()
        print(f"    {fam:<8}{lin:>+9.3f}{quad:>+11.3f}"
              f"{(abs(quad) / abs(lin) if lin else 0):>9.0%}")

    if len(words) > 1:
        print(f"\n  Per-model consistency (decoders, significant in how many words):")
        for model in DECODERS:
            d = df[df.model == model]
            if d.empty:
                continue
            k = (d.partial_p < .05).sum()
            signs = set(np.sign(d[d.partial_p < .05].partial_r))
            note = "" if len(signs) <= 1 else "  [sign varies]"
            print(f"    {model:<22}{k}/{len(d)}{note}")

    out = ensure_dir(get_results_root() / "analysis" / "_cross_word")
    path = out / f"cross_word_{proj}_{measure}.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="spectral_norm", choices=list(COLUMN))
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--words", default="", help="comma-separated; default is all found")
    a = ap.parse_args()
    words = [w.strip() for w in a.words.split(",") if w.strip()] or available_words()
    if not words:
        raise SystemExit("No words found under results/words/")
    run(a.measure, a.projection, words)
