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


def residualise(a, b):
    return a - np.polyval(np.polyfit(b, a, 1), b)


def cell(model, word, measure, proj):
    which = "erank" if measure == "erank" else "params"
    m = merge_geometry_sep(model, word, which=which)
    if m is None:
        return None
    col = COLUMN[measure].format(proj=proj)
    if col not in m.columns or m[col].isna().all() or m[col].std() == 0:
        return None
    depth = m["layer"].values.astype(float)
    rg = residualise(m[col].values, depth)
    rs = residualise(m["separation"].values, depth)
    r, p = stats.pearsonr(rg, rs)
    return {"r": r, "p": p, "n": len(m)}


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
                             "n_layers": c["n"], "sig": star(c["p"])})
            print(line)

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo data.")
        return

    print(f"\n{'=' * 78}")
    print("CONSISTENCY ACROSS WORDS")
    print(f"{'=' * 78}")
    print(f"  {'word':<14}{'decoders sig':>14}{'encoders sig':>14}{'mean r (dec)':>14}{'sign':>8}")
    print("  " + "-" * 64)
    for w in words:
        d = df[(df.word == w) & (df.model.isin(DECODERS))]
        e = df[(df.word == w) & (~df.model.isin(DECODERS))]
        if d.empty:
            continue
        nd = (d.partial_p < .05).sum()
        ne = (e.partial_p < .05).sum()
        mr = d.partial_r.mean()
        print(f"  {w:<14}{f'{nd}/{len(d)}':>14}{f'{ne}/{len(e)}':>14}"
              f"{mr:>+14.3f}{'neg' if mr < 0 else 'pos':>8}")

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
