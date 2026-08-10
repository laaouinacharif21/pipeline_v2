# -*- coding: utf-8 -*-
"""
Summary tables for the cross-word study.

Produces three tables from data already on disk:

    A  dataset properties per word: size, lexical overlap, sentence length,
       context-only and target-only classification accuracy, and the largest
       semantic separation reached
    B  per-model association: mean partial correlation and the number of words
       reaching significance, for one projection and measure
    C  architecture summary: decoder against encoder, with and without a named
       word as a leakage control

Each table is printed and written to CSV. Nothing is recomputed from the
models; the tables read the outputs of cross_word.py, context_baseline.py and
the metrics stage.

Usage
    python -m src.analysis.results_tables
    python -m src.analysis.results_tables --measure spectral_norm --projection up_proj
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import DECODERS, ALL_MODELS, load_sep
from src.utils.paths import get_results_root, get_project_root, ensure_dir

WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]


def _tok(s):
    return re.findall(r"[a-z']+", s.lower())


def dataset_properties(word):
    """Size, class balance, lexical overlap and sentence length.

    The dataset path is taken from the word's config rather than guessed from
    the word name, since not every file is named after its target word.
    """
    from src.config import load_dataset
    try:
        d, _, _ = load_dataset(word)
    except Exception as e:
        print(f"  [warn] {word}: {type(e).__name__}: {e}")
        return {}

    S, L = d["sentences"], d["labels"]
    senses = sorted(set(L))
    A = [s for s, l in zip(S, L) if l == senses[0]]
    B = [s for s, l in zip(S, L) if l == senses[1]]
    VA = set(w for s in A for w in _tok(s))
    VB = set(w for s in B for w in _tok(s))
    la = [len(_tok(s)) for s in A]
    lb = [len(_tok(s)) for s in B]
    return {
        "n": len(S),
        "senses": "/".join(senses),
        "jaccard": len(VA & VB) / len(VA | VB),
        "len_mean": (sum(la) + sum(lb)) / (len(la) + len(lb)),
        "len_diff": abs(sum(la) / len(la) - sum(lb) / len(lb)),
    }


def baseline_accuracies(word):
    """Decoder-mean context-only and target-only accuracy, if the baseline was run."""
    p = (get_results_root() / "words" / word / "_context_baseline"
         / f"context_baseline_{word}.csv")
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[(d.stratum == "all") & (d.model.isin(DECODERS))]
    if d.empty:
        return {}
    piv = d.pivot_table(index="model", columns="representation", values="accuracy")
    return {
        "context_acc": piv.get("context", pd.Series(dtype=float)).mean(),
        "target_acc": piv.get("target", pd.Series(dtype=float)).mean(),
    }


def separation(word):
    """Largest Sep(l) reached, averaged over decoder models, and where it peaks."""
    peaks, depths = [], []
    for m in DECODERS:
        s = load_sep(m, word)
        if s is None:
            continue
        peaks.append(s.separation.max())
        depths.append(s.separation.idxmax() / (len(s) - 1))
    if not peaks:
        return {}
    return {"sep_max": float(np.mean(peaks)),
            "sep_peak_depth": float(np.mean(depths))}


def table_a(words, out_dir):
    rows = []
    for w in words:
        r = {"word": w}
        r.update(dataset_properties(w))
        r.update(baseline_accuracies(w))
        r.update(separation(w))
        rows.append(r)
    df = pd.DataFrame(rows)

    print("\nTable A  Dataset properties")
    print(f"  {'word':<8}{'n':>5}{'senses':>22}{'overlap':>9}{'length':>8}"
          f"{'context':>9}{'target':>8}{'Sep max':>9}{'peak':>7}")
    print("  " + "-" * 85)
    for _, r in df.iterrows():
        print(f"  {r.word:<8}{int(r.get('n', 0)):>5}{r.get('senses', ''):>22}"
              f"{r.get('jaccard', np.nan):>9.3f}{r.get('len_mean', np.nan):>8.1f}"
              f"{r.get('context_acc', np.nan):>9.3f}{r.get('target_acc', np.nan):>8.3f}"
              f"{r.get('sep_max', np.nan):>9.3f}{r.get('sep_peak_depth', np.nan):>7.2f}")
    df.to_csv(out_dir / "table_a_dataset_properties.csv", index=False)
    return df


def table_b(projection, measure, words, out_dir):
    p = (get_results_root() / "analysis" / "_cross_word"
         / f"cross_word_{projection}_{measure}.csv")
    if not p.exists():
        print(f"\n  [skip] Table B: {p.name} not found")
        return None
    d = pd.read_csv(p)
    d = d[d.word.isin(words)]

    rows = []
    for m in ALL_MODELS:
        s = d[d.model == m]
        if s.empty:
            continue
        rows.append({
            "model": m,
            "architecture": "decoder" if m in DECODERS else "encoder",
            "mean_r": s.partial_r.mean(),
            "min_r": s.partial_r.min(),
            "max_r": s.partial_r.max(),
            "n_significant": int((s.partial_p < .05).sum()),
            "n_words": len(s),
            "sign_consistent": bool(len(set(np.sign(s.partial_r))) == 1),
        })
    df = pd.DataFrame(rows)

    print(f"\nTable B  {projection} {measure.replace('_', ' ')} by model")
    print(f"  {'model':<22}{'arch':>9}{'mean r':>9}{'range':>18}"
          f"{'significant':>13}{'sign':>12}")
    print("  " + "-" * 84)
    for _, r in df.iterrows():
        rng = f"{r.min_r:+.2f} to {r.max_r:+.2f}"
        print(f"  {r.model:<22}{r.architecture:>9}{r.mean_r:>+9.3f}{rng:>18}"
              f"{f'{r.n_significant}/{r.n_words}':>13}"
              f"{'consistent' if r.sign_consistent else 'varies':>12}")
    df.to_csv(out_dir / f"table_b_by_model_{projection}_{measure}.csv", index=False)
    return df


def table_c(projection, measure, words, exclude, out_dir):
    p = (get_results_root() / "analysis" / "_cross_word"
         / f"cross_word_{projection}_{measure}.csv")
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.word.isin(words)]

    rows = []
    for label, sub in [("all words", d),
                       (f"excluding {exclude}", d[d.word != exclude])]:
        for arch, mask in [("decoder", sub.model.isin(DECODERS)),
                           ("encoder", ~sub.model.isin(DECODERS))]:
            s = sub[mask]
            if s.empty:
                continue
            rows.append({
                "subset": label, "architecture": arch,
                "mean_r": s.partial_r.mean(),
                "sd_r": s.partial_r.std(),
                "significant_cells": int((s.partial_p < .05).sum()),
                "total_cells": len(s),
                "proportion": (s.partial_p < .05).mean(),
            })
    df = pd.DataFrame(rows)

    print(f"\nTable C  {projection} {measure.replace('_', ' ')} by architecture")
    print(f"  {'subset':<20}{'arch':>9}{'mean r':>9}{'sd':>8}"
          f"{'significant':>15}{'proportion':>12}")
    print("  " + "-" * 74)
    for _, r in df.iterrows():
        cells = f"{r.significant_cells}/{r.total_cells}"
        print(f"  {r.subset:<20}{r.architecture:>9}{r.mean_r:>+9.3f}{r.sd_r:>8.3f}"
              f"{cells:>15}{r.proportion:>12.0%}")
    df.to_csv(out_dir / f"table_c_by_architecture_{projection}_{measure}.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="stable_rank")
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--exclude", default="bank")
    ap.add_argument("--words", default=",".join(WORDS))
    a = ap.parse_args()

    words = [w.strip() for w in a.words.split(",") if w.strip()]
    out_dir = ensure_dir(get_results_root() / "analysis" / "_tables")

    print(f"\n{'=' * 88}")
    print(f"RESULTS TABLES   {a.projection} {a.measure}   words: {len(words)}")
    print(f"{'=' * 88}")

    table_a(words, out_dir)
    table_b(a.projection, a.measure, words, out_dir)
    table_c(a.projection, a.measure, words, a.exclude, out_dir)

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
