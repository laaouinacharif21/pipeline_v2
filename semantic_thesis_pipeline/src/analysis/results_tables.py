# -*- coding: utf-8 -*-
"""
Summary tables for the cross-word study (revised September 2026).

Produces, from data already on disk:

    A   dataset properties per word: size, lexical overlap, sentence length,
        context-only and target-only classification accuracy, and the largest
        semantic separation reached                              (unchanged)
    B   per model: mean partial r over words, range, words positive, and the
        circular-shift permutation p, under linear and quadratic depth control
    C   model-level tests by group (decoders, llama, qwen, encoders) and for
        decoders with a named word excluded: mean of per-model r, sign count,
        leave-one-model-out range, t-test and Wilcoxon p (exploratory)
    S1  supplementary: share of decoder and encoder cells with p < 0.05 under
        the naive per-cell test and under the effective-n correction, with the
        mean effective n and residual lag-1 autocorrelation
    S2  supplementary: pre-block against post-block alignment, model-level

Nothing is recomputed from the models. B, C and S1 read the outputs of
cross_word.py for the chosen alignment; S2 needs both alignments on disk.
No per-cell significance is reported in the main tables.

Usage
    python -m src.analysis.results_tables
    python -m src.analysis.results_tables --measure erank --projection up_proj
    python -m src.analysis.results_tables --alignment pre
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import ALIGNMENTS, ALL_MODELS, DECODERS, load_sep  # noqa: E402
from src.utils.paths import ensure_dir, get_results_root  # noqa: E402

WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]
GROUP_ORDER = ["decoders", "llama", "qwen", "encoders"]


def _tok(s):
    return re.findall(r"[a-z']+", s.lower())


# ---------------------------------------------------------------- Table A

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


# ------------------------------------------------------- cross_word inputs

def cw_stem(projection, measure, alignment):
    sfx = "" if alignment == "post" else f"_{alignment}"
    return get_results_root() / "analysis" / "_cross_word" / f"cross_word_{projection}_{measure}{sfx}"


def read_cw(projection, measure, alignment, kind):
    """kind: '' (cells), '_models' or '_summary'. Returns None if missing."""
    p = Path(f"{cw_stem(projection, measure, alignment)}{kind}.csv")
    if not p.exists():
        print(f"\n  [skip] {p.name} not found -- run cross_word.py "
              f"--measure {measure} --projection {projection} --alignment {alignment}")
        return None
    d = pd.read_csv(p)
    if kind == "" and "partial_r_quad" not in d.columns:
        raise SystemExit(f"{p.name} is in the pre-revision format. Rerun cross_word.py.")
    return d


def fmt_p(p):
    return "   n/a" if pd.isna(p) else f"{p:6.4f}"


# ---------------------------------------------------------------- Table B

def table_b(projection, measure, alignment, words, out_dir, sfx):
    M = read_cw(projection, measure, alignment, "_models")
    if M is None:
        return None
    if set(M.n_words) != {len(words)}:
        print(f"  [warn] Table B: models file has n_words {sorted(set(M.n_words))}, "
              f"expected {len(words)}. Rerun cross_word.py with the same words.")
    order = [m for m in ALL_MODELS if m in set(M.model)]
    M = M.set_index("model").loc[order].reset_index()
    cols = ["model", "arch", "n_layers", "n_words",
            "mean_r", "min_r", "max_r", "n_words_positive", "perm_p",
            "mean_r_quad", "min_r_quad", "max_r_quad", "n_words_positive_quad", "perm_p_quad"]
    df = M[cols]

    print(f"\nTable B  {projection} {measure.replace('_', ' ')} by model  (alignment={alignment})")
    print("  mean partial r over words [range], words positive, circular-shift permutation p")
    print(f"  {'model':<20}{'arch':>8}{'linear':>9}{'range':>17}{'pos':>6}{'perm p':>8}"
          f"{'quad':>9}{'range':>17}{'pos':>6}{'perm p':>8}")
    print("  " + "-" * 108)
    for _, r in df.iterrows():
        print(f"  {r.model:<20}{r.arch:>8}{r.mean_r:>+9.3f}  [{r.min_r:+.2f},{r.max_r:+.2f}]"
              f"{f'{r.n_words_positive}/{r.n_words}':>6}{r.perm_p:>8.3f}"
              f"{r.mean_r_quad:>+9.3f}  [{r.min_r_quad:+.2f},{r.max_r_quad:+.2f}]"
              f"{f'{r.n_words_positive_quad}/{r.n_words}':>6}{r.perm_p_quad:>8.3f}")
    df.to_csv(out_dir / f"table_b_by_model_{projection}_{measure}{sfx}.csv", index=False)
    return df


# ---------------------------------------------------------------- Table C

def table_c(projection, measure, alignment, exclude, out_dir, sfx):
    T = read_cw(projection, measure, alignment, "_summary")
    if T is None:
        return None
    rows = []
    for g in GROUP_ORDER:
        rows.append(T[(T.group == g) & (T.words == "all")])
    if exclude:
        wo = T[(T.group == "decoders") & (T.words == f"without_{exclude}")]
        if wo.empty:
            print(f"  [warn] Table C: no leave-out rows for '{exclude}'")
        rows.append(wo)
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "subset", np.where(df.words == "all", "all words", "excluding " + df.words.str.replace("without_", "")))
    cols = ["subset", "group", "control", "n_models", "mean", "median", "n_positive", "n_negative",
            "loo_min", "loo_max", "loo_min_dropped", "loo_max_dropped", "t_p", "wilcoxon_p"]
    df = df[cols]

    print(f"\nTable C  {projection} {measure.replace('_', ' ')}: model-level tests  (alignment={alignment})")
    print("  one value per model; p-values exploratory (models are not independent samples)")
    print(f"  {'subset':<18}{'group':<10}{'control':<11}{'mean':>8}{'pos':>7}"
          f"{'LOO range':>18}{'t p':>9}{'Wilcoxon p':>12}")
    print("  " + "-" * 93)
    for _, r in df.iterrows():
        print(f"  {r.subset:<18}{r.group:<10}{r.control:<11}{r['mean']:>+8.3f}"
              f"{f'{r.n_positive}/{r.n_models}':>7}"
              f"{f'[{r.loo_min:+.3f},{r.loo_max:+.3f}]':>18}"
              f"{fmt_p(r.t_p):>9}{fmt_p(r.wilcoxon_p):>12}")
    print("  (groups of 4 models: the smallest attainable Wilcoxon p is 0.125)")
    df.to_csv(out_dir / f"table_c_by_architecture_{projection}_{measure}{sfx}.csv", index=False)
    return df


# --------------------------------------------------------------- Table S1

def table_s1(projection, measure, alignment, out_dir, sfx):
    C = read_cw(projection, measure, alignment, "")
    if C is None:
        return None
    rows = []
    for arch in ["decoder", "encoder"]:
        a = C[C.arch == arch]
        if a.empty:
            continue
        for q, control in (("", "linear"), ("_quad", "quadratic")):
            rows.append({
                "architecture": arch, "control": control, "cells": len(a),
                "share_p05_naive": (a[f"partial_p{q}_naive"] < .05).mean(),
                "share_p05_effective_n": (a[f"partial_p{q}_eff"] < .05).mean(),
                "mean_n_layers": a.n_layers.mean(),
                "mean_n_eff": a[f"n_eff{q}"].mean(),
                "mean_lag1_geometry": a[f"lag1_geometry{q}"].mean(),
                "mean_lag1_sep": a[f"lag1_sep{q}"].mean(),
            })
    df = pd.DataFrame(rows)

    print(f"\nTable S1  Sensitivity: per-cell significance under layer dependence  (alignment={alignment})")
    print(f"  {'arch':<9}{'control':<11}{'cells':>6}{'naive':>8}{'eff-n':>8}"
          f"{'layers':>8}{'n_eff':>7}{'lag1 geo':>10}{'lag1 sep':>10}")
    print("  " + "-" * 77)
    for _, r in df.iterrows():
        print(f"  {r.architecture:<9}{r.control:<11}{r.cells:>6}{r.share_p05_naive:>8.1%}"
              f"{r.share_p05_effective_n:>8.1%}{r.mean_n_layers:>8.1f}{r.mean_n_eff:>7.1f}"
              f"{r.mean_lag1_geometry:>10.3f}{r.mean_lag1_sep:>10.3f}")
    print("  (effective n assumes AR(1) dependence; a sensitivity check, not a definitive test)")
    df.to_csv(out_dir / f"table_s1_sensitivity_{projection}_{measure}{sfx}.csv", index=False)
    return df


# --------------------------------------------------------------- Table S2

def table_s2(projection, measure, out_dir):
    frames = {}
    for al in ALIGNMENTS:
        p = Path(f"{cw_stem(projection, measure, al)}_summary.csv")
        if not p.exists():
            print(f"\n  [skip] Table S2: {p.name} not found (needs both alignments)")
            return None
        frames[al] = pd.read_csv(p)
    rows = []
    for g in ["decoders", "llama", "qwen", "encoders"]:
        for control in ["linear", "quadratic"]:
            row = {"group": g, "control": control}
            for al, T in frames.items():
                r = T[(T.group == g) & (T.control == control) & (T.words == "all")]
                if r.empty:
                    continue
                r = r.iloc[0]
                row.update({f"{al}_mean": r["mean"], f"{al}_positive": f"{r.n_positive}/{r.n_models}",
                            f"{al}_t_p": r.t_p, f"{al}_wilcoxon_p": r.wilcoxon_p})
            rows.append(row)
    df = pd.DataFrame(rows)

    print(f"\nTable S2  Sensitivity: pre-block against post-block alignment, model-level")
    print(f"  {'group':<10}{'control':<11}{'post':>8}{'pos':>6}{'W p':>8}{'pre':>9}{'pos':>6}{'W p':>8}")
    print("  " + "-" * 66)
    for _, r in df.iterrows():
        print(f"  {r.group:<10}{r.control:<11}{r.post_mean:>+8.3f}{r.post_positive:>6}"
              f"{fmt_p(r.post_wilcoxon_p):>8}{r.pre_mean:>+9.3f}{r.pre_positive:>6}"
              f"{fmt_p(r.pre_wilcoxon_p):>8}")
    df.to_csv(out_dir / f"table_s2_alignment_{projection}_{measure}.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="stable_rank")
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--exclude", default="bank")
    ap.add_argument("--words", default=",".join(WORDS))
    ap.add_argument("--alignment", default="post", choices=list(ALIGNMENTS))
    a = ap.parse_args()

    words = [w.strip() for w in a.words.split(",") if w.strip()]
    out_dir = ensure_dir(get_results_root() / "analysis" / "_tables")
    sfx = "" if a.alignment == "post" else f"_{a.alignment}"

    print(f"\n{'=' * 88}")
    print(f"RESULTS TABLES   {a.projection} {a.measure}   words: {len(words)}   alignment: {a.alignment}")
    print(f"{'=' * 88}")

    table_a(words, out_dir)
    table_b(a.projection, a.measure, a.alignment, words, out_dir, sfx)
    table_c(a.projection, a.measure, a.alignment, a.exclude, out_dir, sfx)
    table_s1(a.projection, a.measure, a.alignment, out_dir, sfx)
    table_s2(a.projection, a.measure, out_dir)

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
