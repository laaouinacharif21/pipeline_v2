# -*- coding: utf-8 -*-
"""
Table 4: layer-wise TruthfulQA probe accuracy against projection geometry
(September 2026).

Replaces a table previously produced by a one-off command (pre-block
alignment, naive per-model p-values) with a reproducible script. The original
file is preserved in results_pre_alignment_fix/analysis/_tables/.

Probe metric
    probe_accuracy is the cross-validated accuracy of a logistic-regression
    probe trained on hidden states at each layer (src/analysis/layer_probe.py,
    StratifiedGroupKFold; not modified here). It measures information linearly
    decodable by a trained probe. It is NOT the model's factual generation
    accuracy.

Design
    alignment   geometry of block l <-> probe accuracy at hidden state l+1
                ("post", primary) or at hidden state l ("pre", sensitivity),
                via _io.align_geometry_sep
    per model   one probe series, so there is no averaging over words;
                partial r under linear and quadratic depth control;
                circular-shift p and effective-n p as sensitivity analyses
    groups      model-level summary (_stats.model_level_summary) for decoders,
                llama, qwen and encoders; p-values exploratory

Outputs in results/analysis/_tables/  (suffix _pre for the pre alignment)
    table4_probe_geometry.csv            one row per model
    table4_probe_geometry_summary.csv    model-level tests by group and control

Usage
    python -m src.analysis.probe_geometry                  both alignments
    python -m src.analysis.probe_geometry --alignment post
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis import _stats as S  # noqa: E402
from src.analysis._io import (ALIGNMENTS, ALL_MODELS, DECODERS, FAMILIES,  # noqa: E402
                              align_geometry_sep, load_params)
from src.utils.paths import ensure_dir, get_model_result_dir, get_results_root  # noqa: E402

TASK = "truthfulqa"
MEASURE = "q_proj_stable_rank"
CONTROLS = ((1, "", "linear"), (2, "_quad", "quadratic"))
GROUPS = {"decoders": DECODERS, "llama": FAMILIES["llama"],
          "qwen": FAMILIES["qwen"], "encoders": FAMILIES["bert"]}


def family_of(model):
    return next(f for f, ms in FAMILIES.items() if model in ms)


def load_probe(model, task=TASK):
    p = get_model_result_dir(model, task) / "metrics" / "probe_accuracy.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[["layer", "probe_accuracy"]].sort_values("layer").reset_index(drop=True)


def compute(alignment="post", measure=MEASURE, task=TASK):
    """One row per model with a probe file and geometry."""
    rows = []
    for model in ALL_MODELS:
        geo, pr = load_params(model), load_probe(model, task)
        if geo is None or pr is None or measure not in geo.columns:
            continue
        m = align_geometry_sep(geo, pr, alignment=alignment)
        depth = m["layer"].to_numpy(float)
        x = m[measure].to_numpy(float)
        y = m["probe_accuracy"].to_numpy(float)
        row = {"model": model, "family": family_of(model),
               "arch": "decoder" if model in DECODERS else "encoder",
               "task": task, "measure": measure, "alignment": alignment,
               "n_layers": len(m),
               "probe_embedding": float(pr.probe_accuracy.iloc[0]),
               "probe_first_paired": float(y[0]),
               "probe_max": float(pr.probe_accuracy.max())}
        for deg, sfx, _ in CONTROLS:
            rx, ry = S.residualise(x, depth, deg), S.residualise(y, depth, deg)
            r, n_eff, p_eff = S.effective_n_p(rx, ry, deg)
            _, p_perm, _ = S.circular_shift_p(rx, [ry])
            row.update({f"r{sfx}": r,
                        f"p{sfx}_naive": float(stats.pearsonr(rx, ry)[1]),
                        f"n_eff{sfx}": n_eff, f"p{sfx}_eff": p_eff,
                        f"perm_p{sfx}": p_perm})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df):
    rows = []
    for group, members in GROUPS.items():
        sub = df[df.model.isin(members)]
        if sub.empty:
            continue
        for _, sfx, name in CONTROLS:
            s = S.model_level_summary(dict(zip(sub.model, sub[f"r{sfx}"])))
            rows.append({"group": group, "control": name, **s})
    return pd.DataFrame(rows)


def fmt_p(p):
    return "   n/a" if pd.isna(p) else f"{p:6.4f}"


def report(df, T, alignment):
    print(f"\nTable 4  {TASK} probe accuracy vs {MEASURE}  (alignment={alignment})")
    print("  probe accuracy = cross-validated linear-probe accuracy, not generation accuracy")
    print(f"  {'model':<20}{'arch':>8}{'layers':>7}{'emb':>7}{'linear':>9}{'perm p':>8}{'eff p':>8}"
          f"{'quad':>9}{'perm p':>8}{'eff p':>8}")
    print("  " + "-" * 92)
    for _, r in df.iterrows():
        print(f"  {r.model:<20}{r.arch:>8}{r.n_layers:>7}{r.probe_embedding:>7.3f}"
              f"{r.r:>+9.3f}{r.perm_p:>8.3f}{r.p_eff:>8.3f}"
              f"{r.r_quad:>+9.3f}{r.perm_p_quad:>8.3f}{r.p_quad_eff:>8.3f}")
    print(f"\n  Model-level (exploratory)")
    print(f"  {'group':<10}{'control':<11}{'mean':>8}{'pos':>7}{'LOO range':>18}{'t p':>9}{'Wilcoxon p':>12}")
    for _, r in T.iterrows():
        loo = "" if pd.isna(r.loo_min) else f"[{r.loo_min:+.3f},{r.loo_max:+.3f}]"
        print(f"  {r.group:<10}{r.control:<11}{r['mean']:>+8.3f}{f'{r.n_positive}/{r.n_models}':>7}"
              f"{loo:>18}{fmt_p(r.t_p):>9}{fmt_p(r.wilcoxon_p):>12}")
    dec = df[df.arch == "decoder"]
    if not dec.empty:
        print(f"\n  Sensitivity, decoders: p<0.05 naive {int((dec.p_naive < .05).sum())}/{len(dec)} "
              f"-> effective-n {int((dec.p_eff < .05).sum())}/{len(dec)} "
              f"-> circular shift {int((dec.perm_p < .05).sum())}/{len(dec)} (linear);  "
              f"quadratic {int((dec.p_quad_naive < .05).sum())} -> {int((dec.p_quad_eff < .05).sum())} "
              f"-> {int((dec.perm_p_quad < .05).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", default="both", choices=["both", *ALIGNMENTS])
    a = ap.parse_args()
    out_dir = ensure_dir(get_results_root() / "analysis" / "_tables")
    aligns = list(ALIGNMENTS) if a.alignment == "both" else [a.alignment]
    means = {}
    for al in aligns:
        df = compute(al)
        if df.empty:
            raise SystemExit(f"No probe results found for task '{TASK}'.")
        T = summarise(df)
        report(df, T, al)
        sfx = "" if al == "post" else f"_{al}"
        df.to_csv(out_dir / f"table4_probe_geometry{sfx}.csv", index=False)
        T.to_csv(out_dir / f"table4_probe_geometry_summary{sfx}.csv", index=False)
        dec = T[T.group == "decoders"].set_index("control")["mean"]
        means[al] = dec
    if len(means) == 2:
        print("\n  Alignment comparison, decoder mean r:  "
              + "   ".join(f"{c}: post {means['post'][c]:+.3f} / pre {means['pre'][c]:+.3f}"
                           for c in ("linear", "quadratic")))
    print(f"\nSaved -> {out_dir}/table4_probe_geometry*.csv")


if __name__ == "__main__":
    main()
