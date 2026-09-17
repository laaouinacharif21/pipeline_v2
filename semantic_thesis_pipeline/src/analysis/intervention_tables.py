# -*- coding: utf-8 -*-
"""
Table 5: spectral intervention results (September 2026).

Replaces a one-off table (results/analysis/_tables/table5_intervention.csv,
19 rows, only 4 of the 27 main cells) with a reproducible script reading
every per-model file in results/analysis/_intervention/.

Retention
    Every value is computed against the ratio = 1.0 row of the same
    (model, word, projection, condition) file, i.e. the unmodified model run
    through the same code path. The script stops on a missing or duplicate
    baseline row.
        peak retention  sep_max(after)  / sep_max(baseline)
        mean retention  sep_mean(after) / sep_mean(baseline)
    Peak Sep can move to a different layer after the intervention, in which
    case peak retention compares different layers; mean retention is reported
    alongside for that reason, with the peak layer before and after.

Tables (results/analysis/_tables/)
    table5a_qproj_truncate_cells.csv    q_proj truncate 0.80, 9 models x 3 words
    table5b_qproj_truncate_models.csv   one row per model (mean over words)
    table5c_model_level.csv             model-level tests on retention - 1 and
                                        capability deltas (exploratory; a
                                        non-significant result does not show
                                        that separation was preserved)
    table5d_vproj_truncate_cells.csv    v_proj truncate 0.80 cells, plus models
    table5e_capability.csv              GSM8K answer log-probability (primary)
                                        and C-Eval accuracy per model/condition
    table5f_probe_intervention.csv      TruthfulQA probe under intervention
    table5g_case_study.csv              every other q_proj condition/ratio
                                        (llama-3-8b, qwen2.5-7b): descriptive
                                        only, not part of cross-model tests

Capability notes
    gsm8k_delta is the change in reference-answer log-probability, not
    generated-answer accuracy. C-Eval is flagged near chance when the baseline
    accuracy is below 0.35 (chance 0.25); such rows cannot detect degradation.

Usage
    python -m src.analysis.intervention_tables
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis import _stats as S  # noqa: E402
from src.analysis._io import DECODERS, FAMILIES  # noqa: E402
from src.utils.paths import ensure_dir, get_results_root  # noqa: E402

KEY = ["model", "word", "projection", "condition"]
CEVAL_NEAR_CHANCE = 0.35


def intervention_dir():
    return get_results_root() / "analysis" / "_intervention"


def family_of(model):
    return next((f for f, ms in FAMILIES.items() if model in ms), "")


def order_models(df):
    rank = {m: i for i, m in enumerate(DECODERS)}
    return df.assign(_o=df.model.map(rank)).sort_values(["_o"] + [c for c in ("word",) if c in df]).drop(columns="_o")


# ----------------------------------------------------------------- loading

def load_interventions():
    files = sorted(intervention_dir().glob("intervention_*.csv"))
    if not files:
        raise SystemExit(f"No intervention_*.csv under {intervention_dir()}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("ratio", "sep_max", "sep_mean", "stable_rank_retained", "fro_change"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def retention_cells(d):
    is_base = np.isclose(d.ratio, 1.0)
    base = d[is_base].set_index(KEY)
    if not base.index.is_unique:
        dup = base.index[base.index.duplicated()].tolist()[:3]
        raise SystemExit(f"Duplicate baseline rows: {dup}")
    rows = []
    for _, r in d[~is_base].iterrows():
        k = tuple(r[c] for c in KEY)
        if k not in base.index:
            raise SystemExit(f"No ratio=1.0 baseline row for {k}")
        b = base.loc[k]
        rows.append({
            "model": r.model, "family": family_of(r.model), "word": r.word,
            "projection": r.projection, "condition": r.condition, "ratio": float(r.ratio),
            "stable_rank_retained": float(r.stable_rank_retained), "fro_change": float(r.fro_change),
            "sep_max_base": float(b.sep_max), "sep_max_after": float(r.sep_max),
            "peak_retention": float(r.sep_max / b.sep_max),
            "sep_mean_base": float(b.sep_mean), "sep_mean_after": float(r.sep_mean),
            "mean_retention": float(r.sep_mean / b.sep_mean),
            "peak_layer_base": int(b.sep_peak_layer), "peak_layer_after": int(r.sep_peak_layer),
            "peak_shifted": bool(int(b.sep_peak_layer) != int(r.sep_peak_layer)),
            "peak_layer_shift": int(r.sep_peak_layer) - int(b.sep_peak_layer),
        })
    return pd.DataFrame(rows)


def grid(cells, projection, condition, ratio):
    g = cells[(cells.projection == projection) & (cells.condition == condition)
              & np.isclose(cells.ratio, ratio)]
    return order_models(g).reset_index(drop=True)


def by_model(cells):
    M = cells.groupby("model").agg(
        family=("family", "first"), n_words=("word", "count"),
        stable_rank_retained=("stable_rank_retained", "mean"),
        peak_retention=("peak_retention", "mean"),
        mean_retention=("mean_retention", "mean"),
        min_peak_retention=("peak_retention", "min"),
        max_peak_retention=("peak_retention", "max"),
        words_peak_up=("peak_retention", lambda s: int((s > 1).sum())),
        words_mean_up=("mean_retention", lambda s: int((s > 1).sum())),
        peak_shifts=("peak_shifted", "sum"),
    ).reset_index()
    M["peak_shifts"] = M.peak_shifts.astype(int)
    return order_models(M).reset_index(drop=True)


def load_capability():
    files = sorted(intervention_dir().glob("capability_*.csv"))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    base = d[d.condition == "baseline"].set_index("model").ceval_accuracy.rename("ceval_baseline")
    d = d.join(base, on="model")
    d["family"] = d.model.map(family_of)
    d["ceval_near_chance"] = d.ceval_baseline < CEVAL_NEAR_CHANCE
    cols = ["model", "family", "condition", "stable_rank_retained", "ceval_accuracy",
            "ceval_baseline", "ceval_relative", "ceval_near_chance", "gsm8k_logprob", "gsm8k_delta"]
    return order_models(d[cols]).reset_index(drop=True)


def load_probe_intervention():
    rows = []
    for f in sorted(intervention_dir().glob("probe_intervention_*.csv")):
        d = pd.read_csv(f)
        b = d[np.isclose(d.ratio, 1.0)]
        if len(b) != 1:
            raise SystemExit(f"{f.name}: expected one ratio=1.0 row, found {len(b)}")
        b = b.iloc[0]
        for _, r in d[~np.isclose(d.ratio, 1.0)].iterrows():
            rows.append({"model": r.model, "projection": r.projection, "condition": r.condition,
                         "ratio": float(r.ratio), "stable_rank_retained": float(r.stable_rank_retained),
                         "probe_peak_base": float(b.probe_peak), "probe_peak_after": float(r.probe_peak),
                         "probe_peak_retention": float(r.probe_peak / b.probe_peak),
                         "probe_mean_retention": float(r.probe_mean / b.probe_mean),
                         "peak_layer_base": int(b.peak_layer), "peak_layer_after": int(r.peak_layer)})
    return order_models(pd.DataFrame(rows)).reset_index(drop=True) if rows else pd.DataFrame()


# ----------------------------------------------------------- model level

def model_level(models_q, models_v, capability):
    rows = []

    def add(analysis, values, group="decoders", note=""):
        if not values:
            return
        rows.append({"analysis": analysis, "group": group, "note": note,
                     **S.model_level_summary(values)})

    for label, M in (("q_proj truncate 0.80", models_q), ("v_proj truncate 0.80", models_v)):
        for col in ("peak_retention", "mean_retention"):
            add(f"{label}: {col} - 1", dict(zip(M.model, M[col] - 1)))
    if not capability.empty:
        for cond in ("q_proj truncate 0.80", "v_proj truncate 0.80", "q_proj rotate"):
            c = capability[capability.condition == cond]
            add(f"{cond}: gsm8k_delta", dict(zip(c.model, c.gsm8k_delta)),
                note="answer log-probability, not accuracy")
            q = c[~c.ceval_near_chance]
            add(f"{cond}: ceval_relative - 1", dict(zip(q.model, q.ceval_relative - 1)),
                group="ceval above near-chance threshold",
                note=f"baseline >= {CEVAL_NEAR_CHANCE}; n={len(q)}")
    return pd.DataFrame(rows)


def build():
    cells = retention_cells(load_interventions())
    cells_q = grid(cells, "q_proj", "truncate", 0.80)
    cells_v = grid(cells, "v_proj", "truncate", 0.80)
    main_keys = set(map(tuple, pd.concat([cells_q, cells_v])[["model", "word", "projection", "condition", "ratio"]].values))
    case = cells[[tuple(x) not in main_keys for x in cells[["model", "word", "projection", "condition", "ratio"]].values]]
    case = case.sort_values(["model", "condition", "ratio", "word"]).reset_index(drop=True)
    models_q, models_v = by_model(cells_q), by_model(cells_v)
    capability = load_capability()
    return {"cells_q": cells_q, "models_q": models_q, "cells_v": cells_v, "models_v": models_v,
            "model_level": model_level(models_q, models_v, capability),
            "capability": capability, "probe": load_probe_intervention(), "case": case}


# ---------------------------------------------------------------- output

def fmt_p(p):
    return "   n/a" if pd.isna(p) else f"{p:6.4f}"


def print_cells(C, title):
    print(f"\n{title}")
    print(f"  {'model':<14}{'word':<7}{'SR ret':>7}{'peak ret':>9}{'mean ret':>9}{'layer':>10}{'shift':>7}")
    for _, r in C.iterrows():
        print(f"  {r.model:<14}{r.word:<7}{r.stable_rank_retained:>7.1%}{r.peak_retention:>9.1%}"
              f"{r.mean_retention:>9.1%}{f'{r.peak_layer_base}->{r.peak_layer_after}':>10}"
              f"{'yes' if r.peak_shifted else '':>7}")
    print(f"  cells: peak up {int((C.peak_retention > 1).sum())} / down {int((C.peak_retention < 1).sum())}, "
          f"range {C.peak_retention.min():.1%}-{C.peak_retention.max():.1%};  "
          f"mean up {int((C.mean_retention > 1).sum())} / down {int((C.mean_retention < 1).sum())};  "
          f"peak shifted in {int(C.peak_shifted.sum())}")


def print_models(M, title):
    print(f"\n{title}")
    print(f"  {'model':<14}{'words':>6}{'peak ret':>9}{'[min, max]':>16}{'mean ret':>9}{'up (peak/mean)':>16}{'shifts':>7}")
    for _, r in M.iterrows():
        print(f"  {r.model:<14}{r.n_words:>6}{r.peak_retention:>9.1%}"
              f"{f'[{r.min_peak_retention:.0%}, {r.max_peak_retention:.0%}]':>16}{r.mean_retention:>9.1%}"
              f"{f'{r.words_peak_up}/{r.words_mean_up}':>16}{r.peak_shifts:>7}")


def main():
    T = build()
    out = ensure_dir(get_results_root() / "analysis" / "_tables")

    print(f"{'=' * 88}\nTABLE 5  Spectral intervention (retention = after / unmodified baseline)\n{'=' * 88}")
    print_cells(T["cells_q"], "5a  q_proj truncate 0.80, all cells")
    print_models(T["models_q"], "5b  q_proj truncate 0.80, per model (mean over words)")

    L = T["model_level"]
    print("\n5c  Model-level tests (exploratory; non-significance does not show preservation)")
    print(f"  {'analysis':<46}{'group':<10}{'mean':>9}{'pos':>7}{'t p':>9}{'Wilcoxon p':>12}")
    for _, r in L.iterrows():
        grp = "decoders" if r.group == "decoders" else "C-Eval ok"
        print(f"  {r.analysis:<46}{grp:<10}{r['mean']:>+9.4f}{f'{r.n_positive}/{r.n_models}':>7}"
              f"{fmt_p(r.t_p):>9}{fmt_p(r.wilcoxon_p):>12}")

    print_models(T["models_v"], "5d  v_proj truncate 0.80, per model")
    Cv = T["cells_v"]
    print(f"  cells: {len(Cv)}, models: {Cv.model.nunique()}, peak retention range "
          f"{Cv.peak_retention.min():.1%}-{Cv.peak_retention.max():.1%}")

    Cap = T["capability"]
    if not Cap.empty:
        print("\n5e  Capability (GSM8K = change in answer log-probability; C-Eval accuracy)")
        piv = Cap[Cap.condition != "baseline"].pivot_table(
            index="model", columns="condition", values="gsm8k_delta", sort=False)
        print("  GSM8K delta")
        print("  " + piv.round(3).to_string().replace("\n", "\n  "))
        base = Cap[Cap.condition == "baseline"][["model", "ceval_accuracy", "ceval_near_chance"]]
        print("  C-Eval baseline (near chance flagged): "
              + ", ".join(f"{r.model} {r.ceval_accuracy:.3f}{'*' if r.ceval_near_chance else ''}"
                          for _, r in base.iterrows()))
        for cond in ("q_proj truncate 0.80", "v_proj truncate 0.80", "q_proj rotate"):
            c = Cap[Cap.condition == cond]
            print(f"  {cond:<22} gsm8k_delta {c.gsm8k_delta.min():+.3f} to {c.gsm8k_delta.max():+.3f}   "
                  f"ceval_relative {c.ceval_relative.min():.3f} to {c.ceval_relative.max():.3f}")

    P = T["probe"]
    if not P.empty:
        print("\n5f  TruthfulQA probe under intervention")
        for _, r in P.iterrows():
            print(f"  {r.model:<14}{r.projection:<8}{r.condition:<9}{r.ratio:>5.2f}  peak {r.probe_peak_retention:>7.1%}"
                  f"  mean {r.probe_mean_retention:>7.1%}  layer {r.peak_layer_base}->{r.peak_layer_after}")

    Cs = T["case"]
    if not Cs.empty:
        print("\n5g  Case study (single-model conditions; descriptive only)")
        print(f"  {'model':<12}{'cond':<9}{'ratio':>6}{'word':>7}{'SR ret':>8}{'fro':>8}{'peak ret':>9}{'mean ret':>9}{'layer':>9}")
        for _, r in Cs.iterrows():
            print(f"  {r.model:<12}{r.condition:<9}{r.ratio:>6.2f}{r.word:>7}{r.stable_rank_retained:>8.1%}"
                  f"{r.fro_change:>+8.3f}{r.peak_retention:>9.1%}{r.mean_retention:>9.1%}"
                  f"{f'{r.peak_layer_base}->{r.peak_layer_after}':>9}")

    T["cells_q"].to_csv(out / "table5a_qproj_truncate_cells.csv", index=False)
    T["models_q"].to_csv(out / "table5b_qproj_truncate_models.csv", index=False)
    L.to_csv(out / "table5c_model_level.csv", index=False)
    pd.concat([T["cells_v"].assign(level="cell"), T["models_v"].assign(level="model")],
              ignore_index=True).to_csv(out / "table5d_vproj_truncate_cells.csv", index=False)
    Cap.to_csv(out / "table5e_capability.csv", index=False)
    P.to_csv(out / "table5f_probe_intervention.csv", index=False)
    Cs.to_csv(out / "table5g_case_study.csv", index=False)
    print(f"\nSaved -> {out}/table5[a-g]_*.csv")


if __name__ == "__main__":
    main()
