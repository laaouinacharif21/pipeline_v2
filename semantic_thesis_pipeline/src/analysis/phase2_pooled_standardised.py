"""
Phase 2 pooled layer-level analysis.

Raw pooling across models mixes between-model scale differences with the
within-model layer relationship of interest, so both variables are z-scored
within each model before pooling. Layer depth co-varies with both variables,
so the depth-controlled partial correlation is reported as the primary
result, consistent with the Phase 1A protocol. Spearman is reported
alongside Pearson as a monotonicity check.

All p-values are exploratory: layers within a model are not independent
observations and no multiple-comparison correction is applied.

Usage:
    python -m src.analysis.phase2_pooled_standardised
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import get_results_root, infer_family

RESULTS_ROOT = get_results_root()

ALL_MODELS = [
    "llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b",
    "qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b",
    "bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base",
]

MEASURE = "up_proj_erank"


def z(v):
    v = np.asarray(v, dtype=float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def residualise(a, b):
    """Remove the linear component of b from a."""
    return a - np.polyval(np.polyfit(b, a, 1), b)


def report(label, x, y, n=None):
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    n = n if n is not None else len(x)
    print(f"  {label:<16} n={n:>4}  "
          f"Pearson r={pr:+.3f} p={pp:.5f} {star(pp):<5} "
          f"Spearman r={sr:+.3f} p={sp:.5f} {star(sp)}")
    return {"pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp, "n": n}


def build_frame():
    rows = []
    for model in ALL_MODELS:
        fam = infer_family(model)
        ep = RESULTS_ROOT / fam / model / "parameters" / "effective_rank.csv"
        sp = RESULTS_ROOT / fam / model / "metrics" / "semantic_separation.csv"
        if not ep.exists() or not sp.exists():
            print(f"  [SKIP] {model}")
            continue
        m = pd.read_csv(ep).merge(pd.read_csv(sp), on="layer").dropna(
            subset=[MEASURE, "separation"])
        if len(m) < 5:
            continue
        rows.append(pd.DataFrame({
            "model": model,
            "family": fam,
            "layer": m["layer"],
            "depth": m["layer"] / m["layer"].max(),
            "erank_z": z(m[MEASURE]),
            "sep_z": z(m["separation"]),
        }))
    return pd.concat(rows, ignore_index=True)


def run():
    df = build_frame()
    summary = []

    print(f"\n{'='*84}")
    print(f"PHASE 2 POOLED  --  {MEASURE} vs Sep(l), within-model standardised")
    print(f"results root: {RESULTS_ROOT}")
    print(f"{'='*84}")

    print("\n[1] Uncontrolled")
    for fam in ["llama", "qwen", "bert"]:
        d = df[df.family == fam]
        if len(d) >= 5:
            r = report(fam.upper(), d.erank_z, d.sep_z)
            summary.append({"group": fam, "control": "none", **r})
    d = df[df.family != "bert"]
    r = report("ALL DECODERS", d.erank_z, d.sep_z)
    summary.append({"group": "decoders", "control": "none", **r})

    print("\n[2] Depth-controlled (primary)")
    for fam in ["llama", "qwen", "bert"]:
        d = df[df.family == fam]
        if len(d) >= 5:
            r = report(fam.upper(),
                       residualise(d.erank_z.values, d.depth.values),
                       residualise(d.sep_z.values, d.depth.values))
            summary.append({"group": fam, "control": "depth", **r})
    d = df[df.family != "bert"]
    r = report("ALL DECODERS",
               residualise(d.erank_z.values, d.depth.values),
               residualise(d.sep_z.values, d.depth.values))
    summary.append({"group": "decoders", "control": "depth", **r})

    print("\n[3] Depth co-variation")
    dd = df[df.family != "bert"]
    print("  depth~erank r=%+.3f p=%.5f" % pearsonr(dd.depth, dd.erank_z))
    print("  depth~sep   r=%+.3f p=%.5f" % pearsonr(dd.depth, dd.sep_z))

    print("\n[4] Leave-one-model-out (decoders, depth-controlled)")
    for m in sorted(dd.model.unique()):
        s = dd[dd.model != m]
        report(f"w/o {m}",
               residualise(s.erank_z.values, s.depth.values),
               residualise(s.sep_z.values, s.depth.values))

    out = RESULTS_ROOT / "phase2_pooled_summary.csv"
    pd.DataFrame(summary).to_csv(out, index=False)
    df.to_csv(RESULTS_ROOT / "phase2_pooled_standardised.csv", index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    run()
