"""Audit check: layer dependence and significance of q_proj stable rank vs Sep(l). READ-ONLY.

For each decoder, alignment (pre / post) and depth control (linear / quadratic):
  1. lag-1 autocorrelation of the depth-residualised series (neighbouring-layer dependence)
  2. per-cell significance: naive Pearson p vs effective-n corrected p
       n_eff = n * (1 - rho_x*rho_y) / (1 + rho_x*rho_y)
  3. per-model circular-shift permutation test on the word-averaged r
       (the same shift is applied to all 7 words, because they share one geometry)
       NOTE: with n layers there are only n-1 shifts, so the smallest possible p is 1/n (~0.03)
  4. across-model test: 9 word-averaged r values -> sign count, t-test, Wilcoxon
  5. TruthfulQA probe accuracy vs q_proj stable rank under both alignments

Run from the pipeline root:
    python _audit/check_dependence.py
"""
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from src.analysis._io import load_params, load_sep  # noqa: E402

LLAMA = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"]
QWEN = ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]
DEC = LLAMA + QWEN
WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]
MEAS = "q_proj_stable_rank"
ALIGNS = (("pre", 0), ("post", 1))
DEGS = ((1, "linear"), (2, "quadratic"))


def resid(L, v, deg):
    X = np.vander(L.astype(float), deg + 1)
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def lag1(v):
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


def pval(r, df):
    df = max(df, 1.0)
    t = r * np.sqrt(df / max(1.0 - r * r, 1e-12))
    return float(2 * stats.t.sf(abs(t), df))


def sep_col(df):
    for c in ("separation", "sep", "semantic_separation"):
        if c in df.columns:
            return c
    raise SystemExit(f"no separation column in {list(df.columns)}")


def shifted(layers, table, col, shift):
    d = dict(zip(table["layer"], table[col]))
    return np.array([d[int(l) + shift] for l in layers], dtype=float)


cells, models = [], []
for model in DEC:
    geo = load_params(model)
    L = geo["layer"].to_numpy()
    x = geo[MEAS].to_numpy(float)
    n = len(L)
    seps = {}
    for w in WORDS:
        s = load_sep(model, w)
        seps[w] = (s, sep_col(s))
    for align, shift in ALIGNS:
        for deg, dname in DEGS:
            rx = resid(L, x, deg)
            rho_x = lag1(rx)
            rys = {}
            for w in WORDS:
                s, c = seps[w]
                ry = resid(L, shifted(L, s, c, shift), deg)
                rys[w] = ry
                r = float(np.corrcoef(rx, ry)[0, 1])
                rho_y = lag1(ry)
                k = rho_x * rho_y
                n_eff = n * (1 - k) / (1 + k)
                cells.append(dict(model=model, word=w, align=align, control=dname, r=r,
                                  rho_x=rho_x, rho_y=rho_y, n=n, n_eff=n_eff,
                                  p_naive=pval(r, n - 2 - deg),
                                  p_eff=pval(r, max(n_eff - 2 - deg, 1.0))))
            obs = np.mean([np.corrcoef(rx, rys[w])[0, 1] for w in WORDS])
            null = [np.mean([np.corrcoef(rx, np.roll(rys[w], k))[0, 1] for w in WORDS])
                    for k in range(1, n)]
            p_perm = (1 + sum(abs(v) >= abs(obs) for v in null)) / n
            models.append(dict(model=model, align=align, control=dname, r_mean=obs, p_perm=p_perm))

C = pd.DataFrame(cells)
M = pd.DataFrame(models)
C["family"] = np.where(C.model.isin(LLAMA), "llama", "qwen")
pd.set_option("display.width", 200)

print("=== 1. lag-1 autocorrelation of depth residuals (mean) ===")
print(C.groupby(["control", "align", "family"])[["rho_x", "rho_y", "n_eff"]].mean().round(3).to_string())

print("\n=== 2. share of the 63 decoder cells with p < 0.05 (sign ignored) ===")
C["sig_naive"] = C.p_naive < 0.05
C["sig_eff"] = C.p_eff < 0.05
C["sig_naive_pos"] = (C.p_naive < 0.05) & (C.r > 0)
C["sig_eff_pos"] = (C.p_eff < 0.05) & (C.r > 0)
print(C.groupby(["control", "align"])[["sig_naive", "sig_eff", "sig_naive_pos", "sig_eff_pos"]].mean().round(3).to_string())
print("(handoff claimed 71% of decoder cells significant: linear / pre)")

print("\n=== 3. per-model word-averaged r and circular-shift permutation p ===")
print(M.pivot_table(index="model", columns=["control", "align"], values=["r_mean", "p_perm"]).round(3).to_string())

print("\n=== 4. across the 9 models (one r per model) ===")
for (ctl, al), g in M.groupby(["control", "align"]):
    v = g.r_mean.to_numpy()
    t = stats.ttest_1samp(v, 0.0)
    wx = stats.wilcoxon(v)
    print(f"  {ctl:9s} {al:4s}  mean {v.mean():+.3f}  positive {int((v > 0).sum())}/9  "
          f"t-test p {t.pvalue:.4f}  wilcoxon p {wx.pvalue:.4f}  "
          f"models perm p<0.05: {int((g.p_perm < 0.05).sum())}/9")

print("\n=== 5. TruthfulQA probe accuracy vs q_proj stable rank ===")
rows = []
for model in DEC:
    hits = glob.glob(f"results/words/truthfulqa/*/{model}/metrics/probe_accuracy.csv")
    if len(hits) != 1:
        print(f"  {model}: probe file not found ({hits})")
        continue
    p = pd.read_csv(hits[0])
    geo = load_params(model)
    L = geo["layer"].to_numpy()
    x = geo[MEAS].to_numpy(float)
    n = len(L)
    row = dict(model=model, probe_layers=len(p), probe_layer0=round(float(p.sort_values("layer").probe_accuracy.iloc[0]), 3))
    for align, shift in ALIGNS:
        y_all = shifted(L, p, "probe_accuracy", shift)
        for deg, dname in DEGS:
            rx, ry = resid(L, x, deg), resid(L, y_all, deg)
            r = float(np.corrcoef(rx, ry)[0, 1])
            k = lag1(rx) * lag1(ry)
            n_eff = n * (1 - k) / (1 + k)
            row[f"{dname[:3]}_{align}_r"] = round(r, 3)
            row[f"{dname[:3]}_{align}_p_eff"] = round(pval(r, max(n_eff - 2 - deg, 1.0)), 3)
    rows.append(row)
if rows:
    P = pd.DataFrame(rows)
    print(P.to_string(index=False))
    for col in [c for c in P.columns if c.endswith("_r")]:
        print(f"  {col:12s} mean {P[col].mean():+.3f}")
    print("(handoff: linear mean +0.700, llama-7b +0.840, qwen3-8b +0.433)")
