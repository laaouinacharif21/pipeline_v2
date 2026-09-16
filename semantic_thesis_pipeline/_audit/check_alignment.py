"""Audit check: geometry vs Sep(l) under two layer alignments. READ-ONLY.

pre  : G(block l) <-> Sep(hidden state l)    (current pipeline, _io.merge on "layer")
post : G(block l) <-> Sep(hidden state l+1)  (auditor's proposal: output of block l)

Run from the pipeline root:
    python _audit/check_alignment.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.analysis._io import load_sep, load_params, load_erank  # noqa: E402

LLAMA = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"]
QWEN = ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"]
DEC = LLAMA + QWEN
WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]
MEASURES = ["q_proj_stable_rank", "q_proj_spectral_norm", "up_proj_erank"]


def sep_col(df):
    for c in ("separation", "sep", "semantic_separation"):
        if c in df.columns:
            return c
    raise SystemExit(f"no separation column in {list(df.columns)}")


def partial_r(layer, x, y, deg):
    X = np.vander(layer.astype(float), deg + 1)
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


rows = []
for model in DEC:
    params = load_params(model)
    erank = load_erank(model)
    geo = params.merge(erank, on="layer")
    for word in WORDS:
        sep = load_sep(model, word)
        s = sep[["layer", sep_col(sep)]].rename(columns={sep_col(sep): "sep"})
        for align, shift in (("pre", 0), ("post", 1)):
            g = geo.copy()
            g["sep_layer"] = g["layer"] + shift
            m = g.merge(s, left_on="sep_layer", right_on="layer", suffixes=("", "_s"))
            for meas in MEASURES:
                x = m[meas].to_numpy(float)
                y = m["sep"].to_numpy(float)
                L = m["layer"].to_numpy()
                for deg, name in ((1, "linear"), (2, "quadratic")):
                    rows.append(dict(model=model, word=word, align=align, measure=meas,
                                     control=name, n=len(m), r=partial_r(L, x, y, deg),
                                     first_sep=float(y[0]), n_geo=len(geo), n_sep=len(s)))

R = pd.DataFrame(rows)
R["family"] = np.where(R.model.isin(LLAMA), "llama", "qwen")
pd.set_option("display.width", 200)

print("=== layer counts (geometry rows / separation rows / merged rows) ===")
print(R.groupby(["model", "align"])[["n_geo", "n_sep", "n"]].first().to_string())

print("\n=== Sep at the first merged row (pre should be the embedding, ~0) ===")
print(R.groupby(["model", "align"]).first_sep.mean().unstack().round(4).to_string())

print("\n=== DECODER MEAN r over 9 models x 7 words ===")
print(R.pivot_table(index="measure", columns=["control", "align"], values="r", aggfunc="mean").round(3).to_string())

print("\n=== FAMILY MEAN r ===")
print(R.pivot_table(index=["measure", "control"], columns=["family", "align"], values="r", aggfunc="mean").round(3).to_string())

print("\n=== q_proj_stable_rank per model (mean over 7 words) ===")
q = R[R.measure == "q_proj_stable_rank"]
print(q.pivot_table(index="model", columns=["control", "align"], values="r", aggfunc="mean").round(3).to_string())

print("\nAuditor's numbers to compare (pre -> post):")
print("  q_proj_stable_rank   linear +0.464 -> +0.501   quadratic +0.070 -> +0.137")
print("  q_proj_spectral_norm linear -0.378 -> -0.431   quadratic +0.099 -> +0.012")
print("  up_proj_erank        linear +0.156 -> +0.236   quadratic -0.239 -> -0.117")
