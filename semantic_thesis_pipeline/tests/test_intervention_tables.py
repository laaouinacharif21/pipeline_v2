# -*- coding: utf-8 -*-
"""
Reproducibility test for Table 5 (src/analysis/intervention_tables.py).

Expected values were computed directly from the raw per-model CSVs during the
September 2026 audit, independently of this script.

Run:  python -m tests.test_intervention_tables
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.intervention_tables import build  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail and not condition else ''}")


def close(a, b, tol=5e-4):
    return abs(float(a) - float(b)) <= tol


if __name__ == "__main__":
    print("Table 5 reproducibility tests")
    T = build()
    Cq, Mq, Cv, Mv = T["cells_q"], T["models_q"], T["cells_v"], T["models_v"]

    print("\n[A] q_proj truncate 0.80")
    check("27 cells", len(Cq) == 27, f"{len(Cq)}")
    check("9 models x 3 words", Cq.model.nunique() == 9 and Cq.word.nunique() == 3)
    check("every cell has stable rank ~80%", bool(np.allclose(Cq.stable_rank_retained, 0.80, atol=0.002)))
    m_peak = (Mq.peak_retention - 1).mean()
    m_mean = (Mq.mean_retention - 1).mean()
    check("model mean peak change +2.53%", close(m_peak, 0.0253), f"{m_peak:+.4f}")
    check("model mean Sep-mean change +2.28%", close(m_mean, 0.0228), f"{m_mean:+.4f}")
    check("models peak up 5 / down 4",
          (Mq.peak_retention > 1).sum() == 5 and (Mq.peak_retention < 1).sum() == 4)
    check("models mean up 5 / down 4",
          (Mq.mean_retention > 1).sum() == 5 and (Mq.mean_retention < 1).sum() == 4)
    check("cells peak up 17 / down 10",
          (Cq.peak_retention > 1).sum() == 17 and (Cq.peak_retention < 1).sum() == 10)
    check("cells mean up 16 / down 11",
          (Cq.mean_retention > 1).sum() == 16 and (Cq.mean_retention < 1).sum() == 11)
    check("peak range 0.881 to 1.168",
          close(Cq.peak_retention.min(), 0.881, 6e-4) and close(Cq.peak_retention.max(), 1.168, 6e-4),
          f"{Cq.peak_retention.min():.4f} to {Cq.peak_retention.max():.4f}")
    c = Cq.set_index(["model", "word"])
    check("llama-3-8b pupil 1.168", close(c.loc[("llama-3-8b", "pupil"), "peak_retention"], 1.168, 6e-4))
    check("qwen-7b club 0.881", close(c.loc[("qwen-7b", "club"), "peak_retention"], 0.881, 6e-4))
    m = Mq.set_index("model")
    check("llama-2-7b model peak 0.9684 / mean 0.9423",
          close(m.loc["llama-2-7b", "peak_retention"], 0.9684) and close(m.loc["llama-2-7b", "mean_retention"], 0.9423))
    check("qwen3-8b model peak 1.0628 / mean 1.0641",
          close(m.loc["qwen3-8b", "peak_retention"], 1.0628) and close(m.loc["qwen3-8b", "mean_retention"], 1.0641))

    expected_shifts = {("llama-2-7b", "pupil", 12, 13), ("llama-3-8b", "pupil", 11, 23),
                       ("llama-3.1-8b", "pupil", 11, 23), ("llama-7b", "club", 22, 23),
                       ("qwen2-7b", "bat", 24, 13), ("qwen2-7b", "club", 28, 23)}
    got = set(map(tuple, Cq[Cq.peak_shifted][["model", "word", "peak_layer_base", "peak_layer_after"]].values))
    check("exactly the 6 audited peak shifts", got == expected_shifts, f"got {sorted(got)}")

    print("\n[B] v_proj truncate 0.80")
    check("24 cells, 8 models", len(Cv) == 24 and Cv.model.nunique() == 8, f"{len(Cv)} cells")
    check("qwen-7b absent", "qwen-7b" not in set(Cv.model))
    check("peak range 0.000 to 1.215",
          close(Cv.peak_retention.min(), 0.0, 6e-4) and close(Cv.peak_retention.max(), 1.215, 6e-4),
          f"{Cv.peak_retention.min():.4f} to {Cv.peak_retention.max():.4f}")

    print("\n[C] capability")
    Cap = T["capability"]
    for cond, lo, hi in [("q_proj truncate 0.80", -0.022, 0.058), ("v_proj truncate 0.80", -3.185, -0.015),
                         ("q_proj rotate", -8.371, -4.393)]:
        x = Cap[Cap.condition == cond].gsm8k_delta
        check(f"{cond} gsm8k_delta {lo:+.3f} to {hi:+.3f}",
              len(x) == 9 and close(x.min(), lo, 6e-4) and close(x.max(), hi, 6e-4),
              f"n={len(x)} {x.min():+.4f} to {x.max():+.4f}")
    q = Cap[Cap.condition == "q_proj truncate 0.80"].ceval_relative
    check("q_proj truncate ceval_relative 0.929 to 1.065", close(q.min(), 0.929, 6e-4) and close(q.max(), 1.065, 6e-4),
          f"{q.min():.4f} to {q.max():.4f}")
    near = set(Cap[(Cap.condition == "baseline") & Cap.ceval_near_chance].model)
    check("C-Eval near-chance flag = the 4 LLaMA models", near == set(["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"]),
          f"{sorted(near)}")

    print("\n[D] probe under intervention")
    P = T["probe"].set_index(["model", "projection"])
    for model, proj, want in [("llama-2-7b", "q_proj", 0.9856), ("llama-3-8b", "q_proj", 1.0060),
                              ("qwen2.5-7b", "q_proj", 0.9845), ("llama-2-7b", "v_proj", 0.8426),
                              ("llama-3-8b", "v_proj", 0.8353), ("qwen2.5-7b", "v_proj", 0.8579)]:
        v = P.loc[(model, proj), "probe_peak_retention"]
        check(f"{model} {proj} probe peak retention {want:.4f}", close(v, want), f"{float(v):.4f}")

    print("\n[E] case study")
    Cs = T["case"]
    check("case study excludes the main grids",
          not ((Cs.projection == "q_proj") & (Cs.condition == "truncate") & np.isclose(Cs.ratio, 0.8)).any())
    check("rotation keeps stable rank and removes separation (llama-3-8b bat)",
          bool(((Cs.model == "llama-3-8b") & (Cs.word == "bat") & (Cs.condition == "rotate")
                & (Cs.stable_rank_retained > 0.99) & (Cs.peak_retention < 0.05)).any()))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
