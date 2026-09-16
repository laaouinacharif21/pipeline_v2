# -*- coding: utf-8 -*-
"""
Tests for src/analysis/_stats.py (September 2026 revision).

    [A] synthetic checks with known answers
    [B] real data: reproduce the audited model-level numbers for
        q_proj stable rank vs Sep, post-block alignment, 9 decoders x 7 words
        (values from _audit/check_dependence.py)

Run:  python -m tests.test_stats
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import _stats as S  # noqa: E402
from src.analysis._io import FAMILIES, merge_geometry_sep  # noqa: E402

WORDS = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]
PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail and not condition else ''}")


def close(a, b, tol):
    return abs(a - b) <= tol


def test_synthetic():
    print("\n[A] synthetic checks")
    depth = np.arange(32, dtype=float)
    u = np.sin(depth * 1.7) + 0.3 * np.cos(depth * 0.9)

    x, y = 2 * depth + 3 + u, -depth + 1 + u
    check("linear trend removed exactly (partial r = 1)",
          close(S.partial_r(x, y, depth, 1), 1.0, 1e-9), f"{S.partial_r(x, y, depth, 1)}")
    x2, y2 = 0.5 * depth ** 2 - depth + u, -0.2 * depth ** 2 + 4 + u
    check("quadratic trend removed exactly (partial r = 1)",
          close(S.partial_r(x2, y2, depth, 2), 1.0, 1e-9), f"{S.partial_r(x2, y2, depth, 2)}")

    smooth = np.sin(depth / 5.0)
    n_eff = S.effective_n(smooth, smooth, 1)
    check("effective n below n for smooth series", n_eff < 32, f"n_eff={n_eff:.2f}")
    alt = np.array([(-1) ** i for i in range(32)], float) + 0.01 * depth
    check("effective n capped at n", S.effective_n(alt, alt * 1.0, 1) <= 32)
    check("effective n floored at degree + 3", S.effective_n(smooth, smooth, 2) >= 5)

    rx = S.residualise(u, depth, 1)
    obs, p, n = S.circular_shift_p(rx, [rx, rx])
    check("permutation: identical series gives r = 1", close(obs, 1.0, 1e-9))
    check("permutation: smallest attainable p is 1/n", p >= 1 / n - 1e-12, f"p={p}, 1/n={1 / n}")

    s = S.model_level_summary({f"m{i}": 0.1 * (i + 1) for i in range(9)})
    check("9/9 positive counted", s["n_positive"] == 9)
    check("Wilcoxon exact p for 9/9 positive = 2/512", close(s["wilcoxon_p"], 2 / 512, 1e-9),
          f"{s['wilcoxon_p']}")
    check("leave-one-out range brackets the mean", s["loo_min"] <= s["mean"] <= s["loo_max"])


def real_model_values(degree):
    means, perm = {}, {}
    for model in FAMILIES["llama"] + FAMILIES["qwen"]:
        rx, rys = None, []
        for w in WORDS:
            m = merge_geometry_sep(model, w, which="params")
            if m is None:
                raise SystemExit(f"missing data for {model}/{w}")
            rx = S.residualise(m["q_proj_stable_rank"], m["layer"], degree)
            rys.append(S.residualise(m["separation"], m["layer"], degree))
        obs, p, _ = S.circular_shift_p(rx, rys)
        means[model], perm[model] = obs, p
    return means, perm


def test_real():
    print("\n[B] real data: audited model-level numbers (post-block, q_proj stable rank)")
    lin, lin_perm = real_model_values(1)
    quad, quad_perm = real_model_values(2)
    L, Q = S.model_level_summary(lin), S.model_level_summary(quad)

    check("linear decoder mean +0.501", close(L["mean"], 0.501, 6e-4), f"{L['mean']:+.4f}")
    check("linear 9/9 positive", L["n_positive"] == 9, f"{L['n_positive']}/9")
    check("linear Wilcoxon p 0.0039", close(L["wilcoxon_p"], 0.0039, 6e-5), f"{L['wilcoxon_p']:.4f}")
    check("linear t-test p < 0.001", L["t_p"] < 0.001, f"{L['t_p']:.5f}")
    check("quadratic decoder mean +0.137", close(Q["mean"], 0.137, 6e-4), f"{Q['mean']:+.4f}")
    check("quadratic 7/9 positive", Q["n_positive"] == 7, f"{Q['n_positive']}/9")
    check("quadratic t-test p 0.0384", close(Q["t_p"], 0.0384, 6e-5), f"{Q['t_p']:.4f}")
    check("quadratic Wilcoxon p 0.0547", close(Q["wilcoxon_p"], 0.0547, 6e-5), f"{Q['wilcoxon_p']:.4f}")

    expected = [("llama-3-8b", lin, 0.629), ("qwen3-8b", lin, 0.179),
                ("llama-2-7b", quad, 0.368), ("qwen-7b", quad, -0.007)]
    for model, table, want in expected:
        check(f"{model} model mean {want:+.3f}", close(table[model], want, 6e-4), f"{table[model]:+.4f}")
    for model, table, want in [("llama-3-8b", lin_perm, 0.031), ("qwen2-7b", lin_perm, 0.143),
                               ("llama-3.1-8b", quad_perm, 0.125)]:
        check(f"{model} permutation p {want:.3f}", close(table[model], want, 6e-4), f"{table[model]:.4f}")

    print(f"\n  linear    LOO mean range {L['loo_min']:+.3f} (drop {L['loo_min_dropped']}) "
          f"to {L['loo_max']:+.3f} (drop {L['loo_max_dropped']})")
    print(f"  quadratic LOO mean range {Q['loo_min']:+.3f} (drop {Q['loo_min_dropped']}) "
          f"to {Q['loo_max']:+.3f} (drop {Q['loo_max_dropped']})")


if __name__ == "__main__":
    print("Statistics module tests")
    test_synthetic()
    test_real()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
