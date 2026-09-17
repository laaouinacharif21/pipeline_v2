# -*- coding: utf-8 -*-
"""
Reproducibility test for Table 4 (src/analysis/probe_geometry.py).

Checks the new script against values computed independently by
_audit/check_dependence.py (section 5):

    post  decoder mean  linear +0.635  quadratic +0.321   plus all 18 per-model values
    pre   decoder mean  linear +0.700  quadratic +0.500   (original table: llama-7b +0.840)
    the embedding layer is excluded under post (paired probe != layer-0 probe)

Run:  python -m tests.test_probe_geometry
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.probe_geometry import compute, summarise  # noqa: E402

PASS, FAIL = [], []
TOL = 6e-4

POST_LINEAR = {"llama-7b": 0.799, "llama-2-7b": 0.845, "llama-3-8b": 0.715, "llama-3.1-8b": 0.761,
               "qwen-7b": 0.703, "qwen1.5-7b": 0.371, "qwen2-7b": 0.572, "qwen2.5-7b": 0.643,
               "qwen3-8b": 0.302}
POST_QUAD = {"llama-7b": 0.766, "llama-2-7b": 0.723, "llama-3-8b": 0.398, "llama-3.1-8b": 0.516,
             "qwen-7b": 0.573, "qwen1.5-7b": 0.143, "qwen2-7b": -0.183, "qwen2.5-7b": -0.194,
             "qwen3-8b": 0.143}


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail and not condition else ''}")


def dec_mean(T, control):
    return float(T[(T.group == "decoders") & (T.control == control)]["mean"].iloc[0])


if __name__ == "__main__":
    print("Table 4 reproducibility tests")

    post = compute("post")
    pre = compute("pre")
    Tpost, Tpre = summarise(post), summarise(pre)

    print("\n[A] decoder means")
    for T, al, control, want in [(Tpost, "post", "linear", 0.635), (Tpost, "post", "quadratic", 0.321),
                                 (Tpre, "pre", "linear", 0.700), (Tpre, "pre", "quadratic", 0.500)]:
        got = dec_mean(T, control)
        check(f"{al} {control} mean {want:+.3f}", abs(got - want) <= TOL, f"got {got:+.4f}")

    print("\n[B] per-model values, post alignment")
    p = post.set_index("model")
    for table, col, label in [(POST_LINEAR, "r", "linear"), (POST_QUAD, "r_quad", "quadratic")]:
        for model, want in table.items():
            ok = model in p.index and abs(float(p.loc[model, col]) - want) <= TOL
            got = f"{float(p.loc[model, col]):+.4f}" if model in p.index else "missing"
            check(f"{model} {label} {want:+.3f}", ok, f"got {got}")

    print("\n[C] alignment sanity")
    q = pre.set_index("model")
    check("pre llama-7b linear +0.840 (original table)", abs(float(q.loc["llama-7b", "r"]) - 0.840) <= TOL,
          f"got {float(q.loc['llama-7b', 'r']):+.4f}")
    dec = post[post.arch == "decoder"]
    check("post excludes the embedding layer for every decoder",
          bool((dec.probe_first_paired != dec.probe_embedding).all()))
    check("layer counts match geometry blocks (28/32/36)",
          set(dec.n_layers) <= {28, 32, 36}, f"{sorted(set(dec.n_layers))}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
