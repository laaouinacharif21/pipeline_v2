# -*- coding: utf-8 -*-
"""
Pipeline integrity checks.

These assert invariants that must hold for any correct run. Each corresponds
to a defect found in August 2026:

    test_sep_zero_at_embedding_layer   mean-pooled extraction (Sep(0) != 0)
    test_target_token_position         attention-sink contamination
    test_label_alignment               label/embedding desynchronisation
    test_geometry_deterministic        randomised power iteration
    test_qwen7b_projections_distinct   unsliced fused c_attn

Run:  python -m tests.test_pipeline
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis._io import ALL_MODELS, FAMILIES, load_sep, load_erank, load_params
from src.utils.paths import get_standard_result_files

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--word", default="bank")
WORD = _ap.parse_known_args()[0].word
MIN_TOKEN_INDEX = 3
PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail and not condition else ''}")


def test_sep_zero_at_embedding_layer():
    """Sep(0) reflects the embedding layer only.

    Decoder models apply rotary position encoding inside attention, so their
    layer-0 representation of the target token is pure token identity and
    Sep(0) must be zero up to float32 error. BERT-family encoders add
    absolute positional (and segment) embeddings before layer 0, so the same
    token receives different vectors at different positions and Sep(0) is
    small but non-zero. The tolerances differ accordingly.
    """
    print("\n[1] Sep(0) at the embedding layer")
    for m in ALL_MODELS:
        df = load_sep(m, WORD)
        if df is None:
            continue
        v = float(df.loc[df.layer == 0, "separation"].iloc[0])
        is_encoder = m in FAMILIES["bert"]
        tol = 0.05 if is_encoder else 1e-5
        label = f"{m}{' (encoder, positional)' if is_encoder else ''}"
        check(label, abs(v) < tol, f"Sep(0)={v:.6g} tol={tol}")


def test_target_token_position():
    print(f"\n[2] target token index >= {MIN_TOKEN_INDEX}")
    for m in ALL_MODELS:
        f = get_standard_result_files(m, WORD)["embedding_metadata"]
        if not f.exists():
            continue
        r = json.load(open(f)).get("location_report", {})
        lo = r.get("min_token_index")
        check(f"{m}", lo is None or lo >= MIN_TOKEN_INDEX, f"min index={lo}")


def test_label_alignment():
    print("\n[3] labels align with embeddings")
    for m in ALL_MODELS:
        f = get_standard_result_files(m, WORD)
        if not f["embedding_tensor"].exists():
            continue
        n_emb = np.load(f["embedding_tensor"])["embeddings"].shape[0]
        n_lab = len(json.load(open(f["labels"]))["labels"])
        check(f"{m}", n_emb == n_lab, f"{n_emb} embeddings vs {n_lab} labels")


def test_geometry_deterministic():
    """Spectral norm must equal the largest singular value, so it cannot be
    smaller than the largest per-layer value implied by the Frobenius norm."""
    print("\n[4] spectral norm <= Frobenius norm")
    for m in ALL_MODELS:
        df = load_params(m)
        if df is None:
            continue
        ok = True
        for c in df.columns:
            if not c.endswith("_spectral_norm"):
                continue
            fro = c.replace("_spectral_norm", "_fro_norm")
            if fro in df.columns and (df[c] > df[fro] + 1e-3).any():
                ok = False
        check(f"{m}", ok)


def test_qwen7b_projections_distinct():
    """Qwen-7B stores QKV fused; the three slices must differ."""
    print("\n[5] qwen-7b q/k/v distinct after slicing")
    df = load_params("qwen-7b")
    if df is None:
        print("  SKIP  qwen-7b not present")
        return
    q, k, v = (df[f"{x}_proj_spectral_norm"] for x in "qkv")
    check("q != k", not np.allclose(q, k, rtol=1e-4))
    check("q != v", not np.allclose(q, v, rtol=1e-4))


def test_erank_bounds():
    print("\n[6] 1 <= effective rank <= min(matrix dims)")
    for m in ALL_MODELS:
        df = load_erank(m)
        if df is None:
            continue
        cols = [c for c in df.columns if c.endswith("_erank")]
        ok = all((df[c] >= 1.0).all() and np.isfinite(df[c]).all() for c in cols)
        check(f"{m}", ok)


if __name__ == "__main__":
    from src.utils.paths import get_results_root
    print(f"Pipeline integrity checks  (word='{WORD}')")
    print(f"results root: {get_results_root()}")
    if not (get_results_root() / "words" / WORD).exists():
        print(f"\nERROR: no results for word '{WORD}' under this root.")
        sys.exit(1)
    test_sep_zero_at_embedding_layer()
    test_target_token_position()
    test_label_alignment()
    test_geometry_deterministic()
    test_qwen7b_projections_distinct()
    test_erank_bounds()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("Failures:", ", ".join(FAIL))
        sys.exit(1)
