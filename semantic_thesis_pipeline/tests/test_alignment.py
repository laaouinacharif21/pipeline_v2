# -*- coding: utf-8 -*-
"""
Layer-alignment regression tests (defect found September 2026).

Index conventions
    geometry     index l   = transformer block l            (n_blocks rows)
    separation   index 0   = embedding output               (n_blocks + 1 rows)
                 index l+1 = output of transformer block l

The original merge joined the two on equal indices, so block 0 was paired
with the embedding and the output of the last block was never used. The
primary analysis pairs block l with hidden state l+1 ("post"); the original
pairing is kept as the named option "pre".

    [A] pairing logic on synthetic data (no files needed)
    [B] merge_geometry_sep on real results, every model

Run:  python -m tests.test_alignment [--word bank]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import _io  # noqa: E402
from src.analysis._io import ALL_MODELS, load_erank, load_params, load_sep, merge_geometry_sep  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--word", default="bank")
WORD = _ap.parse_known_args()[0].word
PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail and not condition else ''}")


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


def synthetic(n_blocks=4):
    geo = pd.DataFrame({"layer": list(range(n_blocks)),
                        "q_proj_stable_rank": np.arange(n_blocks) * 10.0})
    sep = pd.DataFrame({"layer": list(range(n_blocks + 1)),
                        "separation": np.arange(n_blocks + 1) / 100.0})
    return geo, sep


def test_pairing_logic():
    print("\n[A] pairing logic on synthetic data")
    align = getattr(_io, "align_geometry_sep", None)
    if align is None:
        check("align_geometry_sep exists in _io.py", False, "not implemented")
        return
    geo, sep = synthetic(4)

    post = align(geo, sep)
    check("default alignment is post",
          post["hidden_layer"].tolist() == [1, 2, 3, 4], f"hidden_layer={post['hidden_layer'].tolist()}")
    first = float(post.loc[post["layer"] == 0, "separation"].iloc[0])
    check("block 0 paired with hidden state 1", abs(first - 0.01) < 1e-12, f"got {first}")
    check("embedding excluded under post", 0 not in post["hidden_layer"].tolist())
    check("all blocks retained, layer is the block index",
          post["layer"].tolist() == [0, 1, 2, 3], f"layer={post['layer'].tolist()}")
    check("geometry values unchanged",
          post["q_proj_stable_rank"].tolist() == [0.0, 10.0, 20.0, 30.0])

    pre = align(geo, sep, alignment="pre")
    check("pre pairs block l with hidden state l",
          pre["hidden_layer"].tolist() == [0, 1, 2, 3], f"hidden_layer={pre['hidden_layer'].tolist()}")

    check("unknown alignment rejected",
          raises_value_error(lambda: align(geo, sep, alignment="middle")))
    check("equal row counts rejected (would hide the off-by-one)",
          raises_value_error(lambda: align(geo, sep.iloc[:4].reset_index(drop=True))))
    check("non-contiguous layer indices rejected",
          raises_value_error(lambda: align(geo, sep.assign(layer=[0, 1, 2, 3, 5]))))


def test_real_merge():
    print(f"\n[B] merge_geometry_sep on real results (word='{WORD}')")
    for m in ALL_MODELS:
        sep = load_sep(m, WORD)
        if sep is None:
            continue
        sep1 = float(sep.loc[sep["layer"] == 1, "separation"].iloc[0])
        for which, loader in (("params", load_params), ("erank", load_erank)):
            geo = loader(m)
            if geo is None:
                continue
            name = f"{m} [{which}]: block 0 <-> Sep(1), {len(geo)} blocks"
            try:
                merged = merge_geometry_sep(m, WORD, which=which)
            except ValueError as e:
                check(name, False, f"ValueError: {e}")
                continue
            if merged is None:
                check(name, False, "merge returned None")
                continue
            merged = merged.sort_values("layer")
            first = float(merged["separation"].iloc[0])
            ok = len(merged) == len(geo) and abs(first - sep1) < 1e-12
            check(name, ok, f"rows {len(merged)}/{len(geo)}, first Sep {first:.6g}, Sep(1) {sep1:.6g}")


if __name__ == "__main__":
    print(f"Layer-alignment regression tests  (word='{WORD}')")
    test_pairing_logic()
    test_real_merge()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)
