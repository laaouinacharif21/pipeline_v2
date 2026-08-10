# -*- coding: utf-8 -*-
"""
Factual probing under spectral intervention.

The intervention was evaluated against word-sense separation and against
general capability. This stage applies it to the second semantic measure:
layer-wise linear decodability of truth value on TruthfulQA statements.

If reducing the spectral concentration of a projection leaves both semantic
separation and factual probe accuracy unchanged, the null generalises across
two unrelated semantic properties rather than resting on one.

The model is patched in memory, statements are extracted at the final
non-padding token, and a logistic regression is fitted per layer with folds
formed over questions so that the two answers to the same question cannot be
split between training and test.

Usage
    python -m src.analysis.intervention_probe --model llama-3-8b
    python -m src.analysis.intervention_probe --model llama-3-8b \
        --projection v_proj --ratios 1.0,0.80
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_word_config, resolve_dataset
from src.models.hf_loader import load_model_and_tokenizer
from src.analysis.intervention import patch_model
from src.pipeline.extract_embeddings_final_token import extract_final_token
from src.utils.paths import get_results_root, ensure_dir

SEED = 0


def probe_by_layer(emb, labels, groups):
    accs = []
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=SEED))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for l in range(emb.shape[1]):
        s = cross_val_score(clf, emb[:, l, :], labels, cv=cv,
                            groups=groups, scoring="accuracy")
        accs.append(float(s.mean()))
    return np.array(accs)


def run(model_name, word, projection, ratios, condition, batch_size):
    cfg = load_word_config(word)
    data = json.load(open(resolve_dataset(cfg), encoding="utf-8"))
    sentences = data["sentences"]
    labels = np.array(data["labels"])
    groups = np.array(data.get("groups", [i // 2 for i in range(len(sentences))]))

    rows, curves = [], []
    for ratio in ratios:
        print(f"\n  {condition}  {projection}  ratio {ratio:.2f}")
        tokenizer, model = load_model_and_tokenizer(model_name)[:2]
        model.eval()

        sr = 1.0
        if ratio < 1.0:
            stats = patch_model(model, projection, ratio, condition)
            sr = stats.after_stable_rank.mean() / stats.before_stable_rank.mean()
            print(f"    stable rank retained {sr:.1%}")

        emb, _ = extract_final_token(tokenizer, model, sentences,
                                     batch_size=batch_size)
        acc = probe_by_layer(emb, labels, groups)

        peak, at = float(acc.max()), int(acc.argmax())
        rows.append({"model": model_name, "projection": projection,
                     "condition": condition, "ratio": ratio,
                     "stable_rank_retained": sr,
                     "probe_peak": peak, "peak_layer": at,
                     "probe_mean": float(acc.mean()),
                     "probe_layer0": float(acc[0])})
        curves.append(pd.DataFrame({"ratio": ratio,
                                    "layer": np.arange(len(acc)),
                                    "probe_accuracy": acc}))
        print(f"    probe peak {peak:.3f} at layer {at}   mean {acc.mean():.3f}")

        del model, emb
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = ensure_dir(get_results_root() / "analysis" / "_intervention")
    tag = f"{model_name}_{word}_{projection}_{condition}"
    df.to_csv(out / f"probe_intervention_{tag}.csv", index=False)
    pd.concat(curves).to_csv(out / f"probe_curves_{tag}.csv", index=False)

    base = df.probe_peak.iloc[0]
    print(f"\n  {'ratio':>7}{'stable rank':>13}{'probe peak':>13}{'vs baseline':>13}")
    print("  " + "-" * 46)
    for _, r in df.iterrows():
        print(f"  {r.ratio:>7.2f}{r.stable_rank_retained:>12.0%}"
              f"{r.probe_peak:>13.3f}{r.probe_peak / base:>13.0%}")
    print(f"\n  Saved -> {out / f'probe_intervention_{tag}.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3-8b")
    ap.add_argument("--word", default="truthfulqa")
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--condition", default="truncate",
                    choices=["truncate", "rotate", "noise"])
    ap.add_argument("--ratios", default="1.0,0.80")
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    ratios = [float(x) for x in a.ratios.split(",")]
    print(f"\nProbe under intervention   {a.model}   {a.projection}   {a.condition}")
    run(a.model, a.word, a.projection, ratios, a.condition, a.batch_size)


if __name__ == "__main__":
    main()
