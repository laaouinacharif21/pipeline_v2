# -*- coding: utf-8 -*-
"""
Layer-wise linear probe accuracy.

Semantic separation Sep(l) measures whether the two sense classes form
distinguishable clusters in cosine space. A linear probe asks a different
question: whether the sense is linearly decodable from the representation at
that layer. The two can diverge, so both are reported.

For every layer, a logistic regression is fitted on the target token's hidden
state under stratified cross-validation. Where the dataset is built as pairs
that share a frame, folds are formed over pair index rather than over
individual sentences, so that a near-copy of a test item cannot appear in
training.

Accuracy per layer is written alongside semantic_separation.csv and can be
correlated against projection geometry in the same way.

Usage
    python -m src.analysis.layer_probe --word bank --model llama-7b
    python -m src.analysis.layer_probe --word bank --all-models
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import get_model_result_dir, get_standard_result_files

SEED = 0

ALL_MODELS = [
    "llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b",
    "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b",
    "bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base",
]


def probe(X, y, groups, n_splits=5):
    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    if k < 2 or len(set(y)) < 2:
        return float("nan"), float("nan")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=SEED))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    s = cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="accuracy")
    return float(s.mean()), float(s.std())


def run_model(model_name, word, paired=True):
    files = get_standard_result_files(model_name, word)
    if not files["embedding_tensor"].exists():
        print(f"  [skip] {model_name}: no embeddings for '{word}'")
        return None

    emb = np.load(files["embedding_tensor"])["embeddings"]
    labels = np.array(json.load(open(files["labels"]))["labels"])
    n, n_layers, _ = emb.shape

    # Sentences are emitted one per sense in sequence, so consecutive pairs
    # share a frame. Without pairing, each sentence is its own group.
    groups = np.array([i // 2 for i in range(n)]) if paired else np.arange(n)

    rows = []
    for l in range(n_layers):
        acc, sd = probe(emb[:, l, :], labels, groups)
        rows.append({"model": model_name, "word": word, "layer": l,
                     "probe_accuracy": acc, "probe_sd": sd, "n": n})

    df = pd.DataFrame(rows)
    out = get_model_result_dir(model_name, word) / "metrics" / "probe_accuracy.csv"
    df.to_csv(out, index=False)
    peak = df.loc[df.probe_accuracy.idxmax()]
    print(f"  {model_name:<22} peak {peak.probe_accuracy:.3f} at layer "
          f"{int(peak.layer)}/{n_layers - 1}   layer0 {df.probe_accuracy.iloc[0]:.3f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="bank")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model")
    g.add_argument("--all-models", action="store_true")
    ap.add_argument("--unpaired", action="store_true",
                    help="Treat every sentence as its own group")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    models = [m for m in ALL_MODELS if m not in skip] if a.all_models else [a.model]

    print(f"\nLAYER-WISE PROBE  --  '{a.word}'"
          f"   folds grouped by {'sentence' if a.unpaired else 'pair'}\n")
    frames = [f for f in (run_model(m, a.word, not a.unpaired) for m in models)
              if f is not None]
    if not frames:
        raise SystemExit("no results")

    df = pd.concat(frames, ignore_index=True)

    # Correlation with Sep(l), where available
    print("\n  probe accuracy against Sep(l), per model")
    print(f"    {'model':<22}{'pearson r':>11}{'p':>10}")
    print("    " + "-" * 43)
    from scipy.stats import pearsonr
    for m in df.model.unique():
        sep_p = get_model_result_dir(m, a.word) / "metrics" / "semantic_separation.csv"
        if not sep_p.exists():
            continue
        sep = pd.read_csv(sep_p)
        d = df[df.model == m].merge(sep, on="layer").dropna()
        if len(d) < 5:
            continue
        r, p = pearsonr(d.probe_accuracy, d.separation)
        print(f"    {m:<22}{r:>+11.3f}{p:>10.4f}")


if __name__ == "__main__":
    main()
