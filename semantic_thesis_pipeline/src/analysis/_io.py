"""
Shared I/O for analysis scripts.

Centralises the layout so that per-word results and word-independent
parameter geometry are read consistently:

    results/parameters/{family}/{model}/{effective_rank,parameter_stats}.csv
    results/words/{word}/{family}/{model}/metrics/semantic_separation.csv
    results/analysis/{word}/...
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import (
    get_results_root, get_parameters_dir, get_model_result_dir,
    get_analysis_dir, infer_family, ensure_dir,
)

FAMILIES = {
    "llama": ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"],
    "qwen": ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"],
    "bert": ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"],
}

ALL_MODELS = FAMILIES["llama"] + FAMILIES["qwen"] + FAMILIES["bert"]
DECODERS = FAMILIES["llama"] + FAMILIES["qwen"]
# Primary projections, pre-specified before analysis.
#   q_proj, k_proj  attention scores are formed as Q K^T, so the two are
#                   analysed together rather than one in isolation
#   v_proj          attention value path
#   up_proj         feed-forward expansion, present in every architecture
PRIMARY_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "up_proj"]

# Supplementary, reported without primary claims. gate_proj is absent from
# BERT-family encoders, so it cannot enter the cross-architecture comparison.
SUPPLEMENTARY_PROJECTIONS = ["o_proj", "gate_proj", "down_proj"]

ALL_PROJECTIONS = PRIMARY_PROJECTIONS + SUPPLEMENTARY_PROJECTIONS

# Backwards-compatible default
PROJECTIONS = PRIMARY_PROJECTIONS


def sep_path(model: str, word: str) -> Path:
    return get_model_result_dir(model, word) / "metrics" / "semantic_separation.csv"


def load_sep(model: str, word: str):
    p = sep_path(model, word)
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("layer").reset_index(drop=True)


def load_erank(model: str):
    p = get_parameters_dir(model) / "effective_rank.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("layer").reset_index(drop=True)


def load_params(model: str):
    p = get_parameters_dir(model) / "parameter_stats.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).sort_values("layer").reset_index(drop=True)


def merge_geometry_sep(model: str, word: str, which: str = "erank"):
    """Merge geometry with Sep(l) on shared layer indices."""
    geo = load_erank(model) if which == "erank" else load_params(model)
    sep = load_sep(model, word)
    if geo is None or sep is None:
        return None
    m = geo.merge(sep, on="layer")
    return m if len(m) >= 5 else None


def analysis_dir(word: str, subdir: str = "") -> Path:
    d = get_analysis_dir() / word
    if subdir:
        d = d / subdir
    return ensure_dir(d)


def analysis_plot_dir(word: str, subdir: str = "") -> Path:
    return analysis_dir(word, "plots" if not subdir else f"plots/{subdir}")


def star(p) -> str:
    if p != p:
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def available(word: str) -> list:
    """Models that have Sep(l) for this word."""
    return [m for m in ALL_MODELS if sep_path(m, word).exists()]
