# -*- coding: utf-8 -*-
"""
effective_rank_analysis.py
--------------------------
Computes the effective rank of projection matrices (q_proj, v_proj, up_proj)
across all transformer layers for all model families.

Effective rank (Roy & Vetterli, 2007) is defined as:
    erank(A) = exp(H(p))
where p_i = sigma_i / sum(sigma) are the normalised singular values and
H(p) = -sum(p_i * log(p_i)) is the Shannon entropy of that distribution.

Interpretation:
    - A matrix with erank = k behaves as if it has k equally important dimensions
    - High erank = information spread across many dimensions (distributed)
    - Low erank = information concentrated in few dimensions (low-rank structure)

This metric goes beyond spectral norm (top singular value only) and Frobenius
norm (sum of all squared singular values) to capture the full singular value
distribution -- directly justifying the word "geometric" in the thesis title.

Output:
    - Per-model CSV: results/{family}/{model}/parameters/effective_rank.csv
    - Family-level plots: results/plots/effective_rank/
    - Summary CSV: results/effective_rank/effective_rank_summary.csv

Usage:
    python -m src.analysis.effective_rank_analysis --family llama
    python -m src.analysis.effective_rank_analysis --family qwen
    python -m src.analysis.effective_rank_analysis --family bert
    python -m src.analysis.effective_rank_analysis --all
"""

import sys
# Stub out transformers_stream_generator which breaks on newer transformers versions
# (imports DisjunctiveConstraint which was removed). Required for qwen-7b loading.
if "transformers_stream_generator" not in sys.modules:
    import types
    sys.modules["transformers_stream_generator"] = types.ModuleType("transformers_stream_generator")

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr

# -- project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import get_results_root, infer_family
from src.models.hf_loader import load_model_and_tokenizer

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "axes.spines.top": False,
    "figure.dpi":      150,
})

FAMILIES = {
    "llama": ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"],
    "qwen":  ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b"],
    "bert":  ["bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base"],
}

PROJECTIONS = ["q_proj", "v_proj", "up_proj"]

RESULTS_ROOT = get_results_root()
ERANK_PLOT_DIR = RESULTS_ROOT / "plots" / "effective_rank"
ERANK_SUMMARY_DIR = RESULTS_ROOT / "effective_rank"
ERANK_PLOT_DIR.mkdir(parents=True, exist_ok=True)
ERANK_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# -- core math -----------------------------------------------------------------

def effective_rank(matrix: torch.Tensor) -> float:
    """
    Compute effective rank of a 2D matrix using singular value entropy.
    erank(A) = exp(-sum(p_i * log(p_i)))  where p_i = sigma_i / sum(sigma)
    """
    with torch.no_grad():
        # Use float32 for SVD stability
        mat = matrix.float()
        try:
            singular_values = torch.linalg.svdvals(mat).cpu().numpy()
        except Exception:
            # fallback: numpy SVD
            singular_values = np.linalg.svd(mat.cpu().numpy(), compute_uv=False)

        singular_values = singular_values[singular_values > 1e-10]  # remove near-zero
        if len(singular_values) == 0:
            return 1.0

        p = singular_values / singular_values.sum()
        # Shannon entropy in nats
        entropy = -np.sum(p * np.log(p + 1e-12))
        return float(np.exp(entropy))


def get_projection_matrix(model, layer_idx: int, proj_name: str, model_name: str,
                           _visited: set = None):
    """
    Extract a projection weight matrix from a transformer layer.
    Handles both decoder (LLaMA/Qwen) and encoder (BERT) architectures.
    """
    if _visited is None:
        _visited = set()
    if id(model) in _visited:
        return None
    _visited.add(id(model))

    try:
        # -- Decoder models (LLaMA, Qwen) via CausalLM wrapper ----------------
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            mlp  = layer.mlp

            if proj_name == "q_proj" and hasattr(attn, "q_proj"):
                return attn.q_proj.weight.data
            if proj_name == "v_proj" and hasattr(attn, "v_proj"):
                return attn.v_proj.weight.data
            if proj_name == "up_proj" and hasattr(mlp, "up_proj"):
                return mlp.up_proj.weight.data
            # Qwen fused attention fallback
            if proj_name == "q_proj" and hasattr(attn, "c_attn"):
                W = attn.c_attn.weight.data
                dim = W.shape[0] // 3
                return W[:dim, :]

        # -- Decoder models loaded via AutoModel (llama-7b, llama-2-7b) -------
        if hasattr(model, "layers") and not hasattr(model, "encoder"):
            layer = model.layers[layer_idx]
            attn = layer.self_attn
            mlp  = layer.mlp

            if proj_name == "q_proj" and hasattr(attn, "q_proj"):
                return attn.q_proj.weight.data
            if proj_name == "v_proj" and hasattr(attn, "v_proj"):
                return attn.v_proj.weight.data
            if proj_name == "up_proj" and hasattr(mlp, "up_proj"):
                return mlp.up_proj.weight.data

        # -- Encoder models (BERT family) --------------------------------------
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layer = model.encoder.layer[layer_idx]
            attn  = layer.attention.self

            if proj_name == "q_proj" and hasattr(attn, "query"):
                return attn.query.weight.data
            if proj_name == "v_proj" and hasattr(attn, "value"):
                return attn.value.weight.data
            if proj_name == "up_proj":
                # BERT uses intermediate dense as the up-projection equivalent
                if hasattr(layer, "intermediate") and hasattr(layer.intermediate, "dense"):
                    return layer.intermediate.dense.weight.data

        # -- QWen-7b GPT-style (transformer.h) --------------------------------
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layer = model.transformer.h[layer_idx]
            attn = layer.attn
            mlp  = layer.mlp

            if proj_name == "q_proj":
                if hasattr(attn, "c_attn"):
                    W = attn.c_attn.weight.data  # [3*hidden, hidden]
                    dim = W.shape[0] // 3
                    return W[:dim, :]            # Q slice
            if proj_name == "v_proj":
                if hasattr(attn, "c_attn"):
                    W = attn.c_attn.weight.data
                    dim = W.shape[0] // 3
                    return W[2*dim:, :]          # V slice
            if proj_name == "up_proj":
                if hasattr(mlp, "w2"):
                    return mlp.w2.weight.data    # [11008, 4096]

        # -- BaseModel wrapper (last resort) -----------------------------------
        if hasattr(model, "base_model"):
            return get_projection_matrix(model.base_model, layer_idx, proj_name,
                                         model_name, _visited)

    except (IndexError, AttributeError) as e:
        pass

    return None


def count_layers(model, _visited: set = None) -> int:
    """Return the number of transformer layers in the model."""
    if _visited is None:
        _visited = set()
    if id(model) in _visited:
        return 0
    _visited.add(id(model))

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):                          # AutoModel direct (llama-7b, llama-2-7b)
        return len(model.layers)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return len(model.encoder.layer)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # qwen-7b GPT-style
        return len(model.transformer.h)
    if hasattr(model, "base_model"):
        return count_layers(model.base_model, _visited)
    return 0


# -- per-model analysis --------------------------------------------------------

def analyse_model(model_name: str, device: str = "cuda") -> pd.DataFrame | None:
    """Load model, compute effective rank per layer per projection, return DataFrame."""
    family = infer_family(model_name)
    out_path = RESULTS_ROOT / family / model_name / "parameters" / "effective_rank.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  [CACHED] {model_name} -- loading from {out_path}")
        return pd.read_csv(out_path)

    print(f"\n  Loading model: {model_name} ...")
    try:
        result = load_model_and_tokenizer(model_name, device=device)
        # handle both (model, tokenizer) and model-only returns
        if isinstance(result, tuple):
            model = result[0]
        else:
            model = result
        model.eval()
    except Exception as e:
        print(f"  [ERROR] Could not load {model_name}: {e}")
        return None

    n_layers = count_layers(model)
    if n_layers == 0:
        print(f"  [ERROR] Could not determine layer count for {model_name}")
        return None

    print(f"  Layers: {n_layers}  |  Computing effective rank ...")

    records = []
    for layer_idx in range(n_layers):
        row = {"layer": layer_idx}
        for proj in PROJECTIONS:
            W = get_projection_matrix(model, layer_idx, proj, model_name)
            if W is not None:
                er = effective_rank(W)
                row[f"{proj}_erank"] = er
            else:
                row[f"{proj}_erank"] = float("nan")
        records.append(row)
        if (layer_idx + 1) % 8 == 0:
            print(f"    Layer {layer_idx + 1}/{n_layers} done")

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    # free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return df


# -- correlation with semantic separation --------------------------------------

def correlate_with_separation(model_name: str, erank_df: pd.DataFrame) -> dict:
    """Compute Pearson r between effective rank and semantic separation per projection."""
    family = infer_family(model_name)
    sep_path = RESULTS_ROOT / family / model_name / "metrics" / "semantic_separation.csv"

    if not sep_path.exists():
        return {}

    sep_df = pd.read_csv(sep_path).sort_values("layer").reset_index(drop=True)
    erank_df = erank_df.sort_values("layer").reset_index(drop=True)

    shared = set(sep_df["layer"]).intersection(set(erank_df["layer"]))
    sep_df   = sep_df[sep_df["layer"].isin(shared)].reset_index(drop=True)
    erank_df = erank_df[erank_df["layer"].isin(shared)].reset_index(drop=True)

    sep = sep_df["separation"].values
    corrs = {}
    for proj in PROJECTIONS:
        col = f"{proj}_erank"
        if col in erank_df.columns and not erank_df[col].isna().all():
            er = erank_df[col].values
            r, p = pearsonr(er, sep)
            corrs[proj] = {"r": float(r), "p": float(p)}
    return corrs


# -- plotting ------------------------------------------------------------------

def plot_family_erank(family: str, models: list, all_dfs: dict):
    """Plot effective rank across layers for each projection in a family."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for proj in PROJECTIONS:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, model_name in enumerate(models):
            df = all_dfs.get(model_name)
            if df is None:
                continue
            col = f"{proj}_erank"
            if col not in df.columns:
                continue
            ax.plot(df["layer"], df[col],
                    label=model_name, color=colors[i % len(colors)],
                    linewidth=2, marker="o", markersize=4)

        ax.set_title(f"{family.upper()} Family -- Effective Rank of {proj}", fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Effective Rank")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        plt.tight_layout()

        fname = ERANK_PLOT_DIR / f"{family}_erank_{proj}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot -> {fname}")


def plot_erank_vs_separation(family: str, models: list, all_dfs: dict):
    """Scatter plot: effective rank vs semantic separation per layer."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for proj in PROJECTIONS:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, model_name in enumerate(models):
            df = all_dfs.get(model_name)
            if df is None:
                continue
            family_name = infer_family(model_name)
            sep_path = RESULTS_ROOT / family_name / model_name / "metrics" / "semantic_separation.csv"
            if not sep_path.exists():
                continue
            sep_df = pd.read_csv(sep_path)
            merged = df.merge(sep_df, on="layer")
            col = f"{proj}_erank"
            if col not in merged.columns:
                continue
            ax.scatter(merged[col], merged["separation"],
                       label=model_name, color=colors[i % len(colors)],
                       alpha=0.7, s=40)

        ax.set_title(f"{family.upper()} -- Effective Rank vs Semantic Separation ({proj})",
                     fontweight="bold")
        ax.set_xlabel(f"Effective Rank ({proj})")
        ax.set_ylabel("Semantic Separation")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        plt.tight_layout()

        fname = ERANK_PLOT_DIR / f"{family}_erank_vs_separation_{proj}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot -> {fname}")


# -- main ----------------------------------------------------------------------

def run_family(family: str, device: str = "cuda"):
    models = FAMILIES[family]
    print(f"\n{'='*60}")
    print(f"Family: {family.upper()}")
    print(f"{'='*60}")

    all_dfs = {}
    summary_rows = []

    for model_name in models:
        print(f"\nModel: {model_name}")
        df = analyse_model(model_name, device=device)
        if df is None:
            continue
        all_dfs[model_name] = df

        # correlation with semantic separation
        corrs = correlate_with_separation(model_name, df)

        print(f"  Correlation (effective rank vs semantic separation):")
        for proj, vals in corrs.items():
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else "n.s."
            print(f"    {proj:12s}  r={vals['r']:+.3f}  p={vals['p']:.4f}  {sig}")

            summary_rows.append({
                "family":  family,
                "model":   model_name,
                "proj":    proj,
                "erank_sep_r": vals["r"],
                "erank_sep_p": vals["p"],
                "mean_erank":  float(df[f"{proj}_erank"].mean()) if f"{proj}_erank" in df else float("nan"),
                "max_erank":   float(df[f"{proj}_erank"].max())  if f"{proj}_erank" in df else float("nan"),
            })

    # plots
    if all_dfs:
        print(f"\n  Generating plots for {family} ...")
        plot_family_erank(family, models, all_dfs)
        plot_erank_vs_separation(family, models, all_dfs)

    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Compute effective rank of projection matrices")
    parser.add_argument("--family", type=str, choices=["llama", "qwen", "bert"],
                        help="Model family to analyse")
    parser.add_argument("--all", action="store_true", help="Run all families")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if not args.all and not args.family:
        parser.error("Specify --family <name> or --all")

    families_to_run = list(FAMILIES.keys()) if args.all else [args.family]

    all_summary = []
    for family in families_to_run:
        rows = run_family(family, device=args.device)
        all_summary.extend(rows)

    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        out_csv = ERANK_SUMMARY_DIR / "effective_rank_summary.csv"
        summary_df.to_csv(out_csv, index=False)
        print(f"\nSaved summary -> {out_csv}")

        print("\n" + "="*70)
        print("SUMMARY: Effective Rank vs Semantic Separation Correlations")
        print("="*70)
        print(f"{'Model':<22} {'Proj':<12} {'r':>8} {'p':>8} {'Mean erank':>12} {'Sig':>8}")
        print("-"*70)
        for _, row in summary_df.iterrows():
            sig = "***" if row["erank_sep_p"] < 0.001 else "**" if row["erank_sep_p"] < 0.01 \
                  else "*" if row["erank_sep_p"] < 0.05 else "n.s."
            print(f"{row['model']:<22} {row['proj']:<12} {row['erank_sep_r']:>+8.3f} "
                  f"{row['erank_sep_p']:>8.4f} {row['mean_erank']:>12.1f} {sig:>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()