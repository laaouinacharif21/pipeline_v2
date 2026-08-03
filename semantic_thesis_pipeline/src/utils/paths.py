from __future__ import annotations

from pathlib import Path
from typing import Dict


VALID_FAMILIES = {"llama", "bert", "qwen"}


def get_project_root() -> Path:
    """
    Return the root directory of the project.
    Assumes this file is located at:
    project_root/src/utils/paths.py
    """
    return Path(__file__).resolve().parents[2]


def infer_family(model_name: str) -> str:
    """
    Infer the model family from the model name.

    Examples:
        llama-7b              -> llama
        llama-3-8b            -> llama
        bert-base             -> bert
        roberta-base          -> bert
        deberta-v3-base       -> bert
        modernbert-base       -> bert
        xlm-roberta-base      -> bert
        gte-multilingual-base -> bert
        qwen-7b               -> qwen
    """
    model_name = model_name.strip().lower()

    if model_name.startswith("llama"):
        return "llama"

    if (
        model_name.startswith("bert")
        or model_name.startswith("roberta")
        or model_name.startswith("deberta")
        or model_name.startswith("modernbert")
        or model_name.startswith("xlm-roberta")
        or model_name.startswith("gte")
        or model_name.startswith("spanbert")
    ):
        return "bert"

    if model_name.startswith("qwen"):
        return "qwen"

    raise ValueError(
        f"Could not infer family from model name: '{model_name}'. "
        f"Expected name starting with one of: llama, bert, roberta, deberta, modernbert, xlm-roberta, gte, modernbert ,qwen."
    )


def get_results_root() -> Path:
    """Return the global results root directory."""
    return get_project_root() / "results"


def get_model_result_dir(model_name: str) -> Path:
    """
    Return the model-specific results directory:
    results/{family}/{model_name}/
    """
    family = infer_family(model_name)
    return get_results_root() / family / model_name


def get_model_paths(model_name: str) -> Dict[str, Path]:
    """
    Build all canonical output directories for a given model.
    """
    model_root = get_model_result_dir(model_name)

    paths = {
        "root": model_root,
        "embeddings": model_root / "embeddings",
        "metrics": model_root / "metrics",
        "plots": model_root / "plots",
        "pca": model_root / "plots" / "pca",
        "heatmaps": model_root / "plots" / "heatmaps",
        "trends": model_root / "plots" / "trends",
        "summaries": model_root / "summaries",
        "logs": model_root / "logs",
    }

    return paths


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_model_result_dirs(model_name: str) -> Dict[str, Path]:
    """
    Create all standard result directories for a given model.
    Returns the dictionary of created paths.
    """
    paths = get_model_paths(model_name)
    for path in paths.values():
        ensure_dir(path)
    return paths


def get_standard_result_files(model_name: str) -> Dict[str, Path]:
    """
    Return standard output file paths for a model.
    This keeps naming consistent across the project.
    """
    paths = get_model_paths(model_name)

    return {
        "embedding_tensor": paths["embeddings"] / "hidden_states.npz",
        "embedding_metadata": paths["embeddings"] / "embedding_metadata.json",
        "labels": paths["embeddings"] / "labels.json",
        "sentences": paths["embeddings"] / "sentences.json",
        "cosine_metrics_csv": paths["metrics"] / "cosine_layerwise.csv",
        "l2_metrics_csv": paths["metrics"] / "l2_layerwise.csv",
        "drift_metrics_csv": paths["metrics"] / "drift_layerwise.csv",
        "geometry_metrics_csv": paths["metrics"] / "geometry_layerwise.csv",
        "layer_ranking_csv": paths["summaries"] / "layer_ranking.csv",
        "top_layers_json": paths["summaries"] / "top_layers.json",
        "interpretation_summary_txt": paths["summaries"] / "interpretation_summary.txt",
        "run_log": paths["logs"] / "run.log",
        "run_summary_json": paths["summaries"] / "run_summary.json",
    }