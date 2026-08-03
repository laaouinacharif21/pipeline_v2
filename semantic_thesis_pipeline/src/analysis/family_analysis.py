import json
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def get_family_models(family):
    if family == "llama":
        return [
            "llama-7b",
            "llama-2-7b",
            "llama-3-8b",
            "llama-3.1-8b",
        ]
    elif family == "qwen":
        return [
            "qwen-7b",
            "qwen1.5-7b",
            "qwen2-7b",
            "qwen2.5-7b",
            "qwen3-8b",
        ]
    elif family == "bert":
        return [
            "bert-base",
            "roberta-base",
            "spanbert-base-cased",
            "xlm-roberta-base",
        ]
    else:
        raise ValueError(f"Unknown family: {family}")


def load_metric_csv(results_root: Path, model_name: str, filename: str) -> pd.DataFrame:
    path = results_root / model_name / "metrics" / filename
    return pd.read_csv(path)


def load_top_layers(results_root: Path, model_name: str):
    path = results_root / model_name / "summaries" / "top_layers.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["top_layers"]


def plot_metric(results_root: Path, models, filename: str, value_col: str, title: str, output_name: str):
    plt.figure()

    for model_name in models:
        df = load_metric_csv(results_root, model_name, filename)
        plt.plot(df["layer"], df[value_col], label=model_name)

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_root / output_name)
    plt.close()


def save_top_layers_summary(results_root: Path, models, family):
    rows = []

    for model_name in models:
        top_layers = load_top_layers(results_root, model_name)
        rows.append({
            "model": model_name,
            "top_1": top_layers[0],
            "top_2": top_layers[1],
            "top_3": top_layers[2],
            "top_4": top_layers[3],
        })

    df = pd.DataFrame(rows)
    df.to_csv(results_root / f"{family}_top_layers_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, required=True)
    args = parser.parse_args()

    family = args.family
    models = get_family_models(family)

    results_root = Path(f"results/{family}")

    plot_metric(
        results_root,
        models,
        "cosine_layerwise.csv",
        "avg_cosine",
        f"{family.upper()} Family - Average Cosine Across Layers",
        f"{family}_family_cosine.png",
    )

    plot_metric(
        results_root,
        models,
        "l2_layerwise.csv",
        "avg_l2",
        f"{family.upper()} Family - Average L2 Across Layers",
        f"{family}_family_l2.png",
    )

    plot_metric(
        results_root,
        models,
        "layer_drift.csv",
        "drift_to_next",
        f"{family.upper()} Family - Layer Drift",
        f"{family}_family_drift.png",
    )

    plot_metric(
        results_root,
        models,
        "semantic_separation.csv",
        "separation",
        f"{family.upper()} Family - Semantic Separation Across Layers",
        f"{family}_family_semantic_separation.png",
    )

    save_top_layers_summary(results_root, models, family)

    print(f"{family.upper()} family analysis saved in results/{family}/")


if __name__ == "__main__":
    main()