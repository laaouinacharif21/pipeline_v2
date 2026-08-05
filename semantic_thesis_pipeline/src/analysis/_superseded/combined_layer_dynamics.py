import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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


def load_metrics(family, model_name):
    base = Path("results") / family / model_name

    drift = pd.read_csv(base / "metrics" / "layer_drift.csv")
    sep = pd.read_csv(base / "metrics" / "semantic_separation.csv")

    return drift, sep


def load_param(family, model_name):
    path = Path("results") / family / model_name / "parameters" / "parameter_stats.csv"
    return pd.read_csv(path)


def plot_combined(family, model_name):
    drift, sep = load_metrics(family, model_name)
    param = load_param(family, model_name)

    plt.figure(figsize=(8, 5))

    if "up_proj_spectral_norm" in param.columns:
        plt.plot(
            param["layer"],
            param["up_proj_spectral_norm"],
            label="MLP spectral norm",
        )

    plt.plot(
        drift["layer"],
        drift["drift_to_next"],
        label="representation drift",
    )

    plt.plot(
        sep["layer"],
        sep["separation"],
        label="semantic separation",
    )

    plt.xlabel("Layer")
    plt.ylabel("Value")
    plt.title(f"{model_name} Layer Dynamics")

    plt.legend()
    plt.tight_layout()

    out = Path("results") / family / f"{model_name}_combined_dynamics.png"
    plt.savefig(out)
    plt.close()

    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, required=True)
    args = parser.parse_args()

    models = get_family_models(args.family)

    for m in models:
        plot_combined(args.family, m)


if __name__ == "__main__":
    main()