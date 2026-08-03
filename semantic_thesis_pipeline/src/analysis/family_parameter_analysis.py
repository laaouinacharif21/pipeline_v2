import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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


def load_param_csv(family, model_name):
    path = Path("results") / family / model_name / "parameters" / "parameter_stats.csv"
    return pd.read_csv(path)


def plot_metric(df, models, metric_name, output_path, title):
    if metric_name not in df.columns:
        print(f"Skipping {metric_name} (not found)")
        return

    plt.figure()

    for model_name in models:
        sub = df[df["model"] == model_name]
        if metric_name in sub.columns:
            plt.plot(sub["layer"], sub[metric_name], label=model_name)

    plt.xlabel("Layer")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, required=True)
    args = parser.parse_args()

    family = args.family
    models = get_family_models(family)

    all_dfs = []

    for model_name in models:
        print(f"Loading {model_name}")
        df_model = load_param_csv(family, model_name)
        df_model["model"] = model_name
        all_dfs.append(df_model)

    df = pd.concat(all_dfs, ignore_index=True)

    out_dir = Path("results") / family
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / f"{family}_family_parameter_geometry.csv", index=False)

    plot_metric(
        df,
        models,
        "q_proj_spectral_norm",
        out_dir / f"{family}_family_qproj_spectral.png",
        f"{family.upper()} - q_proj Spectral Norm",
    )

    plot_metric(
        df,
        models,
        "v_proj_spectral_norm",
        out_dir / f"{family}_family_vproj_spectral.png",
        f"{family.upper()} - v_proj Spectral Norm",
    )

    plot_metric(
        df,
        models,
        "up_proj_spectral_norm",
        out_dir / f"{family}_family_upproj_spectral.png",
        f"{family.upper()} - up_proj Spectral Norm",
    )

    print(f"Saved {family} parameter analysis")


if __name__ == "__main__":
    main()