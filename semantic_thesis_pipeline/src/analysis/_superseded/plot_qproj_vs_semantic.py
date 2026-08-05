import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# ROOT
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
FAMILY_CONFIG = {
    "qwen": {
        "models": [
            "qwen-7b",
            "qwen1.5-7b",
            "qwen2-7b",
            "qwen2.5-7b",
            "qwen3-8b",
        ],
        "param_file": "qwen_family_parameter_geometry.csv",
    },
    "llama": {
        "models": [
            "llama-7b",
            "llama-2-7b",
            "llama-3-8b",
            "llama-3.1-8b",
        ],
        "param_file": "llama_family_parameter_geometry.csv",
    },
    "bert": {
        "models": [
            "bert-base",
            "roberta-base",
            "spanbert-base-cased",
            "xlm-roberta-base",
        ],
        "param_file": "bert_family_parameter_geometry.csv",
    },
}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
def load_data(family, model_name):
    base = PROJECT_ROOT / "results" / family / model_name

    sep = pd.read_csv(base / "metrics/semantic_separation.csv")

    param_all = pd.read_csv(
        PROJECT_ROOT / "results" / family / FAMILY_CONFIG[family]["param_file"]
    )

    param = param_all[param_all["model"] == model_name]

    return sep, param


# --------------------------------------------------
# PLOT ONE MODEL
# --------------------------------------------------
def plot_model(family, model_name, threshold, normalize):
    sep, param = load_data(family, model_name)

    merged = pd.merge(sep, param, on="layer")

    if merged.empty:
        print(f"[WARNING] Empty merge for {model_name}")
        return

    if "q_proj_spectral_norm" not in merged.columns:
        print(f"[WARNING] q_proj_spectral_norm missing for {model_name}")
        return

    # -----------------------------
    # RAW CORRELATION
    # -----------------------------
    x_raw = merged["q_proj_spectral_norm"]
    y_raw = merged["separation"]

    raw_corr = x_raw.corr(y_raw)

    # -----------------------------
    # FILTER IMPORTANT LAYERS
    # -----------------------------
    filtered = merged[merged["separation"] > threshold]

    if len(filtered) < 2:
        print(f"{model_name} | raw: {raw_corr:.4f} | NOT ENOUGH POINTS after filtering")
        return

    x = filtered["q_proj_spectral_norm"]
    y = filtered["separation"]

    # -----------------------------
    # OPTIONAL NORMALIZATION
    # -----------------------------
    if normalize:
        if x.std() != 0:
            x = (x - x.mean()) / x.std()
        if y.std() != 0:
            y = (y - y.mean()) / y.std()

    corr = x.corr(y)

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.scatter(x, y, alpha=0.7, label=model_name)

    print(f"{model_name} | raw: {raw_corr:.4f} | filtered: {corr:.4f}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--family",
        required=True,
        choices=["qwen", "llama", "bert"]
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--normalize",
        action="store_true"
    )

    args = parser.parse_args()

    models = FAMILY_CONFIG[args.family]["models"]

    plt.figure(figsize=(8, 6))

    for m in models:
        plot_model(
            args.family,
            m,
            threshold=args.threshold,
            normalize=args.normalize
        )

    plt.xlabel("q_proj spectral norm")
    plt.ylabel("semantic separation")
    plt.title(f"{args.family.upper()} (filtered): q_proj vs semantic")

    plt.legend()
    plt.grid()

    out_path = PROJECT_ROOT / f"results/{args.family}/{args.family}_qproj_vs_semantic.png"
    plt.savefig(out_path, dpi=300)

    print(f"\nSaved to: {out_path}")

    plt.close()


# --------------------------------------------------
if __name__ == "__main__":
    main()