import argparse
from pathlib import Path

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


def safe_corr(df, x_col, y_col="separation"):
    if x_col not in df.columns:
        return None
    if y_col not in df.columns:
        return None
    if df[x_col].isna().all():
        return None
    return df[x_col].corr(df[y_col])


def compute_correlation(family):
    models = get_family_models(family)

    rows = []

    for model in models:
        print(f"Processing {model}")

        param_path = Path("results") / family / model / "parameters" / "parameter_stats.csv"
        param = pd.read_csv(param_path)

        sep_path = Path("results") / family / model / "metrics" / "semantic_separation.csv"
        sep = pd.read_csv(sep_path)

        merged = pd.merge(param, sep, on="layer")

        corr_up = safe_corr(merged, "up_proj_spectral_norm")
        corr_q = safe_corr(merged, "q_proj_spectral_norm")
        corr_v = safe_corr(merged, "v_proj_spectral_norm")

        rows.append(
            {
                "model": model,
                "corr_up_proj_vs_semantic": corr_up,
                "corr_q_proj_vs_semantic": corr_q,
                "corr_v_proj_vs_semantic": corr_v,
            }
        )

    df = pd.DataFrame(rows)

    out_path = Path("results") / family / "parameter_semantic_correlation.csv"
    df.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, required=True)
    args = parser.parse_args()

    compute_correlation(args.family)


if __name__ == "__main__":
    main()