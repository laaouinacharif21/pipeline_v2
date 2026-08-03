import argparse
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.models.hf_loader import load_model_and_tokenizer


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


def power_iteration_spectral_norm(matrix, num_iters=20):
    matrix = matrix.float()
    device = matrix.device

    v = torch.randn(matrix.shape[1], device=device)
    v = v / (torch.norm(v) + 1e-12)

    for _ in range(num_iters):
        u = matrix @ v
        u = u / (torch.norm(u) + 1e-12)
        v = matrix.T @ u
        v = v / (torch.norm(v) + 1e-12)

    sigma = torch.dot(u, matrix @ v)
    return sigma.item()


def frobenius_norm(matrix):
    return torch.norm(matrix.float(), p="fro").item()


def get_layers(model):
    if hasattr(model, "layers"):
        return model.layers

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "h"):
            return model.transformer.h
        if hasattr(model.transformer, "layers"):
            return model.transformer.layers

    # ===== BERT / RoBERTa =====
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer

    # ===== DeBERTa =====
    if hasattr(model, "deberta") and hasattr(model.deberta, "encoder"):
        if hasattr(model.deberta.encoder, "layer"):
            return model.deberta.encoder.layer

    raise ValueError("Unknown model structure")


def extract_matrices(layer):
    matrices = {}

    # ===== LLaMA-style =====
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
        matrices["q_proj"] = layer.self_attn.q_proj.weight
        matrices["k_proj"] = layer.self_attn.k_proj.weight
        matrices["v_proj"] = layer.self_attn.v_proj.weight
        matrices["o_proj"] = layer.self_attn.o_proj.weight

    # ===== Qwen-style =====
    elif hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
        matrices["q_proj"] = layer.attn.c_attn.weight
        matrices["k_proj"] = layer.attn.c_attn.weight
        matrices["v_proj"] = layer.attn.c_attn.weight
        matrices["o_proj"] = layer.attn.c_proj.weight

    # ===== DeBERTa / BERT / RoBERTa / XLM-R attention =====
    elif hasattr(layer, "attention") and hasattr(layer.attention, "self"):
        attn_self = layer.attention.self

        if hasattr(attn_self, "query_proj"):
            matrices["q_proj"] = attn_self.query_proj.weight
        elif hasattr(attn_self, "query"):
            matrices["q_proj"] = attn_self.query.weight

        if hasattr(attn_self, "key_proj"):
            matrices["k_proj"] = attn_self.key_proj.weight
        elif hasattr(attn_self, "key"):
            matrices["k_proj"] = attn_self.key.weight

        if hasattr(attn_self, "value_proj"):
            matrices["v_proj"] = attn_self.value_proj.weight
        elif hasattr(attn_self, "value"):
            matrices["v_proj"] = attn_self.value.weight

        if hasattr(layer.attention, "output") and hasattr(layer.attention.output, "dense"):
            matrices["o_proj"] = layer.attention.output.dense.weight

    # ===== MLP =====
    if hasattr(layer, "mlp"):
        if hasattr(layer.mlp, "gate_proj"):
            matrices["gate_proj"] = layer.mlp.gate_proj.weight
            matrices["up_proj"] = layer.mlp.up_proj.weight
            matrices["down_proj"] = layer.mlp.down_proj.weight

        elif hasattr(layer.mlp, "w1"):
            matrices["gate_proj"] = layer.mlp.w1.weight
            matrices["up_proj"] = layer.mlp.w2.weight
            matrices["down_proj"] = layer.mlp.c_proj.weight

        elif hasattr(layer.mlp, "Wi"):
            matrices["up_proj"] = layer.mlp.Wi.weight
            if hasattr(layer.mlp, "Wo"):
                matrices["down_proj"] = layer.mlp.Wo.weight

    # ===== BERT / RoBERTa / DeBERTa / XLM-R FFN =====
    if hasattr(layer, "intermediate") and hasattr(layer.intermediate, "dense"):
        matrices["up_proj"] = layer.intermediate.dense.weight

    if hasattr(layer, "output") and hasattr(layer.output, "dense"):
        matrices["down_proj"] = layer.output.dense.weight

    return matrices


def analyze_model(family, model_name):
    print(f"\nRunning {model_name}")

    _, model = load_model_and_tokenizer(model_name)
    layers = get_layers(model)

    rows = []

    for i, layer in enumerate(layers):
        print(f"{model_name} | layer {i}")

        matrices = extract_matrices(layer)

        res = {"layer": i}

        for name, mat in matrices.items():
            mat = mat.detach().cpu().float()
            res[f"{name}_spectral_norm"] = power_iteration_spectral_norm(mat)
            res[f"{name}_fro_norm"] = frobenius_norm(mat)

        rows.append(res)

    df = pd.DataFrame(rows)

    out_dir = Path("results") / family / model_name / "parameters"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "parameter_stats.csv", index=False)

    plt.figure()

    if "q_proj_spectral_norm" in df.columns:
        plt.plot(df["layer"], df["q_proj_spectral_norm"], label="q_proj")
    if "v_proj_spectral_norm" in df.columns:
        plt.plot(df["layer"], df["v_proj_spectral_norm"], label="v_proj")
    if "up_proj_spectral_norm" in df.columns:
        plt.plot(df["layer"], df["up_proj_spectral_norm"], label="up_proj")

    plt.xlabel("Layer")
    plt.ylabel("Spectral Norm")
    plt.title(f"{model_name} Parameter Geometry")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "parameter_spectral_norm.png")
    plt.close()

    print(f"Saved {model_name} to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, required=True)
    args = parser.parse_args()

    family = args.family
    models = get_family_models(family)

    for model_name in models:
        analyze_model(family, model_name)

    print(f"\nFinished {family} family")


if __name__ == "__main__":
    main()