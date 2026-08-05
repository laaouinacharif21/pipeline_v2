import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)


BASE_MODEL_PATH = "/new_raid/nanhangproj/tianyu/models"


MODEL_PATHS = {

    # LLaMA
    "llama-7b": os.path.join(BASE_MODEL_PATH, "Llama", "llama-7b"),
    "llama-2-7b": os.path.join(BASE_MODEL_PATH, "Llama", "Llama-2-7b"),
    "llama-3-8b": os.path.join(BASE_MODEL_PATH, "Llama", "Llama-3-8b"),
    "llama-3.1-8b": os.path.join(BASE_MODEL_PATH, "Llama", "Llama-3.1-8b"),

    # Qwen
    "qwen-7b": os.path.join(BASE_MODEL_PATH, "Qwen", "qwen-7b"),
    "qwen1.5-7b": os.path.join(BASE_MODEL_PATH, "Qwen", "qwen1.5-7b"),
    "qwen2-7b": os.path.join(BASE_MODEL_PATH, "Qwen", "qwen2-7b"),
    "qwen2.5-7b": os.path.join(BASE_MODEL_PATH, "Qwen", "qwen2.5-7b"),
    "qwen3-8b": os.path.join(BASE_MODEL_PATH, "Qwen", "qwen3-8b"),

    # Encoder models
    "bert-base": os.path.join(BASE_MODEL_PATH, "Bert", "bert-base"),
    "roberta-base": os.path.join(BASE_MODEL_PATH, "Bert", "roberta-base"),
    "spanbert-base": os.path.join(BASE_MODEL_PATH, "Bert", "spanbert-base"),
    "xlm-roberta-base": os.path.join(BASE_MODEL_PATH, "Bert", "xlm-roberta-base"),
}


def load_model_and_tokenizer(model_name, device=None):

    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model {model_name}. Available: {list(MODEL_PATHS.keys())}"
        )

    model_path = MODEL_PATHS[model_name]

    print(f"Loading model: {model_name}")
    print(f"Path: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)


    # LLaMA models
    if "llama" in model_name.lower():

        tokenizer = LlamaTokenizer.from_pretrained(
            model_path
        )

        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )


    # Qwen models
    elif "qwen" in model_name.lower():

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )


    # Encoder models
    else:

        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        model = AutoModel.from_pretrained(
            model_path
        )


    if device is not None:
        model.to(device)


    model.eval()

    return tokenizer, model