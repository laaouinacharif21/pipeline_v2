import json
import numpy as np

from src.models.hf_loader import load_model_and_tokenizer
from src.extraction.hidden_state_extractor import extract_hidden_states
from src.utils.paths import ensure_model_result_dirs, get_standard_result_files


def run_extraction(model_name: str, dataset_path: str):
    paths = ensure_model_result_dirs(model_name)
    files = get_standard_result_files(model_name)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data["sentences"]
    labels = data["labels"]
    target_word = data.get("target_word", "")

    tokenizer, model = load_model_and_tokenizer(model_name)
    embeddings = extract_hidden_states(tokenizer, model, sentences)

    np.savez_compressed(files["embedding_tensor"], embeddings=embeddings)

    with open(files["labels"], "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, indent=2)

    with open(files["sentences"], "w", encoding="utf-8") as f:
        json.dump({"sentences": sentences}, f, indent=2)

    with open(files["embedding_metadata"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "target_word": target_word,
                "num_sentences": len(sentences),
                "num_layers": int(embeddings.shape[1]),
                "hidden_size": int(embeddings.shape[2])
            },
            f,
            indent=2
        )

    print(f"Saved embeddings to: {files['embedding_tensor']}")
    print(f"Shape: {embeddings.shape}")
