import json

import numpy as np

from src.models.hf_loader import load_model_and_tokenizer
from src.extraction.target_token_extractor import extract_target_token_hidden_states
from src.utils.paths import ensure_model_result_dirs, get_standard_result_files


def run_extraction_target(model_name: str, dataset_path: str, batch_size: int = 8):
    ensure_model_result_dirs(model_name)
    files = get_standard_result_files(model_name)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data["sentences"]
    labels = data["labels"]
    target_word = data.get("target_word", "")

    if not target_word:
        raise ValueError("Dataset has no 'target_word' field.")

    result = load_model_and_tokenizer(model_name)
    tokenizer, model = result[0], result[1]

    embeddings, keep_mask, report = extract_target_token_hidden_states(
        tokenizer, model, sentences, target_word, batch_size=batch_size
    )

    kept_sentences = [s for s, k in zip(sentences, keep_mask) if k]
    kept_labels = [l for l, k in zip(labels, keep_mask) if k]

    np.savez_compressed(files["embedding_tensor"], embeddings=embeddings)

    with open(files["labels"], "w", encoding="utf-8") as f:
        json.dump({"labels": kept_labels}, f, indent=2)

    with open(files["sentences"], "w", encoding="utf-8") as f:
        json.dump({"sentences": kept_sentences}, f, indent=2)

    with open(files["embedding_metadata"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "target_word": target_word,
                "pooling": "target_token_first_subword",
                "num_sentences": int(embeddings.shape[0]),
                "num_layers": int(embeddings.shape[1]),
                "hidden_size": int(embeddings.shape[2]),
                "location_report": report,
            },
            f,
            indent=2,
        )

    print(f"[{model_name}] located {report['n_located']}/{report['n_sentences']}, "
          f"validated {report['n_validated']}/{report['n_sentences']} "
          f"({report['validation_rate']:.1%})")
    print(f"Saved embeddings to: {files['embedding_tensor']}")
    print(f"Shape: {embeddings.shape}")
