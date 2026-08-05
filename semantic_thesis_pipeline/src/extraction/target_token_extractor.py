"""
Target-token hidden-state extraction.

For each sentence, locates the FIRST SUBWORD TOKEN of the target word and
returns that token's hidden state at every layer. Tokenizer-agnostic: it
locates the token by decoding cumulative prefixes rather than relying on
offset mappings, which slow tokenizers do not provide.
"""

import re

import numpy as np
import torch


def _nows_len(s: str) -> int:
    return len(re.sub(r"\s+", "", s))


def find_target_token_index(tokenizer, sentence, target_word, max_length=128):
    """Return (token_index_of_first_subword, validated_flag)."""
    pattern = r"\b" + re.escape(target_word) + r"\b"
    m = re.search(pattern, sentence, flags=re.IGNORECASE)
    if m is None:
        m = re.search(re.escape(target_word), sentence, flags=re.IGNORECASE)
    if m is None:
        return None, False

    start_nows = _nows_len(sentence[: m.start()])

    ids = tokenizer(
        sentence,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    idx = None
    for k in range(1, len(ids) + 1):
        decoded = tokenizer.decode(ids[:k], skip_special_tokens=True)
        if _nows_len(decoded) > start_nows:
            idx = k - 1
            break

    if idx is None:
        return None, False

    piece = tokenizer.decode([ids[idx]], skip_special_tokens=True)
    piece_clean = re.sub(r"\s+", "", piece).lower()
    tw = target_word.lower()
    validated = bool(piece_clean) and (
        tw.startswith(piece_clean) or piece_clean.startswith(tw)
    )

    return idx, validated


def extract_target_token_hidden_states(
    tokenizer,
    model,
    sentences,
    target_word,
    max_length=128,
    batch_size=8,
):
    """
    Returns
    -------
    embeddings : np.ndarray [n_kept, n_layers, hidden_size]
    keep_mask  : list[bool]
    report     : dict
    """
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    indices = []
    validated = []
    for s in sentences:
        i, v = find_target_token_index(tokenizer, s, target_word, max_length)
        indices.append(i)
        validated.append(v)

    keep_mask = [i is not None for i in indices]

    device = next(model.parameters()).device
    collected = []

    for b in range(0, len(sentences), batch_size):
        chunk = sentences[b : b + batch_size]
        chunk_idx = indices[b : b + batch_size]

        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)

        hs = torch.stack(out.hidden_states, dim=0).permute(1, 0, 2, 3)
        seq_len = hs.shape[2]

        for j, ti in enumerate(chunk_idx):
            if ti is None:
                continue
            ti_use = min(ti, seq_len - 1)
            collected.append(hs[j, :, ti_use, :].float().cpu().numpy())

        del out, hs
        torch.cuda.empty_cache()

    embeddings = np.stack(collected, axis=0)

    report = {
        "n_sentences": len(sentences),
        "n_located": int(sum(keep_mask)),
        "n_validated": int(sum(validated)),
        "validation_rate": float(sum(validated)) / max(1, len(sentences)),
        "example_indices": indices[:10],
    }

    return embeddings, keep_mask, report
