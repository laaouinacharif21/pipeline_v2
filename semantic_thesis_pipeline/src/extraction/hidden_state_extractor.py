"""
DEPRECATED -- mean-pooled sentence representations.

This function averages hidden states over all tokens in the sentence. It does
not implement the target-token extraction described in the methodology, and
was the source of a measurement error corrected in August 2026: pooled
representations are dominated by outlier feature dimensions, leaving Sep(l)
near zero through the middle layers.

Retained only to reproduce the mean-pooled vs target-token comparison. Use
src/extraction/target_token_extractor.py for all new work.
"""

import torch


def extract_hidden_states(tokenizer, model, sentences, device="cuda", max_length=128):
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # ?? FIX: send inputs to SAME device as model (not forced "cuda")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,   # ?? CRITICAL FIX
            return_dict=True
        )

    hidden_states = outputs.hidden_states  # now NOT None

    # [layers, batch, seq, dim] ? stack to [batch, layers, seq, dim]
    stacked = torch.stack(hidden_states, dim=0).permute(1, 0, 2, 3)

    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]
    masked = stacked * mask

    lengths = attention_mask.sum(dim=1).view(-1, 1, 1).clamp(min=1)

    sentence_embeddings = masked.sum(dim=2) / lengths  # mean pooling

    return sentence_embeddings.float().cpu().numpy()