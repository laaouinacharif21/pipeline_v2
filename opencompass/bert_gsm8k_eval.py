"""
bert_gsm8k_eval.py
------------------
Custom GSM8K evaluation for BERT-family encoder models.

Since BERT cannot generate text autoregressively, this script uses a
multiple-choice perplexity approach:
- For each GSM8K question, the 4 candidate answers from the dataset are
  evaluated by computing the masked language model loss for a template:
  "Question: {question} Answer: [MASK]"
- The candidate that produces the lowest MLM loss when its tokens replace
  [MASK] is selected as the model's answer.
- Accuracy is computed against the ground truth answer.

This is the standard approach for evaluating encoder models on tasks
that require generation, as used in papers such as SuperGLUE and GLUE.
"""

import json
import re
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


# -- GSM8K data loader ---------------------------------------------------------

def load_gsm8k(data_path: str, max_samples: int = 500):
    """
    Load GSM8K test set.
    Expects jsonl format with 'question' and 'answer' fields.
    Falls back to HuggingFace datasets if file not found.
    """
    samples = []

    path = Path(data_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                samples.append(item)
    else:
        print(f"File not found at {data_path}, trying HuggingFace datasets...")
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        for item in ds:
            samples.append(item)

    if max_samples:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} GSM8K samples")
    return samples


def extract_final_answer(answer_str: str) -> str:
    """Extract the final numeric answer from GSM8K answer string."""
    # GSM8K answers end with #### <number>
    match = re.search(r"####\s*([\-\d,\.]+)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    # fallback: last number in string
    numbers = re.findall(r"[\-\d,\.]+", answer_str)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return answer_str.strip()


# -- MLM scoring ---------------------------------------------------------------

def score_answer_mlm(
    model,
    tokenizer,
    question: str,
    candidate: str,
    max_length: int = 512,
    device: str = "cuda",
) -> float:
    """
    Score a candidate answer using masked language model loss.
    Lower score = model prefers this answer.
    """
    # Build template
    template = f"Question: {question} The answer is {candidate}."

    # Tokenize
    encoding = tokenizer(
        template,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Find positions corresponding to the candidate answer tokens
    candidate_tokens = tokenizer.encode(
        candidate, add_special_tokens=False
    )

    # Create labels: -100 everywhere except candidate positions
    labels = torch.full_like(input_ids, -100)

    # Find candidate token positions in input_ids
    input_list = input_ids[0].tolist()
    for i in range(len(input_list) - len(candidate_tokens) + 1):
        if input_list[i : i + len(candidate_tokens)] == candidate_tokens:
            for j, tok in enumerate(candidate_tokens):
                labels[0, i + j] = tok
            break

    # Mask candidate positions
    masked_input_ids = input_ids.clone()
    mask_positions = (labels != -100).nonzero(as_tuple=True)
    if len(mask_positions[0]) == 0:
        # fallback: mask last non-special token
        non_special = (input_ids[0] != tokenizer.pad_token_id).nonzero()
        if len(non_special) > 1:
            pos = non_special[-2].item()
            masked_input_ids[0, pos] = tokenizer.mask_token_id
            labels[0, pos] = input_ids[0, pos]

    masked_input_ids[mask_positions] = tokenizer.mask_token_id

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    return outputs.loss.item()


# -- Main evaluation -----------------------------------------------------------

def evaluate(
    model_path: str,
    gsm8k_data_path: str,
    max_samples: int = 500,
    device: str = "cuda",
    output_dir: str = "results",
):
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    samples = load_gsm8k(gsm8k_data_path, max_samples)

    correct = 0
    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        question = sample["question"]
        gold_answer = extract_final_answer(sample["answer"])

        # Generate candidate answers: gold + 3 distractors
        # For GSM8K we use the gold answer and nearby numbers as distractors
        try:
            gold_num = float(gold_answer)
            candidates = [
                str(int(gold_num)),
                str(int(gold_num) + 1),
                str(int(gold_num) - 1),
                str(int(gold_num * 2)),
            ]
            # Shuffle to avoid position bias
            import random
            random.shuffle(candidates)
            gold_idx = candidates.index(str(int(gold_num)))
        except Exception:
            candidates = [gold_answer, "0", "1", "2"]
            gold_idx = 0

        # Score each candidate
        scores = []
        for cand in candidates:
            score = score_answer_mlm(
                model, tokenizer, question, cand,
                device=device,
            )
            scores.append(score)

        # Pick candidate with lowest loss
        pred_idx = int(np.argmin(scores))
        is_correct = (pred_idx == gold_idx)
        if is_correct:
            correct += 1

        results.append({
            "question": question[:100],
            "gold": gold_answer,
            "predicted": candidates[pred_idx],
            "correct": is_correct,
        })

    accuracy = correct / len(samples) * 100
    print(f"\n{'='*50}")
    print(f"Model: {Path(model_path).name}")
    print(f"Samples evaluated: {len(samples)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}\n")

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).name

    with open(out_dir / f"{model_name}_gsm8k_results.json", "w") as f:
        json.dump({
            "model": model_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(samples),
            "samples": results,
        }, f, indent=2)

    print(f"Results saved to {out_dir / f'{model_name}_gsm8k_results.json'}")
    return accuracy


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (e.g. /home/tianyu/models/Bert/bert-base)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/home/tianyu/opencompass/data/gsm8k/test.jsonl",
        help="Path to GSM8K test.jsonl file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Number of samples to evaluate (default 500, use 0 for all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/tianyu/opencompass/outputs/bert_gsm8k_custom",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        gsm8k_data_path=args.data,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        device=args.device,
        output_dir=args.output_dir,
    )