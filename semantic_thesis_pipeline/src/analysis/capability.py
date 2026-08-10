# -*- coding: utf-8 -*-
"""
Capability measurement under spectral intervention.

The intervention reduces semantic separation under some conditions. That is
only informative if the model still functions: a manipulation that destroys
the model would reduce separation trivially. This stage measures general
capability under the same conditions, with the model patched in memory so
that no modified weights are written to disk.

Two measures, both avoiding free generation so that scoring is deterministic
and no answer parsing is required:

    C-Eval      Each option is scored as a continuation of the question and
                the highest-likelihood option is taken as the answer. Length
                normalised, since options differ in length.

    GSM8K       Mean token log-likelihood of the reference solution given the
                problem. This does not measure whether the model solves the
                problem, only whether the reference reasoning remains
                well-predicted; a large fall indicates that the manipulation
                has damaged the model's language modelling.

Both are relative measures. The comparison of interest is each condition
against the unmodified baseline in the same run, not the absolute value
against published benchmark scores.

Usage
    python -m src.analysis.capability --model llama-3-8b
    python -m src.analysis.capability --model llama-3-8b --n-ceval 300 --n-gsm8k 150
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.hf_loader import load_model_and_tokenizer
from src.analysis.intervention import patch_model
from src.utils.paths import get_results_root, ensure_dir

CEVAL = ("/new_raid/nanhangproj/tianyu/opencompass/data/ceval/formal_ceval/val")
GSM8K = ("/new_raid/nanhangproj/tianyu/opencompass/data/gsm8k/test.jsonl")

SEED = 0
CHOICES = ["A", "B", "C", "D"]


def load_ceval(n, rng):
    rows = []
    for f in sorted(glob.glob(f"{CEVAL}/*.csv")):
        subject = Path(f).stem.replace("_val", "")
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "answer" not in d.columns:
            continue
        for _, r in d.iterrows():
            if any(pd.isna(r.get(c)) for c in CHOICES + ["question", "answer"]):
                continue
            rows.append({"subject": subject, "question": str(r["question"]),
                         "options": [str(r[c]) for c in CHOICES],
                         "answer": CHOICES.index(str(r["answer"]).strip()[:1])})
    rng.shuffle(rows)
    return rows[:n]


def load_gsm8k(n, rng):
    rows = [json.loads(l) for l in open(GSM8K, encoding="utf-8")]
    rng.shuffle(rows)
    return rows[:n]


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt, continuation, device):
    """Mean log-probability per token of `continuation` given `prompt`."""
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full = tokenizer(prompt + continuation, return_tensors="pt").input_ids
    if full.shape[1] <= p_ids.shape[1]:
        return float("nan")
    full = full.to(device)
    logits = model(full).logits[:, :-1, :].float()
    targets = full[:, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    start = p_ids.shape[1] - 1
    return float(lp[0, start:].mean())


def eval_ceval(model, tokenizer, items, device):
    correct = 0
    for it in items:
        prompt = f"以下是一道单项选择题。\n问题：{it['question']}\n答案："
        scores = [sequence_logprob(model, tokenizer, prompt, " " + o, device)
                  for o in it["options"]]
        if int(np.nanargmax(scores)) == it["answer"]:
            correct += 1
    return correct / max(1, len(items))


def eval_gsm8k(model, tokenizer, items, device):
    vals = []
    for it in items:
        prompt = f"Question: {it['question']}\nAnswer:"
        v = sequence_logprob(model, tokenizer, prompt, " " + it["answer"], device)
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


CONDITIONS = [
    ("baseline", None, None, 1.0),
    ("q_proj rotate", "q_proj", "rotate", 0.90),
    ("q_proj truncate 0.80", "q_proj", "truncate", 0.80),
    ("v_proj truncate 0.80", "v_proj", "truncate", 0.80),
]


def run(model_name, n_ceval, n_gsm8k):
    import random
    rng = random.Random(SEED)
    ceval = load_ceval(n_ceval, rng)
    rng = random.Random(SEED)
    gsm = load_gsm8k(n_gsm8k, rng)
    print(f"  C-Eval {len(ceval)} questions from "
          f"{len({c['subject'] for c in ceval})} subjects, GSM8K {len(gsm)} problems\n")

    rows = []
    for label, proj, cond, ratio in CONDITIONS:
        print(f"  {label}")
        tokenizer, model = load_model_and_tokenizer(model_name)[:2]
        model.eval()
        device = next(model.parameters()).device

        sr = float("nan")
        if proj is not None:
            stats = patch_model(model, proj, ratio, cond)
            sr = stats.after_stable_rank.mean() / stats.before_stable_rank.mean()

        acc = eval_ceval(model, tokenizer, ceval, device)
        lp = eval_gsm8k(model, tokenizer, gsm, device)
        rows.append({"model": model_name, "condition": label,
                     "stable_rank_retained": sr,
                     "ceval_accuracy": acc, "gsm8k_logprob": lp})
        print(f"    C-Eval {acc:.3f}   GSM8K mean log-prob {lp:.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    base = df.iloc[0]
    df["ceval_relative"] = df.ceval_accuracy / base.ceval_accuracy
    df["gsm8k_delta"] = df.gsm8k_logprob - base.gsm8k_logprob

    out = ensure_dir(get_results_root() / "analysis" / "_intervention")
    df.to_csv(out / f"capability_{model_name}.csv", index=False)

    print(f"\n  {'condition':<24}{'C-Eval':>9}{'relative':>11}"
          f"{'GSM8K logp':>13}{'delta':>9}")
    print("  " + "-" * 66)
    for _, r in df.iterrows():
        print(f"  {r.condition:<24}{r.ceval_accuracy:>9.3f}{r.ceval_relative:>11.0%}"
              f"{r.gsm8k_logprob:>13.4f}{r.gsm8k_delta:>+9.4f}")
    print(f"\n  Saved -> {out / f'capability_{model_name}.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3-8b")
    ap.add_argument("--n-ceval", type=int, default=200)
    ap.add_argument("--n-gsm8k", type=int, default=100)
    a = ap.parse_args()
    print(f"\nCapability under intervention   {a.model}\n")
    run(a.model, a.n_ceval, a.n_gsm8k)


if __name__ == "__main__":
    main()
