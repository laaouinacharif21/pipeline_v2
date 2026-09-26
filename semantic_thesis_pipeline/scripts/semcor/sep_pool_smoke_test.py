"""GPU smoke test and baseline for the within-word separation pool.

Checks, on a real model:
  1. separation() returns a finite scalar
  2. it carries a gradient back to the model parameters
  3. the batch fits in memory
  4. the baseline value of the term, averaged over many batches, which is what
     the manipulation check of the fine-tuning runs compares against

Usage:
    python scripts/semcor/sep_pool_smoke_test.py --model llama-3-8b
    python scripts/semcor/sep_pool_smoke_test.py --model llama-3-8b \
        --layers 29,30,31,32 --layers-alt 8,9,10,11,12 --batches 40
"""
import argparse, random, sys, pathlib, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.finetune.separation_pool_semcor import SeparationBatchSemCor
from src.models.hf_loader import load_model_and_tokenizer


def measure(pool, model, layers, batches, seed=0):
    rng = random.Random(seed)
    vals = []
    with torch.no_grad():
        for _ in range(batches):
            b = pool.sample(0, rng)
            s = pool.separation(model, b, layers)
            if s is not None:
                vals.append(float(s))
    v = np.array(vals)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3-8b")
    ap.add_argument("--layers", default="29,30,31,32",
                    help="layers used by the current regulariser")
    ap.add_argument("--layers-alt", default="",
                    help="a second set to compare, e.g. mid-network")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--words-per-step", type=int, default=4)
    ap.add_argument("--per-sense", type=int, default=4)
    a = ap.parse_args()

    tokenizer, model = load_model_and_tokenizer(a.model)[:2]
    model.eval()
    device = next(model.parameters()).device

    pool = SeparationBatchSemCor(tokenizer, device,
                                 words_per_step=a.words_per_step,
                                 per_sense=a.per_sense)

    rng = random.Random(0)
    batch = pool.sample(0, rng)
    layers = [int(x) for x in a.layers.split(",")]

    # 1-3. a forward pass with gradient
    print("\n1. forward pass with gradient")
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()
    emb.weight.requires_grad_(True)          # a cheap parameter to test the path
    sep = pool.separation(model, batch, layers)
    if sep is None:
        raise SystemExit("separation() returned None: no valid pairs in the batch")
    print(f"   value {float(sep):+.6f}   requires_grad {sep.requires_grad}")
    sep.backward()
    g = emb.weight.grad
    print(f"   gradient reaches the embedding: norm {float(g.norm()):.4e}")
    if not torch.isfinite(g).all():
        raise SystemExit("gradient contains non-finite values")
    emb.weight.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    print(f"   peak GPU memory {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

    # 4. baseline over many batches
    print(f"\n2. baseline over {a.batches} batches, layers {layers}")
    t0 = time.time()
    v = measure(pool, model, layers, a.batches)
    print(f"   mean {v.mean():+.6f}   sd {v.std():.6f}   "
          f"min {v.min():+.6f}   max {v.max():+.6f}   "
          f"({time.time()-t0:.0f}s, {len(v)} batches)")

    if a.layers_alt:
        alt = [int(x) for x in a.layers_alt.split(",")]
        print(f"\n3. the same, layers {alt}")
        v2 = measure(pool, model, alt, a.batches)
        print(f"   mean {v2.mean():+.6f}   sd {v2.std():.6f}   "
              f"min {v2.min():+.6f}   max {v2.max():+.6f}")
        print(f"\n   ratio alt/current: {v2.mean()/v.mean():.2f}x")


if __name__ == "__main__":
    main()
