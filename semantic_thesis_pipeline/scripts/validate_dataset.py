# -*- coding: utf-8 -*-
"""
Dataset validation for a target-word sentence set.

Reports the checks that a controlled word-sense dataset has to pass before it
enters the pipeline, and the statistics that should accompany it in the
methods section.

Structural checks
    class balance, duplicates, sentence-initial targets, capitalisation leak,
    verb use of the target word

Lexical checks
    vocabulary overlap between the two sense classes (Jaccard), and the words
    most strongly associated with either class (log-odds). A high overlap can
    still hide a single dominant cue, so both are reported.

Length
    mean and range per class; a large difference would let context length
    stand in for sense.

Position
    the token index of the target word, computed with every model's own
    tokenizer. Position is tokenizer-dependent, so this cannot be checked with
    one model and assumed for the rest. Indices below MIN_TOKEN_INDEX fall in
    the region where decoder attention-sink activations dominate.

Usage
    python -m scripts.validate_dataset --dataset data/raw/semantic_sentences/crane.json
    python -m scripts.validate_dataset --dataset ... --no-tokenizers
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_TOKEN_INDEX = 3

ALL_MODELS = [
    "llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b",
    "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b",
    "bert-base", "roberta-base", "spanbert-base-cased", "xlm-roberta-base",
]

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def structural(data):
    S, L = data["sentences"], data["labels"]
    tw = data["target_word"]
    print("\n[1] Structure")
    counts = Counter(L)
    check("two sense classes", len(counts) == 2, str(dict(counts)))
    check("classes balanced", len(set(counts.values())) == 1, str(dict(counts)))
    check("no duplicates", len(S) == len(set(S)), f"{len(S) - len(set(S))} repeated")

    prefix = data.get("prefix", "")
    bodies = [s[len(prefix):] if prefix and s.startswith(prefix) else s for s in S]
    initial = sum(1 for b in bodies if b.lower().startswith(tw.lower()))
    check("target never sentence-initial", initial == 0, f"{initial} sentences")

    cap = sum(1 for s in S if re.search(rf"\s{tw.capitalize()}\b", s))
    check("no capitalisation leak", cap == 0, f"{cap} sentences")

    verb = sum(1 for s in S
               if re.search(rf"\b(to {tw}|{tw}d|{tw[:-1] if tw.endswith('e') else tw}ing)\b",
                            s.lower()))
    check("noun uses only", verb == 0, f"{verb} possible verb uses")

    missing = sum(1 for s in S if not re.search(rf"\b{tw}\b", s, flags=re.I))
    check("target present in every sentence", missing == 0, f"{missing} missing")


def lexical(data):
    S, L = data["sentences"], data["labels"]
    senses = sorted(set(L))
    A = [s for s, l in zip(S, L) if l == senses[0]]
    B = [s for s, l in zip(S, L) if l == senses[1]]
    VA = set(w for s in A for w in words(s))
    VB = set(w for s in B for w in words(s))
    j = len(VA & VB) / len(VA | VB)

    print("\n[2] Lexical overlap")
    print(f"  Jaccard overlap: {j:.3f}")
    check("overlap in 0.40-0.75", 0.40 <= j <= 0.75, f"{j:.3f}")

    ca = Counter(w for s in A for w in set(words(s)))
    cb = Counter(w for s in B for w in set(words(s)))
    lo = {}
    for w in VA | VB:
        a, b = ca.get(w, 0), cb.get(w, 0)
        if a + b >= 6:
            lo[w] = (math.log((a + .5) / (len(A) - a + .5))
                     - math.log((b + .5) / (len(B) - b + .5)))
    top = sorted(lo.items(), key=lambda x: -abs(x[1]))[:8]
    print(f"\n  Strongest class-associated words (positive favours '{senses[0]}'):")
    for w, v in top:
        print(f"    {w:16}{v:+6.2f}   {senses[0]} {ca.get(w,0):3}   {senses[1]} {cb.get(w,0):3}")
    worst = abs(top[0][1]) if top else 0.0
    check("no dominant lexical cue (|log-odds| < 2.5)", worst < 2.5, f"max {worst:.2f}")

    exclusive = [w for w in (VA - VB) | (VB - VA)
                 if max(ca.get(w, 0), cb.get(w, 0)) >= 8]
    check("no frequent class-exclusive word", not exclusive, str(exclusive[:5]))


def lengths(data):
    S, L = data["sentences"], data["labels"]
    senses = sorted(set(L))
    la = [len(words(s)) for s, l in zip(S, L) if l == senses[0]]
    lb = [len(words(s)) for s, l in zip(S, L) if l == senses[1]]
    mu = lambda x: sum(x) / len(x)
    print("\n[3] Length")
    print(f"  {senses[0]:10} mean {mu(la):.2f}  range {min(la)}-{max(la)}")
    print(f"  {senses[1]:10} mean {mu(lb):.2f}  range {min(lb)}-{max(lb)}")
    check("mean length difference < 1 token", abs(mu(la) - mu(lb)) < 1.0,
          f"{abs(mu(la) - mu(lb)):.2f}")


def composition(data):
    print("\n[4] Composition")
    for key in ("stratum", "syntax"):
        if key in data:
            c = Counter(data[key])
            print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    if "syntax" in data:
        c = Counter(data["syntax"])
        check("at least 6 syntactic structures", len(c) >= 6, f"{len(c)} present")
        check("no structure exceeds 40% of the set",
              max(c.values()) / sum(c.values()) < 0.40,
              f"max {max(c.values())/sum(c.values()):.0%}")


def positions(data, models):
    from src.models.hf_loader import load_model_and_tokenizer
    from src.extraction.target_token_extractor import find_target_token_index

    S, tw = data["sentences"], data["target_word"]
    print("\n[5] Target token position, per tokenizer")
    print(f"  {'model':<22}{'min':>6}{'mean':>8}{'max':>6}{'below gate':>13}{'unlocated':>11}")
    print("  " + "-" * 66)

    all_ok = True
    for m in models:
        try:
            tok = load_model_and_tokenizer(m)[0]
        except Exception as e:
            print(f"  {m:<22}  [load failed: {type(e).__name__}]")
            continue
        idx = [find_target_token_index(tok, s, tw)[0] for s in S]
        located = [i for i in idx if i is not None]
        if not located:
            print(f"  {m:<22}  [target not located in any sentence]")
            all_ok = False
            continue
        low = sum(1 for i in located if i < MIN_TOKEN_INDEX)
        unl = len(idx) - len(located)
        if low or unl:
            all_ok = False
        print(f"  {m:<22}{min(located):>6}{sum(located)/len(located):>8.1f}"
              f"{max(located):>6}{low:>13}{unl:>11}")

    check(f"all targets at index >= {MIN_TOKEN_INDEX} in every tokenizer", all_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--no-tokenizers", action="store_true",
                    help="Skip the per-model position check")
    ap.add_argument("--models", default="",
                    help="Comma-separated subset; default is all except qwen-7b")
    a = ap.parse_args()

    data = json.load(open(a.dataset, encoding="utf-8"))
    print(f"\n{'=' * 74}")
    print(f"DATASET VALIDATION  --  '{data['target_word']}'  ({len(data['sentences'])} sentences)")
    print(f"{a.dataset}")
    print(f"{'=' * 74}")

    structural(data)
    lexical(data)
    lengths(data)
    composition(data)
    if not a.no_tokenizers:
        models = [m.strip() for m in a.models.split(",") if m.strip()] or ALL_MODELS
        positions(data, models)

    print(f"\n{'=' * 74}")
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("Failed: " + ", ".join(FAIL))
        sys.exit(1)
    print("Dataset meets the specification.")


if __name__ == "__main__":
    main()
