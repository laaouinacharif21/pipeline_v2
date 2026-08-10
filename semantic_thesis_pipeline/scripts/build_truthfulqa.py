# -*- coding: utf-8 -*-
"""
Build a balanced true/false statement set from TruthfulQA.

Each question contributes matched pairs: one statement formed from a correct
answer and one from an incorrect answer to the same question. Pairing within a
question holds the topic constant across the two classes, so a classifier
cannot separate them on subject matter alone, and it lets cross-validation
folds be formed over questions rather than statements.

The statement is written as a declarative continuation rather than a question
and answer pair, so that the representation at the final token reflects the
claim itself rather than the interrogative frame.

Output matches the pipeline's dataset schema, so the extraction, metrics and
probe stages run unchanged.

Usage
    python scripts/build_truthfulqa.py
    python scripts/build_truthfulqa.py --pairs-per-question 2
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

SOURCE = ("/new_raid/nanhangproj/tianyu/opencompass/data/truthful_qa"
          "/generation/validation-00000-of-00001.parquet")

PREFIX = "Consider the following statement: "
SEED = 0


def to_statement(question, answer):
    """Join a question and an answer into a single declarative line."""
    q = question.strip().rstrip("?").strip()
    a = answer.strip().rstrip(".").strip()
    if not a:
        return None
    return f"{PREFIX}{q}. {a[0].upper() + a[1:]}."


def build(pairs_per_question=1, max_words=40):
    rng = random.Random(SEED)
    df = pd.read_parquet(SOURCE)

    sentences, labels, groups, categories = [], [], [], []
    for qi, row in df.iterrows():
        correct = [a for a in list(row["correct_answers"]) if a and a.strip()]
        wrong = [a for a in list(row["incorrect_answers"]) if a and a.strip()]
        if not correct or not wrong:
            continue

        # Prefer answers of similar length so that length cannot stand in for
        # the label; pair the closest match rather than sampling blindly.
        rng.shuffle(correct)
        for k in range(min(pairs_per_question, len(correct), len(wrong))):
            c = correct[k]
            w = min(wrong, key=lambda x: abs(len(x.split()) - len(c.split())))
            wrong = [x for x in wrong if x is not w]

            sc, sw = to_statement(row["question"], c), to_statement(row["question"], w)
            if sc is None or sw is None:
                continue
            if max(len(sc.split()), len(sw.split())) > max_words:
                continue

            sentences += [sc, sw]
            labels += ["true", "false"]
            groups += [int(qi), int(qi)]
            categories += [row["category"], row["category"]]

    return sentences, labels, groups, categories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-per-question", type=int, default=1)
    ap.add_argument("--max-words", type=int, default=40)
    a = ap.parse_args()

    S, L, G, C = build(a.pairs_per_question, a.max_words)

    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_word": "",
        "sentences": S,
        "labels": L,
        "groups": G,
        "category": C,
        "prefix": PREFIX,
        "design": "TruthfulQA generation split; each question contributes a "
                  "matched pair of statements formed from a correct and an "
                  "incorrect answer, chosen to be similar in length",
    }, open(out / "truthfulqa.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "truthfulqa.yaml").write_text(
        "word: truthfulqa\n"
        "dataset: data/raw/semantic_sentences/truthfulqa.json\n"
        "senses:\n  - false\n  - true\n"
        f"n_per_sense: {L.count('true')}\n"
        f'prefix: "{PREFIX}"\n'
    )

    import statistics as st
    lt = [len(s.split()) for s, l in zip(S, L) if l == "true"]
    lf = [len(s.split()) for s, l in zip(S, L) if l == "false"]
    print(f"{len(S)} statements from {len(set(G))} questions")
    print(f"  true {L.count('true')}   false {L.count('false')}")
    print(f"  length  true {st.mean(lt):.1f}   false {st.mean(lf):.1f}   "
          f"diff {abs(st.mean(lt) - st.mean(lf)):.2f}")
    print("\nexamples:")
    for s in S[:4]:
        print("  ", s)


if __name__ == "__main__":
    main()
