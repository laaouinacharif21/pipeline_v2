import json
import random
from pathlib import Path


random.seed(42)


def unique_extend(target, candidates, max_count):
    for s in candidates:
        if s not in target:
            target.append(s)
        if len(target) >= max_count:
            break


def build_finance_sentences():
    people = [
        "She", "He", "They", "My father", "My sister",
        "The customer", "The manager", "A student", "A woman", "A man"
    ]

    starts = [
        "At the bank,", "Inside the bank,", "Near the bank,", "This morning at the bank,",
        "Yesterday at the bank,", "Before noon at the bank,", "After work at the bank,",
        "During lunch at the bank,", "Early in the day at the bank,", "Late in the afternoon at the bank,"
    ]

    transaction_verbs = [
        "deposited money", "withdrew cash", "transferred funds", "paid a bill",
        "cashed a check", "opened an account", "closed an account",
        "requested a statement", "updated account details", "checked a balance"
    ]

    service_actions = [
        "spoke with a teller", "asked the clerk for help", "waited in line",
        "met a financial advisor", "discussed a loan", "reported a card problem",
        "requested account help", "asked about bank fees", "reviewed mortgage options",
        "confirmed a transfer"
    ]

    finance_actions = [
        "asked about a savings account", "reviewed a credit card offer",
        "discussed a mortgage payment", "applied for a personal loan",
        "checked a checking account", "fixed a debit card issue",
        "confirmed a wire transfer", "read a bank statement",
        "planned a monthly payment", "changed an account limit"
    ]

    sentences = []

    candidates = []

    for p in people:
        for a in transaction_verbs:
            candidates.append(f"{p} {a} at the bank.")
            candidates.append(f"{random.choice(starts)} {p.lower()} {a}.")
            candidates.append(f"The bank helped {p.lower()} when {p.lower()} {a}.")

    for p in people:
        for a in service_actions:
            candidates.append(f"{p} {a} at the bank.")
            candidates.append(f"{random.choice(starts)} {p.lower()} {a}.")
            candidates.append(f"The bank was busy while {p.lower()} {a}.")

    for p in people:
        for a in finance_actions:
            candidates.append(f"{p} {a} at the bank.")
            candidates.append(f"{random.choice(starts)} {p.lower()} {a}.")
            candidates.append(f"The bank called after {p.lower()} {a}.")

    random.shuffle(candidates)
    unique_extend(sentences, candidates, 100)
    return sentences


def build_river_sentences():
    people = [
        "She", "He", "They", "My father", "My sister",
        "The child", "The fisherman", "A hiker", "A woman", "A man"
    ]

    river_templates = [
        "{p} sat quietly on the bank beside the river.",
        "{p} walked slowly along the bank near the water.",
        "{p} watched ducks from the bank of the river.",
        "{p} looked for fish from the bank near the stream.",
        "{p} stood on the bank and watched the current.",
        "{p} climbed the muddy bank after crossing the stream.",
        "{p} rested on the grassy bank beside the water.",
        "{p} watched a small boat from the bank.",
        "{p} heard frogs while standing on the bank near the river.",
        "{p} saw reeds moving beside the bank in the wind.",
        "{p} looked at the ripples from the bank near the water.",
        "{p} sat in the grass on the bank beside the stream.",
        "{p} watched birds from the rocky bank of the river.",
        "{p} stood on the bank near the flood water.",
        "{p} followed turtles along the bank near the river.",
        "{p} watched the water from the shaded bank.",
        "{p} stood by the bank where stones lined the river.",
        "{p} rested near the bank after walking beside the stream.",
        "{p} watched insects over the water from the bank.",
        "{p} looked down from the steep bank at the river below.",
    ]

    nature_templates = [
        "The bank beside the river was muddy after the rain.",
        "The bank near the water was covered with grass.",
        "The bank along the stream was steep and rocky.",
        "The bank beside the river was wet from the flood.",
        "The bank near the current was lined with stones.",
        "The bank by the water was shaded by trees.",
        "The bank beside the river was soft underfoot.",
        "The bank near the stream was uneven and slippery.",
        "The bank beside the water was bright in the sun.",
        "The bank along the river was dark after sunset.",
    ]

    sentences = []
    candidates = []

    for p in people:
        for t in river_templates:
            candidates.append(t.format(p=p))

    for t in nature_templates:
        candidates.append(t)
        candidates.append("After the storm, " + t[0].lower() + t[1:])
        candidates.append("Near the river, " + t[0].lower() + t[1:])

    random.shuffle(candidates)
    unique_extend(sentences, candidates, 100)
    return sentences


def main():
    finance = build_finance_sentences()
    river = build_river_sentences()

    if len(finance) < 100 or len(river) < 100:
        raise ValueError(f"Not enough sentences generated: finance={len(finance)}, river={len(river)}")

    sentences = finance + river
    labels = ["finance"] * 100 + ["river"] * 100

    combined = list(zip(sentences, labels))
    random.shuffle(combined)

    sentences = [x[0] for x in combined]
    labels = [x[1] for x in combined]

    out = {
        "target_word": "bank",
        "sentences": sentences,
        "labels": labels,
    }

    output_path = Path("data/raw/semantic_sentences/sentences.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to {output_path}")
    print(f"Total sentences: {len(sentences)}")
    print(f"Finance: {labels.count('finance')}")
    print(f"River: {labels.count('river')}")


if __name__ == "__main__":
    main()