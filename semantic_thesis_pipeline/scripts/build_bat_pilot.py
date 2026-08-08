# -*- coding: utf-8 -*-
"""
Pilot sentence set for the ambiguous noun "bat" (mammal / sports implement).

Forty sentences, twenty per sense, written to test whether the construction
recipe produces a usable measurement before the full set is written.

Acceptance criteria, measured before scaling:
    context-only accuracy   0.80 - 0.95
    target-token probe      above 0.70

Below 0.80 on context the label is not reliably recoverable and separation
tends to collapse; above 0.95 the sentence pair is a topic classifier and the
target token is redundant.

Design
    Cues precede the target word. A calibration probe showed decoder models
    produce zero separation when the disambiguating phrase follows the target,
    since causal attention cannot read it at that position.

    Cues name properties of the referent (wings, roost, claws; handle, grip,
    willow) rather than whole topic fields (caves and nocturnal hunting versus
    stadiums and innings), so the two classes are not separable by subject
    matter alone.

    Sentences are written in pairs sharing a frame, so length and syntax are
    matched and folds can be grouped by pair.
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

PAIRS = [
    ("Hanging by its {cue}, the bat stayed completely still.",
     "small claws", "leather grip"),
    ("With its {cue} folded in, the bat took up very little room.",
     "thin wings", "narrow handle"),
    ("Because its {cue} was damaged, the bat was set aside.",
     "left wing", "wooden shaft"),
    ("Its {cue} worn smooth with use, the bat had clearly seen years of service.",
     "claws", "handle"),
    ("Resting against the {cue}, the bat remained where it had been left.",
     "cave wall", "changing room wall"),
    ("With the {cue} finally quiet, the bat settled for the night.",
     "colony", "clubhouse"),
    ("Its {cue} unusually pale, the bat stood out from the others.",
     "fur", "grain"),
    ("Because the {cue} had shifted, the bat was moved to a safer spot.",
     "roost", "rack"),
    ("Covered in fine {cue}, the bat looked older than it was.",
     "grey hair", "wood dust"),
    ("With its {cue} extended, the bat seemed much larger than before.",
     "wingspan", "reach"),
    ("Its {cue} carefully examined, the bat was recorded in the register.",
     "wing membrane", "surface grain"),
    ("Since the {cue} had been cleaned, the bat looked almost new.",
     "fur", "willow"),
    ("Guided by its {cue}, the bat found its way without difficulty.",
     "hearing", "balance"),
    ("With the {cue} approaching, the bat was brought back inside.",
     "cold season", "end of the over"),
    ("Its {cue} slightly bent, the bat still worked as it should.",
     "wing bone", "lower edge"),
    ("Because the {cue} was too narrow, the bat could not pass through.",
     "gap in the roof", "gap in the rack"),
    ("With its {cue} tightly closed, the bat waited out the noise.",
     "wings", "case"),
    ("Its {cue} counted carefully, the bat entered the survey record.",
     "colony size", "weight in ounces"),
    ("Following an inspection of its {cue}, the bat was approved for use.",
     "wing condition", "blade condition"),
    ("With its {cue} finally repaired, the bat returned to service.",
     "torn wing", "cracked handle"),
]


def build():
    rows = []
    for frame, mammal, sport in PAIRS:
        rows.append((PREFIX + frame.format(cue=mammal), "mammal"))
        rows.append((PREFIX + frame.format(cue=sport), "sport"))
    return rows


if __name__ == "__main__":
    rows = build()
    labels = [r[1] for r in rows]

    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_word": "bat",
        "sentences": [r[0] for r in rows],
        "labels": labels,
        "stratum": ["pilot"] * len(rows),
        "prefix": PREFIX,
        "design": "pilot, 20 pairs; cue precedes the target and names a "
                  "referent property rather than a topic field",
    }, open(out / "bat_pilot.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "bat_pilot.yaml").write_text(
        "word: bat\n"
        "dataset: data/raw/semantic_sentences/bat_pilot.json\n"
        "senses:\n  - mammal\n  - sport\n"
        "n_per_sense: 20\n"
        f'prefix: "{PREFIX}"\n'
    )
    print(f"{len(PAIRS)} pairs -> {len(rows)} sentences "
          f"(mammal {labels.count('mammal')}, sport {labels.count('sport')})")
