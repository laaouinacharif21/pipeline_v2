# -*- coding: utf-8 -*-
"""
Cue-strength calibration probe for the target word "crane".

The controlled crane set produced Sep(l) at the 1e-3 level across all models,
including on minimal pairs where the senses differ by an explicit clause. This
probe establishes how much sense-bearing context is needed before separation
appears at all, so that dataset strength can be calibrated from measurement
rather than argument.

Three levels, 20 sentences each (10 per sense), sharing the same target word
and the same prefix:

  strong    Referent properties named early and unambiguously. Cues appear
            before the target so that causal decoders can use them.
  medium    Referent properties named, but later in the sentence and with
            less specific wording.
  weak      The level used in the failed set: cues arrive after the target
            and describe the situation rather than the referent.

Cues describe the referent (wings, beak, hook, cable) rather than the topic
field (marsh, construction), so a positive result cannot be attributed to
topic separation alone.
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

STRONG = [
    ("With its wings folded back, the crane stood at the edge of the shallows.", "bird"),
    ("Feathers ruffled by the wind, the crane held its place on one leg.", "bird"),
    ("Stretching its long beak forward, the crane reached into the mud.", "bird"),
    ("Its narrow legs half submerged, the crane waited without moving.", "bird"),
    ("Wading between the reeds, the crane searched for something to eat.", "bird"),
    ("The migrating flock settled, and the crane folded its wings.", "bird"),
    ("Preening slowly, the crane stayed where the water was shallow.", "bird"),
    ("With a sudden beat of feathers, the crane lifted from the ground.", "bird"),
    ("Its plumage grey against the sky, the crane circled once.", "bird"),
    ("Nesting season had begun, and the crane returned to the marsh.", "bird"),

    ("With its steel cable drawn tight, the crane held the load steady.", "machine"),
    ("Hook lowered toward the platform, the crane waited for the signal.", "machine"),
    ("Extending its hydraulic boom, the crane reached the upper floor.", "machine"),
    ("Its counterweight swinging slightly, the crane rotated on its base.", "machine"),
    ("Operating from the cab, the driver moved the crane into position.", "machine"),
    ("The diesel engine idling, the crane stood ready for the next lift.", "machine"),
    ("Bolted to a concrete footing, the crane rose above the scaffolding.", "machine"),
    ("With a grinding of gears, the crane raised the steel girder.", "machine"),
    ("Its jib painted yellow against the sky, the crane turned slowly.", "machine"),
    ("The lifting certificate had expired, so the crane was taken out of service.", "machine"),
]

MEDIUM = [
    ("The crane stood at the edge of the shallows with its wings folded.", "bird"),
    ("The crane held its place on one leg, feathers ruffled by the wind.", "bird"),
    ("The crane reached into the mud, stretching its long beak forward.", "bird"),
    ("The crane waited without moving, its narrow legs half submerged.", "bird"),
    ("The crane searched for food while wading between the reeds.", "bird"),
    ("The crane folded its wings once the flock had settled.", "bird"),
    ("The crane stayed in the shallow water, preening slowly.", "bird"),
    ("The crane lifted from the ground with a sudden beat of feathers.", "bird"),
    ("The crane circled once, its plumage grey against the sky.", "bird"),
    ("The crane returned to the marsh at the start of nesting season.", "bird"),

    ("The crane held the load steady with its steel cable drawn tight.", "machine"),
    ("The crane waited for the signal, hook lowered toward the platform.", "machine"),
    ("The crane reached the upper floor, extending its hydraulic boom.", "machine"),
    ("The crane rotated on its base, its counterweight swinging slightly.", "machine"),
    ("The crane was moved into position by the driver in the cab.", "machine"),
    ("The crane stood ready for the next lift, its diesel engine idling.", "machine"),
    ("The crane rose above the scaffolding, bolted to a concrete footing.", "machine"),
    ("The crane raised the steel girder with a grinding of gears.", "machine"),
    ("The crane turned slowly, its jib painted yellow against the sky.", "machine"),
    ("The crane was taken out of service once the lifting certificate expired.", "machine"),
]

WEAK = [
    ("The crane stood near the water while it watched for movement.", "bird"),
    ("The crane remained still while the others searched the shallows.", "bird"),
    ("The crane stayed at the edge of the area until it moved off toward the reeds.", "bird"),
    ("The crane held its position while the light faded over the marsh.", "bird"),
    ("They watched the crane closely while it fed along the bank.", "bird"),
    ("We noticed the crane again after it returned to the wetland.", "bird"),
    ("The team observed the crane until it settled for the evening.", "bird"),
    ("The crane was noticed by everyone once it stretched its neck upward.", "bird"),
    ("The crane was recorded in the log after it left the shallow pool.", "bird"),
    ("The crane that stood by the water had not moved its wings for a long time.", "bird"),

    ("The crane stood near the water while it waited for operation.", "machine"),
    ("The crane remained still while the others secured the load.", "machine"),
    ("The crane stayed at the edge of the area until it moved off toward the next bay.", "machine"),
    ("The crane held its position while the light faded over the yard.", "machine"),
    ("They watched the crane closely while it worked along the quay.", "machine"),
    ("We noticed the crane again after it returned to the site.", "machine"),
    ("The team observed the crane until it shut down for the evening.", "machine"),
    ("The crane was noticed by everyone once it extended its arm upward.", "machine"),
    ("The crane was recorded in the log after it left the loading point.", "machine"),
    ("The crane that stood by the water had not moved its cable for a long time.", "machine"),
]

LEVELS = {"strong": STRONG, "medium": MEDIUM, "weak": WEAK}


if __name__ == "__main__":
    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)

    for level, rows in LEVELS.items():
        name = f"crane_probe_{level}"
        json.dump({
            "target_word": "crane",
            "sentences": [PREFIX + s for s, _ in rows],
            "labels": [l for _, l in rows],
            "prefix": PREFIX,
            "design": f"cue-strength calibration probe, level={level}",
        }, open(out / f"{name}.json", "w"), indent=2, ensure_ascii=False)

        (cfg / f"{name}.yaml").write_text(
            f"word: crane\n"
            f"dataset: data/raw/semantic_sentences/{name}.json\n"
            f"senses:\n  - bird\n  - machine\n"
            f"n_per_sense: 10\n"
            f'prefix: "{PREFIX}"\n'
        )
        print(f"wrote {name}: {len(rows)} sentences")
