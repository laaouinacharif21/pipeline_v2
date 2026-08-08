# -*- coding: utf-8 -*-
"""
Crane minimal-pair set, extended to 100 pairs.

Each pair shares a frame and differs only in a short phrase naming a property
of the referent. The disambiguating phrase always precedes the target word, so
that causal decoder models can use it: a calibration probe showed decoders
produce exactly zero separation when the cue follows the target, while encoder
models are unaffected.

Cues name referent properties (wings, beak, plumage; cable, boom, counterweight)
rather than topic fields (marsh, construction site), so that the two classes are
not separable by subject matter alone.

This set exists to test whether the target-token representation predicts the
sense better than the surrounding context does. An earlier run at 20 pairs
suggested it does, but with five folds over twenty groups the estimate rested on
eight test items per fold. One hundred pairs makes it measurable.

Output: data/raw/semantic_sentences/crane_minimal.json
        configs/words/crane_minimal.yaml
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

# (frame with {cue}, bird cue, machine cue)
PAIRS = [
    ("With its {cue} clearly visible, the crane stood at the edge of the water.", "folded wings", "steel cable"),
    ("With its {cue} lowered slightly, the crane waited without moving.", "long beak", "heavy hook"),
    ("Its {cue} catching the light, the crane turned slowly to one side.", "grey plumage", "yellow jib"),
    ("Because its {cue} had been damaged, the crane could not move properly.", "left wing", "main cable"),
    ("Balanced on its {cue}, the crane held still for several minutes.", "thin legs", "steel base"),
    ("Since its {cue} was fully extended, the crane looked much larger.", "wingspan", "boom"),
    ("Under its {cue}, the crane kept something out of sight.", "folded wing", "raised arm"),
    ("Guided by its {cue}, the crane moved with unexpected precision.", "sharp eyesight", "control system"),
    ("Although its {cue} was worn, the crane continued as before.", "outer plumage", "outer casing"),
    ("Without its {cue}, the crane would not have managed the task.", "long neck", "extending arm"),
    ("Covered in {cue}, the crane looked older than it was.", "damp feathers", "dried cement"),
    ("Its {cue} folded away, the crane occupied very little space.", "broad wings", "long boom"),
    ("With its {cue} fully open, the crane looked considerably wider.", "wing feathers", "outrigger legs"),
    ("Given the state of its {cue}, the crane clearly needed attention.", "worn feathers", "worn cables"),
    ("Its {cue} marked with a tag, the crane was easy to identify.", "left leg", "lower frame"),
    ("Because of its {cue}, the crane could reach further than expected.", "long neck", "long jib"),
    ("Its {cue} moving slightly, the crane adjusted its balance.", "tail feathers", "counterweight"),
    ("With its {cue} tucked in, the crane took up less room.", "head and neck", "boom and hook"),
    ("Its {cue} clearly worn down, the crane had seen many seasons.", "beak", "hook"),
    ("Judging by its {cue}, the crane was fully mature.", "adult plumage", "service record"),
    ("Its {cue} raised high, the crane surveyed the area around it.", "head", "boom"),
    ("With its {cue} braced, the crane resisted the strong wind.", "legs", "outriggers"),
    ("Its {cue} dark against the sky, the crane was easy to pick out.", "silhouette", "lattice frame"),
    ("Because its {cue} was so distinctive, the crane was recognised at once.", "call", "livery"),
    ("Its {cue} slowly extending, the crane prepared for what came next.", "neck", "arm"),
    ("With its {cue} unusually still, the crane seemed to be resting.", "wings", "cables"),
    ("Its {cue} needed cleaning, so the crane was left alone for a while.", "feathers", "housing"),
    ("Given the length of its {cue}, the crane could cover a wide area.", "wingspan", "jib"),
    ("Its {cue} bent slightly, the crane leaned toward the left.", "long neck", "steel mast"),
    ("With its {cue} recently checked, the crane was ready to continue.", "ring and tag", "load certificate"),
    ("Its {cue} exposed to the weather, the crane had begun to look worn.", "outer feathers", "outer panels"),
    ("Because of the shape of its {cue}, the crane was unmistakable.", "beak", "hook block"),
    ("Its {cue} lowered to the ground, the crane came fully to rest.", "head", "hook"),
    ("With its {cue} clearly numbered, the crane was easy to track.", "leg ring", "chassis plate"),
    ("Its {cue} spread wide, the crane occupied the whole clearing.", "wings", "outriggers"),
    ("Because its {cue} had stiffened, the crane moved more slowly.", "joints", "bearings"),
    ("Its {cue} pointing forward, the crane faced into the wind.", "beak", "jib"),
    ("With its {cue} secured, the crane was safe to leave overnight.", "nest", "load"),
    ("Its {cue} pale in the low light, the crane blended into the background.", "plumage", "paintwork"),
    ("Because of its {cue}, the crane was heavier than it looked.", "dense bones", "steel ballast"),
    ("Its {cue} held steady, the crane did not shift at all.", "wings", "cables"),
    ("With its {cue} freshly marked, the crane could be identified from afar.", "leg band", "boom number"),
    ("Its {cue} tucked close, the crane waited out the rain.", "head", "hook"),
    ("Because its {cue} was so long, the crane needed a wide clearance.", "wingspan", "boom"),
    ("Its {cue} covered in dust, the crane looked as though it had travelled far.", "feathers", "panels"),
    ("With its {cue} fully retracted, the crane looked much smaller.", "neck", "boom"),
    ("Its {cue} scarred from earlier damage, the crane still functioned well.", "wing", "frame"),
    ("Because of its {cue}, the crane could balance on uneven ground.", "long toes", "levelling jacks"),
    ("Its {cue} shifting slowly, the crane adjusted to the changing light.", "head", "cab"),
    ("With its {cue} in poor condition, the crane was not expected to last.", "flight feathers", "lifting cables"),
    ("Its {cue} clean and unmarked, the crane appeared to be new.", "plumage", "paint"),
    ("Because its {cue} was blocked, the crane could not proceed.", "flight path", "access route"),
    ("Its {cue} angled downward, the crane looked toward the ground.", "beak", "hook"),
    ("With its {cue} folded, the crane fitted easily into the space.", "wings", "outriggers"),
    ("Its {cue} recorded in the survey, the crane was formally logged.", "ring number", "serial number"),
    ("Because of the weight of its {cue}, the crane moved carefully.", "full crop", "loaded hook"),
    ("Its {cue} damp from the morning, the crane took time to dry out.", "feathers", "decking"),
    ("With its {cue} extended fully, the crane reached across the gap.", "neck", "jib"),
    ("Its {cue} unusually pale, the crane stood out from the others.", "plumage", "paintwork"),
    ("Because its {cue} needed repair, the crane was taken aside.", "damaged wing", "damaged cable"),
    ("Its {cue} lifted clear, the crane began to move forward.", "feet", "outriggers"),
    ("With its {cue} clearly marked, the crane was easy to follow.", "tagged leg", "numbered boom"),
    ("Its {cue} still folded, the crane had not yet prepared to move.", "wings", "arms"),
    ("Because of its {cue}, the crane could be spotted from the far bank.", "white plumage", "white paintwork"),
    ("Its {cue} turning slowly, the crane scanned the area.", "head", "cab"),
    ("With its {cue} secured for the night, the crane was left in place.", "roost", "load"),
    ("Its {cue} noticeably longer, the crane differed from the others nearby.", "neck", "boom"),
    ("Because its {cue} had been treated, the crane recovered quickly.", "injured wing", "worn cable"),
    ("Its {cue} bright in the sunlight, the crane was impossible to miss.", "plumage", "paint"),
    ("With its {cue} resting on the ground, the crane stayed completely still.", "feet", "outriggers"),
    ("Its {cue} rebuilt that season, the crane returned in better condition.", "nest", "gearbox"),
    ("Because of its {cue}, the crane managed the distance without difficulty.", "strong wings", "strong cables"),
    ("Its {cue} kept low, the crane avoided drawing attention.", "head", "boom"),
    ("With its {cue} inspected, the crane was cleared to continue.", "leg ring", "load chart"),
    ("Its {cue} ruffled by the wind, the crane turned away from it.", "feathers", "tarpaulins"),
    ("Because its {cue} was so wide, the crane needed space to manoeuvre.", "wingspan", "track width"),
    ("Its {cue} lowered carefully, the crane came to rest on the ground.", "body", "hook"),
    ("With its {cue} in view, the crane was easy to photograph.", "full plumage", "full boom"),
    ("Its {cue} newly fitted, the crane performed noticeably better.", "tail feathers", "brake system"),
    ("Because of its {cue}, the crane could be heard before it was seen.", "call", "engine"),
    ("Its {cue} held above the water, the crane moved slowly forward.", "feet", "hook"),
    ("With its {cue} tightly closed, the crane rested where it stood.", "beak", "grab"),
    ("Its {cue} weathered and grey, the crane looked older than expected.", "plumage", "steelwork"),
    ("Because its {cue} had grown back, the crane was able to continue.", "flight feathers", "replacement cables"),
    ("Its {cue} moving in rhythm, the crane made steady progress.", "wings", "pulleys"),
    ("With its {cue} logged and checked, the crane was ready for the season.", "ring", "certificate"),
    ("Its {cue} unusually short, the crane looked different from the rest.", "neck", "jib"),
    ("Because of its {cue}, the crane remained stable in the wind.", "wide stance", "wide base"),
    ("Its {cue} folded neatly, the crane occupied a small area.", "wings", "outriggers"),
    ("With its {cue} freshly repaired, the crane was back in use.", "wing", "winch"),
    ("Its {cue} pointing skyward, the crane held that position for some time.", "beak", "boom"),
    ("Because its {cue} was distinctive, the crane was easy to recognise.", "marking", "livery"),
    ("Its {cue} settled on the ground, the crane showed no sign of moving.", "body", "base"),
    ("With its {cue} spread out, the crane covered a surprising width.", "wings", "outriggers"),
    ("Its {cue} carefully counted, the crane was recorded in the register.", "eggs", "lifts"),
    ("Because of the condition of its {cue}, the crane needed checking.", "plumage", "cabling"),
    ("Its {cue} raised slightly, the crane prepared to move off.", "wings", "hook"),
    ("With its {cue} noted in the file, the crane was tracked over time.", "ring number", "asset number"),
    ("Its {cue} tucked beneath it, the crane looked smaller than usual.", "legs", "jacks"),
    ("Because its {cue} was in good order, the crane continued without trouble.", "plumage", "machinery"),
    ("Its {cue} clearly aged, the crane had been in the area for years.", "feathers", "paintwork"),
]


def build():
    rows = []
    for frame, bird, machine in PAIRS:
        rows.append((PREFIX + frame.format(cue=bird), "bird"))
        rows.append((PREFIX + frame.format(cue=machine), "machine"))
    return rows


if __name__ == "__main__":
    rows = build()
    labels = [r[1] for r in rows]

    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_word": "crane",
        "sentences": [r[0] for r in rows],
        "labels": labels,
        "stratum": ["minimal"] * len(rows),
        "prefix": PREFIX,
        "design": "minimal pairs only; shared frame, one cue phrase differing, "
                  "cue precedes the target; cues name referent properties",
    }, open(out / "crane_minimal.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "crane_minimal.yaml").write_text(
        "word: crane\n"
        "dataset: data/raw/semantic_sentences/crane_minimal.json\n"
        "senses:\n  - bird\n  - machine\n"
        f"n_per_sense: {labels.count('bird')}\n"
        f'prefix: "{PREFIX}"\n'
    )
    print(f"{len(PAIRS)} pairs -> {len(rows)} sentences "
          f"(bird {labels.count('bird')}, machine {labels.count('machine')})")
