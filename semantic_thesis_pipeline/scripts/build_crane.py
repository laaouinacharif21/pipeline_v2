# -*- coding: utf-8 -*-
"""
Sentence set for the ambiguous noun "crane" (long-necked bird / lifting machine).

Two hundred sentences, one hundred per sense, in two strata:

    controlled  50 pairs sharing a frame, differing in one cue phrase
    natural     50 pairs with freer wording and no shared frame

This replaces two earlier attempts. The first placed the disambiguating clause
after the target word, which produced exactly zero separation in every decoder
model, since causal attention cannot read past the current token. The second
raised the cue too little to be measurable. The recipe used here was validated
on a bat pilot: cue before the target, naming a property of the referent rather
than a topic field.

Noun uses only. The target is never sentence-initial, never capitalised
mid-sentence, and never appears inside a longer word.
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

CONTROLLED = [
    ("With its {cue} clearly visible, the crane was easy to identify.", "folded wings", "steel cable"),
    ("Because its {cue} was damaged, the crane could not operate.", "left wing", "main hoist"),
    ("Its {cue} worn with age, the crane showed its years.", "plumage", "paintwork"),
    ("Standing beside the {cue}, the crane looked out of place.", "reed bed", "site office"),
    ("With its {cue} still wet, the crane stayed where it was.", "feathers", "surfaces"),
    ("Its {cue} unusually pale, the crane stood out from the rest.", "colouring", "casing"),
    ("Because the {cue} had shifted, the crane was checked again.", "nest", "ballast"),
    ("Covered in fine {cue}, the crane looked neglected.", "down", "rust"),
    ("With its {cue} fully extended, the crane reached its limit.", "neck", "boom"),
    ("Its {cue} recorded carefully, the crane entered the register.", "ring number", "serial number"),
    ("Since the {cue} had been cleaned, the crane looked almost new.", "feathers", "cab"),
    ("With the {cue} approaching, the crane was moved to safety.", "storm", "deadline"),
    ("Its {cue} slightly bent, the crane still functioned normally.", "wing bone", "jib section"),
    ("Because the {cue} was too narrow, the crane could not pass.", "channel", "gateway"),
    ("With its {cue} drawn back, the crane took up far less room.", "wings", "arm"),
    ("Its {cue} counted carefully, the crane entered the annual record.", "brood", "hours"),
    ("Following an inspection of its {cue}, the crane was approved.", "condition", "certification"),
    ("With its {cue} finally repaired, the crane returned to normal.", "injured wing", "broken winch"),
    ("Its {cue} measured precisely, the crane was catalogued that day.", "wingspan", "reach"),
    ("Because the {cue} felt unstable, the crane was moved.", "branch", "ground"),
    ("With its {cue} coated in dust, the crane needed cleaning.", "feathers", "windows"),
    ("Its {cue} the largest of the group, the crane was unmistakable.", "wingspan", "counterweight"),
    ("Since the {cue} was in poor shape, the crane was withdrawn.", "wing tissue", "cable"),
    ("With the {cue} removed, the crane could be examined properly.", "netting", "cover"),
    ("Its {cue} showing clear damage, the crane needed attention.", "wing tip", "hook"),
    ("Because the {cue} was warm, the crane settled comfortably.", "shallow water", "engine housing"),
    ("With its {cue} spread wide, the crane covered a surprising distance.", "wings", "outriggers"),
    ("Its {cue} noted in detail, the crane was added to the record.", "species", "specification"),
    ("Following the loss of a {cue}, the crane adapted slowly.", "flight feather", "safety pin"),
    ("With the {cue} closed for the night, the crane was left alone.", "reserve", "compound"),
    ("Its {cue} finer than expected, the crane surprised the observer.", "plumage", "engineering"),
    ("Because the {cue} had dried out, the crane needed treatment.", "wetland", "hydraulic line"),
    ("With its {cue} clearly marked, the crane was easy to trace.", "leg ring", "chassis plate"),
    ("Its {cue} heavier than the others, the crane needed more space.", "build", "base"),
    ("Since the {cue} had been renewed, the crane performed better.", "tag", "cable"),
    ("With the {cue} switched off, the crane became still.", "floodlight", "motor"),
    ("Its {cue} inspected each year, the crane remained in good order.", "health", "gear"),
    ("Because the {cue} was uneven, the crane leaned to one side.", "mud", "surface"),
    ("With its {cue} tucked in, the crane looked much smaller.", "head", "boom"),
    ("Its {cue} confirmed by the specialist, the crane was accepted.", "species", "load rating"),
    ("Following a check of its {cue}, the crane was cleared.", "flight", "brakes"),
    ("With the {cue} blocked, the crane could not be reached.", "path", "access road"),
    ("Its {cue} unusually dense, the crane attracted comment.", "plumage", "rigging"),
    ("Because the {cue} had loosened, the crane was secured again.", "nest fixing", "bolt"),
    ("With its {cue} clearly visible, the crane was photographed closely.", "wing veins", "cable drum"),
    ("Its {cue} thicker than usual, the crane seemed unusually solid.", "body", "frame"),
    ("Since the {cue} had settled, the crane was left undisturbed.", "flock", "dust"),
    ("With its {cue} fully dry, the crane was ready again.", "feathers", "surface"),
    ("Its {cue} written on the form, the crane was formally logged.", "ring number", "plant number"),
    ("Because the {cue} was soft, the crane sank slightly.", "mud", "ground"),
]

NATURAL = [
    ("Wading slowly through the shallows, the crane searched for food.",
     "Swinging slowly above the platform, the crane searched for the mark."),
    ("Stretching its long neck upward, the crane looked toward the water.",
     "Extending its steel arm upward, the crane reached the top floor."),
    ("Nesting in the same marsh each spring, the crane returned reliably.",
     "Working on the same contract each spring, the crane returned reliably."),
    ("Startled by the noise from the path, the crane lifted away.",
     "Halted by the alarm from the cab, the crane stopped short."),
    ("Feeding steadily since first light, the crane showed no sign of leaving.",
     "Running steadily since first light, the crane showed no sign of stopping."),
    ("Preening its feathers on the bank, the crane ignored the visitors.",
     "Idling between deliveries on site, the crane waited for instructions."),
    ("Migrating south for the winter months, the crane travelled far.",
     "Transported north for the winter contract, the crane travelled far."),
    ("Standing motionless on one leg, the crane appeared asleep.",
     "Standing motionless on its outriggers, the crane appeared abandoned."),
    ("Gliding low over the wet meadow, the crane came down softly.",
     "Tracking low along the rail, the crane came to a smooth stop."),
    ("Ringed as a chick several years ago, the crane was well studied.",
     "Commissioned several years ago for the docks, the crane was well used."),
    ("Calling loudly across the water, the crane announced itself.",
     "Sounding its warning across the yard, the crane announced its movement."),
    ("Sheltering from the wind among the reeds, the crane stayed low.",
     "Secured against the wind by heavy chains, the crane stayed put."),
    ("Weighing only a few kilograms, the crane was surprisingly light.",
     "Weighing nearly sixty tonnes, the crane needed a firm base."),
    ("Watched daily by local birdwatchers, the crane became familiar.",
     "Checked daily by the site inspector, the crane stayed compliant."),
    ("Raising two young that season, the crane was successful.",
     "Completing two contracts that season, the crane was productive."),
    ("Hunting in the shallow water at dusk, the crane moved carefully.",
     "Operating under floodlights at dusk, the crane moved carefully."),
    ("Recovering after an injury last year, the crane flew again.",
     "Repaired after a failure last year, the crane worked again."),
    ("Photographed at dawn from the hide, the crane looked magnificent.",
     "Photographed at dawn from the road, the crane dominated the skyline."),
    ("Protected under conservation law, the crane could not be disturbed.",
     "Regulated under safety law, the crane could not be operated untested."),
    ("Circling twice before landing, the crane chose a quiet spot.",
     "Swinging twice before stopping, the crane reached the exact spot."),
    ("Living in the wetland for a decade, the crane grew old there.",
     "Standing on the site for a decade, the crane became a landmark."),
    ("Tagged for a migration study, the crane was tracked by satellite.",
     "Fitted with a load monitor, the crane was tracked by computer."),
    ("Disturbed by drainage work nearby, the crane moved on.",
     "Displaced by new construction nearby, the crane was relocated."),
    ("Shaking water from its plumage, the crane moved to drier ground.",
     "Shedding dust from its housing, the crane moved to firmer ground."),
    ("Silhouetted against the evening sky, the crane was unmistakable.",
     "Outlined against the evening sky, the crane dominated the view."),
    ("Feeding alongside herons in the marsh, the crane was hard to pick out.",
     "Working alongside smaller machines, the crane was hard to miss."),
    ("Weakened after a long migration, the crane rested for days.",
     "Worn after a long project, the crane was serviced for days."),
    ("Nesting on a small island for safety, the crane raised its young.",
     "Sited on a reinforced pad for safety, the crane lifted its loads."),
    ("Returning to the estuary each autumn, the crane was expected.",
     "Returning to the depot each autumn, the crane was overhauled."),
    ("Startling easily when approached, the crane kept its distance.",
     "Stopping automatically when overloaded, the crane protected itself."),
    ("Studied by ornithologists for years, the crane was well documented.",
     "Operated by the same driver for years, the crane was well understood."),
    ("Growing rapidly in its first summer, the crane soon flew.",
     "Assembled rapidly in a single week, the crane soon worked."),
    ("Sleeping standing in the shallows, the crane was undisturbed.",
     "Parked upright on the hardstanding, the crane was undisturbed."),
    ("Competing with others for nesting space, the crane claimed a spot.",
     "Competing with newer machines for work, the crane held its place."),
    ("Fed by hand at the rescue centre, the crane became tame.",
     "Guided by hand signals on site, the crane moved precisely."),
    ("Emerging from the reeds at first light, the crane began feeding.",
     "Emerging from the mist at first light, the crane began lifting."),
    ("Navigating by landmarks over long distances, the crane found its way.",
     "Guided by markers over short distances, the crane found its position."),
    ("Wintering in a sheltered valley, the crane survived the season.",
     "Wintering in an unheated yard, the crane seized up."),
    ("Barely visible among the tall grasses, the crane was easily missed.",
     "Barely visible behind the hoarding, the crane was easily missed."),
    ("Alarmed by a passing dog, the crane took to the air.",
     "Jolted by a sudden gust, the crane swayed noticeably."),
    ("Observed through a telescope from the shore, the crane was clearly active.",
     "Observed through binoculars from the gate, the crane was clearly working."),
    ("Living undisturbed for many years, the crane reached old age.",
     "Maintained carefully for many years, the crane stayed serviceable."),
    ("Folding its wings neatly against its body, the crane settled.",
     "Folding its jib neatly against its mast, the crane shut down."),
    ("Trailing a small tracking device, the crane was monitored daily.",
     "Bearing a small inspection tag, the crane was checked daily."),
    ("Chosen as the emblem of the reserve, the crane was celebrated.",
     "Chosen as the largest on the fleet, the crane was in demand."),
    ("Slipping quietly into the water, the crane began to feed.",
     "Slipping slightly on the wet surface, the crane was repositioned."),
    ("Recorded on the wetland survey, the crane was noted twice.",
     "Recorded in the equipment log, the crane was noted twice."),
    ("Balanced carefully on the muddy bank, the crane held still.",
     "Balanced carefully on temporary matting, the crane held firm."),
    ("Adapting well to the restored habitat, the crane settled quickly.",
     "Adapting well to the confined site, the crane worked efficiently."),
    ("Recognised instantly by its call, the crane was identified.",
     "Recognised instantly by its livery, the crane was identified."),
]


def build():
    rows = []
    for frame, bird, machine in CONTROLLED:
        rows.append((PREFIX + frame.format(cue=bird), "bird", "controlled"))
        rows.append((PREFIX + frame.format(cue=machine), "machine", "controlled"))
    for bird, machine in NATURAL:
        rows.append((PREFIX + bird, "bird", "natural"))
        rows.append((PREFIX + machine, "machine", "natural"))
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
        "stratum": [r[2] for r in rows],
        "prefix": PREFIX,
        "design": "50 controlled pairs and 50 natural pairs; cue precedes the "
                  "target and names a referent property rather than a topic field",
    }, open(out / "crane.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "crane.yaml").write_text(
        "word: crane\n"
        "dataset: data/raw/semantic_sentences/crane.json\n"
        "senses:\n  - bird\n  - machine\n"
        "n_per_sense: 100\n"
        f'prefix: "{PREFIX}"\n'
    )
    print(f"{len(rows)} sentences: bird {labels.count('bird')}, "
          f"machine {labels.count('machine')}")
