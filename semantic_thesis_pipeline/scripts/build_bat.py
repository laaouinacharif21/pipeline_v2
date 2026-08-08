# -*- coding: utf-8 -*-
"""
Sentence set for the ambiguous noun "bat" (mammal / sports implement).

Two hundred sentences, one hundred per sense, in two strata:

    controlled  50 pairs sharing a frame, differing in one cue phrase
    natural     50 pairs with freer wording and no shared frame

Construction rules, established by pilot measurement:

    The disambiguating cue precedes the target word. Decoder models produce no
    separation when it follows, since causal attention cannot read it at the
    target position.

    Cues name properties of the referent (wings, claws, roost; handle, grip,
    willow) rather than whole topic fields (caves and night hunting against
    stadiums and innings), so the classes are not separable by subject alone.

    Noun uses only. The target is never sentence-initial and never capitalised
    mid-sentence.

Pilot values for the controlled construction, qwen2.5-7b:
    context-only 0.900, target probe 1.000, Sep(l) 0.086 at layer 13.
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

CONTROLLED = [
    ("Hanging by its {cue}, the bat stayed completely still.", "small claws", "leather grip"),
    ("With its {cue} folded in, the bat took up very little room.", "thin wings", "protective cover"),
    ("Because its {cue} was damaged, the bat was set aside.", "left wing", "wooden shaft"),
    ("Its {cue} worn smooth with use, the bat had seen years of service.", "claws", "handle"),
    ("Resting against the {cue}, the bat remained where it had been left.", "cave wall", "dressing room wall"),
    ("With the {cue} finally quiet, the bat settled for the night.", "colony", "clubhouse"),
    ("Its {cue} unusually pale, the bat stood out from the others.", "fur", "grain"),
    ("Because the {cue} had shifted, the bat was moved to a safer spot.", "roost", "rack"),
    ("Covered in fine {cue}, the bat looked older than it was.", "grey hair", "wood dust"),
    ("With its {cue} extended, the bat seemed much larger than before.", "wingspan", "full length"),
    ("Its {cue} carefully examined, the bat was recorded in the register.", "wing membrane", "surface grain"),
    ("Since the {cue} had been cleaned, the bat looked almost new.", "fur", "willow"),
    ("Guided by its {cue}, the bat found its way without difficulty.", "hearing", "balance"),
    ("With the {cue} approaching, the bat was brought back inside.", "cold season", "end of play"),
    ("Its {cue} slightly bent, the bat still worked as it should.", "wing bone", "lower edge"),
    ("Because the {cue} was too narrow, the bat could not pass through.", "gap in the roof", "gap in the rack"),
    ("With its {cue} tightly closed, the bat waited out the noise.", "wings", "case"),
    ("Its {cue} counted carefully, the bat entered the survey record.", "colony size", "weight in ounces"),
    ("Following an inspection of its {cue}, the bat was approved for use.", "wing condition", "blade condition"),
    ("With its {cue} finally repaired, the bat returned to service.", "torn wing", "cracked handle"),
    ("Its {cue} measured precisely, the bat was catalogued that afternoon.", "forearm", "blade width"),
    ("Because the {cue} felt rough, the bat was handled with care.", "membrane", "splintered edge"),
    ("With its {cue} coated in dust, the bat had clearly been undisturbed.", "fur", "surface"),
    ("Its {cue} the darkest of the group, the bat was easy to pick out.", "colouring", "varnish"),
    ("Since the {cue} was in poor shape, the bat was withdrawn.", "wing tissue", "grip binding"),
    ("With the {cue} removed, the bat could be examined properly.", "netting", "cover"),
    ("Its {cue} showing clear damage, the bat needed attention.", "wing tip", "toe end"),
    ("Because the {cue} was warm, the bat became more active.", "roosting space", "practice hall"),
    ("With its {cue} laid flat, the bat occupied the whole shelf.", "wings", "length"),
    ("Its {cue} recorded in detail, the bat was added to the collection.", "species", "maker's mark"),
    ("Following the loss of a {cue}, the bat was noticeably less effective.", "wing claw", "grip layer"),
    ("With the {cue} secured, the bat was left overnight.", "enclosure", "locker"),
    ("Its {cue} smoother than expected, the bat surprised the handler.", "fur", "finish"),
    ("Because the {cue} had dried out, the bat needed treatment.", "membrane", "willow"),
    ("With its {cue} in view, the bat could be identified at once.", "marking", "sticker"),
    ("Its {cue} lighter than the others, the bat was preferred.", "frame", "weight"),
    ("Since the {cue} had been replaced, the bat performed better.", "damaged claw", "worn grip"),
    ("With the {cue} switched off, the bat became still.", "lamp", "floodlight"),
    ("Its {cue} inspected annually, the bat remained in good order.", "wing health", "blade condition"),
    ("Because the {cue} was uneven, the bat rested at an angle.", "ledge", "shelf"),
    ("With its {cue} drawn close, the bat looked smaller than before.", "wings", "cover"),
    ("Its {cue} noted by the specialist, the bat was passed as sound.", "wing structure", "grain structure"),
    ("Following a check of its {cue}, the bat was returned.", "flight ability", "balance point"),
    ("With the {cue} sealed, the bat could not be reached.", "cave entrance", "storage case"),
    ("Its {cue} unusually fine, the bat drew attention immediately.", "fur", "grain"),
    ("Because the {cue} had loosened, the bat was fixed the same day.", "roost fitting", "handle binding"),
    ("With its {cue} clearly visible, the bat was photographed closely.", "wing veins", "grain lines"),
    ("Its {cue} thicker than usual, the bat felt heavier in the hand.", "body", "blade"),
    ("Since the {cue} had settled, the bat was left in peace.", "colony", "dust"),
    ("With its {cue} fully dry, the bat was ready again.", "wings", "surface"),
]

NATURAL = [
    ("Flitting between the rafters above, the bat was hard to follow.",
     "Passing between the players above, the bat was hard to follow."),
    ("Emerging at dusk from the roof space, the bat crossed the yard.",
     "Emerging at dusk from the kit bag, the bat crossed the yard."),
    ("Wrapped in a soft cloth for safety, the bat was carried carefully.",
     "Wrapped in a soft cloth for storage, the bat was carried carefully."),
    ("Small enough to sit in one hand, the bat weighed almost nothing.",
     "Light enough to hold in one hand, the bat felt well balanced."),
    ("Roosting quietly in the corner, the bat had gone unnoticed for weeks.",
     "Standing quietly in the corner, the bat had gone unused for weeks."),
    ("Startled by the sudden light, the bat retreated further back.",
     "Knocked by the sudden movement, the bat slid further back."),
    ("Feeding on insects through the evening, the bat rarely settled.",
     "Striking cleanly through the evening, the bat rarely missed."),
    ("Marked with a small numbered ring, the bat was easy to trace.",
     "Marked with a small numbered label, the bat was easy to trace."),
    ("Sheltering from the rain under the eaves, the bat waited it out.",
     "Sheltering from the rain under the cover, the bat stayed dry."),
    ("Weighing less than a matchbox, the bat was surprisingly delicate.",
     "Weighing more than expected, the bat was surprisingly solid."),
    ("Returning each night to the same beam, the bat had a fixed routine.",
     "Returning each week to the same locker, the bat had a fixed place."),
    ("Squeaking faintly in the darkness, the bat gave itself away.",
     "Cracking faintly on contact, the bat gave itself away."),
    ("Cared for by the local group, the bat was in good condition.",
     "Cared for by the equipment manager, the bat was in good condition."),
    ("Folded neatly against its body, the bat's wings were barely visible.",
     "Placed neatly against the bench, the bat's handle was barely visible."),
    ("Counted during the annual survey, the bat was one of many.",
     "Counted during the annual audit, the bat was one of many."),
    ("Living in the roof for several seasons, the bat had settled well.",
     "Sitting in the store for several seasons, the bat had aged well."),
    ("Disturbed by building work nearby, the bat found a new home.",
     "Damaged by rough handling nearby, the bat needed a new grip."),
    ("Warmed by the afternoon sun, the bat became restless.",
     "Warped by the afternoon sun, the bat became unusable."),
    ("Photographed in flight at dusk, the bat appeared as a blur.",
     "Photographed in motion at dusk, the bat appeared as a blur."),
    ("Handled only with gloves, the bat was treated with care.",
     "Handled only with clean hands, the bat was treated with care."),
    ("Sleeping upside down all day, the bat woke only at dusk.",
     "Standing upright all day, the bat was used only at dusk."),
    ("Nursed back to health slowly, the bat recovered fully.",
     "Restored to condition slowly, the bat performed well again."),
    ("Detected by its high calls, the bat was located quickly.",
     "Detected by its distinctive sound, the bat was located quickly."),
    ("Hunting low over the water, the bat covered the same route.",
     "Sweeping low through the air, the bat followed the same arc."),
    ("Protected under local regulations, the bat could not be moved.",
     "Protected under tournament rules, the bat could not be changed."),
    ("Occupying a crack in the stonework, the bat stayed hidden.",
     "Occupying a slot in the rack, the bat stayed upright."),
    ("Fed on a diet of moths, the bat grew steadily.",
     "Treated with linseed oil, the bat lasted longer."),
    ("Ringed as a juvenile years ago, the bat was still healthy.",
     "Bought as a spare years ago, the bat was still usable."),
    ("Chased by an owl one evening, the bat escaped narrowly.",
     "Chipped against the fence one evening, the bat survived narrowly."),
    ("Sharing the loft with dozens of others, the bat was unremarkable.",
     "Sharing the rack with dozens of others, the bat was unremarkable."),
    ("Roosting close to the chimney breast, the bat kept warm.",
     "Resting close to the pavilion door, the bat stayed dry."),
    ("Identified by an expert on sight, the bat was a rare species.",
     "Identified by an expert on sight, the bat was a rare model."),
    ("Circling twice before settling, the bat chose a high spot.",
     "Turning twice before settling, the bat found its balance."),
    ("Weakened by the long winter, the bat emerged late.",
     "Weakened by repeated use, the bat cracked early."),
    ("Recorded on the survey form, the bat was one of three found.",
     "Recorded on the inventory form, the bat was one of three found."),
    ("Clinging to the underside of the beam, the bat held firm.",
     "Leaning against the side of the bench, the bat stayed put."),
    ("Released back into the wild that evening, the bat flew off.",
     "Returned to the kit that evening, the bat was packed away."),
    ("Threatened by loss of habitat, the bat had fewer options.",
     "Threatened by wear and splitting, the bat had less life left."),
    ("Nesting in an old barn nearby, the bat was rarely seen.",
     "Stored in an old bag nearby, the bat was rarely used."),
    ("Silhouetted against the evening sky, the bat was unmistakable.",
     "Silhouetted against the evening sky, the bat was unmistakable in shape."),
    ("Caught briefly in the torch beam, the bat vanished again.",
     "Caught briefly in the floodlight, the bat gleamed once."),
    ("Studied by researchers for years, the bat was well documented.",
     "Used by the same player for years, the bat was well known."),
    ("Emerging from hibernation in spring, the bat was thin.",
     "Emerging from storage in spring, the bat was dry."),
    ("Navigating without any light, the bat moved confidently.",
     "Swinging without any hesitation, the bat moved cleanly."),
    ("Sheltered in a hollow tree, the bat passed the winter.",
     "Sheltered in a padded case, the bat passed the winter."),
    ("Grooming itself between flights, the bat kept clean.",
     "Wiped down between innings, the bat stayed clean."),
    ("Weighing only a few grams, the bat was easily overlooked.",
     "Weighing well over a kilo, the bat was hard to swing."),
    ("Alarmed by the noise below, the bat did not move.",
     "Jarred by the impact below, the bat did not break."),
    ("Watched through an infrared camera, the bat was clearly active.",
     "Watched through a slow-motion camera, the bat was clearly effective."),
    ("Living undisturbed for a decade, the bat had a long life.",
     "Kept in good repair for a decade, the bat had a long life."),
]


def build():
    rows = []
    for frame, mammal, sport in CONTROLLED:
        rows.append((PREFIX + frame.format(cue=mammal), "mammal", "controlled"))
        rows.append((PREFIX + frame.format(cue=sport), "sport", "controlled"))
    for mammal, sport in NATURAL:
        rows.append((PREFIX + mammal, "mammal", "natural"))
        rows.append((PREFIX + sport, "sport", "natural"))
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
        "stratum": [r[2] for r in rows],
        "prefix": PREFIX,
        "design": "50 controlled pairs and 50 natural pairs; cue precedes the "
                  "target and names a referent property rather than a topic field",
    }, open(out / "bat.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "bat.yaml").write_text(
        "word: bat\n"
        "dataset: data/raw/semantic_sentences/bat.json\n"
        "senses:\n  - mammal\n  - sport\n"
        "n_per_sense: 100\n"
        f'prefix: "{PREFIX}"\n'
    )
    print(f"{len(rows)} sentences: mammal {labels.count('mammal')}, "
          f"sport {labels.count('sport')}")
