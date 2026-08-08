# -*- coding: utf-8 -*-
"""
Sentence set for the ambiguous noun "seal" (marine mammal / closure).

Two hundred sentences, one hundred per sense, in two strata:

    controlled  50 pairs sharing a frame, differing in one cue phrase
    natural     50 pairs with freer wording and no shared frame

Construction follows the recipe validated on the bat pilot:

    The disambiguating cue precedes the target word. Decoder models produce no
    separation when it follows, since causal attention cannot read it at the
    target position.

    Cues name properties of the referent (whiskers, flippers, pup; wax, rubber,
    rim) rather than whole topic fields, so the classes are not separable by
    subject matter alone.

    Noun uses only. The verb sense of "seal" is excluded, and inflected forms
    such as "sealed" and "sealing" do not appear. The target is never
    sentence-initial and never capitalised mid-sentence.
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

CONTROLLED = [
    ("With its {cue} clearly visible, the seal was easy to identify.", "long whiskers", "raised emblem"),
    ("Because its {cue} was damaged, the seal no longer functioned properly.", "front flipper", "rubber rim"),
    ("Its {cue} worn with age, the seal showed its years.", "coat", "surface"),
    ("Resting against the {cue}, the seal stayed where it was.", "harbour wall", "door frame"),
    ("With its {cue} still wet, the seal glistened in the light.", "fur", "wax"),
    ("Its {cue} unusually dark, the seal stood out from the rest.", "colouring", "impression"),
    ("Because the {cue} had shifted, the seal was checked again.", "sandbank", "casing"),
    ("Covered in fine {cue}, the seal looked neglected.", "sand", "dust"),
    ("With its {cue} pressed flat, the seal took up almost no space.", "flippers", "edges"),
    ("Its {cue} examined closely, the seal was recorded in the log.", "markings", "lettering"),
    ("Since the {cue} had been cleaned, the seal looked almost new.", "fur", "brass"),
    ("With the {cue} approaching, the seal was moved to safety.", "storm", "deadline"),
    ("Its {cue} slightly cracked, the seal still held together.", "muzzle", "wax"),
    ("Because the {cue} was too tight, the seal could not settle.", "netting", "housing"),
    ("With its {cue} fully extended, the seal reached further than expected.", "neck", "rim"),
    ("Its {cue} counted carefully, the seal entered the official record.", "colony", "batch"),
    ("Following an inspection of its {cue}, the seal was approved.", "condition", "integrity"),
    ("With its {cue} finally repaired, the seal was back in place.", "injured flipper", "broken rim"),
    ("Its {cue} measured precisely, the seal was catalogued that day.", "length", "diameter"),
    ("Because the {cue} felt rough, the seal was handled with care.", "skin", "edge"),
    ("With its {cue} coated in salt, the seal needed cleaning.", "fur", "surface"),
    ("Its {cue} the palest of the group, the seal was easy to spot.", "pelt", "wax"),
    ("Since the {cue} was in poor shape, the seal was replaced.", "hind flipper", "gasket"),
    ("With the {cue} lifted, the seal could be examined properly.", "netting", "lid"),
    ("Its {cue} showing clear damage, the seal needed attention.", "eye", "rim"),
    ("Because the {cue} was warm, the seal was more comfortable.", "shallow water", "workshop"),
    ("With its {cue} laid out, the seal covered the whole surface.", "body", "impression"),
    ("Its {cue} noted in detail, the seal was added to the archive.", "species", "design"),
    ("Following the loss of a {cue}, the seal was less effective.", "whisker", "ring"),
    ("With the {cue} closed, the seal was left undisturbed overnight.", "gate", "case"),
    ("Its {cue} smoother than expected, the seal surprised everyone.", "skin", "finish"),
    ("Because the {cue} had dried out, the seal needed treatment.", "hide", "wax"),
    ("With its {cue} in view, the seal was recognised at once.", "tag", "crest"),
    ("Its {cue} heavier than the others, the seal was set apart.", "build", "casing"),
    ("Since the {cue} had been renewed, the seal performed well again.", "tag", "lining"),
    ("With the {cue} switched off, the seal became still.", "lamp", "press"),
    ("Its {cue} inspected each year, the seal remained in good order.", "health", "condition"),
    ("Because the {cue} was uneven, the seal did not sit correctly.", "rock ledge", "flange"),
    ("With its {cue} drawn in, the seal looked smaller than before.", "flippers", "edges"),
    ("Its {cue} confirmed by the specialist, the seal was accepted.", "species", "authenticity"),
    ("Following a check of its {cue}, the seal was cleared.", "breathing", "tightness"),
    ("With the {cue} blocked, the seal could not be reached.", "channel", "opening"),
    ("Its {cue} unusually fine, the seal attracted attention.", "fur", "engraving"),
    ("Because the {cue} had loosened, the seal was fixed that morning.", "collar", "fitting"),
    ("With its {cue} clearly marked, the seal was photographed closely.", "flipper tag", "date stamp"),
    ("Its {cue} thicker than usual, the seal felt more substantial.", "blubber", "wax layer"),
    ("Since the {cue} had settled, the seal was left alone.", "colony", "resin"),
    ("With its {cue} fully dry, the seal was ready again.", "coat", "surface"),
    ("Its {cue} recorded on the form, the seal was formally logged.", "tag number", "reference number"),
    ("Because the {cue} was narrow, the seal barely fitted.", "gap in the rocks", "gap in the frame"),
]

NATURAL = [
    ("Basking on the rocks in the afternoon sun, the seal barely moved.",
     "Resting in the drawer since the spring, the seal had not been used."),
    ("Diving below the surface for several minutes, the seal disappeared.",
     "Pressed into the warm wax by hand, the seal left a clear mark."),
    ("Raised in captivity from a young pup, the seal was unusually tame.",
     "Cast in brass by a local maker, the seal was unusually heavy."),
    ("Feeding close to the shoreline all morning, the seal stayed nearby.",
     "Fitted around the door frame last year, the seal kept out the draught."),
    ("Tagged by researchers the previous season, the seal was easy to track.",
     "Stamped with an official mark, the seal was easy to verify."),
    ("Hauled out on the sandbank at low tide, the seal rested quietly.",
     "Laid out on the desk beside the letters, the seal waited to be used."),
    ("Startled by the boat passing close by, the seal slipped away.",
     "Loosened by the heat of the engine, the seal began to fail."),
    ("Recovering from an injury to its flipper, the seal was kept inshore.",
     "Perished after years of exposure, the seal was replaced entirely."),
    ("Watched from the cliff path each evening, the seal became familiar.",
     "Kept in the same box for years, the seal became worn."),
    ("Living in the estuary through the winter, the seal grew heavier.",
     "Sitting in the toolbox through the winter, the seal grew brittle."),
    ("Photographed from a distance at dawn, the seal looked almost still.",
     "Photographed under strong light, the seal showed every detail."),
    ("Protected under conservation rules, the seal could not be disturbed.",
     "Protected under the terms of the contract, the seal could not be broken."),
    ("Hunting for fish in the shallow bay, the seal moved constantly.",
     "Holding the pressure in the pipe, the seal did its work quietly."),
    ("Weighing well over a hundred kilograms, the seal was substantial.",
     "Weighing only a few grams, the seal was easily mislaid."),
    ("Sheltering from the wind behind the rocks, the seal stayed low.",
     "Set behind a protective cover, the seal stayed clean."),
    ("Nursed at the rescue centre for weeks, the seal recovered slowly.",
     "Reconditioned at the workshop for days, the seal was serviceable again."),
    ("Calling loudly across the water, the seal drew attention.",
     "Cracking audibly under pressure, the seal drew attention."),
    ("Returning to the same beach each year, the seal was well known.",
     "Reused on every document that year, the seal was well known."),
    ("Grey and mottled along its back, the seal blended with the stones.",
     "Grey and slightly discoloured with age, the seal blended with the metal."),
    ("Swimming against a strong current, the seal made slow progress.",
     "Pressed against an uneven surface, the seal made poor contact."),
    ("Counted during the annual survey, the seal was one of forty.",
     "Counted during the annual audit, the seal was one of forty."),
    ("Disturbed by construction along the coast, the seal moved on.",
     "Damaged by repeated opening, the seal was discarded."),
    ("Sleeping in the shallows near the pier, the seal was undisturbed.",
     "Stored in the cabinet near the press, the seal was undisturbed."),
    ("Following the fishing boats out to sea, the seal kept pace.",
     "Following the manufacturer's specification exactly, the seal fitted well."),
    ("Identified by an expert on sight, the seal was a rare species.",
     "Identified by an expert on sight, the seal was a rare design."),
    ("Cared for by the local wildlife group, the seal thrived.",
     "Cared for by the archive staff, the seal survived intact."),
    ("Weakened after a long winter, the seal was underweight.",
     "Weakened after long exposure, the seal was ineffective."),
    ("Surfacing briefly to breathe, the seal vanished again.",
     "Softening briefly under heat, the seal set again."),
    ("Released back into open water, the seal swam strongly.",
     "Returned to the storage case, the seal was put away."),
    ("Threatened by falling fish stocks, the seal had fewer options.",
     "Threatened by constant vibration, the seal had a shorter life."),
    ("Gathered with others on the shingle, the seal was hard to distinguish.",
     "Grouped with others in the tray, the seal was hard to distinguish."),
    ("Silhouetted against the bright water, the seal was unmistakable.",
     "Outlined against the pale paper, the seal was unmistakable."),
    ("Caught briefly on the camera trap, the seal appeared healthy.",
     "Caught briefly under the microscope, the seal appeared sound."),
    ("Studied by biologists for a decade, the seal was well documented.",
     "Used by the same office for a decade, the seal was well known."),
    ("Emerging from the water onto the ice, the seal moved awkwardly.",
     "Emerging from the mould still warm, the seal held its shape."),
    ("Navigating by sound in murky water, the seal hunted successfully.",
     "Working under constant pressure, the seal performed reliably."),
    ("Wintering in the sheltered inlet, the seal survived the season.",
     "Wintering in the unheated store, the seal became stiff."),
    ("Grooming itself on the warm rock, the seal seemed content.",
     "Wiped clean after every use, the seal stayed serviceable."),
    ("Barely visible among the waves, the seal was easily missed.",
     "Barely visible against the paper, the seal was easily missed."),
    ("Alarmed by the sudden noise, the seal returned to the water.",
     "Affected by the sudden heat, the seal lost its shape."),
    ("Observed through a long lens, the seal was clearly active.",
     "Observed under strong magnification, the seal was clearly intact."),
    ("Living undisturbed for many years, the seal reached old age.",
     "Kept in good condition for many years, the seal remained usable."),
    ("Rolling onto its side in the sun, the seal looked comfortable.",
     "Set firmly into its housing, the seal looked secure."),
    ("Trailing a small tracking device, the seal was monitored daily.",
     "Bearing a small reference code, the seal was checked daily."),
    ("Competing with others for space, the seal claimed a good spot.",
     "Competing with cheaper alternatives, the seal held its reputation."),
    ("Fed by hand at the sanctuary, the seal became trusting.",
     "Applied by hand at the bindery, the seal became a signature detail."),
    ("Marked with a numbered flipper tag, the seal was traceable.",
     "Marked with a numbered die, the seal was traceable."),
    ("Slipping quietly off the rocks, the seal entered the water.",
     "Slipping quietly out of position, the seal allowed a leak."),
    ("Recorded on the coastal survey, the seal was noted twice.",
     "Recorded in the inventory list, the seal was noted twice."),
    ("Growing rapidly through the summer, the seal doubled in size.",
     "Hardening steadily over the summer, the seal became rigid."),
]


def build():
    rows = []
    for frame, animal, closure in CONTROLLED:
        rows.append((PREFIX + frame.format(cue=animal), "animal", "controlled"))
        rows.append((PREFIX + frame.format(cue=closure), "closure", "controlled"))
    for animal, closure in NATURAL:
        rows.append((PREFIX + animal, "animal", "natural"))
        rows.append((PREFIX + closure, "closure", "natural"))
    return rows


if __name__ == "__main__":
    rows = build()
    labels = [r[1] for r in rows]
    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_word": "seal",
        "sentences": [r[0] for r in rows],
        "labels": labels,
        "stratum": [r[2] for r in rows],
        "prefix": PREFIX,
        "design": "50 controlled pairs and 50 natural pairs; cue precedes the "
                  "target and names a referent property rather than a topic field",
    }, open(out / "seal.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "seal.yaml").write_text(
        "word: seal\n"
        "dataset: data/raw/semantic_sentences/seal.json\n"
        "senses:\n  - animal\n  - closure\n"
        "n_per_sense: 100\n"
        f'prefix: "{PREFIX}"\n'
    )
    print(f"{len(rows)} sentences: animal {labels.count('animal')}, "
          f"closure {labels.count('closure')}")
