# -*- coding: utf-8 -*-
"""
Controlled sentence set for the ambiguous noun "crane" (bird / machine),
second version.

Why this differs from the first version
---------------------------------------
In the first version the disambiguating clause followed the target word. A
calibration probe showed that decoder models then produce exactly zero
separation, while encoder models are unaffected:

    cue before target    llama-7b 0.183   bert-base 0.088
    cue after target     llama-7b 0.000   bert-base 0.086

Causal attention cannot read past the current token, so a cue placed after
the target is invisible to a decoder at the moment the target is encoded.
A dataset built that way yields an apparent architecture difference that is
an artefact of sentence construction rather than of the models.

Every sentence here therefore places its disambiguating content before the
target word.

Design
------
Cues name properties of the referent (wings, beak, plumage; cable, boom,
hook) rather than topic fields (marsh, construction site), so separation
cannot be attributed to the two classes discussing different subjects.

Composition per sense (100 each)
    20  minimal pairs   identical frame, one cue noun swapped
    60  natural         varied phrasing, cue before the target
    20  hard            weaker cues, still before the target

Constraints
    noun uses only; "crane" is never a verb
    never sentence-initial, never capitalised mid-sentence
    frames are shared between senses, so length is matched by construction
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

# (syntax, frame with {cue}, bird cue, machine cue)
MINIMAL = [
    ("participial", "With its {cue} clearly visible, the crane stood at the edge of the water.",
     "folded wings", "steel cable"),
    ("participial", "With its {cue} lowered slightly, the crane waited without moving.",
     "long beak", "heavy hook"),
    ("prepositional", "Behind the {cue}, the crane remained in the same position all morning.",
     "tall reeds", "site fence"),
    ("participial", "Its {cue} catching the light, the crane turned slowly to one side.",
     "grey plumage", "yellow jib"),
    ("subordinate", "Because its {cue} had been damaged, the crane could not move properly.",
     "left wing", "main cable"),
    ("temporal", "After the {cue} arrived, the crane settled into its usual place.",
     "rest of the flock", "rest of the equipment"),
    ("participial", "Balanced on its {cue}, the crane held still for several minutes.",
     "thin legs", "steel base"),
    ("prepositional", "Among the {cue}, the crane was difficult to see at first.",
     "marsh grasses", "stacked containers"),
    ("subordinate", "Since its {cue} was fully extended, the crane looked much larger.",
     "wingspan", "boom"),
    ("participial", "Feathers and {cue} aside, the crane was an impressive sight.",
     "hollow bones", "counterweights"),
    ("coordination", "The {cue} was inspected first and the crane was checked afterwards.",
     "nest", "footing"),
    ("temporal", "During the {cue}, the crane stayed close to the same spot.",
     "moulting season", "night shift"),
    ("prepositional", "Under its {cue}, the crane kept something out of sight.",
     "folded wing", "raised arm"),
    ("participial", "Guided by its {cue}, the crane moved with unexpected precision.",
     "sharp eyesight", "control system"),
    ("subordinate", "Although its {cue} was worn, the crane continued as before.",
     "outer plumage", "outer casing"),
    ("object", "Noting the {cue}, the inspector recorded the crane in the log.",
     "unusual markings", "serial number"),
    ("existential", "Beside the {cue} there was a crane that had not moved in hours.",
     "shallow pool", "concrete slab"),
    ("prepositional", "Without its {cue}, the crane would not have managed the task.",
     "long neck", "extending arm"),
    ("participial", "Covered in {cue}, the crane looked older than it was.",
     "damp feathers", "dried cement"),
    ("subordinate", "Once the {cue} had been checked, the crane was left alone.",
     "nesting site", "lifting gear"),
]

NATURAL = [
    ("participial", "Wading slowly through the shallows, the crane searched for something small.",
     "Swinging slowly above the platform, the crane searched for the marked point."),
    ("participial", "Stretching its neck upward, the crane looked toward the far side.",
     "Extending its boom upward, the crane reached toward the far side."),
    ("prepositional", "In the middle of the wetland, the crane stood entirely alone.",
     "In the middle of the compound, the crane stood entirely alone."),
    ("subordinate", "Because the flock had already left, the crane seemed unusually quiet.",
     "Because the shift had already ended, the crane seemed unusually quiet."),
    ("temporal", "Just before sunrise, the crane began to stir in the reeds.",
     "Just before sunrise, the crane began to move above the yard."),
    ("participial", "Preening carefully, the crane paid no attention to the group.",
     "Idling quietly, the crane drew no attention from the group."),
    ("prepositional", "At the edge of the marsh, the crane had left clear footprints.",
     "At the edge of the slab, the crane had left clear tyre marks."),
    ("participial", "Feeding steadily since dawn, the crane showed no sign of leaving.",
     "Working steadily since dawn, the crane showed no sign of stopping."),
    ("subordinate", "When the water level dropped, the crane moved further out.",
     "When the load was released, the crane moved further along."),
    ("coordination", "The reeds parted and the crane stepped carefully into the open.",
     "The gates opened and the crane rolled carefully into the open."),
    ("participial", "Watching the surface intently, the crane waited for movement.",
     "Holding the cable steady, the crane waited for the signal."),
    ("prepositional", "Along the muddy shoreline, the crane had walked a considerable way.",
     "Along the concrete apron, the crane had travelled a considerable way."),
    ("object", "Following the flock with binoculars, they lost sight of the crane.",
     "Following the schedule on paper, they lost track of the crane."),
    ("temporal", "Throughout the nesting season, the crane returned to the same island.",
     "Throughout the building season, the crane returned to the same bay."),
    ("participial", "Standing on one leg, the crane appeared almost asleep.",
     "Resting on its outriggers, the crane appeared almost abandoned."),
    ("subordinate", "If the wind picks up, the crane usually moves to sheltered ground.",
     "If the wind picks up, the crane usually stops for safety reasons."),
    ("prepositional", "Beyond the line of trees, the crane was just about visible.",
     "Beyond the line of hoardings, the crane was just about visible."),
    ("participial", "Calling once across the water, the crane announced its presence.",
     "Sounding its alarm across the yard, the crane announced its movement."),
    ("existential", "Near the flooded field there was a crane that nobody disturbed.",
     "Near the excavated pit there was a crane that nobody approached."),
    ("temporal", "Each spring without fail, the crane appeared in the same wetland.",
     "Each spring without fail, the crane appeared on the same contract."),
    ("participial", "Startled by the noise, the crane lifted away from the bank.",
     "Halted by the alarm, the crane stopped short of the wall."),
    ("subordinate", "As the light faded, the crane settled among the tall grass.",
     "As the light faded, the crane settled onto its locking pins."),
    ("prepositional", "Across the shallow lagoon, the crane could be seen clearly.",
     "Across the loading area, the crane could be seen clearly."),
    ("object", "Tracking its migration route, the researchers found the crane again.",
     "Tracking its service record, the engineers found the crane again."),
    ("participial", "Shaking water from its feathers, the crane moved to drier ground.",
     "Shaking dust from its housing, the crane moved to firmer ground."),
    ("coordination", "The tide went out and the crane followed the retreating water.",
     "The delivery arrived and the crane followed the marked route."),
    ("subordinate", "Whenever the reeds were cut, the crane found somewhere else to feed.",
     "Whenever the ground was cleared, the crane found somewhere else to stand."),
    ("temporal", "Long before the others woke, the crane was already active.",
     "Long before the others arrived, the crane was already running."),
    ("participial", "Gliding low over the marsh, the crane came down beside the water.",
     "Tracking low along the rail, the crane came to rest beside the wall."),
    ("prepositional", "Within sight of the nest, the crane refused to move away.",
     "Within sight of the foundation, the crane remained exactly in place."),
    ("participial", "Ringed and tagged the previous year, the crane was easy to identify.",
     "Numbered and certified the previous year, the crane was easy to identify."),
    ("subordinate", "Although the flock had scattered, the crane stayed behind.",
     "Although the crew had left, the crane stayed in position."),
    ("object", "Counting the birds one by one, the warden included the crane.",
     "Checking the machines one by one, the foreman included the crane."),
    ("prepositional", "Between the reed beds, the crane had made a shallow nest.",
     "Between the storage bays, the crane had left a clear track."),
    ("temporal", "Some hours after first light, the crane finally took off.",
     "Some hours after first light, the crane finally started up."),
    ("participial", "Fed and rested, the crane prepared to continue its journey.",
     "Fuelled and serviced, the crane prepared to continue the work."),
    ("coordination", "The mist cleared and the crane became visible across the water.",
     "The tarpaulin came off and the crane became visible across the site."),
    ("existential", "Deep in the reeds there was a crane that had nested for years.",
     "Deep in the compound there was a crane that had stood for years."),
    ("subordinate", "Because the marsh had flooded, the crane found feeding easier.",
     "Because the ground had frozen, the crane found footing easier."),
    ("participial", "Alarmed by the movement, the crane retreated toward the reeds.",
     "Alerted by the movement, the crane stopped short of the trench."),
    ("prepositional", "Over the wet meadow, the crane circled once and landed.",
     "Over the levelled ground, the crane swung once and stopped."),
    ("temporal", "By the end of the migration, the crane had travelled a long way.",
     "By the end of the contract, the crane had lifted a great deal."),
    ("object", "Photographing the wetland at dawn, she captured the crane clearly.",
     "Photographing the site at dawn, she captured the crane clearly."),
    ("participial", "Nesting again that year, the crane raised two young successfully.",
     "Operating again that year, the crane completed two contracts successfully."),
    ("subordinate", "Since the shallows had dried, the crane moved to the river.",
     "Since the yard had cleared, the crane moved to the dock."),
    ("coordination", "The reeds swayed and the crane shifted its weight slightly.",
     "The cables tightened and the crane shifted its load slightly."),
    ("prepositional", "From the observation hide, the crane was clearly in view.",
     "From the site office, the crane was clearly in view."),
    ("participial", "Weighing barely six kilograms, the crane was lighter than expected.",
     "Weighing nearly sixty tonnes, the crane was heavier than expected."),
    ("temporal", "In the weeks after hatching, the crane rarely left the nest.",
     "In the weeks after delivery, the crane rarely left the bay."),
    ("subordinate", "While the flock fed nearby, the crane kept watch.",
     "While the crew worked nearby, the crane stood ready."),
    ("participial", "Perched at the water's edge, the crane looked out across the marsh.",
     "Positioned at the dock's edge, the crane looked out across the water."),
    ("object", "Recording every sighting that month, the survey listed the crane twice.",
     "Recording every movement that month, the log listed the crane twice."),
    ("prepositional", "Through the morning mist, the crane was barely distinguishable.",
     "Through the morning mist, the crane was barely distinguishable from the tower."),
    ("participial", "Having flown several hundred miles, the crane rested for two days.",
     "Having lifted several hundred tonnes, the crane was serviced for two days."),
    ("subordinate", "As soon as the ice melted, the crane returned to the wetland.",
     "As soon as the permit cleared, the crane returned to the site."),
    ("temporal", "On the coldest morning that winter, the crane did not move at all.",
     "On the coldest morning that winter, the crane would not start at all."),
    ("coordination", "The water rose and the crane retreated to higher ground.",
     "The water rose and the crane was moved to higher ground."),
    ("prepositional", "Against the pale sky, the crane made a distinctive shape.",
     "Against the pale sky, the crane made a distinctive outline."),
    ("participial", "Circling twice before landing, the crane chose a quiet spot.",
     "Swinging twice before stopping, the crane reached the marked spot."),
    ("existential", "Out on the mudflats there was a crane feeding by itself.",
     "Out on the hardstanding there was a crane standing by itself."),
]

HARD = [
    ("prepositional", "In the quiet of the early morning, the crane had not moved.",
     "In the quiet of the early morning, the crane had not started."),
    ("subordinate", "Because nothing had changed overnight, the crane stayed as it was.",
     "Because nothing had changed overnight, the crane stood as it was."),
    ("temporal", "For the third day running, the crane was exactly where it had been.",
     "For the third day running, the crane was exactly where it had been left."),
    ("participial", "Standing perfectly still, the crane gave nothing away.",
     "Standing perfectly still, the crane gave no sign of use."),
    ("object", "Looking out across the flat ground, they noticed the crane at once.",
     "Looking out across the flat ground, they spotted the crane at once."),
    ("prepositional", "At that distance, the crane was little more than an outline.",
     "At that distance, the crane was little more than a silhouette."),
    ("subordinate", "Although the weather had turned, the crane remained in place.",
     "Although the weather had turned, the crane remained on site."),
    ("coordination", "The light changed and the crane seemed to shift slightly.",
     "The light changed and the crane appeared to move slightly."),
    ("temporal", "Later that afternoon, the crane was still in the same position.",
     "Later that afternoon, the crane was still in the same bay."),
    ("participial", "Seen from the road, the crane looked smaller than expected.",
     "Seen from the road, the crane looked shorter than expected."),
    ("existential", "Further along there was a crane that nobody had mentioned.",
     "Further along there was a crane that nobody had logged."),
    ("prepositional", "Beyond the low wall, the crane was almost out of sight.",
     "Beyond the low wall, the crane was almost out of view."),
    ("subordinate", "When the noise stopped, the crane was the only thing left.",
     "When the noise stopped, the crane was the only thing running."),
    ("object", "Making a note of everything, the observer included the crane.",
     "Making a note of everything, the supervisor included the crane."),
    ("temporal", "By late evening, the crane had settled for the night.",
     "By late evening, the crane had shut down for the night."),
    ("participial", "Half hidden by the mist, the crane was hard to make out.",
     "Half hidden by the mist, the crane was hard to identify."),
    ("prepositional", "On the far side, the crane had been there since morning.",
     "On the far side, the crane had been there since the handover."),
    ("subordinate", "Since nobody was watching, the crane went unrecorded.",
     "Since nobody was watching, the crane went unattended."),
    ("coordination", "The ground was wet and the crane had left clear marks.",
     "The ground was wet and the crane had left deep marks."),
    ("temporal", "All through the week, the crane stayed exactly where it was.",
     "All through the week, the crane stayed exactly where it stood."),
]


def build():
    rows = []
    for tag, frame, bird, machine in MINIMAL:
        rows.append((PREFIX + frame.format(cue=bird), "bird", tag, "minimal"))
        rows.append((PREFIX + frame.format(cue=machine), "machine", tag, "minimal"))
    for tag, bird, machine in NATURAL:
        rows.append((PREFIX + bird, "bird", tag, "natural"))
        rows.append((PREFIX + machine, "machine", tag, "natural"))
    for tag, bird, machine in HARD:
        rows.append((PREFIX + bird, "bird", tag, "hard"))
        rows.append((PREFIX + machine, "machine", tag, "hard"))
    return rows


if __name__ == "__main__":
    rows = build()
    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_word": "crane",
        "sentences": [r[0] for r in rows],
        "labels": [r[1] for r in rows],
        "syntax": [r[2] for r in rows],
        "stratum": [r[3] for r in rows],
        "prefix": PREFIX,
        "design": "disambiguating cues precede the target word; cues name "
                  "referent properties rather than topic fields",
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
    labels = [r[1] for r in rows]
    print(f"{len(rows)} sentences: bird {labels.count('bird')}, machine {labels.count('machine')}")
