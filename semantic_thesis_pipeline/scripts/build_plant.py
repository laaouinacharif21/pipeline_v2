# -*- coding: utf-8 -*-
"""
Sentence set for the ambiguous noun "plant" (vegetation / industrial facility).

Two hundred sentences, one hundred per sense, in two strata:

    controlled  50 pairs sharing a frame, differing in one cue phrase
    natural     50 pairs with freer wording and no shared frame

Construction follows the recipe validated on the bat pilot:

    The disambiguating cue precedes the target word. Decoder models produce no
    separation when it follows, since causal attention cannot read it at the
    target position.

    Cues name properties of the referent (leaves, roots, stem; boiler, output,
    shift) rather than whole topic fields, so the classes are not separable by
    subject matter alone.

    Noun uses only. The verb sense of "plant" is excluded, and inflected forms
    such as "planted" and "planting" do not appear. The target is never
    sentence-initial and never capitalised mid-sentence.
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

CONTROLLED = [
    ("With its {cue} clearly visible, the plant was easy to identify.", "broad leaves", "tall chimneys"),
    ("Because its {cue} had been damaged, the plant struggled to recover.", "root system", "cooling system"),
    ("Its {cue} showing signs of age, the plant needed attention.", "lower leaves", "outer casing"),
    ("Standing beside the {cue}, the plant looked out of place.", "garden path", "access road"),
    ("With its {cue} still wet, the plant was left to dry.", "foliage", "concrete floor"),
    ("Its {cue} unusually pale, the plant drew immediate concern.", "leaves", "exhaust"),
    ("Because the {cue} had shifted, the plant was inspected again.", "soil", "foundation"),
    ("Covered in fine {cue}, the plant looked neglected.", "dust on its leaves", "grey dust"),
    ("With its {cue} fully open, the plant reached its greatest spread.", "leaves", "vents"),
    ("Its {cue} recorded carefully, the plant was entered in the register.", "species", "capacity"),
    ("Since the {cue} had been cleaned, the plant looked much better.", "leaves", "windows"),
    ("With the {cue} approaching, the plant was prepared in advance.", "frost", "shutdown"),
    ("Its {cue} slightly damaged, the plant continued to function.", "outer leaf", "outer wall"),
    ("Because the {cue} was too small, the plant could not develop.", "container", "site"),
    ("With its {cue} fully extended, the plant occupied the whole space.", "branches", "conveyor"),
    ("Its {cue} counted carefully, the plant entered the annual survey.", "shoots", "units"),
    ("Following an inspection of its {cue}, the plant was approved.", "condition", "safety record"),
    ("With its {cue} finally repaired, the plant returned to normal.", "damaged stem", "damaged pipe"),
    ("Its {cue} measured precisely, the plant was catalogued that week.", "height", "output"),
    ("Because the {cue} felt dry, the plant was given attention.", "soil", "air"),
    ("With its {cue} coated in grime, the plant needed cleaning.", "leaves", "panels"),
    ("Its {cue} the largest in the group, the plant stood out clearly.", "canopy", "structure"),
    ("Since the {cue} was in poor condition, the plant was replaced.", "root ball", "boiler"),
    ("With the {cue} removed, the plant could be examined properly.", "netting", "casing"),
    ("Its {cue} showing clear stress, the plant needed intervention.", "foliage", "machinery"),
    ("Because the {cue} was warm, the plant did better than expected.", "greenhouse", "workshop"),
    ("With its {cue} spread out, the plant covered a wide area.", "roots", "buildings"),
    ("Its {cue} noted in detail, the plant was added to the record.", "variety", "specification"),
    ("Following the loss of a {cue}, the plant recovered slowly.", "main stem", "main pump"),
    ("With the {cue} closed for the night, the plant was left alone.", "greenhouse", "gate"),
    ("Its {cue} finer than expected, the plant surprised the inspector.", "leaf structure", "filtration"),
    ("Because the {cue} had dried out, the plant needed urgent care.", "soil", "coolant"),
    ("With its {cue} clearly labelled, the plant was easy to find.", "pot", "unit"),
    ("Its {cue} heavier than the others, the plant needed more support.", "growth", "machinery"),
    ("Since the {cue} had been renewed, the plant performed better.", "compost", "equipment"),
    ("With the {cue} switched off, the plant became quiet.", "irrigation", "turbine"),
    ("Its {cue} checked every season, the plant stayed in good order.", "growth", "output"),
    ("Because the {cue} was uneven, the plant leaned to one side.", "ground", "floor"),
    ("With its {cue} drawn in, the plant took up far less room.", "leaves", "scaffolding"),
    ("Its {cue} confirmed by the specialist, the plant was accepted.", "species", "certification"),
    ("Following a check of its {cue}, the plant was cleared.", "health", "emissions"),
    ("With the {cue} blocked, the plant could not be reached.", "path", "gate"),
    ("Its {cue} unusually dense, the plant attracted comment.", "foliage", "piping"),
    ("Because the {cue} had loosened, the plant was secured again.", "stake", "bolt"),
    ("With its {cue} clearly marked, the plant was photographed for the file.", "label", "sign"),
    ("Its {cue} thicker than usual, the plant seemed unusually robust.", "stem", "walls"),
    ("Since the {cue} had settled, the plant was left undisturbed.", "soil", "dust"),
    ("With its {cue} fully dry, the plant was ready to be moved.", "roots", "flooring"),
    ("Its {cue} written on the form, the plant was formally logged.", "variety name", "site number"),
    ("Because the {cue} was narrow, the plant barely fitted the space.", "border", "access route"),
]

NATURAL = [
    ("Growing steadily through the spring months, the plant doubled in size.",
     "Running steadily through the spring months, the plant met every target."),
    ("Placed near the window for the light, the plant thrived.",
     "Placed near the river for the water, the plant operated year round."),
    ("Watered every second day without fail, the plant stayed healthy.",
     "Serviced every second month without fail, the plant stayed reliable."),
    ("Grown from a cutting taken last year, the plant was already large.",
     "Built from a design drawn up last year, the plant was already busy."),
    ("Wilting badly after the long dry spell, the plant needed help.",
     "Idling badly after the long dispute, the plant needed investment."),
    ("Repotted into fresh compost that morning, the plant settled quickly.",
     "Refitted with new controls that month, the plant restarted quickly."),
    ("Reaching almost to the ceiling now, the plant needed cutting back.",
     "Employing almost four hundred people now, the plant needed more space."),
    ("Sheltered from the wind by a low wall, the plant did well.",
     "Screened from the road by a low wall, the plant drew few complaints."),
    ("Bought at the market for very little, the plant grew rapidly.",
     "Bought at auction for very little, the plant proved a good investment."),
    ("Struggling in the poor light of the hallway, the plant faded.",
     "Struggling with the falling demand of that year, the plant closed."),
    ("Flowering earlier than expected this season, the plant surprised everyone.",
     "Reopening earlier than expected this season, the plant surprised everyone."),
    ("Identified by a specialist as a rare variety, the plant was valuable.",
     "Identified by inspectors as a rare design, the plant was noteworthy."),
    ("Trimmed back hard the previous autumn, the plant recovered well.",
     "Scaled back sharply the previous autumn, the plant recovered slowly."),
    ("Standing alone in the corner of the room, the plant was easily missed.",
     "Standing alone at the edge of the town, the plant was hard to miss."),
    ("Suffering from an infestation of aphids, the plant lost leaves.",
     "Suffering from a shortage of parts, the plant lost output."),
    ("Given a larger pot in the spring, the plant expanded quickly.",
     "Given a larger contract in the spring, the plant expanded quickly."),
    ("Left unattended for several weeks, the plant became overgrown.",
     "Left unattended for several weeks, the plant fell behind schedule."),
    ("Producing new shoots every few days, the plant was clearly thriving.",
     "Producing new components every few minutes, the plant was clearly efficient."),
    ("Kept indoors through the coldest months, the plant survived.",
     "Kept running through the coldest months, the plant met demand."),
    ("Photographed for the catalogue that week, the plant looked its best.",
     "Photographed for the annual report that week, the plant looked impressive."),
    ("Nurtured by the same gardener for years, the plant flourished.",
     "Managed by the same director for years, the plant prospered."),
    ("Dying back naturally in late autumn, the plant returned each spring.",
     "Winding down slowly in late autumn, the plant reopened each spring."),
    ("Tolerating poor soil better than most, the plant persisted.",
     "Tolerating tight margins better than most, the plant persisted."),
    ("Rescued from a neglected garden, the plant recovered fully.",
     "Rescued from an uncertain closure, the plant recovered fully."),
    ("Shaded by the larger tree beside it, the plant grew slowly.",
     "Overshadowed by the larger site beside it, the plant grew slowly."),
    ("Divided into three smaller pots, the plant spread further.",
     "Divided into three smaller units, the plant spread its risk."),
    ("Blooming for only a few days each year, the plant was prized.",
     "Operating at full capacity a few weeks each year, the plant was profitable."),
    ("Fed with a weak solution each month, the plant stayed strong.",
     "Supplied with raw material each month, the plant stayed busy."),
    ("Threatened by an early frost, the plant was covered overnight.",
     "Threatened by an early closure, the plant was kept open."),
    ("Trailing over the edge of the shelf, the plant needed support.",
     "Extending beyond the edge of the site, the plant needed permission."),
    ("Studied by botanists for many years, the plant was well documented.",
     "Studied by engineers for many years, the plant was well documented."),
    ("Recovering after being cut back hard, the plant grew stronger.",
     "Recovering after being closed for repairs, the plant ran better."),
    ("Sitting in the same spot for a decade, the plant became familiar.",
     "Operating in the same town for a decade, the plant became familiar."),
    ("Losing colour in the low winter light, the plant looked tired.",
     "Losing orders in the difficult winter market, the plant looked vulnerable."),
    ("Propagated easily from a single cutting, the plant spread widely.",
     "Replicated easily from a single design, the plant model spread widely."),
    ("Sensitive to any change in temperature, the plant needed care.",
     "Sensitive to any change in regulation, the plant needed monitoring."),
    ("Positioned to catch the morning sun, the plant did well.",
     "Positioned to reach the main railway, the plant did well."),
    ("Overwatered by a well-meaning visitor, the plant suffered.",
     "Overloaded by an ambitious schedule, the plant suffered."),
    ("Displayed prominently in the entrance hall, the plant was admired.",
     "Featured prominently in the local news, the plant was well known."),
    ("Surviving on very little water, the plant proved hardy.",
     "Surviving on very thin margins, the plant proved resilient."),
    ("Attracting insects throughout the summer, the plant was busy.",
     "Attracting investment throughout the year, the plant expanded."),
    ("Grown organically without any treatment, the plant stayed healthy.",
     "Run efficiently without any subsidy, the plant stayed profitable."),
    ("Rooted deeply in heavy clay, the plant was hard to move.",
     "Established firmly in the local economy, the plant was hard to replace."),
    ("Wrapped in fleece against the cold, the plant came through winter.",
     "Insulated against the worst of the cold, the plant kept running."),
    ("Bearing fruit for the first time, the plant justified the wait.",
     "Turning a profit for the first time, the plant justified the investment."),
    ("Chosen for its hardiness in the region, the plant did well.",
     "Chosen for its location in the region, the plant did well."),
    ("Losing several leaves after the move, the plant took time to settle.",
     "Losing several staff after the merger, the plant took time to settle."),
    ("Inspected weekly for signs of disease, the plant stayed clean.",
     "Inspected weekly for signs of wear, the plant stayed compliant."),
    ("Growing well beyond its expected size, the plant needed repotting.",
     "Growing well beyond its original capacity, the plant needed extension."),
    ("Sold on to a new owner last month, the plant moved house.",
     "Sold on to a new owner last month, the plant changed hands."),
]


def build():
    rows = []
    for frame, veg, factory in CONTROLLED:
        rows.append((PREFIX + frame.format(cue=veg), "vegetation", "controlled"))
        rows.append((PREFIX + frame.format(cue=factory), "factory", "controlled"))
    for veg, factory in NATURAL:
        rows.append((PREFIX + veg, "vegetation", "natural"))
        rows.append((PREFIX + factory, "factory", "natural"))
    return rows


if __name__ == "__main__":
    rows = build()
    labels = [r[1] for r in rows]
    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_word": "plant",
        "sentences": [r[0] for r in rows],
        "labels": labels,
        "stratum": [r[2] for r in rows],
        "prefix": PREFIX,
        "design": "50 controlled pairs and 50 natural pairs; cue precedes the "
                  "target and names a referent property rather than a topic field",
    }, open(out / "plant.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "plant.yaml").write_text(
        "word: plant\n"
        "dataset: data/raw/semantic_sentences/plant.json\n"
        "senses:\n  - factory\n  - vegetation\n"
        "n_per_sense: 100\n"
        f'prefix: "{PREFIX}"\n'
    )
    print(f"{len(rows)} sentences: vegetation {labels.count('vegetation')}, "
          f"factory {labels.count('factory')}")
