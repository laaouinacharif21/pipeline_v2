# -*- coding: utf-8 -*-
"""
Sentence set for the ambiguous noun "club" (organisation / heavy stick).

Two hundred sentences, one hundred per sense, in two strata:

    controlled  50 pairs sharing a frame, differing in one cue phrase
    natural     50 pairs with freer wording and no shared frame

Construction follows the recipe validated on the bat pilot:

    The disambiguating cue precedes the target word. Decoder models produce no
    separation when it follows, since causal attention cannot read it at the
    target position.

    Cues name properties of the referent (membership, committee, subscription;
    shaft, weight, grip) rather than whole topic fields, so the classes are not
    separable by subject matter alone.

    Noun uses only. The target is never sentence-initial and never capitalised
    mid-sentence.
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

CONTROLLED = [
    ("With its {cue} clearly listed, the club was easy to identify.", "membership rules", "maker's mark"),
    ("Because its {cue} had lapsed, the club was no longer usable.", "registration", "binding"),
    ("Its {cue} worn with time, the club showed its age.", "reputation", "handle"),
    ("Standing beside the {cue}, the club looked out of place.", "town hall", "display case"),
    ("With its {cue} recently renewed, the club was in good standing.", "licence", "grip"),
    ("Its {cue} unusually large, the club drew comment locally.", "membership", "head"),
    ("Because the {cue} had changed, the club was reviewed again.", "committee", "weighting"),
    ("Covered in fine {cue}, the club looked neglected.", "dust in the hall", "dust"),
    ("With its {cue} fully settled, the club functioned smoothly.", "committee", "balance"),
    ("Its {cue} recorded carefully, the club entered the register.", "membership number", "serial number"),
    ("Since the {cue} had been cleaned, the club looked much better.", "meeting room", "shaft"),
    ("With the {cue} approaching, the club was prepared in advance.", "annual meeting", "tournament"),
    ("Its {cue} slightly damaged, the club still served its purpose.", "reputation", "shaft"),
    ("Because the {cue} was too small, the club could not expand.", "venue", "grip"),
    ("With its {cue} fully extended, the club reached its limit.", "waiting list", "length"),
    ("Its {cue} counted carefully, the club entered the annual return.", "members", "grams"),
    ("Following an inspection of its {cue}, the club was approved.", "accounts", "condition"),
    ("With its {cue} finally repaired, the club was back in use.", "roof", "cracked shaft"),
    ("Its {cue} measured precisely, the club was catalogued that week.", "turnover", "length"),
    ("Because the {cue} felt unwelcoming, the club lost interest.", "atmosphere", "grip"),
    ("With its {cue} coated in grime, the club needed cleaning.", "windows", "head"),
    ("Its {cue} the largest in the area, the club stood out.", "membership", "head"),
    ("Since the {cue} was in poor shape, the club was replaced.", "premises", "shaft"),
    ("With the {cue} removed, the club could be examined properly.", "cover charge", "cover"),
    ("Its {cue} showing clear strain, the club needed support.", "finances", "joint"),
    ("Because the {cue} was warm, the club was more comfortable.", "meeting room", "workshop"),
    ("With its {cue} laid out, the club occupied the whole space.", "stalls", "length"),
    ("Its {cue} noted in detail, the club was added to the record.", "constitution", "specification"),
    ("Following the loss of a {cue}, the club struggled on.", "key member", "grip layer"),
    ("With the {cue} locked for the night, the club was left alone.", "hall", "case"),
    ("Its {cue} finer than expected, the club surprised the visitor.", "facilities", "finish"),
    ("Because the {cue} had dried out, the club needed treatment.", "funding", "wood"),
    ("With its {cue} clearly marked, the club was easy to locate.", "sign", "label"),
    ("Its {cue} heavier than the others, the club was set apart.", "workload", "head"),
    ("Since the {cue} had been renewed, the club performed better.", "committee", "grip"),
    ("With the {cue} switched off, the club became quiet.", "music", "light"),
    ("Its {cue} reviewed each year, the club remained in good order.", "accounts", "condition"),
    ("Because the {cue} was uneven, the club did not sit well.", "floor", "shaft"),
    ("With its {cue} drawn in, the club took up far less room.", "activities", "cover"),
    ("Its {cue} confirmed by the inspector, the club was accepted.", "status", "authenticity"),
    ("Following a check of its {cue}, the club was cleared.", "records", "balance"),
    ("With the {cue} blocked, the club could not be reached.", "entrance", "rack"),
    ("Its {cue} unusually dense, the club attracted attention.", "programme", "wood"),
    ("Because the {cue} had loosened, the club was fixed that day.", "committee structure", "grip binding"),
    ("With its {cue} clearly visible, the club was photographed for the file.", "emblem", "engraving"),
    ("Its {cue} thicker than usual, the club felt more substantial.", "handbook", "shaft"),
    ("Since the {cue} had settled, the club continued as before.", "dispute", "resin"),
    ("With its {cue} fully dry, the club was ready again.", "hall floor", "surface"),
    ("Its {cue} written on the form, the club was formally logged.", "reference number", "model number"),
    ("Because the {cue} was narrow, the club barely fitted.", "doorway", "slot"),
]

NATURAL = [
    ("Founded by a group of local residents, the club grew steadily.",
     "Carved by a local craftsman from oak, the club lasted for years."),
    ("Meeting every Thursday in the village hall, the club was well attended.",
     "Resting every winter in the wooden rack, the club was well preserved."),
    ("Struggling to attract new members that year, the club nearly closed.",
     "Splitting along the grain that winter, the club nearly broke."),
    ("Supported by a small annual subscription, the club stayed open.",
     "Wrapped in a protective leather cover, the club stayed intact."),
    ("Run by volunteers for over a decade, the club thrived.",
     "Used by the same player for over a decade, the club performed well."),
    ("Holding its meetings in the back room, the club kept a low profile.",
     "Kept in the corner of the back room, the club gathered dust."),
    ("Welcoming anyone who wished to join, the club expanded quickly.",
     "Weighted evenly along its length, the club handled well."),
    ("Losing its lease on the premises, the club had to move.",
     "Losing the binding on its handle, the club had to be repaired."),
    ("Celebrating fifty years since its founding, the club held an event.",
     "Showing fifty years of steady use, the club was still serviceable."),
    ("Advertised only by word of mouth, the club stayed small.",
     "Marked only with a faint stamp, the club was hard to date."),
    ("Governed by a written constitution, the club operated formally.",
     "Balanced by a weight in the head, the club swung smoothly."),
    ("Divided over the proposed changes, the club lost members.",
     "Cracked across the middle section, the club was set aside."),
    ("Recognised by the national body last year, the club gained status.",
     "Refinished by a specialist last year, the club looked new."),
    ("Charging very little to join, the club attracted many.",
     "Weighing very little in the hand, the club suited beginners."),
    ("Organising trips throughout the summer, the club stayed active.",
     "Standing unused throughout the summer, the club stayed dry."),
    ("Founded originally as a reading group, the club changed over time.",
     "Made originally from a single piece, the club changed hands often."),
    ("Meeting resistance from the council, the club delayed its plans.",
     "Meeting the ball squarely each time, the club performed reliably."),
    ("Growing from twenty members to two hundred, the club prospered.",
     "Passing from father to son over years, the club was treasured."),
    ("Relying entirely on donations that year, the club survived.",
     "Relying on a single repair that year, the club survived."),
    ("Publishing a newsletter every month, the club kept in touch.",
     "Polished carefully every month, the club stayed in condition."),
    ("Suspended briefly after the dispute, the club reopened later.",
     "Stored briefly after the season, the club was used again later."),
    ("Located above a shop in the high street, the club was easy to miss.",
     "Located at the back of the cupboard, the club was easy to miss."),
    ("Chaired by the same person for years, the club was stable.",
     "Owned by the same person for years, the club was well cared for."),
    ("Attracting members from several villages, the club drew widely.",
     "Attracting interest from several collectors, the club sold quickly."),
    ("Recorded in the county directory, the club was well established.",
     "Recorded in the auction catalogue, the club was well described."),
    ("Struggling with rising costs that winter, the club raised its fees.",
     "Suffering from damp that winter, the club warped slightly."),
    ("Formed after the old society disbanded, the club started fresh.",
     "Reshaped after the old head was replaced, the club felt different."),
    ("Hosting a competition every spring, the club was busy.",
     "Cleaned and checked every spring, the club stayed ready."),
    ("Running at a loss for two seasons, the club needed help.",
     "Sitting unused for two seasons, the club needed attention."),
    ("Admired throughout the region for its work, the club was respected.",
     "Admired by collectors for its craftsmanship, the club was valued."),
    ("Reduced to a handful of members, the club nearly folded.",
     "Reduced to a splintered handle, the club was discarded."),
    ("Renting a room in the community centre, the club made do.",
     "Hanging on a hook in the storeroom, the club waited."),
    ("Merging with a neighbouring group, the club doubled in size.",
     "Matched with a similar piece, the club made a pair."),
    ("Encouraging younger people to take part, the club renewed itself.",
     "Adjusted slightly for a shorter player, the club suited them better."),
    ("Operating without any paid staff, the club kept costs low.",
     "Made without any modern materials, the club was traditional."),
    ("Facing closure at the end of the year, the club appealed for support.",
     "Facing replacement at the end of the season, the club was retired."),
    ("Named after the founder's home village, the club had local roots.",
     "Named after its maker's workshop, the club had a clear history."),
    ("Restricted to those living nearby, the club stayed local.",
     "Restricted to a single approved design, the club met the rules."),
    ("Rebuilt after the fire two years ago, the club recovered.",
     "Rebuilt after the break two years ago, the club was usable again."),
    ("Depending on a few loyal helpers, the club continued.",
     "Depending on careful storage, the club stayed sound."),
    ("Praised in the local paper last month, the club gained interest.",
     "Featured in a collectors' magazine last month, the club gained value."),
    ("Sharing premises with a similar group, the club saved money.",
     "Sharing a case with a similar piece, the club stayed protected."),
    ("Established well before the war, the club had a long history.",
     "Produced well before the war, the club was genuinely old."),
    ("Welcoming beginners at every session, the club grew friendly.",
     "Suiting beginners in every respect, the club was recommended."),
    ("Falling behind with its paperwork, the club faced questions.",
     "Falling short in the final round, the club was blamed unfairly."),
    ("Meeting less often during the summer, the club went quiet.",
     "Used less often during the summer, the club stayed clean."),
    ("Funded partly by the local council, the club continued its work.",
     "Backed partly by a metal insert, the club held together."),
    ("Drawing people from all backgrounds, the club was inclusive.",
     "Drawing praise from experienced players, the club was well made."),
    ("Deciding to change its name last year, the club moved on.",
     "Losing its original finish last year, the club looked older."),
    ("Keeping careful minutes of every meeting, the club stayed organised.",
     "Keeping its shape after heavy use, the club stayed reliable."),
]


def build():
    rows = []
    for frame, org, stick in CONTROLLED:
        rows.append((PREFIX + frame.format(cue=org), "organisation", "controlled"))
        rows.append((PREFIX + frame.format(cue=stick), "implement", "controlled"))
    for org, stick in NATURAL:
        rows.append((PREFIX + org, "organisation", "natural"))
        rows.append((PREFIX + stick, "implement", "natural"))
    return rows


if __name__ == "__main__":
    rows = build()
    labels = [r[1] for r in rows]
    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_word": "club",
        "sentences": [r[0] for r in rows],
        "labels": labels,
        "stratum": [r[2] for r in rows],
        "prefix": PREFIX,
        "design": "50 controlled pairs and 50 natural pairs; cue precedes the "
                  "target and names a referent property rather than a topic field",
    }, open(out / "club.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "club.yaml").write_text(
        "word: club\n"
        "dataset: data/raw/semantic_sentences/club.json\n"
        "senses:\n  - implement\n  - organisation\n"
        "n_per_sense: 100\n"
        f'prefix: "{PREFIX}"\n'
    )
    print(f"{len(rows)} sentences: organisation {labels.count('organisation')}, "
          f"implement {labels.count('implement')}")
