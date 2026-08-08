# -*- coding: utf-8 -*-
"""
Sentence set for the ambiguous noun "pupil" (part of the eye / student).

Two hundred sentences, one hundred per sense, in two strata:

    controlled  50 pairs sharing a frame, differing in one cue phrase
    natural     50 pairs with freer wording and no shared frame

Construction follows the recipe validated on the bat pilot:

    The disambiguating cue precedes the target word. Decoder models produce no
    separation when it follows, since causal attention cannot read it at the
    target position.

    Cues name properties of the referent (iris, retina, dilation; homework,
    uniform, attendance) rather than whole topic fields, so the classes are not
    separable by subject matter alone.

    Noun uses only. The target is never sentence-initial and never capitalised
    mid-sentence.
"""

import json
from pathlib import Path

PREFIX = "Example sentence: "

CONTROLLED = [
    ("With its {cue} clearly visible, the pupil was examined closely.", "surrounding iris", "name on the register"),
    ("Because the {cue} was too bright, the pupil narrowed noticeably.", "overhead light", "afternoon sun"),
    ("Its {cue} recorded in the notes, the pupil was assessed carefully.", "diameter", "progress"),
    ("Standing near the {cue}, the pupil was easy to observe.", "slit lamp", "classroom door"),
    ("With the {cue} dimmed, the pupil responded within seconds.", "room light", "hall lights"),
    ("Its {cue} unusually large, the pupil attracted immediate attention.", "opening", "workload"),
    ("Because the {cue} had changed, the pupil was checked again.", "medication", "timetable"),
    ("With its {cue} steady, the pupil showed no cause for concern.", "response", "attendance"),
    ("Its {cue} measured precisely, the pupil was entered in the record.", "width", "score"),
    ("Since the {cue} had been adjusted, the pupil settled quickly.", "lens", "seating"),
    ("With the {cue} approaching, the pupil was prepared in advance.", "examination light", "examination week"),
    ("Its {cue} slightly uneven, the pupil was referred for review.", "shape", "handwriting"),
    ("Because the {cue} was unfamiliar, the pupil reacted slowly.", "stimulus", "material"),
    ("With its {cue} fully open, the pupil admitted as much as possible.", "aperture", "schedule"),
    ("Its {cue} noted at each visit, the pupil was tracked over time.", "size", "progress"),
    ("Following an assessment of its {cue}, the pupil was cleared.", "reflex", "coursework"),
    ("With the {cue} corrected, the pupil improved considerably.", "prescription", "timetable"),
    ("Its {cue} compared with the other side, the pupil looked normal.", "reaction", "results"),
    ("Because the {cue} was uncomfortable, the pupil could not be examined.", "bright beam", "hard chair"),
    ("With its {cue} restored, the pupil functioned as expected.", "reflex", "confidence"),
    ("Its {cue} checked every visit, the pupil remained stable.", "diameter", "attendance"),
    ("Since the {cue} was obstructed, the pupil was hard to assess.", "view", "record"),
    ("With the {cue} removed, the pupil could be observed properly.", "lens", "screen"),
    ("Its {cue} slower than expected, the pupil raised some concern.", "constriction", "progress"),
    ("Because the {cue} was warm, the pupil relaxed noticeably.", "examination room", "common room"),
    ("With its {cue} recorded, the pupil was added to the study.", "measurement", "consent form"),
    ("Its {cue} the smallest of the group, the pupil stood out.", "opening", "class"),
    ("Following a change in {cue}, the pupil responded differently.", "lighting", "teaching"),
    ("With the {cue} closed, the pupil was left in darkness.", "shutter", "blinds"),
    ("Its {cue} finer than expected, the pupil surprised the examiner.", "margin", "reasoning"),
    ("Because the {cue} had faded, the pupil was easier to examine.", "glare", "noise"),
    ("With its {cue} clearly marked, the pupil was easy to locate.", "position", "seat"),
    ("Its {cue} greater than average, the pupil was noted separately.", "dilation", "attainment"),
    ("Since the {cue} had improved, the pupil was reassessed.", "clarity", "effort"),
    ("With the {cue} switched off, the pupil widened again.", "lamp", "projector"),
    ("Its {cue} reviewed each term, the pupil was monitored closely.", "condition", "record"),
    ("Because the {cue} was uneven, the pupil was difficult to judge.", "illumination", "marking"),
    ("With its {cue} narrowed, the pupil admitted very little.", "aperture", "focus"),
    ("Its {cue} confirmed by the specialist, the pupil was accepted.", "measurement", "placement"),
    ("Following a check of its {cue}, the pupil was passed as normal.", "reaction", "coursework"),
    ("With the {cue} blocked, the pupil could not be reached.", "line of sight", "corridor"),
    ("Its {cue} unusually consistent, the pupil drew comment.", "response", "attendance"),
    ("Because the {cue} had loosened, the pupil was repositioned.", "headrest", "seat"),
    ("With its {cue} photographed, the pupil was included in the report.", "image", "work"),
    ("Its {cue} thicker than usual, the pupil appeared different.", "surrounding tissue", "portfolio"),
    ("Since the {cue} had settled, the pupil was examined again.", "swelling", "class"),
    ("With its {cue} fully adjusted, the pupil was ready for testing.", "focus", "timetable"),
    ("Its {cue} written on the form, the pupil was formally logged.", "reference number", "student number"),
    ("Because the {cue} was narrow, the pupil was hard to see.", "gap in the eyelid", "gap in the schedule"),
    ("With its {cue} in view, the pupil was assessed without difficulty.", "reflex", "record"),
]

NATURAL = [
    ("Reacting sharply to the sudden light, the pupil contracted at once.",
     "Reacting sharply to the sudden question, the pupil answered at once."),
    ("Widening steadily in the dim room, the pupil adjusted slowly.",
     "Settling steadily into the new school, the pupil adjusted slowly."),
    ("Examined under magnification that morning, the pupil looked healthy.",
     "Assessed under supervision that morning, the pupil performed well."),
    ("Responding well to the drops applied earlier, the pupil dilated fully.",
     "Responding well to the support offered earlier, the pupil improved quickly."),
    ("Slower than usual to react, the pupil concerned the doctor.",
     "Slower than usual to respond, the pupil concerned the teacher."),
    ("Photographed during the routine check, the pupil appeared normal.",
     "Photographed during the school assembly, the pupil looked cheerful."),
    ("Measured at four millimetres across, the pupil was within range.",
     "Ranked fourth in the year group, the pupil was doing well."),
    ("Constricting quickly under the lamp, the pupil worked as expected.",
     "Improving quickly under the new teacher, the pupil made progress."),
    ("Unequal to the other side, the pupil required investigation.",
     "Different from the rest of the class, the pupil required attention."),
    ("Observed over several appointments, the pupil remained unchanged.",
     "Observed over several terms, the pupil remained consistent."),
    ("Affected by the medication taken that week, the pupil stayed wide.",
     "Affected by the disruption that week, the pupil fell behind."),
    ("Difficult to assess in poor lighting, the pupil was rechecked later.",
     "Difficult to assess after a short absence, the pupil was rechecked later."),
    ("Reacting normally to accommodation, the pupil was recorded as healthy.",
     "Adapting normally to the new syllabus, the pupil was recorded as settled."),
    ("Noted as slightly irregular in shape, the pupil was monitored.",
     "Noted as slightly irregular in attendance, the pupil was monitored."),
    ("Dilated for the retinal examination, the pupil stayed open for hours.",
     "Selected for the science competition, the pupil prepared for weeks."),
    ("Shielded from the bright window, the pupil relaxed again.",
     "Seated away from the bright window, the pupil concentrated better."),
    ("Compared against the standard chart, the pupil measured normally.",
     "Compared against the year group average, the pupil scored highly."),
    ("Reacting more slowly in the left eye, the pupil was flagged.",
     "Struggling more noticeably in written work, the pupil was flagged."),
    ("Checked at every consultation, the pupil showed no change.",
     "Checked at every parents' evening, the pupil showed steady progress."),
    ("Sensitive to any change in brightness, the pupil moved constantly.",
     "Sensitive to any change in routine, the pupil needed reassurance."),
    ("Recovering after the drops wore off, the pupil returned to normal.",
     "Recovering after a difficult term, the pupil returned to form."),
    ("Recorded in the case notes that afternoon, the pupil was unremarkable.",
     "Recorded in the class register that afternoon, the pupil was present."),
    ("Reacting well to the standard test, the pupil passed easily.",
     "Reacting well to the standard test, the pupil scored highly."),
    ("Appearing cloudy in the photograph, the pupil needed a second look.",
     "Appearing anxious in the interview, the pupil needed reassurance."),
    ("Held open with a small instrument, the pupil could be inspected.",
     "Helped along by a patient mentor, the pupil could be encouraged."),
    ("Smaller in the older patient, the pupil reacted less.",
     "Younger than the rest of the class, the pupil struggled initially."),
    ("Affected by the low light indoors, the pupil widened.",
     "Affected by the long journey each day, the pupil arrived tired."),
    ("Assessed as part of the annual review, the pupil was fine.",
     "Assessed as part of the annual review, the pupil was progressing."),
    ("Showing a slight tremor under the light, the pupil was unusual.",
     "Showing real promise in mathematics, the pupil was exceptional."),
    ("Returning to size after a few minutes, the pupil behaved normally.",
     "Returning to school after a few weeks, the pupil caught up quickly."),
    ("Studied by researchers for the trial, the pupil was documented.",
     "Supported by staff throughout the year, the pupil was encouraged."),
    ("Reacting less than expected to light, the pupil raised questions.",
     "Achieving less than expected that term, the pupil raised questions."),
    ("Clear and well defined in the image, the pupil looked healthy.",
     "Clear and well organised in the written work, the pupil impressed."),
    ("Blocked partly by the swollen lid, the pupil was hard to see.",
     "Held back partly by poor attendance, the pupil was hard to help."),
    ("Adjusting between indoor and outdoor light, the pupil worked constantly.",
     "Moving between two different schools, the pupil adapted constantly."),
    ("Reviewed by a second clinician, the pupil was confirmed normal.",
     "Reviewed by a second teacher, the pupil was confirmed capable."),
    ("Slightly larger in dim conditions, the pupil behaved as expected.",
     "Slightly older than the others, the pupil led the group naturally."),
    ("Tested with a handheld torch, the pupil responded briskly.",
     "Tested at the end of term, the pupil responded confidently."),
    ("Unchanged since the last appointment, the pupil was stable.",
     "Unchanged since the last report, the pupil was consistent."),
    ("Photographed under infrared conditions, the pupil showed detail.",
     "Interviewed under gentle questioning, the pupil showed insight."),
    ("Reacting to a moving target, the pupil tracked smoothly.",
     "Reacting to a difficult topic, the pupil coped smoothly."),
    ("Wider than average for the age group, the pupil was noted.",
     "Ahead of average for the age group, the pupil was noted."),
    ("Obscured by a reflection in the image, the pupil was rephotographed.",
     "Overlooked in a crowded classroom, the pupil was reassessed."),
    ("Sluggish after a long day, the pupil reacted weakly.",
     "Sluggish after a long day, the pupil worked slowly."),
    ("Documented across three separate visits, the pupil stayed constant.",
     "Documented across three separate terms, the pupil stayed consistent."),
    ("Reacting fully within a second, the pupil was healthy.",
     "Replying fully within a moment, the pupil was confident."),
    ("Examined again the following week, the pupil had not changed.",
     "Assessed again the following week, the pupil had improved."),
    ("Different in size from the other, the pupil needed investigation.",
     "Different in ability from the others, the pupil needed support."),
    ("Reacting to the change in focus, the pupil adjusted quickly.",
     "Reacting to the change in teacher, the pupil adjusted quickly."),
    ("Noted as entirely normal on examination, the pupil needed no action.",
     "Noted as entirely settled at school, the pupil needed no support."),
]


def build():
    rows = []
    for frame, eye, student in CONTROLLED:
        rows.append((PREFIX + frame.format(cue=eye), "eye", "controlled"))
        rows.append((PREFIX + frame.format(cue=student), "student", "controlled"))
    for eye, student in NATURAL:
        rows.append((PREFIX + eye, "eye", "natural"))
        rows.append((PREFIX + student, "student", "natural"))
    return rows


if __name__ == "__main__":
    rows = build()
    labels = [r[1] for r in rows]
    out = Path("data/raw/semantic_sentences")
    out.mkdir(parents=True, exist_ok=True)
    json.dump({
        "target_word": "pupil",
        "sentences": [r[0] for r in rows],
        "labels": labels,
        "stratum": [r[2] for r in rows],
        "prefix": PREFIX,
        "design": "50 controlled pairs and 50 natural pairs; cue precedes the "
                  "target and names a referent property rather than a topic field",
    }, open(out / "pupil.json", "w"), indent=2, ensure_ascii=False)

    cfg = Path("configs/words")
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "pupil.yaml").write_text(
        "word: pupil\n"
        "dataset: data/raw/semantic_sentences/pupil.json\n"
        "senses:\n  - eye\n  - student\n"
        "n_per_sense: 100\n"
        f'prefix: "{PREFIX}"\n'
    )
    print(f"{len(rows)} sentences: eye {labels.count('eye')}, "
          f"student {labels.count('student')}")
