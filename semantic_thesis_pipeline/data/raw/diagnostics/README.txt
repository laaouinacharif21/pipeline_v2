Diagnostic sentence sets. Not part of the seven-word study.

crane_probe_strong / medium / weak
    Cue-strength and cue-position calibration. The three levels share their
    vocabulary and differ in where the disambiguating phrase sits relative to
    the target word. Decoder models separate the senses only when the cue
    precedes the target (llama-7b: 0.183 before, 0.000 after), while encoder
    models are largely unaffected (bert-base: 0.088 and 0.086). This
    established the construction rule used for all seven words.

crane_minimal
    One hundred minimal pairs sharing a frame and differing in a single cue
    phrase. Context-only classification reaches 0.935, since the cue alone
    determines the label, and the target token adds 0.002. Retained as the
    over-control condition: eliminating shared context also eliminates the
    contribution being measured.
