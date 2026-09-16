"""Apply the layer-alignment fix to src/analysis/_io.py (September 2026).

Replaces merge_geometry_sep with an explicit alignment:
    post (default)  block l <-> hidden state l+1   output of the block
    pre             block l <-> hidden state l     the original pairing

Safety: aborts without writing if the function is not found exactly once,
if its body is not the version audited, or if the result does not compile.
The original is preserved in Git at tag pre-alignment-fix.

Run from the pipeline root:
    python _audit/fix_alignment.py
"""
import sys
from pathlib import Path

NEW = '''
ALIGNMENTS = ("post", "pre")


def align_geometry_sep(geo, sep, alignment: str = "post"):
    """Pair per-block geometry with per-hidden-state separation.

    Index conventions
        geo  "layer" l   = transformer block l            (n_blocks rows)
        sep  "layer" 0   = embedding output               (n_blocks + 1 rows)
             "layer" l+1 = output of transformer block l

    alignment
        "post"  block l <-> hidden state l+1, the output of the block (primary)
        "pre"   block l <-> hidden state l, the input to the block
                (the pairing used before September 2026)

    "layer" in the result stays the block index, so depth control is identical
    under both alignments; "hidden_layer" records the hidden state used.
    """
    if alignment not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {alignment!r}")
    g_idx = geo["layer"].astype(int).tolist()
    s_idx = sep["layer"].astype(int).tolist()
    if g_idx != list(range(len(g_idx))):
        raise ValueError(f"geometry layers not contiguous from 0: {g_idx[:6]}")
    if s_idx != list(range(len(s_idx))):
        raise ValueError(f"separation layers not contiguous from 0: {s_idx[:6]}")
    if len(s_idx) != len(g_idx) + 1:
        raise ValueError(
            f"expected {len(g_idx) + 1} hidden states for {len(g_idx)} blocks "
            f"(embedding + one per block), got {len(s_idx)}")
    shift = 1 if alignment == "post" else 0
    g = geo.copy()
    g["hidden_layer"] = g["layer"].astype(int) + shift
    s = sep.rename(columns={"layer": "hidden_layer"})
    s["hidden_layer"] = s["hidden_layer"].astype(int)
    return g.merge(s, on="hidden_layer", how="inner", validate="one_to_one")


def merge_geometry_sep(model: str, word: str, which: str = "erank",
                       alignment: str = "post"):
    """Merge block geometry with Sep(l). See align_geometry_sep for alignment."""
    geo = load_erank(model) if which == "erank" else load_params(model)
    sep = load_sep(model, word)
    if geo is None or sep is None:
        return None
    m = align_geometry_sep(geo, sep, alignment=alignment)
    return m if len(m) >= 5 else None
'''

p = Path("src/analysis/_io.py")
if not p.exists():
    sys.exit("src/analysis/_io.py not found - run from the pipeline root. Nothing changed.")
s = p.read_text()

if "def align_geometry_sep(" in s:
    sys.exit("Fix already applied. Nothing changed.")

head = "def merge_geometry_sep("
tail = "return m if len(m) >= 5 else None"
if s.count(head) != 1:
    sys.exit(f"Found {s.count(head)} definitions of merge_geometry_sep, expected 1. Nothing changed.")
start = s.index(head)
end = s.find(tail, start)
if end < 0:
    sys.exit("End of merge_geometry_sep not found. Nothing changed.")
end += len(tail)
old = s[start:end]
if 'geo.merge(sep, on="layer")' not in old:
    sys.exit("merge_geometry_sep body differs from the audited version. Nothing changed.")

patched = s[:start] + NEW.strip("\n") + s[end:]
try:
    compile(patched, str(p), "exec")
except SyntaxError as e:
    sys.exit(f"Patched file would not compile ({e}). Nothing changed.")

p.write_text(patched)
print(f"Patched {p}")
print("---- removed ----")
print(old)
print("-----------------")
