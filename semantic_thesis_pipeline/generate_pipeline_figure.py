# -*- coding: utf-8 -*-
# generate_pipeline_figure.py
# Usage: python generate_pipeline_figure.py
# Output: results/plots/publication/figure_2_1_pipeline.png / .svg

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = os.path.join("results", "plots", "publication")
os.makedirs(OUTPUT_DIR, exist_ok=True)

C = {
    "ib": "#E6F1FB", "ie": "#378ADD", "it": "#0C447C",
    "ii": "#ffffff", "iie": "#378ADD", "iit": "#185FA5",

    "1b": "#E1F5EE", "1e": "#1D9E75", "1t": "#085041",
    "1i": "#ffffff", "1ie": "#1D9E75", "1it": "#0F6E56",

    "2ab": "#FAEEDA", "2ae": "#BA7517", "2at": "#633806",
    "2ai": "#ffffff", "2aie": "#BA7517", "2ait": "#854F0B",

    "2bb": "#EEEDFE", "2be": "#7F77DD", "2bt": "#26215C",
    "2bi": "#ffffff", "2bie": "#7F77DD", "2bit": "#3C3489",

    "3b": "#FAECE7", "3e": "#D85A30", "3t": "#4A1B0C", "3it": "#993C1D",

    "4b": "#EAF3DE", "4e": "#639922", "4t": "#173404", "4it": "#3B6D11",

    "ar": "#555555",
    "cap": "#444444",
}

F = "Times New Roman"


def box(ax, x, y, w, h, fc, ec, lw=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
    )


def hdr(ax, x, y, txt, col, fs=11):
    ax.text(
        x, y, txt,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=col,
        fontfamily=F,
        zorder=3,
    )


def blk(ax, x, cy, rows, col, fs=9, sp=0.22):
    n = len(rows)
    top = cy + (n - 1) * sp / 2.0
    for i, t in enumerate(rows):
        ax.text(
            x, top - i * sp, t,
            ha="center",
            va="center",
            fontsize=fs,
            color=col,
            fontfamily=F,
            zorder=3,
        )


def arr(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C["ar"],
            lw=1.2,
            mutation_scale=12,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )


def seg(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-",
            color=C["ar"],
            lw=1.0,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=5,
    )


# ============================================================
# FIGURE CANVAS
# ============================================================

fig, ax = plt.subplots(figsize=(9, 10))
ax.set_xlim(0, 9)
ax.set_ylim(0, 10)
ax.axis("off")

LX = 0.25
W = 8.5
CX = 4.5

# ============================================================
# Stage 0: Input resources
# ============================================================

S0Y, S0H = 7.65, 1.55
box(ax, LX, S0Y, W, S0H, C["ib"], C["ie"], 1.2)
hdr(ax, CX, S0Y + S0H - 0.22, "Input resources", C["it"], fs=11)

box(ax, 0.5, 7.8, 3.5, 1.05, C["ii"], C["iie"], 0.8)
hdr(ax, 2.25, 7.8 + 1.05 - 0.22, "Ambiguity dataset", C["iit"], fs=10)
blk(
    ax, 2.25, 7.8 + 0.45,
    ['200 sentences / word "bank"', "100 finance / 100 river"],
    C["iit"], fs=9, sp=0.22,
)

box(ax, 5.0, 7.8, 3.5, 1.05, C["ii"], C["iie"], 0.8)
hdr(ax, 6.75, 7.8 + 1.05 - 0.22, "Local pretrained models", C["iit"], fs=10)
blk(
    ax, 6.75, 7.8 + 0.45,
    ["13 models / offline mode", "LLaMA / Qwen / BERT-family"],
    C["iit"], fs=9, sp=0.22,
)

arr(ax, CX, S0Y, CX, 7.36)

# ============================================================
# Stage 1: Target-token hidden-state extraction
# ============================================================

S1Y, S1H = 5.7, 1.65
box(ax, LX, S1Y, W, S1H, C["1b"], C["1e"], 1.2)
hdr(
    ax, CX, S1Y + S1H - 0.22,
    "Stage 1 - Target-token hidden-state extraction",
    C["1t"], fs=11,
)

I1 = [
    ("Tokenization", ["Fixed first-subword rule", "consistent across models"]),
    ("Forward pass", ["All transformer layers", "HuggingFace Transformers"]),
    ("Embedding cache", ["Stored per layer / model", "Reusable - no recompute"]),
]

BW, BH = 2.55, 1.1
BY = S1Y + 0.1

for k, (ttl, rows) in enumerate(I1):
    BX = 0.5 + k * 2.83
    box(ax, BX, BY, BW, BH, C["1i"], C["1ie"], 0.8)
    hdr(ax, BX + BW / 2, BY + BH - 0.2, ttl, C["1it"], fs=10)
    blk(ax, BX + BW / 2, BY + BH / 2 - 0.1, rows, C["1it"], fs=9, sp=0.22)

    if k < 2:
        arr(ax, BX + BW, BY + BH / 2, BX + BW + 0.28, BY + BH / 2)

arr(ax, CX, S1Y, CX, 5.41)

# ============================================================
# Stage 2A + Stage 2B
# ============================================================

S2Y, S2H = 3.4, 2.0
T2Y = S2Y + S2H - 0.28

IY = S2Y + 0.45
IH = S2H - 0.58
IC = IY + IH / 2.0

# Stage 2A
box(ax, 0.25, S2Y, 4.1, S2H, C["2ab"], C["2ae"], 1.2)
hdr(ax, 2.3, T2Y, "Stage 2A - Representation-space analysis", C["2at"], fs=10)

box(ax, 0.48, IY, 3.64, IH, C["2ai"], C["2aie"], 0.8)
blk(
    ax, 2.3, IC,
    [
        "Cosine similarity / L2 distance",
        "Layer-wise representation drift",
        "Semantic separation Sep(l)",
        "PCA projections / heatmaps",
    ],
    C["2ait"], fs=9, sp=0.22,
)

# Stage 2B
box(ax, 4.65, S2Y, 4.1, S2H, C["2bb"], C["2be"], 1.2)
hdr(ax, 6.7, T2Y, "Stage 2B - Parameter-space geometry", C["2bt"], fs=10)

box(ax, 4.88, IY, 3.64, IH, C["2bi"], C["2bie"], 0.8)
blk(
    ax, 6.7, IC,
    [
        "q_proj / v_proj / up_proj",
        "Spectral norm (Phase 1A)",
        "Effective rank (Phase 1B)",
        "Partial correlation (depth ctrl)",
    ],
    C["2bit"], fs=9, sp=0.22,
)

# ============================================================
# Stage 3: Geometry-semantics association
# IMPORTANT FIX:
# Stage 3 is moved slightly lower, and the Stage 2 merge arrow now has real length.
# ============================================================

S3Y, S3H = 1.95, 1.1
S3_TOP = S3Y + S3H

# Merge lines from bottom of Stage 2A and 2B
MY = S2Y - 0.18

seg(ax, 2.3, S2Y, 2.3, MY)
seg(ax, 6.7, S2Y, 6.7, MY)
seg(ax, 2.3, MY, 6.7, MY)

# Clean visible arrow from merge line into Stage 3
arr(ax, CX, MY, CX, S3_TOP + 0.02)

box(ax, LX, S3Y, W, S3H, C["3b"], C["3e"], 1.2)
hdr(
    ax, CX, S3Y + S3H - 0.22,
    "Stage 3 - Geometry-semantics association",
    C["3t"], fs=11,
)
blk(
    ax, CX, S3Y + 0.45,
    [
        "Phase 1A partial correlation / Phase 1B effective rank / Phase 2 pooled analysis",
        "Cross-family comparison: LLaMA vs Qwen vs BERT-family encoders",
    ],
    C["3it"], fs=9, sp=0.25,
)

# ============================================================
# Stage 4: Outputs and cross-model comparison
# ============================================================

S4Y, S4H = 0.7, 1.1
S4_TOP = S4Y + S4H

# Clean arrow from Stage 3 to Stage 4
arr(ax, CX, S3Y, CX, S4_TOP + 0.02)

box(ax, LX, S4Y, W, S4H, C["4b"], C["4e"], 1.2)
hdr(
    ax, CX, S4Y + S4H - 0.22,
    "Stage 4 - Outputs and cross-model comparison",
    C["4t"], fs=11,
)
blk(
    ax, CX, S4Y + 0.45,
    [
        "Publication figures / CSV summaries / layer-evolution curves",
        "External benchmark reference: C-Eval / CMMLU / GSM8K",
    ],
    C["4it"], fs=9, sp=0.25,
)

# ============================================================
# Caption
# ============================================================

ax.text(
    CX, 0.25,
    "Figure 2-1. Overall experimental pipeline for semantic geometry analysis.",
    ha="center",
    va="center",
    fontsize=9,
    color=C["cap"],
    fontfamily=F,
    style="italic",
)

# ============================================================
# Save outputs
# ============================================================

svg_path = os.path.join(OUTPUT_DIR, "figure_2_1_pipeline.svg")
png_path = os.path.join(OUTPUT_DIR, "figure_2_1_pipeline.png")

plt.tight_layout(pad=0.2)
plt.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
plt.close()

print("Saved SVG -> " + svg_path)
print("Saved PNG -> " + png_path)