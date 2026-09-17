# -*- coding: utf-8 -*-
"""
Cross-word figures for the geometry-separation analysis (revised September 2026).

Reads the three outputs of cross_word.py (cells, models, summary) and produces:

    fig1_heatmap          model x word partial r, linear depth control
    fig1_heatmap_quad     the same under quadratic depth control
    fig1_heatmap_no_X     linear, with word X excluded (robustness)
    fig2_word_means       mean partial r per word, decoders and encoders
    fig3_consistency      per-model mean r over words with its range, under
                          linear and quadratic control; model-level result in
                          the footer
    fig5_compare          (with --compare) per-model mean r for the primary
                          measure against a comparison measure

No per-cell significance marks are drawn: cells are not independent
observations. Inference is model-level (see _stats and cross_word.py).

Usage
    python -m src.analysis.cross_word_figures --measure stable_rank --projection q_proj
    python -m src.analysis.cross_word_figures --compare up_proj:erank
    python -m src.analysis.cross_word_figures --alignment pre
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import ALIGNMENTS, DECODERS, FAMILIES  # noqa: E402
from src.utils.paths import ensure_dir, get_results_root  # noqa: E402

WORD_ORDER = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]
MODEL_ORDER = FAMILIES["llama"] + FAMILIES["qwen"] + FAMILIES["bert"]
DEC_C, ENC_C = "#1f77b4", "#d62728"
CONTROLS = (("", "linear"), ("_quad", "quadratic"))

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def stem(projection, measure, alignment):
    sfx = "" if alignment == "post" else f"_{alignment}"
    return get_results_root() / "analysis" / "_cross_word" / f"cross_word_{projection}_{measure}{sfx}"


def load(projection, measure, alignment):
    base = stem(projection, measure, alignment)
    paths = [Path(f"{base}{k}.csv") for k in ("", "_models", "_summary")]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Not found: {p}\nRun cross_word.py --measure {measure} "
                             f"--projection {projection} --alignment {alignment} first.")
    cells, models, summary = (pd.read_csv(p) for p in paths)
    if "partial_r_quad" not in cells.columns or "mean_r_quad" not in models.columns:
        raise SystemExit(f"{paths[0].name} is in the pre-revision format. Rerun cross_word.py.")
    return cells[cells.word.isin(WORD_ORDER)], models, summary


def srow(summary, group, control, words="all"):
    r = summary[(summary.group == group) & (summary.control == control) & (summary.words == words)]
    return None if r.empty else r.iloc[0]


def fmt_summary(r):
    if r is None:
        return "n/a"
    w = "n/a" if pd.isna(r.wilcoxon_p) else f"{r.wilcoxon_p:.3f}"
    return f"{r['mean']:+.3f}, {r.n_positive}/{r.n_models} positive, Wilcoxon p={w}"


def tag(alignment):
    return "" if alignment == "post" else f"_{alignment}"


def label(projection, measure):
    return f"{projection} {measure.replace('_', ' ')}"


def save(fig, out_dir, name, pdf=True):
    fig.savefig(out_dir / f"{name}.png")
    if pdf:
        fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved -> {out_dir / (name + '.png')}")


def fig_heatmap(cells, projection, measure, alignment, out_dir, control="linear", exclude=None, lim=None):
    col = "partial_r" if control == "linear" else "partial_r_quad"
    words = [w for w in WORD_ORDER if w in set(cells.word) and w != exclude]
    models = [m for m in MODEL_ORDER if m in set(cells.model)]

    M = np.full((len(models), len(words)), np.nan)
    for i, m in enumerate(models):
        for j, w in enumerate(words):
            r = cells[(cells.model == m) & (cells.word == w)]
            if len(r):
                M[i, j] = r[col].iloc[0]

    lim = lim or np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(1.1 * len(words) + 3.2, 0.46 * len(models) + 2))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), aspect="auto")
    for i in range(len(models)):
        for j in range(len(words)):
            if np.isnan(M[i, j]):
                continue
            shade = "white" if abs(M[i, j]) > 0.6 * lim else "black"
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8, color=shade)

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    n_dec = sum(1 for m in models if m in DECODERS)
    ax.axhline(n_dec - 0.5, color="black", linewidth=1.6)
    pad = -(6.2 * max(len(m) for m in models) + 26)
    for name, centre in [("decoder", (n_dec - 1) / 2),
                         ("encoder", n_dec + (len(models) - n_dec - 1) / 2)]:
        ax.annotate(name, xy=(0, centre), xycoords=("axes fraction", "data"),
                    xytext=(pad, 0), textcoords="offset points", rotation=90,
                    va="center", ha="center", fontsize=11)

    title = f"{label(projection, measure)} against Sep, {control} depth control"
    if exclude:
        title += f" (excluding {exclude})"
    ax.set_title(title, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="partial r")

    name = (f"fig1_heatmap_{projection}_{measure}" + ("" if control == "linear" else "_quad")
            + (f"_no_{exclude}" if exclude else "") + tag(alignment))
    save(fig, out_dir, name)


def fig_word_means(cells, projection, measure, alignment, out_dir):
    words = [w for w in WORD_ORDER if w in set(cells.word)]
    dec = cells[cells.model.isin(DECODERS)]
    enc = cells[~cells.model.isin(DECODERS)]
    x = np.arange(len(words))

    fig, ax = plt.subplots(figsize=(8, 4.4))
    dm = [dec[dec.word == w].partial_r.mean() for w in words]
    ds = [dec[dec.word == w].partial_r.std() for w in words]
    dq = [dec[dec.word == w].partial_r_quad.mean() for w in words]
    em = [enc[enc.word == w].partial_r.mean() for w in words]
    ax.errorbar(x, dm, yerr=ds, marker="o", linewidth=2, capsize=4, color=DEC_C,
                label="decoders, linear (mean ± SD across models)")
    ax.plot(x, dq, marker="o", linewidth=1.5, linestyle="--", color=DEC_C, alpha=0.8,
            label="decoders, quadratic")
    ax.plot(x, em, marker="s", linewidth=2, color=ENC_C, label="encoders, linear")
    ax.axhline(0, color="grey", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(words)
    ax.set_ylabel("mean partial r")
    ax.set_title(f"{label(projection, measure)} by target word")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out_dir, f"fig2_word_means_{projection}_{measure}{tag(alignment)}", pdf=False)


def fig_model_consistency(models, summary, projection, measure, alignment, out_dir):
    order = [m for m in MODEL_ORDER if m in set(models.model)]
    M = models.set_index("model").loc[order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.5, 0.46 * len(order) + 2.4))
    colours = [DEC_C if a == "decoder" else ENC_C for a in M.arch]
    for off, (sfx, _), marker in zip((-0.17, 0.17), CONTROLS, ("o", "s")):
        mean = M[f"mean_r{sfx}"].to_numpy()
        lo = mean - M[f"min_r{sfx}"].to_numpy()
        hi = M[f"max_r{sfx}"].to_numpy() - mean
        ax.errorbar(mean, y + off, xerr=[lo, hi], fmt="none", ecolor="grey",
                    elinewidth=1, capsize=2, alpha=0.7, zorder=2)
        face = colours if marker == "o" else "white"
        ax.scatter(mean, y + off, marker=marker, s=40, zorder=3,
                   facecolors=face, edgecolors=colours, linewidths=1.5)

    n_dec = int((M.arch == "decoder").sum())
    ax.axhline(n_dec - 0.5, color="black", linewidth=1)
    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("mean partial r across words (bars: range across words)")
    ax.set_title(f"{label(projection, measure)}: per-model association")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", color="black", linestyle="none", label="linear depth control"),
               Line2D([], [], marker="s", color="black", markerfacecolor="white", linestyle="none",
                      label="quadratic depth control"),
               Patch(color=DEC_C, label="decoder"), Patch(color=ENC_C, label="encoder")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)

    foot = (f"Decoders, model-level: linear {fmt_summary(srow(summary, 'decoders', 'linear'))};  "
            f"quadratic {fmt_summary(srow(summary, 'decoders', 'quadratic'))}. Exploratory.")
    fig.text(0.01, -0.01, foot, fontsize=8.5, ha="left", va="top")
    plt.tight_layout()
    save(fig, out_dir, f"fig3_consistency_{projection}_{measure}{tag(alignment)}")


def fig_compare(primary, other, alignment, out_dir):
    (p1, m1), (p2, m2) = primary, other
    _, A, SA = load(p1, m1, alignment)
    _, B, SB = load(p2, m2, alignment)
    order = [m for m in MODEL_ORDER if m in set(A.model) and m in set(B.model)]
    A, B = A.set_index("model").loc[order], B.set_index("model").loc[order]
    y = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 2.6), sharey=True)
    for ax, (sfx, control) in zip(axes, CONTROLS):
        a, b = A[f"mean_r{sfx}"].to_numpy(), B[f"mean_r{sfx}"].to_numpy()
        for i in range(len(order)):
            ax.plot([a[i], b[i]], [y[i], y[i]], color="lightgrey", linewidth=1.5, zorder=1)
        ax.scatter(a, y, s=45, color="black", zorder=3, label=label(p1, m1))
        ax.scatter(b, y, s=45, facecolors="white", edgecolors="black", linewidths=1.5,
                   zorder=3, label=label(p2, m2))
        n_dec = int((A.arch == "decoder").sum())
        ax.axhline(n_dec - 0.5, color="black", linewidth=1)
        ax.axvline(0, color="grey", linewidth=1, linestyle="--")
        ra, rb = srow(SA, "decoders", control), srow(SB, "decoders", control)
        ax.set_title(f"{control} depth control\n"
                     f"decoders: {ra['mean']:+.3f} ({ra.n_positive}/{ra.n_models}) vs "
                     f"{rb['mean']:+.3f} ({rb.n_positive}/{rb.n_models})", fontsize=11)
        ax.set_xlabel("mean partial r across words")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    fig.suptitle("Per-model association with Sep: primary against comparison measure", y=1.02)
    plt.tight_layout()
    save(fig, out_dir, f"fig5_compare_{p1}_{m1}_vs_{p2}_{m2}{tag(alignment)}")


def robustness(summary, exclude):
    print(f"\n  Decoders, model-level, with and without '{exclude}'")
    for control in ("linear", "quadratic"):
        a, b = srow(summary, "decoders", control), srow(summary, "decoders", control, f"without_{exclude}")
        print(f"    {control:<10} all words  {fmt_summary(a)}")
        print(f"    {'':<10} without {exclude:<5}{fmt_summary(b)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="stable_rank",
                    choices=["spectral_norm", "fro_norm", "erank", "stable_rank"])
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--exclude", default="bank", help="word left out of the robustness heatmap")
    ap.add_argument("--alignment", default="post", choices=list(ALIGNMENTS))
    ap.add_argument("--compare", default="", help="projection:measure, e.g. up_proj:erank")
    a = ap.parse_args()

    cells, models, summary = load(a.projection, a.measure, a.alignment)
    out_dir = ensure_dir(get_results_root() / "analysis" / "_cross_word" / "figures")
    print(f"\n{a.projection}  {a.measure}  alignment={a.alignment}   "
          f"words: {cells.word.nunique()}   models: {cells.model.nunique()}\n")

    lim = float(np.nanmax(np.abs(cells[["partial_r", "partial_r_quad"]].to_numpy())))
    fig_heatmap(cells, a.projection, a.measure, a.alignment, out_dir, "linear", lim=lim)
    fig_heatmap(cells, a.projection, a.measure, a.alignment, out_dir, "quadratic", lim=lim)
    fig_word_means(cells, a.projection, a.measure, a.alignment, out_dir)
    fig_model_consistency(models, summary, a.projection, a.measure, a.alignment, out_dir)
    if a.exclude and a.exclude in set(cells.word):
        fig_heatmap(cells, a.projection, a.measure, a.alignment, out_dir, "linear", exclude=a.exclude, lim=lim)
        robustness(summary, a.exclude)
    if a.compare:
        p2, m2 = a.compare.split(":")
        fig_compare((a.projection, a.measure), (p2, m2), a.alignment, out_dir)


if __name__ == "__main__":
    main()
