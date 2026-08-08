# -*- coding: utf-8 -*-
"""
Cross-word figures for the geometry-separation analysis.

Produces four figures from the depth-controlled partial correlations written
by cross_word.py:

    fig1  model x word heatmap for one projection and measure
    fig2  mean correlation per word, decoders against encoders
    fig3  per-model consistency, ordered by how many words reach significance
    fig4  the same measure with and without a named word, as a robustness check

The heatmap is the primary figure: models that show the association in every
word appear as uniform bands, and models or architectures that do not appear
as noise.

Usage
    python -m src.analysis.cross_word_figures --measure stable_rank --projection q_proj
    python -m src.analysis.cross_word_figures --measure spectral_norm --projection up_proj
    python -m src.analysis.cross_word_figures --measure stable_rank --projection q_proj --exclude bank
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis._io import FAMILIES, DECODERS, analysis_dir
from src.utils.paths import get_results_root, ensure_dir

WORD_ORDER = ["bank", "bat", "crane", "seal", "plant", "pupil", "club"]
MODEL_ORDER = FAMILIES["llama"] + FAMILIES["qwen"] + FAMILIES["bert"]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(projection, measure):
    p = (get_results_root() / "analysis" / "_cross_word"
         / f"cross_word_{projection}_{measure}.csv")
    if not p.exists():
        raise SystemExit(
            f"Not found: {p}\nRun cross_word.py for this projection and measure first.")
    df = pd.read_csv(p)
    df = df[df.word.isin(WORD_ORDER)]
    return df


def fig_heatmap(df, projection, measure, out_dir, exclude=None):
    words = [w for w in WORD_ORDER if w in set(df.word) and w != exclude]
    models = [m for m in MODEL_ORDER if m in set(df.model)]

    M = np.full((len(models), len(words)), np.nan)
    S = np.empty((len(models), len(words)), dtype=object)
    for i, m in enumerate(models):
        for j, w in enumerate(words):
            r = df[(df.model == m) & (df.word == w)]
            if len(r):
                M[i, j] = r.partial_r.iloc[0]
                p = r.partial_p.iloc[0]
                S[i, j] = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            else:
                S[i, j] = ""

    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(1.1 * len(words) + 3.2, 0.46 * len(models) + 2))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                   aspect="auto")

    for i in range(len(models)):
        for j in range(len(words)):
            if np.isnan(M[i, j]):
                continue
            shade = "white" if abs(M[i, j]) > 0.6 * lim else "black"
            ax.text(j, i, f"{M[i, j]:+.2f}\n{S[i, j]}", ha="center", va="center",
                    fontsize=8, color=shade, linespacing=0.9)

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    n_dec = sum(1 for m in models if m in DECODERS)
    ax.axhline(n_dec - 0.5, color="black", linewidth=1.6)
    # Offset the block labels past the longest model name so they cannot
    # overlap the tick labels.
    pad = -(6.2 * max(len(m) for m in models) + 26)
    for label, centre in [("decoder", (n_dec - 1) / 2),
                          ("encoder", n_dec + (len(models) - n_dec - 1) / 2)]:
        ax.annotate(label, xy=(0, centre), xycoords=("axes fraction", "data"),
                    xytext=(pad, 0), textcoords="offset points", rotation=90,
                    va="center", ha="center", fontsize=11)

    title = f"{projection} {measure.replace('_', ' ')} against Sep(l), depth-controlled"
    if exclude:
        title += f"   (excluding {exclude})"
    ax.set_title(title, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="partial r")

    name = f"fig1_heatmap_{projection}_{measure}" + (f"_no_{exclude}" if exclude else "")
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close()
    print(f"  Saved -> {out_dir / (name + '.png')}")


def fig_word_means(df, projection, measure, out_dir):
    words = [w for w in WORD_ORDER if w in set(df.word)]
    dec, enc, dec_sd = [], [], []
    for w in words:
        d = df[(df.word == w) & (df.model.isin(DECODERS))].partial_r
        e = df[(df.word == w) & (~df.model.isin(DECODERS))].partial_r
        dec.append(d.mean()); dec_sd.append(d.std()); enc.append(e.mean())

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(words))
    ax.errorbar(x, dec, yerr=dec_sd, marker="o", linewidth=2, capsize=4,
                color="#1f77b4", label="decoder models")
    ax.plot(x, enc, marker="s", linewidth=2, color="#d62728", label="encoder models")
    ax.axhline(0, color="grey", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(words)
    ax.set_ylabel("mean partial r")
    ax.set_title(f"{projection} {measure.replace('_', ' ')} by target word")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f"fig2_word_means_{projection}_{measure}.png")
    plt.close()
    print(f"  Saved -> {out_dir / f'fig2_word_means_{projection}_{measure}.png'}")


def fig_model_consistency(df, projection, measure, out_dir):
    rows = []
    for m in MODEL_ORDER:
        d = df[df.model == m]
        if d.empty:
            continue
        rows.append({"model": m, "k": int((d.partial_p < .05).sum()),
                     "n": len(d), "mean_r": d.partial_r.mean(),
                     "decoder": m in DECODERS})
    s = pd.DataFrame(rows).sort_values(["decoder", "k"], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(s) + 1.8))
    colours = ["#1f77b4" if d else "#d62728" for d in s.decoder]
    ax.barh(range(len(s)), s.k, color=colours)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.model)
    ax.set_xlabel(f"words with p < 0.05 (of {int(s.n.max())})")
    ax.set_xlim(0, s.n.max())
    for i, (k, r) in enumerate(zip(s.k, s.mean_r)):
        ax.text(k + 0.08, i, f"mean r {r:+.2f}", va="center", fontsize=9)
    ax.set_title(f"{projection} {measure.replace('_', ' ')}: consistency across words")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_dir / f"fig3_consistency_{projection}_{measure}.png")
    plt.close()
    print(f"  Saved -> {out_dir / f'fig3_consistency_{projection}_{measure}.png'}")


def robustness(df, exclude):
    """Report the decoder mean with and without one word."""
    d = df[df.model.isin(DECODERS)]
    all_words = d.partial_r.mean()
    without = d[d.word != exclude].partial_r.mean()
    sig_all = (d.partial_p < .05).mean()
    sig_wo = (d[d.word != exclude].partial_p < .05).mean()
    print(f"\n  Decoder mean partial r")
    print(f"    all seven words     {all_words:+.3f}   significant {sig_all:.0%} of cells")
    print(f"    excluding {exclude:<10}{without:+.3f}   significant {sig_wo:.0%} of cells")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="stable_rank",
                    choices=["spectral_norm", "fro_norm", "erank", "stable_rank"])
    ap.add_argument("--projection", default="q_proj")
    ap.add_argument("--exclude", default="bank",
                    help="Word to leave out of the robustness figure")
    a = ap.parse_args()

    df = load(a.projection, a.measure)
    out_dir = ensure_dir(get_results_root() / "analysis" / "_cross_word" / "figures")

    print(f"\n{a.projection}  {a.measure}   words: {df.word.nunique()}   "
          f"models: {df.model.nunique()}\n")
    fig_heatmap(df, a.projection, a.measure, out_dir)
    fig_word_means(df, a.projection, a.measure, out_dir)
    fig_model_consistency(df, a.projection, a.measure, out_dir)
    if a.exclude and a.exclude in set(df.word):
        fig_heatmap(df, a.projection, a.measure, out_dir, exclude=a.exclude)
        robustness(df, a.exclude)


if __name__ == "__main__":
    main()
