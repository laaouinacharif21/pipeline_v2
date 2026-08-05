from pathlib import Path as _P
import sys as _sys
_sys.path.insert(0, str(_P(__file__).resolve().parent))
from src.utils.paths import get_results_root as _grr
_RESULTS_ROOT = _grr()
# benchmark_plots.py
# -------------------
# Benchmark visualisation pipeline — extended version.
#
# New function added:  plot_family_progression_with_separation()
#   - Same signature as plot_family_progression() but accepts an optional
#     results_root argument so the function auto-reads peak semantic
#     separation from the pipeline CSV outputs.
#   - If the CSV is not found for a model, separation is shown as 0.0 with
#     a warning so the rest of the plot still renders cleanly.
#
# Original functions (plot_family_progression, plot_family_heatmap) are
# kept unchanged so nothing else in your pipeline breaks.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "figure.dpi":        150,
})

# One colour per model slot — shared across bars and the separation line
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

_BENCHMARK_ALPHA = {"C-Eval": 0.55, "CMMLU": 0.70, "GSM8K": 1.00}
_BENCHMARK_HATCH = {"C-Eval": "",   "CMMLU": "//",  "GSM8K": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Helper — read peak semantic separation from pipeline CSV
# ─────────────────────────────────────────────────────────────────────────────

def _read_peak_separation(
    results_root: str,
    family_name: str,
    version: str,
) -> float:
    """
    Looks for:
        <results_root>/<family_name>/<version>/metrics/semantic_separation.csv

    The CSV is expected to have a column named 'separation'.
    Returns the maximum value found, or 0.0 with a printed warning if the
    file does not exist.
    """
    csv_path = (
        Path(results_root)
        / family_name.lower()
        / version
        / "metrics"
        / "semantic_separation.csv"
    )
    if not csv_path.exists():
        print(f"  [WARNING] separation CSV not found: {csv_path}  → using 0.0")
        return 0.0
    df = pd.read_csv(csv_path)
    return float(df["separation"].max())


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL functions — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def plot_family_progression(
    versions,
    ceval,
    cmmlu,
    gsm8k,
    family_name,
    out_dir="results/plots",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(versions))
    plt.figure(figsize=(10, 6))
    plt.plot(x, ceval, marker="o", linewidth=2, label="C-Eval")
    plt.plot(x, cmmlu, marker="s", linewidth=2, label="CMMLU")
    plt.plot(x, gsm8k, marker="^", linewidth=2, label="GSM8K")
    for i, v in enumerate(ceval):
        plt.text(i, v + 1.2, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(cmmlu):
        plt.text(i, v + 1.2, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(gsm8k):
        plt.text(i, v + 1.2, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, versions, rotation=15)
    plt.ylim(0, 100)
    plt.xlabel("Version")
    plt.ylabel("Score")
    plt.title(f"{family_name} Family - Benchmark Progression Across Versions")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{family_name.lower()}_family_benchmark_lines.png", dpi=300
    )
    plt.close()


def plot_family_heatmap(
    versions,
    ceval,
    cmmlu,
    gsm8k,
    family_name,
    out_dir="results/plots",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data    = np.array([ceval, cmmlu, gsm8k]).T
    metrics = ["C-Eval", "CMMLU", "GSM8K"]
    plt.figure(figsize=(7.5, 5.5))
    im = plt.imshow(data, aspect="auto")
    plt.xticks(np.arange(len(metrics)), metrics)
    plt.yticks(np.arange(len(versions)), versions)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center", fontsize=9,
            )
    plt.title(f"{family_name} Family - Benchmark Heatmap")
    plt.xlabel("Benchmark")
    plt.ylabel("Version")
    plt.colorbar(im, label="Score")
    plt.tight_layout()
    plt.savefig(
        out_dir / f"{family_name.lower()}_family_benchmark_heatmap.png", dpi=300
    )
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# NEW function — benchmark bars + semantic separation overlay
# ─────────────────────────────────────────────────────────────────────────────

def plot_family_progression_with_separation(
    versions,
    ceval,
    cmmlu,
    gsm8k,
    family_name,
    out_dir="results/plots",
    results_root="results",
    peak_sep_override=None,
):
    """
    Grouped bar chart of benchmark scores (C-Eval, CMMLU, GSM8K) with the
    peak semantic separation for each model overlaid as a dashed line on a
    second y-axis.

    Peak separation is read automatically from:
        <results_root>/<family_name>/<version>/metrics/semantic_separation.csv
    Or pass peak_sep_override as a list of floats to bypass CSV reading.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n      = len(versions)
    colors = _COLORS[:n]

    # ── peak separation: use override if provided, else auto-read CSVs ────────
    if peak_sep_override is not None:
        peak_sep = peak_sep_override
    else:
        peak_sep = [
            _read_peak_separation(results_root, family_name, v)
            for v in versions
        ]

    # ── bar geometry ──────────────────────────────────────────────────────────
    bar_w   = 0.22
    gap     = 0.12
    group_w = 3 * bar_w + gap
    centres = np.arange(n) * group_w
    offsets = [-bar_w, 0.0, bar_w]
    metrics = ["C-Eval", "CMMLU", "GSM8K"]
    data    = [ceval, cmmlu, gsm8k]

    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    ax2 = ax1.twinx()

    # ── generation shading ────────────────────────────────────────────────────
    mid = (centres[n // 2 - 1] + centres[n // 2]) / 2
    ax1.axvspan(
        centres[0] - group_w * 0.45, mid,
        color="#f0f4ff", zorder=0,
    )
    ax1.axvspan(
        mid, centres[-1] + group_w * 0.45,
        color="#fff4f0", zorder=0,
    )
    ax1.text(
        centres[0] + (mid - centres[0]) / 2, 99,
        "Earlier\ngenerations",
        ha="center", va="top", fontsize=8.5,
        color="#6688aa", style="italic",
    )
    ax1.text(
        mid + (centres[-1] - mid) / 2, 99,
        "Recent\ngenerations",
        ha="center", va="top", fontsize=8.5,
        color="#aa6655", style="italic",
    )

    # ── grouped bars ──────────────────────────────────────────────────────────
    for metric, vals, off in zip(metrics, data, offsets):
        for v_idx in range(n):
            xpos = centres[v_idx] + off
            ax1.bar(
                xpos, vals[v_idx],
                width=bar_w * 0.92,
                color=colors[v_idx],
                alpha=_BENCHMARK_ALPHA[metric],
                hatch=_BENCHMARK_HATCH[metric],
                edgecolor="white", linewidth=0.6,
                zorder=2,
            )
            ax1.text(
                xpos, vals[v_idx] + 1.2,
                f"{vals[v_idx]:.1f}",
                ha="center", va="bottom",
                fontsize=7.5, color="#333333", zorder=3,
            )

    # ── semantic separation line (right axis) ─────────────────────────────────
    ax2.plot(
        centres, peak_sep,
        color="#333333", linewidth=2.2,
        linestyle="--", marker="D",
        markersize=9, zorder=5,
    )
    for v_idx in range(n):
        ax2.plot(
            centres[v_idx], peak_sep[v_idx],
            marker="D", markersize=10,
            color=colors[v_idx],
            markeredgecolor="#333333", markeredgewidth=1.2,
            zorder=6,
        )
        sep_max = max(peak_sep) if max(peak_sep) > 0 else 1.0
        ax2.text(
            centres[v_idx],
            peak_sep[v_idx] + sep_max * 0.05,
            f"{peak_sep[v_idx]:.3f}",
            ha="center", va="bottom",
            fontsize=8.5, color=colors[v_idx],
            fontweight="bold", zorder=7,
        )

    # ── axis formatting ───────────────────────────────────────────────────────
    ax1.set_xticks(centres)
    ax1.set_xticklabels(versions, rotation=18, ha="right")
    ax1.set_xlim(centres[0] - group_w * 0.55, centres[-1] + group_w * 0.55)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Benchmark Score (%)", labelpad=8)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)

    sep_max = max(peak_sep) if max(peak_sep) > 0 else 1.0
    ax2.set_ylim(0, sep_max * 1.55)
    ax2.set_ylabel("Peak Semantic Separation", labelpad=8, color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.spines["right"].set_color("#888888")
    ax2.spines["top"].set_visible(False)

    # ── title & subtitle ──────────────────────────────────────────────────────
    ax1.set_title(
        f"{family_name} Family - Benchmark Scores & Peak Semantic Separation",
        pad=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.97,
        "Bars = benchmark score (left axis)   "
        "◆ = peak semantic separation across layers (right axis)",
        ha="center", va="top",
        fontsize=8.5, color="#555555", style="italic",
    )

    # ── legends ───────────────────────────────────────────────────────────────
    bench_patches = [
        mpatches.Patch(
            facecolor="grey",
            alpha=_BENCHMARK_ALPHA[m],
            hatch=_BENCHMARK_HATCH[m],
            edgecolor="white",
            label=m,
        )
        for m in metrics
    ]
    leg1 = ax1.legend(
        handles=bench_patches,
        title="Benchmark", title_fontsize=9,
        fontsize=9, loc="upper left",
        framealpha=0.9, edgecolor="#cccccc",
    )
    ax1.add_artist(leg1)

    model_handles = [
        Line2D(
            [0], [0], color=colors[i], linewidth=7,
            solid_capstyle="round", label=versions[i],
        )
        for i in range(n)
    ]
    sep_handle = Line2D(
        [0], [0], color="#333333", linewidth=2,
        linestyle="--", marker="D", markersize=7,
        label="Peak semantic sep.",
    )
    ax1.legend(
        handles=model_handles + [sep_handle],
        title="Model / metric", title_fontsize=9,
        fontsize=9, loc="lower right",
        framealpha=0.9, edgecolor="#cccccc",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = out_dir / f"{family_name.lower()}_family_benchmark_with_separation.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── LLaMA ─────────────────────────────────────────────────────────────────
    llama_versions = ["llama-7b", "llama-2-7b", "llama-3-8b", "llama-3.1-8b"]
    llama_ceval    = [27.29, 32.21, 50.02, 51.47]
    llama_cmmlu    = [27.02, 31.89, 50.81, 52.16]
    llama_gsm8k    = [9.86,  16.68, 56.10, 56.48]

    plot_family_progression(
        llama_versions, llama_ceval, llama_cmmlu, llama_gsm8k, "LLaMA"
    )
    plot_family_heatmap(
        llama_versions, llama_ceval, llama_cmmlu, llama_gsm8k, "LLaMA"
    )
    plot_family_progression_with_separation(
        llama_versions, llama_ceval, llama_cmmlu, llama_gsm8k,
        family_name  = "LLaMA",
        results_root = str(_RESULTS_ROOT),
    )

    # ── Qwen ──────────────────────────────────────────────────────────────────
    # Note: qwen3-8b evaluated with transformers==4.51.0; others with 4.40.2
    # † in label is display-only; real peak separation values passed directly
    qwen_versions = ["qwen-7b", "qwen1.5-7b", "qwen2-7b", "qwen2.5-7b", "qwen3-8b†"]
    qwen_ceval    = [63.36, 73.62, 76.75, 81.27, 79.44]
    qwen_cmmlu    = [62.66, 71.80, 83.79, 81.76, 77.31]
    qwen_gsm8k    = [54.13, 54.44, 79.61, 79.68, 87.79]
    # Real peak separation values read from pipeline CSVs
    qwen_peak_sep = [0.137654, 0.139284, 0.158969, 0.076297, 0.068070]

    plot_family_progression(
        qwen_versions, qwen_ceval, qwen_cmmlu, qwen_gsm8k, "Qwen"
    )
    plot_family_heatmap(
        qwen_versions, qwen_ceval, qwen_cmmlu, qwen_gsm8k, "Qwen"
    )
    plot_family_progression_with_separation(
        qwen_versions, qwen_ceval, qwen_cmmlu, qwen_gsm8k,
        family_name      = "Qwen",
        results_root     = str(_RESULTS_ROOT),
        peak_sep_override = qwen_peak_sep,
    )

    print("All benchmark plots saved in results/plots/")

    # ── BERT ──────────────────────────────────────────────────────────────────
    # GSM8K via MLM 4-choice scoring — not directly comparable to decoder scores
    bert_versions = ["bert-base", "roberta-base", "spanbert-base", "xlm-roberta-base"]
    bert_ceval    = [23.09, 26.72, 26.16, 22.35]
    bert_cmmlu    = [24.89, 25.17, 24.94, 25.64]
    bert_gsm8k    = [36.00, 23.80, 26.40, 36.20]
    bert_peak_sep = [0.174371, 0.028604, 0.120567, 0.047670]

    plot_family_progression(
        bert_versions, bert_ceval, bert_cmmlu, bert_gsm8k, "BERT"
    )
    plot_family_heatmap(
        bert_versions, bert_ceval, bert_cmmlu, bert_gsm8k, "BERT"
    )
    plot_family_progression_with_separation(
        bert_versions, bert_ceval, bert_cmmlu, bert_gsm8k,
        family_name       = "BERT",
        results_root      = str(_RESULTS_ROOT),
        peak_sep_override = bert_peak_sep,
    )

    print("BERT benchmark plots saved.")