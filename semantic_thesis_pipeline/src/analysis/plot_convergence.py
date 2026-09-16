# -*- coding: utf-8 -*-
"""
Convergence curves for the separation-regularised fine-tuning runs.

One panel per task. Each panel shows performance against training step, one
curve per condition, averaged over seeds with a band for the spread. The
question is whether the increase condition reaches a given level in fewer
steps than baseline, which a before-and-after measurement cannot answer.

A second figure reports the steps needed to reach a fixed fraction of the
baseline's final performance, which states the same thing as a number rather
than a shape.

Usage
    python -m src.analysis.plot_convergence
    python -m src.analysis.plot_convergence --metric accuracy --tasks wic
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import get_results_root, ensure_dir

CONDITIONS = ["baseline", "increase", "preserve", "decrease"]
COLOURS = {"baseline": "#2C2C2A", "increase": "#1f77b4",
           "preserve": "#2ca02c", "decrease": "#d62728"}

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(metric):
    rows = []
    for f in glob.glob(str(get_results_root() / "analysis" / "_finetune" / "*.json")):
        d = json.load(open(f))
        for pt in d.get("curve", []):
            v = pt.get(metric)
            if v is None or v != v:
                continue
            rows.append({"task": d["task"], "condition": d["condition"],
                         "seed": d.get("seed"), "step": pt["step"],
                         "value": v, "sep": pt.get("sep"),
                         "eval_n": d.get("eval_n", 0)})
    return pd.DataFrame(rows)


def fig_curves(df, metric, out_dir):
    tasks = sorted(df.task.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.8),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        d = df[df.task == task]
        for cond in CONDITIONS:
            c = d[d.condition == cond]
            if c.empty:
                continue
            g = c.groupby("step").value.agg(["mean", "std", "count"])
            ax.plot(g.index, g["mean"], linewidth=2.2, color=COLOURS[cond],
                    label=cond, marker="o", markersize=3.5)
            if g["count"].max() > 1:
                ax.fill_between(g.index, g["mean"] - g["std"].fillna(0),
                                g["mean"] + g["std"].fillna(0),
                                color=COLOURS[cond], alpha=0.12)
        ax.set_title(task)
        ax.set_xlabel("Training step")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel(metric.capitalize())
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                       frameon=False)
    fig.suptitle(f"Convergence under separation regularisation ({metric})",
                 fontsize=17, y=1.02)
    plt.tight_layout()
    out = out_dir / f"fig_convergence_{metric}.png"
    fig.savefig(out); fig.savefig(out.with_suffix(".pdf")); plt.close()
    print(f"  Saved -> {out}")


def steps_to_target(df, metric, frac, out_dir):
    """Steps needed to reach a fraction of the baseline's final value.

    For loss, which falls, the target is approached from above; for accuracy it
    is approached from below. Runs that never reach the target are reported as
    not reached rather than given an arbitrary value.
    """
    rows = []
    for task in sorted(df.task.unique()):
        d = df[df.task == task]
        base = d[d.condition == "baseline"]
        if base.empty:
            continue
        final = base[base.step == base.step.max()].value.mean()
        start = base[base.step == base.step.min()].value.mean()
        target = start + frac * (final - start)

        for cond in CONDITIONS:
            c = d[d.condition == cond]
            if c.empty:
                continue
            reached = []
            for sd, g in c.groupby("seed"):
                g = g.sort_values("step")
                hit = (g.value <= target) if final < start else (g.value >= target)
                idx = np.argmax(hit.values) if hit.any() else None
                reached.append(g.step.values[idx] if idx is not None else np.nan)
            rows.append({"task": task, "condition": cond,
                         "target": round(target, 4),
                         "steps": np.nanmean(reached) if np.any(~np.isnan(reached)) else np.nan,
                         "n_reached": int(np.sum(~np.isnan(reached))),
                         "n": len(reached)})

    t = pd.DataFrame(rows)
    print(f"\n  Steps to reach {frac:.0%} of the baseline's final {metric}")
    print(f"  {'task':<12}{'condition':<11}{'target':>9}{'steps':>8}{'reached':>10}")
    print("  " + "-" * 50)
    for _, r in t.iterrows():
        st = f"{r.steps:.0f}" if r.steps == r.steps else "--"
        print(f"  {r.task:<12}{r.condition:<11}{r.target:>9.4f}{st:>8}"
              f"{f'{r.n_reached}/{r.n}':>10}")
    t.to_csv(out_dir / f"steps_to_target_{metric}.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="loss", choices=["loss", "accuracy"])
    ap.add_argument("--frac", type=float, default=0.9)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--min-eval-n", type=int, default=150,
                    help="Exclude runs evaluated on fewer items than this, so "
                         "that earlier subset runs do not mix with full "
                         "test-set runs")
    a = ap.parse_args()

    df = load(a.metric)
    if df.empty:
        raise SystemExit("No curve data found. Run the fine-tuning with "
                         "--eval-every set first.")
    if a.tasks:
        df = df[df.task.isin([t.strip() for t in a.tasks.split(",")])]
    before = len(df)
    df = df[df.eval_n >= a.min_eval_n]
    if len(df) < before:
        print(f"  excluded {before - len(df)} points from runs evaluated on "
              f"fewer than {a.min_eval_n} items")

    print(f"\nConvergence curves   metric: {a.metric}   "
          f"tasks: {', '.join(sorted(df.task.unique()))}\n")
    out_dir = ensure_dir(get_results_root() / "analysis" / "_finetune" / "figures")
    fig_curves(df, a.metric, out_dir)
    steps_to_target(df, a.metric, a.frac, out_dir)


if __name__ == "__main__":
    main()
