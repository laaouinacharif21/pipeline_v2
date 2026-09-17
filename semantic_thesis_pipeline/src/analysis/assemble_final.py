# -*- coding: utf-8 -*-
"""
Assemble results/final/ for the thesis (September 2026 revision).

Copies the selected figures and tables into results/final/ under their
thesis names and writes README.txt. Every number in the README is read from
the CSV outputs, so the text cannot drift from the tables.

Safety
    - aborts unless the pre-revision backup results_pre_alignment_fix/final/
      exists
    - aborts if any regenerated source is older than that backup (i.e. was
      not produced by the revised pipeline)
    - separation-only outputs unaffected by the alignment fix (fig1, figS1,
      figS2) are carried over from the backup and labelled as such

Usage
    python -m src.analysis.assemble_final
"""

import datetime as dt
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.paths import get_results_root  # noqa: E402

R = get_results_root()
FINAL = R / "final"
BACKUP = ROOT / "results_pre_alignment_fix"
CW = R / "analysis" / "_cross_word" / "figures"
TB = R / "analysis" / "_tables"
FT = R / "analysis" / "_finetune" / "figures"
WORDS = "bank,bat,crane,seal,plant,pupil,club"

# (final name, source path, carried-over flag)
MANIFEST = [
    ("fig1_separation_by_architecture.png", BACKUP / "final" / "fig1_separation_by_architecture.png", True),
    ("fig1_separation_by_architecture.pdf", BACKUP / "final" / "fig1_separation_by_architecture.pdf", True),
    ("fig2_geometry_heatmap.png", CW / "fig1_heatmap_q_proj_stable_rank.png", False),
    ("fig2_geometry_heatmap.pdf", CW / "fig1_heatmap_q_proj_stable_rank.pdf", False),
    ("fig3_model_consistency.png", CW / "fig3_consistency_q_proj_stable_rank.png", False),
    ("fig3_model_consistency.pdf", CW / "fig3_consistency_q_proj_stable_rank.pdf", False),
    ("fig4_convergence_accuracy.png", FT / "fig_convergence_accuracy.png", False),
    ("table1_dataset_properties.csv", TB / "table_a_dataset_properties.csv", False),
    ("table2_by_model.csv", TB / "table_b_by_model_q_proj_stable_rank.csv", False),
    ("table3_model_level.csv", TB / "table_c_by_architecture_q_proj_stable_rank.csv", False),
    ("table4_probe_geometry.csv", TB / "table4_probe_geometry.csv", False),
    ("table4_probe_geometry_summary.csv", TB / "table4_probe_geometry_summary.csv", False),
    ("table5a_qproj_truncate_cells.csv", TB / "table5a_qproj_truncate_cells.csv", False),
    ("table5b_qproj_truncate_models.csv", TB / "table5b_qproj_truncate_models.csv", False),
    ("table5c_model_level.csv", TB / "table5c_model_level.csv", False),
    ("table5d_vproj_truncate.csv", TB / "table5d_vproj_truncate_cells.csv", False),
    ("table5e_capability.csv", TB / "table5e_capability.csv", False),
    ("table5f_probe_intervention.csv", TB / "table5f_probe_intervention.csv", False),
    ("table5g_case_study.csv", TB / "table5g_case_study.csv", False),
    ("figS1_controlled_words_only.png", BACKUP / "final" / "figS1_controlled_words_only.png", True),
    ("figS2_all_words_by_family.png", BACKUP / "final" / "figS2_all_words_by_family.png", True),
    ("figS3_heatmap_without_bank.png", CW / "fig1_heatmap_q_proj_stable_rank_no_bank.png", False),
    ("figS3_heatmap_without_bank.pdf", CW / "fig1_heatmap_q_proj_stable_rank_no_bank.pdf", False),
    ("figS4_measure_comparison.png", CW / "fig5_compare_q_proj_stable_rank_vs_up_proj_erank.png", False),
    ("figS4_measure_comparison.pdf", CW / "fig5_compare_q_proj_stable_rank_vs_up_proj_erank.pdf", False),
    ("figS5_heatmap_quadratic.png", CW / "fig1_heatmap_q_proj_stable_rank_quad.png", False),
    ("figS5_heatmap_quadratic.pdf", CW / "fig1_heatmap_q_proj_stable_rank_quad.pdf", False),
    ("figS6_convergence_loss.png", FT / "fig_convergence_loss.png", False),
    ("tableS1_sensitivity.csv", TB / "table_s1_sensitivity_q_proj_stable_rank.csv", False),
    ("tableS2_alignment.csv", TB / "table_s2_alignment_q_proj_stable_rank.csv", False),
]


def check_sources():
    if not (BACKUP / "final").is_dir():
        raise SystemExit(f"Backup {BACKUP / 'final'} not found. Refusing to overwrite results/final.")
    cutoff = BACKUP.stat().st_mtime
    problems = []
    for name, src, carried in MANIFEST:
        if not src.exists():
            problems.append(f"missing   {name}  <-  {src}")
        elif not carried and src.stat().st_mtime < cutoff:
            problems.append(f"stale     {name}  <-  {src}  (older than the backup)")
    if problems:
        raise SystemExit("Cannot assemble results/final:\n  " + "\n  ".join(problems))


# ------------------------------------------------------------- numbers

def pick(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    r = df[m]
    if len(r) != 1:
        raise SystemExit(f"Expected one row for {kw}, found {len(r)}")
    return r.iloc[0]


def ml(r, digits=3):
    """Model-level summary sentence from a summary row."""
    w = "n/a" if pd.isna(r.wilcoxon_p) else f"{r.wilcoxon_p:.3f}"
    t = "=n/a" if pd.isna(r.t_p) else ("<0.001" if r.t_p < 0.001 else f"={r.t_p:.3f}")
    loo = "" if pd.isna(r.get("loo_min")) else f", leave-one-model-out {r.loo_min:+.3f} to {r.loo_max:+.3f}"
    return f"{r['mean']:+.{digits}f}, {int(r.n_positive)}/{int(r.n_models)} positive{loo}, t p{t}, Wilcoxon p={w}"


def readme_text():
    C = pd.read_csv(TB / "table_c_by_architecture_q_proj_stable_rank.csv")
    S1 = pd.read_csv(TB / "table_s1_sensitivity_q_proj_stable_rank.csv")
    S2 = pd.read_csv(TB / "table_s2_alignment_q_proj_stable_rank.csv")
    T4 = pd.read_csv(TB / "table4_probe_geometry_summary.csv")
    T4m = pd.read_csv(TB / "table4_probe_geometry.csv")
    T5 = pd.read_csv(TB / "table5c_model_level.csv")
    T5c = pd.read_csv(TB / "table5a_qproj_truncate_cells.csv")
    T5v = pd.read_csv(TB / "table5d_vproj_truncate_cells.csv")
    cmp_ = pd.read_csv(R / "analysis" / "_cross_word" / "cross_word_up_proj_erank_summary.csv")

    dl = pick(C, subset="all words", group="decoders", control="linear")
    dq = pick(C, subset="all words", group="decoders", control="quadratic")
    bl = pick(C, subset="excluding bank", group="decoders", control="linear")
    bq = pick(C, subset="excluding bank", group="decoders", control="quadratic")
    ll = pick(C, subset="all words", group="llama", control="quadratic")
    qq = pick(C, subset="all words", group="qwen", control="quadratic")
    el = pick(C, subset="all words", group="encoders", control="linear")
    ul = pick(cmp_, group="decoders", control="linear", words="all")
    uq = pick(cmp_, group="decoders", control="quadratic", words="all")
    s1l = pick(S1, architecture="decoder", control="linear")
    s1q = pick(S1, architecture="decoder", control="quadratic")
    s2l = pick(S2, group="decoders", control="linear")
    s2q = pick(S2, group="decoders", control="quadratic")
    t4l = pick(T4, group="decoders", control="linear")
    t4q = pick(T4, group="decoders", control="quadratic")
    dec4 = T4m[T4m.arch == "decoder"]
    qpk = pick(T5, analysis="q_proj truncate 0.80: peak_retention - 1", group="decoders")
    qmn = pick(T5, analysis="q_proj truncate 0.80: mean_retention - 1", group="decoders")
    vmn = pick(T5, analysis="v_proj truncate 0.80: mean_retention - 1", group="decoders")
    gq = pick(T5, analysis="q_proj truncate 0.80: gsm8k_delta", group="decoders")
    gv = pick(T5, analysis="v_proj truncate 0.80: gsm8k_delta", group="decoders")
    gr = pick(T5, analysis="q_proj rotate: gsm8k_delta", group="decoders")
    vcells = T5v[T5v.level == "cell"]

    return f"""Figures and tables selected for the thesis document.
Assembled {dt.date.today().isoformat()} by src/analysis/assemble_final.py.
Copies, not originals. Pre-revision versions: results_pre_alignment_fix/final/.

CONVENTIONS (September 2026 revision)
    Alignment   geometry of block l is paired with the hidden state produced by
                that block (index l+1). Pre-block pairing: Table S2 only.
    Estimand    per model, the mean over the seven words of the depth-controlled
                partial correlation between q_proj stable rank and Sep.
    Inference   model level: sign count, leave-one-model-out range, one-sample
                t-test and Wilcoxon signed-rank. All p-values are exploratory:
                the nine decoders are not independent samples (two families,
                related versions). No per-cell significance marks are drawn.
    Dependence  adjacent layers are strongly autocorrelated; see Table S1.

REGENERATE
    python -m src.analysis.cross_word --measure stable_rank --projection q_proj --words {WORDS}
    python -m src.analysis.cross_word --measure stable_rank --projection q_proj --words {WORDS} --alignment pre
    python -m src.analysis.cross_word --measure erank --projection up_proj --words {WORDS}
    python -m src.analysis.results_tables
    python -m src.analysis.cross_word_figures --compare up_proj:erank
    python -m src.analysis.probe_geometry
    python -m src.analysis.intervention_tables
    python -m src.analysis.plot_convergence --metric accuracy
    python -m src.analysis.plot_convergence --metric loss
    python -m src.analysis.assemble_final
    tests: python -m tests.run_all

MAIN

fig1_separation_by_architecture   [carried over; separation only, unaffected]
    Layer-wise Sep(l) by family. Six controlled datasets as a mean with
    +/-1 SD across models; bank shown separately.

fig2_geometry_heatmap
    Partial r between q_proj stable rank and Sep, linear depth control,
    13 models by 7 words. Descriptive; no significance marks.

fig3_model_consistency
    Per-model mean r across words with its range, linear and quadratic
    depth control.
    Decoders, linear:     {ml(dl)}
    Decoders, quadratic:  {ml(dq)}
    Quadratic by family (descriptive): llama {ll['mean']:+.3f} ({int(ll.n_positive)}/{int(ll.n_models)}),
    qwen {qq['mean']:+.3f} ({int(qq.n_positive)}/{int(qq.n_models)}). Encoders, linear: {el['mean']:+.3f} ({int(el.n_positive)}/{int(el.n_models)}).

fig4_convergence_accuracy
    Fine-tuning of llama-3-8b with a separation regulariser (baseline,
    increase, decrease), 3 seeds, evaluation every 20 steps. The regulariser
    changed Sep as intended but did not demonstrate a consistent improvement
    in final performance or convergence speed.

table1_dataset_properties
    Per word: size, senses, lexical overlap, sentence length, context-only
    and target-only classification accuracy, largest Sep and its depth.

table2_by_model
    Per model: mean partial r over words, range, words positive, and
    circular-shift permutation p, under linear and quadratic control.

table3_model_level
    Model-level tests for decoders, llama, qwen and encoders, and for
    decoders excluding bank (context leakage):
    excluding bank, linear:     {ml(bl)}
    excluding bank, quadratic:  {ml(bq)}

table4_probe_geometry (+ _summary)
    TruthfulQA linear-probe accuracy (cross-validated, question-grouped
    folds; not generation accuracy) against q_proj stable rank, post-block.
    Decoders, linear:     {ml(t4l)}
    Decoders, quadratic:  {ml(t4q)}
    Per model, circular-shift p<0.05: {int((dec4.perm_p < .05).sum())}/{len(dec4)} linear, {int((dec4.perm_p_quad < .05).sum())}/{len(dec4)} quadratic;
    effective-n p<0.05: {int((dec4.p_eff < .05).sum())}/{len(dec4)} linear, {int((dec4.p_quad_eff < .05).sum())}/{len(dec4)} quadratic.

table5a-g_intervention
    Retention = value after intervention / unmodified baseline.
    5a/5b q_proj truncation to 80% stable rank, {len(T5c)} cells, 9 models.
          peak Sep change {ml(qpk, 4)}
          mean Sep change {ml(qmn, 4)}
          peak layer moved in {int(T5c.peak_shifted.sum())} cells.
    5c    model-level tests, including capability.
    5d    v_proj truncation, {len(vcells)} cells, {vcells.model.nunique()} models:
          mean Sep change {ml(vmn, 3)}
    5e    capability. GSM8K = change in reference-answer log-probability,
          not accuracy. C-Eval near chance for LLaMA (baseline < 0.35).
          q_proj truncate {ml(gq, 3)}
          v_proj truncate {ml(gv, 3)}
          q_proj rotate   {ml(gr, 3)}
    5f    TruthfulQA probe under intervention, 3 models.
    5g    single-model conditions (rotation, noise, other ratios); descriptive
          only, not part of cross-model tests.

SUPPLEMENTARY

figS1_controlled_words_only   [carried over; separation only, unaffected]
figS2_all_words_by_family     [carried over; separation only, unaffected]

figS3_heatmap_without_bank
    As fig2 with bank excluded.

figS4_measure_comparison
    Per-model mean r for q_proj stable rank against up_proj effective rank.
    up_proj erank is a comparison measure, not a negative control.
    up_proj erank, linear:     {ml(ul)}
    up_proj erank, quadratic:  {ml(uq)}

figS5_heatmap_quadratic
    As fig2 under quadratic depth control.

figS6_convergence_loss
    Loss curves for the fine-tuning runs of fig4.

tableS1_sensitivity
    Share of decoder cells with p<0.05 when layers are treated as independent
    against an AR(1) effective-n correction:
    linear {s1l.share_p05_naive:.1%} -> {s1l.share_p05_effective_n:.1%} (mean n_eff {s1l.mean_n_eff:.1f} of {s1l.mean_n_layers:.1f} layers);
    quadratic {s1q.share_p05_naive:.1%} -> {s1q.share_p05_effective_n:.1%} (mean n_eff {s1q.mean_n_eff:.1f}).
    Effective n assumes AR(1) dependence; a sensitivity check only.

tableS2_alignment
    Post-block against pre-block pairing, model level.
    Decoders linear {s2l.post_mean:+.3f} (post) vs {s2l.pre_mean:+.3f} (pre);
    quadratic {s2q.post_mean:+.3f} vs {s2q.pre_mean:+.3f}.
"""


def main():
    check_sources()
    text = readme_text()
    FINAL.mkdir(parents=True, exist_ok=True)
    removed = 0
    for f in FINAL.iterdir():
        if f.is_file():
            f.unlink()
            removed += 1
    for name, src, carried in MANIFEST:
        shutil.copy2(src, FINAL / name)
    (FINAL / "README.txt").write_text(text)
    print(f"results/final: removed {removed} old files, copied {len(MANIFEST)}, wrote README.txt")
    for name, src, carried in MANIFEST:
        print(f"  {'carried ' if carried else 'new     '}{name}")


if __name__ == "__main__":
    main()
