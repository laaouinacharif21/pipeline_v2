Figures and tables selected for the thesis document.
Assembled 2026-09-26 by src/analysis/assemble_final.py.
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
    python -m src.analysis.cross_word --measure stable_rank --projection q_proj --words bank,bat,crane,seal,plant,pupil,club
    python -m src.analysis.cross_word --measure stable_rank --projection q_proj --words bank,bat,crane,seal,plant,pupil,club --alignment pre
    python -m src.analysis.cross_word --measure erank --projection up_proj --words bank,bat,crane,seal,plant,pupil,club
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
    Decoders, linear:     +0.501, 9/9 positive, leave-one-model-out +0.479 to +0.541, t p<0.001, Wilcoxon p=0.004
    Decoders, quadratic:  +0.137, 7/9 positive, leave-one-model-out +0.108 to +0.168, t p=0.038, Wilcoxon p=0.055
    Quadratic by family (descriptive): llama +0.296 (4/4),
    qwen +0.009 (3/5). Encoders, linear: +0.116 (3/4).

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
    excluding bank, linear:     +0.481, 9/9 positive, leave-one-model-out +0.456 to +0.521, t p<0.001, Wilcoxon p=0.004
    excluding bank, quadratic:  +0.094, 6/9 positive, leave-one-model-out +0.068 to +0.124, t p=0.140, Wilcoxon p=0.203

table4_probe_geometry (+ _summary)
    TruthfulQA linear-probe accuracy (cross-validated, question-grouped
    folds; not generation accuracy) against q_proj stable rank, post-block.
    Decoders, linear:     +0.634, 9/9 positive, leave-one-model-out +0.608 to +0.676, t p<0.001, Wilcoxon p=0.004
    Decoders, quadratic:  +0.320, 7/9 positive, leave-one-model-out +0.265 to +0.385, t p=0.029, Wilcoxon p=0.074
    Per model, circular-shift p<0.05: 6/9 linear, 3/9 quadratic;
    effective-n p<0.05: 3/9 linear, 2/9 quadratic.

table5a-g_intervention
    Retention = value after intervention / unmodified baseline.
    5a/5b q_proj truncation to 80% stable rank, 27 cells, 9 models.
          peak Sep change +0.0253, 5/9 positive, leave-one-model-out +0.014 to +0.034, t p=0.268, Wilcoxon p=0.164
          mean Sep change +0.0228, 5/9 positive, leave-one-model-out +0.013 to +0.033, t p=0.351, Wilcoxon p=0.250
          peak layer moved in 6 cells.
    5c    model-level tests, including capability.
    5d    v_proj truncation, 27 cells, 9 models:
          mean Sep change -0.543, 0/9 positive, leave-one-model-out -0.609 to -0.483, t p=0.003, Wilcoxon p=0.004
    5e    capability. GSM8K = change in reference-answer log-probability,
          not accuracy. C-Eval near chance for LLaMA (baseline < 0.35).
          q_proj truncate -0.003, 2/9 positive, leave-one-model-out -0.011 to -0.001, t p=0.721, Wilcoxon p=0.250
          v_proj truncate -1.712, 0/9 positive, leave-one-model-out -1.924 to -1.528, t p=0.001, Wilcoxon p=0.004
          q_proj rotate   -6.346, 0/9 positive, leave-one-model-out -6.590 to -6.092, t p<0.001, Wilcoxon p=0.004
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
    up_proj erank, linear:     +0.235, 7/9 positive, leave-one-model-out +0.204 to +0.274, t p=0.011, Wilcoxon p=0.020
    up_proj erank, quadratic:  -0.117, 2/9 positive, leave-one-model-out -0.141 to -0.089, t p=0.036, Wilcoxon p=0.055

figS5_heatmap_quadratic
    As fig2 under quadratic depth control.

figS6_convergence_loss
    Loss curves for the fine-tuning runs of fig4.

tableS1_sensitivity
    Share of decoder cells with p<0.05 when layers are treated as independent
    against an AR(1) effective-n correction:
    linear 74.6% -> 12.7% (mean n_eff 8.8 of 31.6 layers);
    quadratic 27.0% -> 3.2% (mean n_eff 15.1).
    Effective n assumes AR(1) dependence; a sensitivity check only.

tableS2_alignment
    Post-block against pre-block pairing, model level.
    Decoders linear +0.501 (post) vs +0.464 (pre);
    quadratic +0.137 vs +0.070.
