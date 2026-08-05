# Command reference

All commands run from the project root:
`/new_raid/nanhangproj/tianyu/semantic_thesis_pipeline`

## Environments

    conda activate llm_env      # everything except qwen-7b
    conda activate qwen7_env    # qwen-7b only

qwen-7b requires transformers 4.32.0. Under 5.x it loads but `attn.c_proj`
is randomly initialised instead of read from the checkpoint, and the forward
pass fails on the removed `get_head_mask`. Both its geometry and its
extraction must run in `qwen7_env`.

## 1. Parameter geometry

Once per model. Independent of the dataset — no need to repeat per word.

    python -m src.analysis.compute_geometry --model qwen2.5-7b
    python -m src.analysis.compute_geometry --all-models --skip qwen-7b
    python -m src.analysis.compute_geometry --all-models --skip qwen-7b --continue-on-error
    python -m src.analysis.compute_geometry --model qwen-7b          # qwen7_env
    python -m src.analysis.compute_geometry --all-models --cache     # skip existing

Writes `results/parameters/{family}/{model}/`:
`effective_rank.csv`, `parameter_stats.csv`, `geometry_manifest.json`

## 2. Extraction, metrics, layer selection, plots

    python run.py --model qwen2.5-7b --word bank
    python run.py --model qwen2.5-7b --word bank --stage extract
    python run.py --model qwen2.5-7b --word bank --stage metrics
    python run.py --model qwen2.5-7b --word bank --stage select_layers
    python run.py --model qwen2.5-7b --word bank --stage plots
    python run.py --all-models --word bank --skip qwen-7b
    python run.py --model qwen-7b --word bank                        # qwen7_env
    python run.py --all-models --word bank --continue-on-error
    python run.py --model qwen2.5-7b --word bank --batch-size 4
    python run.py --model qwen2.5-7b --word bank --no-strict         # disable gates

Stages: `extract` -> `metrics` -> `select_layers` -> `plots`. Default `all`.
Writes `results/words/{word}/{family}/{model}/`.

The `plots` stage generates per-layer PCA and heatmaps (~23 MB per model).

## 3. Analysis

    # Partial correlation, depth-controlled (primary statistic)
    python -m src.analysis.partial_correlation --word bank --measure spectral_norm
    python -m src.analysis.partial_correlation --word bank --measure erank

    # Pooled layer-level, within-model standardised
    python -m src.analysis.pooled_layer_level --word bank --column up_proj_erank
    python -m src.analysis.pooled_layer_level --word bank --column q_proj_erank
    python -m src.analysis.pooled_layer_level --word bank --column q_proj_spectral_norm

    # Separation curves by family + peak table
    python -m src.analysis.plot_family_separation --word bank

    # Publication figures
    python -m src.analysis.publication_plots --word bank

Writes `results/analysis/{word}/`.

`publication_plots` requires `effective_rank_summary.csv` in that directory;
regenerate it with the snippet in section 6.

## 4. Tests

    python -m tests.test_pipeline

67 checks: Sep(0), token position, label alignment, norm bounds, qwen-7b
slicing, effective rank bounds.

## 5. Full rebuild from scratch

    conda activate llm_env
    python -m src.analysis.compute_geometry --all-models --skip qwen-7b
    python run.py --all-models --word bank --skip qwen-7b

    conda activate qwen7_env
    python -m src.analysis.compute_geometry --model qwen-7b
    python run.py --model qwen-7b --word bank

    conda activate llm_env
    python -m src.analysis.partial_correlation --word bank --measure spectral_norm
    python -m src.analysis.partial_correlation --word bank --measure erank
    python -m src.analysis.pooled_layer_level --word bank --column up_proj_erank
    python -m src.analysis.plot_family_separation --word bank
    python -m src.analysis.publication_plots --word bank
    python -m tests.test_pipeline

## 6. Utilities

Regenerate the effective rank summary (needed by `publication_plots`):

    python - << 'EOF'
    import pandas as pd
    from scipy.stats import pearsonr
    from src.analysis._io import ALL_MODELS, merge_geometry_sep, analysis_dir
    from src.utils.paths import infer_family
    WORD = "bank"
    rows = []
    for m in ALL_MODELS:
        mg = merge_geometry_sep(m, WORD)
        if mg is None: continue
        for proj in ["q_proj", "v_proj", "up_proj"]:
            c = f"{proj}_erank"
            if c not in mg.columns: continue
            r, p = pearsonr(mg[c], mg["separation"])
            rows.append({"family": infer_family(m), "model": m, "proj": proj,
                         "erank_sep_r": r, "erank_sep_p": p,
                         "mean_erank": mg[c].mean(), "max_erank": mg[c].max()})
    out = analysis_dir(WORD) / "effective_rank_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)
    EOF

Redirect output to an alternative tree:

    export SEMANTIC_RESULTS_ROOT=$PWD/results_experiment
    unset SEMANTIC_RESULTS_ROOT

Back up results:

    cd /new_raid/nanhangproj/tianyu
    tar czf backups/results_$(date +%Y%m%d).tar.gz -C semantic_thesis_pipeline results
    sha256sum backups/results_$(date +%Y%m%d).tar.gz > backups/results_$(date +%Y%m%d).tar.gz.sha256

## 7. Adding a new word

1. Create the sentence set at `data/raw/semantic_sentences/{word}.json`
   with keys `target_word`, `sentences`, `labels`.
2. Create `configs/words/{word}.yaml` following `bank.yaml`.
3. Run:

    python run.py --all-models --word {word} --skip qwen-7b
    conda activate qwen7_env && python run.py --model qwen-7b --word {word}
    conda activate llm_env
    python -m src.analysis.partial_correlation --word {word} --measure spectral_norm
    python -m src.analysis.partial_correlation --word {word} --measure erank
    python -m src.analysis.pooled_layer_level --word {word} --column up_proj_erank
    python -m src.analysis.plot_family_separation --word {word}

Geometry does not need recomputing.

## 8. Model names

    llama-7b  llama-2-7b  llama-3-8b  llama-3.1-8b
    qwen-7b  qwen1.5-7b  qwen2-7b  qwen2.5-7b  qwen3-8b
    bert-base  roberta-base  spanbert-base-cased  xlm-roberta-base
