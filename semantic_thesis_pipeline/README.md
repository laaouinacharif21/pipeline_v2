# Geometric Features and Semantic Associations in the Parameter Space of Large Language Models

Layer-wise analysis of the association between transformer projection-matrix
geometry (spectral norm, effective rank) and semantic separation of word
senses, across 13 pretrained models spanning three families.

## Layout

    configs/words/{word}.yaml              dataset config per target word
    data/raw/semantic_sentences/           sentence sets
    src/
      config.py                            word config loading, dataset hashing
      models/hf_loader.py                  model and tokenizer loading
      extraction/target_token_extractor.py first-subword location + gates
      pipeline/                            extract -> metrics -> select -> plots
      metrics/                             cosine, L2, drift, separation
      analysis/                            cross-model analysis and figures
      utils/paths.py                       canonical path resolution
    tests/test_pipeline.py                 integrity checks
    run.py                                 pipeline entry point
    results/
      parameters/{family}/{model}/         geometry (word-independent)
      words/{word}/{family}/{model}/       per-word results
      analysis/{word}/                     correlations and figures

## Environments

| Models | Environment | transformers |
|---|---|---|
| 12 models | `llm_env` | 5.4.0 |
| qwen-7b | `qwen7_env` | 4.32.0 |

Qwen-7B ships custom remote code written against transformers 4.x and does
not load on 5.x. All other models run under a single environment.

## Reproducing

    conda activate llm_env

    # 1. Parameter geometry (once per model, independent of the dataset)
    python -m src.analysis.compute_geometry --all-models --skip qwen-7b
    conda activate qwen7_env
    python -m src.analysis.compute_geometry --model qwen-7b
    conda activate llm_env

    # 2. Extraction, metrics, layer selection, plots (per word)
    python run.py --all-models --word bank --skip qwen-7b
    conda activate qwen7_env
    python run.py --model qwen-7b --word bank
    conda activate llm_env

    # 3. Analysis
    python -m src.analysis.partial_correlation    --word bank --measure spectral_norm
    python -m src.analysis.partial_correlation    --word bank --measure erank
    python -m src.analysis.pooled_layer_level     --word bank --column up_proj_erank
    python -m src.analysis.plot_family_separation --word bank
    python -m src.analysis.publication_plots      --word bank

    # 4. Integrity checks
    python -m tests.test_pipeline

## Method

**Semantic separation.** For a target word with two senses, Sep(l) is the
mean intra-class cosine similarity minus the mean inter-class cosine
similarity of the target token's hidden state at layer l, excluding
self-pairs.

**Extraction.** The hidden state is taken at the first subword token of the
target word. Token location is tokenizer-agnostic: cumulative prefixes are
decoded until the reconstructed text reaches the target's character offset,
which works for fast and slow tokenizers alike.

Two validation gates guard extraction and fail the run by default:

- *Position.* Target tokens below index 3 are rejected. Decoder models place
  very large activations on the first token positions, so a target token
  there yields a hidden state dominated by position rather than meaning.
- *Norm.* Any representation whose norm exceeds ten times the median is
  treated as anomalous.

The "bank" dataset carries a neutral prefix ("Consider the following
sentence: ") so that no target token falls in the attention-sink region.

**Geometry.** Spectral norm, Frobenius norm and effective rank are derived
from a single exact SVD per matrix, making them deterministic and mutually
consistent. Effective rank follows Roy & Vetterli (2007):
erank(A) = exp(H(p)) with p_i = s_i / sum(s).

Qwen-7B stores Q, K and V fused in one c_attn matrix and is sliced into
thirds. Its MLP computes w1(x) * silu(w2(x)), so w1 is the up-projection
and w2 the gate, which is the reverse of the attribute naming.

**Statistics.** Layer depth co-varies with both geometry and separation, so
the depth-controlled partial correlation is the primary statistic. Pooled
analyses standardise both variables within each model before pooling, since
raw pooling mixes between-model scale differences with the within-model
layer relationship. Spearman is reported alongside Pearson as a
monotonicity check.

All p-values are exploratory. Layers within a model are not independent
observations and no multiple-comparison correction is applied.

## Provenance

Every extraction run writes run_manifest.json (dataset SHA-256, library
versions, host, validation report). Every geometry run writes
geometry_manifest.json (layer container, projection shapes, method,
versions). results_midterm_meanpooled/ and results_targettoken_noprefix/
retain earlier configurations for comparison; see their README files.

Superseded scripts are in src/analysis/_superseded/ with a note on why each
was replaced.
