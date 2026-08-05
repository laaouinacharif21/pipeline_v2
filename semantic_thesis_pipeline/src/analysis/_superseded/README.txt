Superseded analysis scripts, retained for reference only.

  partial_correlation_analysis.py   -> src/analysis/partial_correlation.py
  phase2_pooled_standardised.py     -> src/analysis/pooled_layer_level.py
  phase2_layer_level.py             -> src/analysis/pooled_layer_level.py
  parameter_semantic_correlation.py -> src/analysis/partial_correlation.py
  family_analysis.py                -> src/analysis/partial_correlation.py
  parameter_geometry_family.py      -> src/analysis/compute_geometry.py
  effective_rank_analysis.py        -> src/analysis/compute_geometry.py

These read the pre-August-2026 results layout and do not run against the
current tree. Two known defects motivated their replacement:

  - parameter_geometry_family.py estimated spectral norms by unseeded
    randomised power iteration (non-deterministic, up to 7% error) and did
    not slice the fused c_attn matrix for Qwen-7B.
  - effective_rank_analysis.py cached results unconditionally and mapped
    Qwen-7B's up-projection to mlp.w2, which is the gate.

Both are fixed in compute_geometry.py, which derives spectral norm,
Frobenius norm and effective rank from a single exact SVD.
