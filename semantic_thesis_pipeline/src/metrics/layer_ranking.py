"""
Layer ranking by representational stability.

Layers are scored as (normalised mean cosine + normalised mean L2 distance
- normalised drift to the next layer), ranking them by how stable and
well-spread their representations are.

NOTE: this score is unrelated to semantic separation Sep(l). The layers it
selects are not the layers where sense separation peaks, and it should not
be cited as such.
"""

import numpy as np
import pandas as pd


def rank_layers(cosine_csv, l2_csv, drift_csv):
    cos = pd.read_csv(cosine_csv)
    l2 = pd.read_csv(l2_csv)
    drift = pd.read_csv(drift_csv)

    df = cos.copy()
    df["avg_l2"] = l2["avg_l2"]

    drift_vals = list(drift["drift_to_next"])
    drift_vals.append(drift_vals[-1])
    df["drift"] = drift_vals

    df["cos_norm"] = (df["avg_cosine"] - df["avg_cosine"].min()) / (df["avg_cosine"].max() - df["avg_cosine"].min())
    df["l2_norm"] = (df["avg_l2"] - df["avg_l2"].min()) / (df["avg_l2"].max() - df["avg_l2"].min())
    df["drift_norm"] = (df["drift"] - df["drift"].min()) / (df["drift"].max() - df["drift"].min())

    df["score"] = df["cos_norm"] + df["l2_norm"] - df["drift_norm"]

    df = df.sort_values("score", ascending=False)

    return df
