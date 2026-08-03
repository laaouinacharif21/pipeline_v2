import numpy as np


def safe_l2_distance(a, b):
    diff = a - b
    diff = diff.astype(np.float32)  # ?? force stability
    return np.sqrt(np.sum(diff * diff))


def compute_l2_metrics(embeddings):
    """
    embeddings: [num_sentences, num_layers, hidden_size]
    """

    num_sent, num_layers, dim = embeddings.shape

    avg_l2 = []
    l2_matrices = []

    for l in range(num_layers):
        layer_emb = embeddings[:, l, :].astype(np.float32)

        # ?? normalize AGAIN (defensive)
        norm = np.linalg.norm(layer_emb, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        layer_emb = layer_emb / norm

        mat = np.zeros((num_sent, num_sent), dtype=np.float32)

        for i in range(num_sent):
            for j in range(num_sent):
                mat[i, j] = safe_l2_distance(layer_emb[i], layer_emb[j])

        l2_matrices.append(mat)

        mask = ~np.eye(num_sent, dtype=bool)
        avg_l2.append(float(mat[mask].mean()))

    return np.array(avg_l2), l2_matrices