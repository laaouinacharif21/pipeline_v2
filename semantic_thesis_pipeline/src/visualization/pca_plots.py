import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def generate_pca_plots(embeddings, labels, output_dir):

    num_layers = embeddings.shape[1]

    label_colors = {
        "river": "blue",
        "finance": "red"
    }

    colors = [label_colors[l] for l in labels]

    for layer in range(num_layers):

        layer_emb = embeddings[:, layer, :]

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(layer_emb)

        plt.figure()

        for i in range(len(reduced)):
            plt.scatter(
                reduced[i,0],
                reduced[i,1],
                color=colors[i],
                s=80
            )

        plt.title(f"PCA Layer {layer}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.tight_layout()

        plt.savefig(f"{output_dir}/pca_layer_{layer}.png")
        plt.close()