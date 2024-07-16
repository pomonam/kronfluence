import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tueplots import markers

from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    # Load the scores. You might need to modify the path.
    scores = (
        Analyzer.load_file("influence_results/cifar10/scores_ekfac/pairwise_scores.safetensors")["all_modules"] / 50_000
    )
    half_scores = (
        Analyzer.load_file("influence_results/cifar10/scores_ekfac_half/pairwise_scores.safetensors")[
            "all_modules"
        ].float()
        / 50_000
    )

    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update(markers.with_edge())
    plt.rcParams["axes.axisbelow"] = True

    # Only plot first 3000 points to avoid clutter.
    idx = 0
    plt.scatter(half_scores[idx][:3000], scores[idx][:3000], edgecolor="k")
    plt.grid()
    plt.xlabel("bfloat16")
    plt.ylabel("float32")
    plt.show()

    # Compute the averaged spearman correlation.
    all_corr = []
    for i in range(2000):
        all_corr.append(spearmanr(scores[i], half_scores[i])[0])
    logging.info(f"Averaged Spearman Correlation: {np.array(all_corr).mean()}")
    logging.info(f"Lowest Spearman Correlation: {np.array(all_corr).min()}")
    logging.info(f"Highest Spearman Correlation: {np.array(all_corr).max()}")


if __name__ == "__main__":
    main()
