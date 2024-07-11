import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tueplots import markers

from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    scores1 = Analyzer.load_file("influence_results/swag/scores_ekfac/pairwise_scores.safetensors")[
        "all_modules"
    ].float()
    scores2 = Analyzer.load_file("influence_results/swag/scores_ekfac_half_qlr32/pairwise_scores.safetensors")[
        "all_modules"
    ].float()

    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update(markers.with_edge())
    plt.rcParams["axes.axisbelow"] = True

    # Only plot first 6000 points to avoid clutter.
    idx = 1
    plt.scatter(scores1[idx], scores2[idx], edgecolor="k")
    plt.grid()
    plt.xlabel("score1")
    plt.ylabel("score2")
    plt.show()

    # Compute the averaged spearman correlation.
    all_corr = []
    for i in range(500):
        all_corr.append(spearmanr(scores1[i], scores2[i])[0])
    logging.info(f"Averaged Spearman Correlation: {np.array(all_corr).mean()}")


if __name__ == "__main__":
    main()
