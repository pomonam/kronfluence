import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tueplots import markers

from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    # Load the scores. You might need to modify the path.
    full_scores = Analyzer.load_file("influence_results/wikitext/scores_ekfac/pairwise_scores.safetensors")[
        "all_modules"
    ]
    half_scores = Analyzer.load_file("influence_results/wikitext/scores_ekfac_half/pairwise_scores.safetensors")[
        "all_modules"
    ].float()

    # Only plot first 1000 points to avoid clutter.
    index = 5
    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update(markers.with_edge())
    plt.rcParams["axes.axisbelow"] = True
    plt.scatter(half_scores[index], full_scores[index], edgecolor="k")
    plt.grid()
    plt.xlabel("bfloat32 Score")
    plt.ylabel("float32 Score")
    plt.show()

    # Compute the averaged spearman correlation.
    all_corr = []
    for i in range(481):
        all_corr.append(spearmanr(full_scores[i], half_scores[i])[0])
    logging.info(f"Averaged Spearman Correlation: {np.array(all_corr).mean()}")


if __name__ == "__main__":
    main()
