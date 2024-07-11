import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tueplots import markers

from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    # Load the scores. You might need to modify the path.
    name = "ekfac_half"
    factor = (
        Analyzer.load_file(f"influence_results/cifar10/factors_{name}/gradient_covariance.safetensors")
    )
    print(factor)

    scores = (
        Analyzer.load_file(f"influence_results/cifar10/scores_{name}/pairwise_scores.safetensors")
    )
    print(scores)


if __name__ == "__main__":
    main()
