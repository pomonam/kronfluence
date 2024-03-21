import logging

import numpy as np
import torch
from scipy.stats import spearmanr

from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    scores = Analyzer.load_file("scores_pairwise/pairwise_scores.safetensors")["all_modules"].to(dtype=torch.float64)

    print("a")


if __name__ == "__main__":
    main()
