import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from tueplots import markers
import torch

from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    results = torch.load("lds_results.pt")
    diff_loss = torch.from_numpy(results["diff_loss"])
    mask = torch.from_numpy(results["mask"]).float()
    mask = ((mask + 1) % 2).to(dtype=torch.float32)
    print(results)

    scores = Analyzer.load_file("scores_pairwise/scores_pairwise/pairwise_scores.safetensors")["all_modules"]
    print(scores)

    preds = mask @ scores.T

    corr_lst = []

    for i in range(preds.shape[1]):
        corr_lst.append(spearmanr(diff_loss[:, i], preds[:, i])[0])
    print(np.mean(corr_lst))


if __name__ == "__main__":
    main()
