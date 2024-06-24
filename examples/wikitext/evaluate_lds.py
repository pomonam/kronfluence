import logging

import numpy as np
import torch
from scipy.stats import spearmanr

from kronfluence.analyzer import Analyzer

import tqdm


def evaluate_correlations(scores):
    losses = torch.from_numpy(torch.load(open("files/losses.pt", "rb")))
    masks = torch.from_numpy(torch.load(open("files/masks.pt", "rb"))).float()

    val_indices = np.arange(481)
    preds = -masks @ scores.T

    rs = []
    ps = []
    for j in tqdm.tqdm(val_indices):
        r, p = spearmanr(preds[:, j], losses[:, j])
        rs.append(r)
        ps.append(p)
    rs, ps = np.array(rs), np.array(ps)
    return rs.mean()


def main():
    logging.basicConfig(level=logging.INFO)

    # You might need to change the path.
    # scores = Analyzer.load_file("scores_pairwise/ekfac_pairwise.safetensors")["all_modules"].to(dtype=torch.float64)
    scores = Analyzer.load_file("scores_ekfac_pairwise/pairwise_scores.safetensors")["all_modules"].to(dtype=torch.float32)
    # scores = Analyzer.load_file("scores_ekfac_pairwise_half/pairwise_scores.safetensors")["all_modules"].to(dtype=torch.float32)

    corr_mean = evaluate_correlations(scores)
    logging.info(f"LDS: {np.mean(corr_mean)}")


if __name__ == "__main__":
    main()
