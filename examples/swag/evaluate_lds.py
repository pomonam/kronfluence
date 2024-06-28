import logging

import numpy as np
import torch
import tqdm
from scipy.stats import spearmanr

from examples.glue.pipeline import get_glue_dataset
from kronfluence.analyzer import Analyzer


def evaluate_correlations(scores: torch.Tensor) -> float:
    margins = torch.from_numpy(torch.load(open("files/margins.pt", "rb")))
    masks = torch.from_numpy(torch.load(open("files/masks.pt", "rb"))).float()

    val_indices = np.arange(2000)
    preds = masks @ scores.T

    rs = []
    ps = []
    for j in tqdm.tqdm(val_indices):
        r, p = spearmanr(preds[:, j], margins[:, j])
        rs.append(r)
        ps.append(p)
    rs, ps = np.array(rs), np.array(ps)
    return rs.mean()


def main():
    logging.basicConfig(level=logging.INFO)

    # You might need to change the path.
    strategy = "ekfac"
    scores = Analyzer.load_file(f"influence_results/swag/scores_{strategy}_half/pairwise_scores.safetensors")[
        "all_modules"
    ].to(dtype=torch.float32)

    # Reformat the scores.
    split1 = torch.cat([torch.sum(a, dim=0, keepdim=True) for a in scores.split(4)], dim=0)
    split2 = torch.cat([torch.sum(a, dim=1, keepdim=True) for a in split1.split(4, dim=1)], dim=1)

    corr_mean = evaluate_correlations(split2)
    logging.info(f"LDS: {np.mean(corr_mean)}")


if __name__ == "__main__":
    main()
