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

    val_indices = np.arange(277)
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
    scores = Analyzer.load_file(f"influence_results/rte/scores_{strategy}/pairwise_scores.safetensors")[
        "all_modules"
    ].to(dtype=torch.float32)

    corr_mean = evaluate_correlations(scores)
    logging.info(f"LDS: {np.mean(corr_mean)}")

    # We can also visualize the top influential sequences.
    eval_idx = 79
    train_dataset = get_glue_dataset(
        data_name="rte",
        split="eval_train",
    )
    eval_dataset = get_glue_dataset(
        data_name="rte",
        split="valid",
    )
    print("Query Data Example:")
    print(f"Sentence1: {eval_dataset[eval_idx]['sentence1']}")
    print(f"Sentence2: {eval_dataset[eval_idx]['sentence2']}")
    print(f"Label: {eval_dataset[eval_idx]['label']}")

    top_idx = int(torch.argsort(scores[eval_idx], descending=True)[0])
    print("Top Influential Example:")
    print(f"Sentence1: {train_dataset[top_idx]['sentence1']}")
    print(f"Sentence2: {train_dataset[top_idx]['sentence2']}")
    print(f"Label: {train_dataset[top_idx]['label']}")


if __name__ == "__main__":
    main()
