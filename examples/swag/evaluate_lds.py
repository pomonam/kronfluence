import logging

import numpy as np
import torch
import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from examples.swag.pipeline import get_swag_dataset
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
    scores = Analyzer.load_file(f"influence_results/swag/scores_{strategy}/pairwise_scores.safetensors")[
        "all_modules"
    ].to(dtype=torch.float32)

    corr_mean = evaluate_correlations(scores)
    logging.info(f"LDS: {np.mean(corr_mean)}")

    # We can also visualize the top influential sequences.
    eval_idx = 1004
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", use_fast=True, trust_remote_code=True)

    train_dataset = get_swag_dataset(
        split="eval_train",
    )
    eval_dataset = get_swag_dataset(
        split="valid",
    )

    print("Query Data Example:")
    for i in range(4):
        text = tokenizer.decode(eval_dataset[eval_idx]["input_ids"][i])
        print(f"Option {i}: {text}")
    print(f"Label: {eval_dataset[eval_idx]['labels']}")

    top_idx = int(torch.argsort(scores[eval_idx], descending=True)[0])
    print("Top Influential Example:")
    for i in range(4):
        text = tokenizer.decode(train_dataset[top_idx]["input_ids"][i])
        print(f"Option {i}: {text}")
    print(f"Label: {train_dataset[top_idx]['labels']}")


if __name__ == "__main__":
    main()
