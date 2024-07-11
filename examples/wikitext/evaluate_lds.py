import logging

import numpy as np
import torch
import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from examples.wikitext.pipeline import get_wikitext_dataset
from kronfluence.analyzer import Analyzer


def evaluate_correlations(scores: torch.Tensor) -> float:
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
    scores = Analyzer.load_file("influence_results/wikitext/scores_ekfac/pairwise_scores.safetensors")[
        "all_modules"
    ].to(dtype=torch.float32)
    # scores = Analyzer.load_file("influence_results/wikitext/scores_ekfac_half/pairwise_scores.safetensors")[
    #     "all_modules"
    # ].to(dtype=torch.float32)
    # scores = Analyzer.load_file("influence_results/wikitext/scores_identity/pairwise_scores.safetensors")[
    #     "all_modules"
    # ].to(dtype=torch.float32)

    corr_mean = evaluate_correlations(scores)
    logging.info(f"LDS: {np.mean(corr_mean)}")

    # We can also visualize the top influential sequences.
    eval_idx = 0
    train_dataset = get_wikitext_dataset(
        split="eval_train",
    )
    eval_dataset = get_wikitext_dataset(
        split="valid",
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
    print("Query Data Example:")
    print(tokenizer.decode(eval_dataset[eval_idx]["input_ids"]))

    top_idx = int(torch.argsort(scores[eval_idx], descending=True)[0])
    print("Top Influential Example:")
    print(tokenizer.decode(train_dataset[top_idx]["input_ids"]))


if __name__ == "__main__":
    main()
