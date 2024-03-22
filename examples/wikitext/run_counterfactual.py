import logging
from typing import List

import numpy as np
import torch
from scipy.stats import spearmanr

from examples.wikitext.pipeline import get_wikitext_dataset
from kronfluence.analyzer import Analyzer
from examples.wikitext.train import train, evaluate_model




def main():
    logging.basicConfig(level=logging.INFO)

    train_dataset = get_wikitext_dataset(split="train")
    scores = Analyzer.load_file("scores_pairwise/pairwise_scores.safetensors")["all_modules"][0]

    def get_topk_indices(current_score: torch.Tensor, topk: int = 1) -> torch.Tensor:
        return torch.topk(current_score, topk).indices

    def get_topk_keep_indices(current_score: torch.Tensor, topk: int = 1) -> List[int]:
        remove_indices = get_topk_indices(current_score, topk)
        remove_indices = [tensor.item() for tensor in remove_indices]
        return list(set(list(range(len(train_dataset)))) - set(remove_indices))

    train_dataset = get_wikitext_dataset(split="train")
    eval_train_dataset = get_wikitext_dataset(split="eval_train", indices=[0])


    def train_fnc(indices):
        train_dataset = get_wikitext_dataset(split="train")
        model = train(
            dataset=train_dataset,
            batch_size=8,
            num_train_epochs=3,
            learning_rate=3e-05,
            weight_decay=0.01,
        )
        # return eva

    keep_indices = get_topk_keep_indices(scores, topk=5)



if __name__ == "__main__":
    main()
