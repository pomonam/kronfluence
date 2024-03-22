import logging
import math
from typing import List

import torch

from examples.wikitext.pipeline import get_wikitext_dataset
from examples.wikitext.train import evaluate_model, train
from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    train_dataset = get_wikitext_dataset(split="train")
    # You might need to change the path.
    scores = Analyzer.load_file("analyses/wikitext/scores_ekfac_pairwise/pairwise_scores.safetensors")["all_modules"][
        :50
    ].sum(dim=0)
    # scores = Analyzer.load_file("scores_pairwise/pairwise_scores.safetensors")["all_modules"][:5].sum(dim=0)

    def get_topk_indices(current_score: torch.Tensor, topk: int = 1) -> torch.Tensor:
        return torch.topk(current_score, topk).indices

    def get_topk_keep_indices(current_score: torch.Tensor, topk: int = 1) -> List[int]:
        remove_indices = get_topk_indices(current_score, topk)
        remove_indices = [tensor.item() for tensor in remove_indices]
        return list(set(list(range(len(train_dataset)))) - set(remove_indices))

    eval_train_dataset = get_wikitext_dataset(split="valid", indices=list(range(50)))

    def train_and_evaluate(indices):
        train_dataset = get_wikitext_dataset(split="train", indices=indices)
        model = train(
            dataset=train_dataset,
            batch_size=8,
            num_train_epochs=3,
            learning_rate=3e-05,
            weight_decay=0.01,
        )
        return evaluate_model(model, eval_train_dataset, batch_size=16)

    num_iter = 1
    topk_lst = [0, 50, 100, 150, 200]
    remove_perp_lst = []

    for topk in topk_lst:
        keep_indices = get_topk_keep_indices(scores, topk=topk)

        perp = 0.0
        for _ in range(num_iter):
            new_loss = train_and_evaluate(indices=keep_indices)
            perp += math.exp(new_loss)
        perp /= num_iter
        remove_perp_lst.append(perp)

        logging.info(f"Removed {topk} data points. Perplexity: {perp}")

    logging.info(remove_perp_lst)


if __name__ == "__main__":
    main()
