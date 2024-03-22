import logging
import math
from random import shuffle
from typing import List

import matplotlib.pyplot as plt
import torch
from tueplots import markers

from examples.wikitext.pipeline import get_wikitext_dataset
from examples.wikitext.train import evaluate_model, train
from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    train_dataset = get_wikitext_dataset(split="train")
    # You might need to change the path.
    identity_scores = Analyzer.load_file("analyses/wikitext/scores_identity_pairwise/pairwise_scores.safetensors")[
        "all_modules"
    ][:50].sum(dim=0)
    ekfac_scores = Analyzer.load_file("analyses/wikitext/scores_ekfac_pairwise/pairwise_scores.safetensors")[
        "all_modules"
    ][:50].sum(dim=0)

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

    ekfac_remove_perp_lst = []
    for topk in topk_lst:
        keep_indices = get_topk_keep_indices(ekfac_scores, topk=topk)

        perp = 0.0
        for _ in range(num_iter):
            new_loss = train_and_evaluate(indices=keep_indices)
            perp += math.exp(new_loss)
        perp /= num_iter
        ekfac_remove_perp_lst.append(perp)

        logging.info(f"Removed {topk} data points. Perplexity: {perp}")
    logging.info(f"EKFAC: {ekfac_remove_perp_lst}")

    id_remove_perp_lst = []
    for topk in topk_lst:
        keep_indices = get_topk_keep_indices(identity_scores, topk=topk)

        perp = 0.0
        for _ in range(num_iter):
            new_loss = train_and_evaluate(indices=keep_indices)
            perp += math.exp(new_loss)
        perp /= num_iter
        id_remove_perp_lst.append(perp)

        logging.info(f"Removed {topk} data points. Perplexity: {perp}")
    logging.info(f"TracIn: {id_remove_perp_lst}")

    random_indices = list(range(4656))
    shuffle(random_indices)
    random_remove_perp_lst = []
    for topk in topk_lst:
        keep_indices = random_indices[topk:]

        perp = 0.0
        for _ in range(num_iter):
            new_loss = train_and_evaluate(indices=keep_indices)
            perp += math.exp(new_loss)
        perp /= num_iter
        random_remove_perp_lst.append(perp)

        logging.info(f"Removed {topk} data points. Perplexity: {perp}")
    logging.info(f"Random: {random_remove_perp_lst}")

    plt.rcParams.update({"figure.dpi": 150})
    plt.rcParams.update(markers.with_edge())
    plt.rcParams["axes.axisbelow"] = True
    plt.plot(topk_lst, [ekfac_remove_perp_lst[0]] + random_remove_perp_lst[1:], "o-", label="Random")
    plt.plot(topk_lst, [ekfac_remove_perp_lst[0]] + id_remove_perp_lst[1:], "o-", label="TracIn (Identity)")
    plt.plot(topk_lst, ekfac_remove_perp_lst, "o-", label="IF (EKFAC)")
    plt.grid()
    plt.legend()
    plt.xlabel("Number of Training Samples Removed")
    plt.ylabel("Mean Query Perplexity")
    plt.show()


if __name__ == "__main__":
    main()
