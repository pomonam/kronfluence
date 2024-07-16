import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from examples.openwebtext.pipeline import (
    MODEL_NAME,
    get_custom_dataset,
    get_openwebtext_dataset,
)
from kronfluence.analyzer import Analyzer


def main():
    # scores = Analyzer.load_file("influence_results/openwebtext/scores_raw/pairwise_scores.safetensors")[
    #     "all_modules"
    # ].float()
    scores = Analyzer.load_file("influence_results/scores_raw_margin_scores/pairwise_scores.safetensors")[
        "all_modules"
    ].float()

    train_dataset = get_openwebtext_dataset()
    eval_dataset = get_custom_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)

    eval_idx = 0
    sorted_scores = torch.sort(scores[eval_idx], descending=True)
    top_indices = sorted_scores.indices

    plt.plot(sorted_scores.values)
    plt.grid()
    plt.ylabel("IF Score")
    plt.show()

    print("Query Sequence:")
    print("Prompt:" + eval_dataset[eval_idx]["prompt"] + "; Completion:" + eval_dataset[eval_idx]["completion"] + "\n")

    print("Top Influential Sequences:")
    for i in range(100):
        print("=" * 80)
        print(f"Rank = {i}; Score = {scores[eval_idx][int(top_indices[i])].item()}")
        print(tokenizer.decode(train_dataset[int(top_indices[i])]["input_ids"]))


if __name__ == "__main__":
    main()
