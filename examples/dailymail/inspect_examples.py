import logging

import torch

from examples.dailymail.pipeline import get_dailymail_dataset, get_tokenizer
from kronfluence.analyzer import Analyzer


def main():
    logging.basicConfig(level=logging.INFO)

    # You might need to change the path.
    strategy = "ekfac"
    scores = Analyzer.load_file(f"influence_results/dailymail/scores_{strategy}_half/pairwise_scores.safetensors")[
        "all_modules"
    ].to(dtype=torch.float32)

    eval_idx = 1
    train_dataset = get_dailymail_dataset(
        split="eval_train",
    )
    eval_dataset = get_dailymail_dataset(
        split="valid",
    )
    tokenizer = get_tokenizer()
    print("Query Data Example:")
    print(f"Input: {tokenizer.decode(eval_dataset[eval_idx]['input_ids'])}")
    print(f"Label: {tokenizer.decode(eval_dataset[eval_idx]['labels'])}")

    top_idx = int(torch.argsort(scores[eval_idx], descending=True)[0])
    print("Top Influential Example:")
    print(f"Input: {tokenizer.decode(train_dataset[top_idx]['input_ids'])}")
    print(f"Label: {tokenizer.decode(train_dataset[top_idx]['labels'])}")


if __name__ == "__main__":
    main()
