import logging
import math

import torch
from colorama import Fore, init
from transformers import AutoTokenizer

from examples.wikitext.pipeline import get_wikitext_dataset
from kronfluence.analyzer import Analyzer

init()


def color_strength(word: str, strength: int) -> None:
    strength = max(0, min(1, strength))
    intensity = math.floor(strength * 255)
    color = f"\033[38;2;{intensity};0;0m"
    print(f"{color}{word}{Fore.RESET}", end="")


def main():
    logging.basicConfig(level=logging.INFO)

    # You might need to change the path.
    scores = Analyzer.load_file("influence_results/wikitext/scores_ekfac_half_per_token/pairwise_scores.safetensors")[
        "all_modules"
    ].to(dtype=torch.float32)
    summed_scores = scores.sum(dim=-1)

    # We can also visualize the top influential sequences.
    eval_idx = 5
    train_dataset = get_wikitext_dataset(
        split="eval_train",
    )
    eval_dataset = get_wikitext_dataset(
        split="valid",
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, trust_remote_code=True)
    print("Query Data Example:")
    print(tokenizer.decode(eval_dataset[eval_idx]["input_ids"]))

    top_idx = int(torch.argsort(summed_scores[eval_idx], descending=True)[0])
    tokens = scores[eval_idx][top_idx]

    print("Top Influential Example:")
    words = tokenizer.batch_decode(train_dataset[top_idx]["input_ids"])
    strengths = torch.abs(tokens) / torch.abs(tokens).max()
    strengths = (strengths**0.3).tolist()
    for word, strength in zip(words, strengths):
        color_strength(word, strength)


if __name__ == "__main__":
    main()
