import argparse
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator

from examples.wikitext.pipeline import construct_gpt2, get_wikitext_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Dict[str, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on GPT-2 dataset.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sst2",
        help="A name of GLUE dataset.",
    )

    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for computing query gradients.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path to store the final checkpoint.",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute preconditioning factors.",
    )

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    return args


class LanguageModelingTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        shift_logits = logits[..., :-1, :].contiguous()

        if not sample:
            labels = batch["labels"]
            shift_labels = labels[..., 1:].contiguous()
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            summed_loss = F.cross_entropy(reshaped_shift_logits, shift_labels.view(-1), reduction="sum")
        else:
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            with torch.no_grad():
                probs = torch.nn.functional.softmax(reshaped_shift_logits, dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(reshaped_shift_logits, sampled_labels.detach(), reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # We could also compute the log-likelihood or averaged margin.
        return self.compute_train_loss(batch, model)

    def tracked_modules(self) -> List[str]:
        total_modules = []

        for i in range(11):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")

        for i in range(11):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> Optional[torch.Tensor]:
        return batch["attention_mask"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    train_dataset = get_wikitext_dataset(
        split="eval_train",
    )
    eval_dataset = get_wikitext_dataset(
        split="valid",
    )

    model = construct_gpt2()
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pth")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path))

    task = LanguageModelingTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=args.dataset_name,
        model=model,
        task=task,
    )
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    factor_args = FactorArguments(strategy=args.factor_strategy)
    analyzer.fit_all_factors(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=True,
        initial_per_device_batch_size_attempt=512,
    )
    analyzer.compute_pairwise_scores(
        scores_name="pairwise",
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        query_indices=list(range(1000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores("pairwise")
    print(scores)


if __name__ == "__main__":
    main()
