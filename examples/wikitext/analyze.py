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
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Dict[str, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on WikiText dataset.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path that is storing the final checkpoint of the model.",
    )

    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--use_half_precision",
        type=bool,
        default=False,
        help="Whether to use half precision for computing factors and scores.",
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
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
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

        for i in range(12):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")

        for i in range(12):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> Optional[torch.Tensor]:
        return batch["attention_mask"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_wikitext_dataset(
        split="eval_train",
    )
    eval_dataset = get_wikitext_dataset(
        split="valid",
    )

    # Prepare the trained model.
    model = construct_gpt2()
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pth")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path))

    # Define task and prepare model.
    task = LanguageModelingTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="wikitext",
        model=model,
        task=task,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    if args.use_half_precision:
        factor_args.activation_covariance_dtype = torch.bfloat16
        factor_args.gradient_covariance_dtype = torch.bfloat16
        factor_args.lambda_dtype = torch.bfloat16
        factors_name += "_half"

    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=False,
        initial_per_device_batch_size_attempt=128,
    )

    # Compute pairwise scores.
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = ScoreArguments(query_gradient_rank=rank, query_gradient_svd_dtype=torch.float32)
    scores_name = f"{factor_args.strategy}_pairwise"
    if rank is not None:
        scores_name += f"_qlr{rank}"

    if args.use_half_precision:
        score_args.per_sample_gradient_dtype = torch.bfloat16
        score_args.score_dtype = torch.bfloat16
        score_args.cached_activation_cpu_offload = True
        scores_name += "_half"

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        query_indices=list(range(min([len(eval_dataset), 2000]))),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
