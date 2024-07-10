import argparse
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from transformers import default_data_collator

from examples.openwebtext.pipeline import (
    construct_llama3,
    get_custom_dataset,
    get_openwebtext_dataset,
)
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import (
    extreme_reduce_memory_factor_arguments,
)
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Dict[str, torch.Tensor]

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on Openwebtext dataset.")

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute influence factors.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=4,
        help="Batch size for computing influence factors.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

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
        ).logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))

        if not sample:
            labels = batch["labels"]
            shift_labels = labels[..., 1:].contiguous()
            summed_loss = F.cross_entropy(
                reshaped_shift_logits, shift_labels.view(-1), reduction="sum", ignore_index=-100
            )
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(reshaped_shift_logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(reshaped_shift_logits, sampled_labels, ignore_index=-100, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="sum")

    def tracked_modules(self) -> List[str]:
        total_modules = []

        for i in range(32):
            total_modules.append(f"model.layers.{i}.mlp.gate_proj")
            total_modules.append(f"model.layers.{i}.mlp.up_proj")
            total_modules.append(f"model.layers.{i}.mlp.down_proj")

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> Optional[torch.Tensor]:
        return batch["attention_mask"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_openwebtext_dataset()
    eval_dataset = get_custom_dataset()

    # Prepare the trained model.
    model = construct_llama3()
    print(list(model.parameters())[0])
    print(list(model.parameters())[10])

    # Define task and prepare model.
    task = LanguageModelingTask()
    model = prepare_model(model, task)

    accelerator = Accelerator()
    model = accelerator.prepare_model(model)

    analyzer = Analyzer(
        analysis_name="openwebtext",
        model=model,
        task=task,
        profile=args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factors_name = args.factor_strategy
    factor_args = extreme_reduce_memory_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
    factor_args.covariance_max_examples = 4
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )

    # Compute pairwise scores.
    scores_name = factor_args.strategy
    score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
    score_args.num_query_gradient_accumulations = 10
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    if rank is not None:
        score_args.query_gradient_rank = rank
        scores_name += f"_qlr{rank}"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
