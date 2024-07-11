import argparse
import logging
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DataCollatorForSeq2Seq

from examples.dailymail.pipeline import (
    construct_t5,
    get_dailymail_dataset,
    get_tokenizer,
)
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.model import apply_ddp

BATCH_TYPE = Dict[str, torch.Tensor]
try:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
except KeyError:
    LOCAL_RANK = WORLD_RANK = WORLD_SIZE = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on CNN/DailyMail dataset.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path that is storing the final checkpoint of the model.",
    )

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
        "--use_ddp",
        action="store_true",
        default=False,
        help="Whether to use DDP for computing factors and scores.",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=64,
        help="Batch size for computing influence factors.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=10,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Batch size for computing training gradients.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


class SummarizationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
        ).logits

        if not sample:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), batch["labels"].view(-1), ignore_index=-100, reduction="sum"
            )
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.view(-1, logits.size(-1)).detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
            masks = batch["labels"].view(-1) == -100
            sampled_labels[masks] = -100
        return F.cross_entropy(logits.view(-1, logits.size(-1)), sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
        ).logits
        logits = logits.view(-1, logits.size(-1))

        labels = batch["labels"].view(-1)
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        masks = batch["labels"].view(-1) != -100
        return -margins[masks].sum()

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []

        # Add attention layers:
        for i in range(6):
            total_modules.append(f"encoder.block.{i}.layer.0.SelfAttention.q")
            total_modules.append(f"encoder.block.{i}.layer.0.SelfAttention.k")
            total_modules.append(f"encoder.block.{i}.layer.0.SelfAttention.v")
            total_modules.append(f"encoder.block.{i}.layer.0.SelfAttention.o")

            total_modules.append(f"decoder.block.{i}.layer.0.SelfAttention.q")
            total_modules.append(f"decoder.block.{i}.layer.0.SelfAttention.k")
            total_modules.append(f"decoder.block.{i}.layer.0.SelfAttention.v")
            total_modules.append(f"decoder.block.{i}.layer.0.SelfAttention.o")

            total_modules.append(f"decoder.block.{i}.layer.1.EncDecAttention.q")
            total_modules.append(f"decoder.block.{i}.layer.1.EncDecAttention.k")
            total_modules.append(f"decoder.block.{i}.layer.1.EncDecAttention.v")
            total_modules.append(f"decoder.block.{i}.layer.1.EncDecAttention.o")

        # Add MLP layers:
        for i in range(6):
            total_modules.append(f"encoder.block.{i}.layer.1.DenseReluDense.wi")
            total_modules.append(f"encoder.block.{i}.layer.1.DenseReluDense.wo")

            total_modules.append(f"decoder.block.{i}.layer.2.DenseReluDense.wi")
            total_modules.append(f"decoder.block.{i}.layer.2.DenseReluDense.wo")

        return total_modules

    def get_attention_mask(self, batch: BATCH_TYPE) -> torch.Tensor:
        return batch["attention_mask"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_dailymail_dataset(
        split="eval_train",
    )
    eval_dataset = get_dailymail_dataset(
        split="valid",
    )
    tokenizer = get_tokenizer()

    # Prepare the trained model.
    model = construct_t5()

    # Define task and prepare model.
    task = SummarizationTask()
    model = prepare_model(model, task)

    if args.use_ddp:
        model = apply_ddp(
            model=model,
            local_rank=LOCAL_RANK,
            rank=WORLD_RANK,
            world_size=WORLD_SIZE,
        )

    analyzer = Analyzer(
        analysis_name="dailymail",
        model=model,
        task=task,
        profile=args.profile,
    )
    # Configure parameters for DataLoader.
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    dataloader_kwargs = DataLoaderKwargs(num_workers=4, collate_fn=data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"
    if args.use_ddp:
        factors_name += "_ddp"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )

    # Compute pairwise scores.
    score_args = ScoreArguments()
    scores_name = factor_args.strategy
    if args.use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16)
        scores_name += "_half"
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    if rank is not None:
        score_args.query_gradient_low_rank = rank
        score_args.query_gradient_accumulation_steps = 10
        scores_name += f"_qlr{rank}"
    if args.use_ddp:
        scores_name += "_ddp"
    analyzer.compute_pairwise_scores(
        score_args=score_args,
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=eval_dataset,
        query_indices=list(range(10)),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
