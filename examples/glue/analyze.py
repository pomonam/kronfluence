import argparse
import logging
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator

from examples.glue.pipeline import construct_bert, get_glue_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Dict[str, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on GLUE dataset.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sst2",
        help="A name of GLUE dataset.",
    )
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
        "--query_batch_size",
        type=int,
        default=100,
        help="Batch size for computing query gradients.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Batch size for computing training gradients.",
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
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


class TextClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        if not sample:
            return F.cross_entropy(logits, batch["labels"], reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        labels = batch["labels"]
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_attention_mask(self, batch: BATCH_TYPE) -> Optional[torch.Tensor]:
        return batch["attention_mask"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_glue_dataset(
        data_name=args.dataset_name,
        split="eval_train",
    )
    eval_dataset = get_glue_dataset(
        data_name=args.dataset_name,
        split="valid",
    )

    # Prepare the trained model.
    model = construct_bert()
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pth")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path))

    # Define task and prepare model.
    task = TextClassificationTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=args.dataset_name,
        model=model,
        task=task,
        cpu=False,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factor_args = FactorArguments(strategy=args.factor_strategy)
    analyzer.fit_all_factors(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=True,
        initial_per_device_batch_size_attempt=512,
    )
    # Compute pairwise scores.
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = ScoreArguments(query_gradient_rank=rank, query_gradient_svd_dtype=torch.float32)
    scores_name = args.factor_strategy
    if rank is not None:
        scores_name += f"_qlr{rank}"
    analyzer.compute_pairwise_scores(
        score_args=score_args,
        scores_name=scores_name,
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
