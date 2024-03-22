import argparse
import logging
import math
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from examples.uci.pipeline import construct_regression_mlp, get_regression_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.task import Task

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on UCI datasets.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="concrete",
        help="The name of the UCI regression dataset.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="A folder containing the UCI regression dataset.",
    )
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
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


class RegressionTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = model(inputs)
        if not sample:
            return F.mse_loss(outputs, targets, reduction="sum")
        with torch.no_grad():
            sampled_targets = torch.normal(outputs, std=math.sqrt(0.5))
        return F.mse_loss(outputs, sampled_targets.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # The measurement function is set as a training loss.
        return self.compute_train_loss(batch, model, sample=False)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_regression_dataset(
        data_name=args.dataset_name, split="eval_train", dataset_dir=args.dataset_dir
    )
    eval_dataset = get_regression_dataset(data_name=args.dataset_name, split="valid", dataset_dir=args.dataset_dir)

    # Prepare the trained model.
    model = construct_regression_mlp()
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pth")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path))

    # Define task and prepare model.
    task = RegressionTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=args.dataset_name,
        model=model,
        task=task,
        cpu=True,
    )
    # Compute influence factors.
    factor_args = FactorArguments(strategy=args.factor_strategy)
    analyzer.fit_all_factors(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )
    # Compute pairwise scores.
    analyzer.compute_pairwise_scores(
        scores_name=args.factor_strategy,
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        # Use full batch for computing query gradient.
        per_device_query_batch_size=len(eval_dataset),
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(args.factor_strategy)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
