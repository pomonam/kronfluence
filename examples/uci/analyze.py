import argparse
import logging
import math
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from analyzer import Analyzer, prepare_model
from arguments import FactorArguments, ScoreArguments
from module.utils import wrap_tracked_modules
from task import Task
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function

from examples.uci.pipeline import construct_regression_mlp, get_regression_dataset

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]


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
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute preconditioning factors.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for compute factors and scores.",
    )
    parser.add_argument(
        "--analysis_name",
        type=str,
        default="uci",
        help="Name of the influence analysis.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path to store the final checkpoint.",
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
        batch: BATCH_DTYPE,
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
        batch: BATCH_DTYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # The measurement function is set as a training loss.
        return self.compute_train_loss(batch, model, sample=False)


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    train_dataset = get_regression_dataset(data_name=args.dataset_name, split="train", data_path=args.dataset_dir)
    eval_dataset = get_regression_dataset(data_name=args.dataset_name, split="valid", data_path=args.dataset_dir)

    model = construct_regression_mlp()

    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pth")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path))

    task = RegressionTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=args.analysis_name,
        model=model,
        task=task,
        cpu=True,
    )
    factor_args = FactorArguments(
        strategy=args.factor_strategy,
        covariance_data_partition_size=5,
        covariance_module_partition_size=4,
    )
    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    #     with record_function("covariance"):
    #         analyzer.fit_covariance_matrices(
    #             factors_name=args.factor_strategy,
    #             dataset=train_dataset,
    #             factor_args=factor_args,
    #             per_device_batch_size=args.batch_size,
    #             overwrite_output_dir=True,
    # )
    #
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # cov_factors = analyzer.fit_covariance_matrices(
    #     factors_name=args.factor_strategy,
    #     dataset=train_dataset,
    #     factor_args=factor_args,
    #     per_device_batch_size=args.batch_size,
    #     overwrite_output_dir=True,
    # )
    # print(cov_factors)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function("eigen"):
            res = analyzer.perform_eigendecomposition(
                factors_name=args.factor_strategy,
                factor_args=factor_args,
                overwrite_output_dir=True,
            )
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # print(res)
    res = analyzer.fit_lambda_matrices(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        # factor_args=factor_args,
        per_device_batch_size=None,
        overwrite_output_dir=True,
    )
    # print(res)
    #
    score_args = ScoreArguments(data_partition_size=2, module_partition_size=2)
    analyzer.compute_pairwise_scores(
        scores_name="hello",
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=16,
        per_device_train_batch_size=8,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    # scores = analyzer.load_pairwise_scores(scores_name="hello")
    # print(scores)
    #
    # analyzer.compute_self_scores(
    #     scores_name="hello",
    #     factors_name=args.factor_strategy,
    #     # query_dataset=eval_dataset,
    #     train_dataset=train_dataset,
    #     # per_device_query_batch_size=16,
    #     per_device_train_batch_size=8,
    #     overwrite_output_dir=True,
    # )
    # # scores = analyzer.load_self_scores(scores_name="hello")
    # # print(scores)

    # analyzer.fit_all_factors(
    #     factor_name=args.factor_strategy,
    #     dataset=train_dataset,
    #     factor_args=factor_args,
    #     per_device_batch_size=None,
    #     overwrite_output_dir=True,
    # )
    #
    # score_name = "full_pairwise"
    # analyzer.compute_pairwise_scores(
    #     score_name=score_name,
    #     query_dataset=eval_dataset,
    #     per_device_query_batch_size=len(eval_dataset),
    #     train_dataset=train_dataset,
    #     per_device_train_batch_size=len(train_dataset),
    # )
    # scores = analyzer.load_pairwise_scores(score_name=score_name)
    # print(scores.shape)


if __name__ == "__main__":
    main()
