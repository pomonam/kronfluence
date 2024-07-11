import argparse
import logging
import os

import torch

from examples.imagenet.analyze import ClassificationTask
from examples.imagenet.pipeline import construct_resnet50, get_imagenet_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import all_low_precision_score_arguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.model import apply_ddp

torch.backends.cudnn.benchmark = True
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on ImageNet dataset.")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="PATH_TO_IMAGENET",
        help="A folder containing the ImageNet dataset.",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute preconditioning factors.",
    )
    parser.add_argument(
        "--query_gradient_rank",
        type=int,
        default=-1,
        help="Rank for the low-rank query gradient approximation.",
    )
    parser.add_argument(
        "--factor_batch_size",
        type=int,
        default=512,
        help="Batch size for computing influence factors.",
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
        help="Batch size for computing training gradient.",
    )
    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        default=False,
        help="Whether to use half precision for computing factors and scores.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Boolean flag to profile computations.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_imagenet_dataset(split="eval_train", dataset_dir=args.dataset_dir)
    eval_dataset = get_imagenet_dataset(split="valid", dataset_dir=args.dataset_dir)

    # Prepare the trained model.
    model = construct_resnet50()
    task = ClassificationTask()

    # Define task and prepare model.
    model = prepare_model(model, task)

    # Apply DDP.
    model = apply_ddp(
        model=model,
        local_rank=LOCAL_RANK,
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
    )

    analyzer = Analyzer(
        analysis_name="imagenet_ddp",
        model=model,
        task=task,
        profile=args.profile,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factors_name = args.factor_strategy
    factor_args = FactorArguments(strategy=args.factor_strategy)
    if args.use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=args.factor_strategy, dtype=torch.bfloat16)
        factors_name += "_half"
    analyzer.fit_all_factors(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        per_device_batch_size=None,
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
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        score_args=score_args,
        factors_name=factors_name,
        query_dataset=eval_dataset,
        query_indices=list(range(1000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
