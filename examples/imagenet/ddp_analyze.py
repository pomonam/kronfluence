import argparse
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel

from examples.imagenet.analyze import ClassificationTask
from examples.imagenet.pipeline import construct_resnet50, get_imagenet_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.dataset import DataLoaderKwargs

torch.backends.cudnn.benchmark = True
BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on ImageNet dataset.")

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mfs1/datasets/imagenet_pytorch/",
        help="A folder containing the ImageNet dataset.",
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
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute preconditioning factors.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_imagenet_dataset(split="eval_train", dataset_dir=args.dataset_dir)
    eval_dataset = get_imagenet_dataset(split="valid", dataset_dir=args.dataset_dir)

    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    torch.cuda.set_device(LOCAL_RANK)

    # Prepare the trained model.
    model = construct_resnet50()
    task = ClassificationTask()

    # Define task and prepare model.
    model = prepare_model(model, task)
    model = model.to(device=device)

    # Apply DDP.
    model = DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    analyzer = Analyzer(
        analysis_name="imagenet_ddp",
        model=model,
        task=task,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factor_args = FactorArguments(
        strategy=args.factor_strategy,
    )
    analyzer.fit_all_factors(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        per_device_batch_size=args.factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )

    # Compute pairwise scores.
    rank = args.query_gradient_rank if args.query_gradient_rank != -1 else None
    score_args = ScoreArguments(query_gradient_rank=rank)
    scores_name = args.factor_strategy
    if rank is not None:
        scores_name += f"_qlr{rank}"
    analyzer.compute_pairwise_scores(
        score_args=score_args,
        scores_name=scores_name,
        factors_name=args.factor_strategy,
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
