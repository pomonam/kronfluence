import argparse
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from examples.imagenet.pipeline import construct_resnet50, get_imagenet_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


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


class ClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
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
        inputs, labels = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()


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

    analyzer = Analyzer(
        analysis_name="imagenet",
        model=model,
        task=task,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=4,
    )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

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
        query_indices=list(range(1000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(scores_name)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
