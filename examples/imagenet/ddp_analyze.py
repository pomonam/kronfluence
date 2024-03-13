import argparse
import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.task import Task
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel

from examples.imagenet.pipeline import construct_resnet50, get_imagenet_dataset
from utils.dataset import DataLoaderKwargs

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
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute preconditioning factors.",
    )
    parser.add_argument(
        "--covariance_batch_size",
        type=int,
        default=512,
        help="Batch size for computing covariance matrices.",
    )
    parser.add_argument(
        "--lambda_batch_size",
        type=int,
        default=256,
        help="Batch size for computing Lambda matrices.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=64,
        help="Batch size for computing query gradient.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Batch size for computing training gradient.",
    )

    args = parser.parse_args()
    return args


class ClassificationTask(Task):
    def compute_model_output(self, batch: BATCH_DTYPE, model: nn.Module) -> torch.Tensor:
        inputs, _ = batch
        return model(inputs)

    def compute_train_loss(
        self,
        batch: BATCH_DTYPE,
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
        batch: BATCH_DTYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
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

    train_dataset = get_imagenet_dataset(split="eval_train", dataset_dir=args.dataset_dir)
    eval_dataset = get_imagenet_dataset(split="valid", dataset_dir=args.dataset_dir)

    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    torch.cuda.set_device(LOCAL_RANK)

    model = construct_resnet50()
    task = ClassificationTask()
    model = prepare_model(model, task)
    model = model.to(device=device)
    model = DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    analyzer = Analyzer(
        analysis_name="ddp",
        model=model,
        task=task,
        profile=True,
        disable_model_save=True,
    )
    dataloader_kwargs = DataLoaderKwargs(
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    factor_args = FactorArguments(
        strategy=args.factor_strategy,
    )
    analyzer.fit_covariance_matrices(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=args.covariance_batch_size,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=False,
    )
    analyzer.perform_eigendecomposition(
        factors_name=args.factor_strategy,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )
    analyzer.fit_lambda_matrices(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=args.lambda_batch_size,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=False,
    )
    scores = analyzer.compute_pairwise_scores(
        scores_name="pairwise",
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        train_dataset=train_dataset,
        per_device_train_batch_size=args.train_batch_size,
        per_device_query_batch_size=args.query_batch_size,
        query_indices=list(range(1000)),
        overwrite_output_dir=False,
    )
    logging.info(f"Scores: {scores}")


if __name__ == "__main__":
    main()
