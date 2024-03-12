import argparse
import logging
import math
import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from analyzer import Analyzer, prepare_model
from arguments import FactorArguments
from module.utils import wrap_tracked_modules
from task import Task
from torch import nn
from torch.nn.parallel.distributed import DistributedDataParallel

from examples.imagenet.pipeline import construct_resnet50, get_imagenet_dataset
from examples.mnist.pipeline import construct_mnist_mlp, get_mnist_dataset

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Influence analysis on ImageNet datasets."
    )

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
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for compute factors and scores.",
    )
    parser.add_argument(
        "--analysis_name",
        type=str,
        default="imagenet",
        help="Name of the influence analysis.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path to store the final checkpoint.",
    )

    args = parser.parse_args()
    return args


class ClassificationTask(Task):
    def compute_model_output(
        self, batch: BATCH_DTYPE, model: nn.Module
    ) -> torch.Tensor:
        inputs, _ = batch
        return model(inputs)

    def compute_train_loss(
        self,
        batch: BATCH_DTYPE,
        outputs: torch.Tensor,
        sample: bool = False,
    ) -> torch.Tensor:
        _, labels = batch

        if not sample:
            return F.cross_entropy(outputs, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(outputs, sampled_labels.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_DTYPE,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        _, labels = batch

        bindex = torch.arange(outputs.shape[0]).to(
            device=outputs.device, non_blocking=False
        )
        logits_correct = outputs[bindex, labels]

        cloned_logits = outputs.clone()
        cloned_logits[bindex, labels] = torch.tensor(
            -torch.inf, device=outputs.device, dtype=outputs.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    train_dataset = get_imagenet_dataset(split="eval_train", data_path=args.dataset_dir)
    eval_dataset = get_imagenet_dataset(split="valid", data_path=args.dataset_dir)

    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    device = torch.device("cuda:{}".format(LOCAL_RANK))
    torch.cuda.set_device(LOCAL_RANK)
    print("device")
    print(LOCAL_RANK)

    model = construct_resnet50()

    task = ClassificationTask()
    model = prepare_model(model, task)

    model = model.to(device=device)
    model = DistributedDataParallel(
        model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
    )

    analyzer = Analyzer(
        analysis_name=args.analysis_name,
        model=model,
        task=task,
        disable_model_save=True,
    )

    factor_args = FactorArguments(
        strategy=args.factor_strategy,
        covariance_data_partition_size=1,
        covariance_module_partition_size=1,
    )
    analyzer.fit_covariance_matrices(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=None,
        overwrite_output_dir=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    # analyzer.perform_eigendecomposition(
    #     factor_name=args.factor_strategy,
    #     factor_args=factor_args,
    #     overwrite_output_dir=True,
    # )
    # analyzer.fit_lambda(train_dataset, per_device_batch_size=None)
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
