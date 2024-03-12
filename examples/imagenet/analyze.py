import argparse
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from analyzer import Analyzer, prepare_model
from arguments import FactorArguments
from task import Task
from torch import nn

from examples.imagenet.pipeline import construct_resnet50, get_imagenet_dataset

BATCH_DTYPE = Tuple[torch.Tensor, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on ImageNet datasets.")

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
        default=512,
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
    def compute_model_output(self, batch: BATCH_DTYPE, model: nn.Module) -> torch.Tensor:
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

        bindex = torch.arange(outputs.shape[0]).to(device=outputs.device, non_blocking=False)
        logits_correct = outputs[bindex, labels]

        cloned_logits = outputs.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=outputs.device, dtype=outputs.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    train_dataset = get_imagenet_dataset(split="eval_train", data_path=args.dataset_dir)
    # eval_dataset = get_imagenet_dataset(split="valid", data_path=args.dataset_dir)

    model = construct_resnet50()

    task = ClassificationTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=args.analysis_name,
        model=model,
        task=task,
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
        per_device_batch_size=1024,
        overwrite_output_dir=True,
    )


if __name__ == "__main__":
    main()
