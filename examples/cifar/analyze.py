import argparse
import logging
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from examples.cifar.pipeline import construct_resnet9, get_cifar10_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on CIFAR-10 dataset.")

    parser.add_argument(
        "--corrupt_percentage",
        type=float,
        default=None,
        help="Percentage of the training dataset to corrupt.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="A folder to download or load CIFAR-10 dataset.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path that is storing the final checkpoint of the model.",
    )

    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=1000,
        help="Batch size for computing query gradients.",
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
    train_dataset = get_cifar10_dataset(
        split="eval_train", corrupt_percentage=args.corrupt_percentage, dataset_dir=args.dataset_dir
    )
    eval_dataset = get_cifar10_dataset(split="valid", dataset_dir=args.dataset_dir)

    # Prepare the trained model.
    model = construct_resnet9()
    model_name = "model"
    if args.corrupt_percentage is not None:
        model_name += "_corrupt_" + str(args.corrupt_percentage)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}.pth")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path))

    # Define task and prepare model.
    task = ClassificationTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="cifar10",
        model=model,
        task=task,
    )
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(num_workers=4)
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
    analyzer.compute_pairwise_scores(
        scores_name=args.factor_strategy,
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        query_indices=list(range(2000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        overwrite_output_dir=False,
    )
    scores = analyzer.load_pairwise_scores(args.factor_strategy)["all_modules"]
    logging.info(f"Scores shape: {scores.shape}")


if __name__ == "__main__":
    main()
