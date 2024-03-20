import argparse
import logging
import os
from typing import Tuple

import torch
from arguments import FactorArguments
from examples.cifar.analyze import ClassificationTask
from examples.cifar.pipeline import construct_resnet9, get_cifar10_dataset
from kronfluence.analyzer import Analyzer, prepare_model



def parse_args():
    parser = argparse.ArgumentParser(description="Influence analysis on UCI datasets.")

    parser.add_argument(
        "--corrupt_percentage",
        type=float,
        default=0.1,
        help="Percentage of the training dataset to corrupt.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="A folder to download or load CIFAR-10 dataset.",
    )

    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=1000,
        help="Batch size for computing query gradients.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="A path to store the final checkpoint.",
    )

    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        help="Strategy to compute preconditioning factors.",
    )

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    train_dataset = get_cifar10_dataset(
        split="eval_train", corrupt_percentage=args.corrupt_percentage, dataset_dir=args.dataset_dir
    )
    eval_dataset = get_cifar10_dataset(split="valid", dataset_dir=args.dataset_dir)

    model = construct_resnet9()
    model_name = "model"
    if args.corrupt_percentage is not None:
        model_name += "_corrupt_" + str(args.corrupt_percentage)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_name}.pth")
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"No checkpoint found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path))

    task = ClassificationTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="cifar10",
        model=model,
        task=task,
        cpu=False,
    )

    factor_args = FactorArguments(strategy=args.factor_strategy)
    analyzer.fit_all_factors(
        factors_name=args.factor_strategy,
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    analyzer.compute_pairwise_scores(
        scores_name="pairwise",
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        query_indices=list(range(2000)),
        train_dataset=train_dataset,
        per_device_query_batch_size=args.query_batch_size,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores("pairwise")
    print(scores)


if __name__ == "__main__":
    main()
