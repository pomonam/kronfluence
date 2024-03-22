import argparse
import logging
import os

import torch

from examples.cifar.analyze import ClassificationTask
from examples.cifar.pipeline import construct_resnet9, get_cifar10_dataset
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Detecting mislabeled CIFAR-10 data points.")

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

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Prepare the dataset.
    train_dataset = get_cifar10_dataset(
        split="eval_train", corrupt_percentage=args.corrupt_percentage, dataset_dir=args.dataset_dir
    )

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
        analysis_name="mislabeled",
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
    # Compute self-influence scores.
    analyzer.compute_self_scores(
        scores_name=args.factor_strategy,
        factors_name=args.factor_strategy,
        train_dataset=train_dataset,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(args.factor_strategy)["all_modules"]

    total_corrupt_size = int(args.corrupt_percentage * len(train_dataset))
    corrupted_indices = list(range(int(args.corrupt_percentage * len(train_dataset))))
    intervals = torch.arange(0.1, 1, 0.1)

    accuracies = []
    for interval in intervals:
        interval = interval.item()
        predicted_indices = torch.argsort(scores, descending=True)[: int(interval * len(train_dataset))]
        predicted_indices = list(predicted_indices.numpy())
        accuracies.append(len(set(predicted_indices) & set(corrupted_indices)) / total_corrupt_size)

    logging.info(f"Inspect Interval: {list(intervals.numpy())}")
    logging.info(f"Detection Accuracy: {accuracies}")


if __name__ == "__main__":
    main()
