import argparse
import logging
import os

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator

from examples.glue.pipeline import construct_bert, get_glue_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train text classification models on GLUE datasets.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sst",
        help="A folder containing the MNIST dataset.",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="A folder containing the MNIST dataset.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="Batch size for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.03,
        help="Fixed learning rate to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay to train the model.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of epochs to train the model.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1004,
        help="A seed for reproducible training pipeline.",
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

    return args


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.seed is not None:
        set_seed(args.seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_dataset = get_glue_dataset(data_name=args.dataset_name, split="train", data_path=args.dataset_dir)
    # train_dataloader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=args.train_batch_size,
    #     shuffle=True,
    #     collate_fn=default_data_collator,
    #     drop_last=True,
    # )


if __name__ == "__main__":
    main()
