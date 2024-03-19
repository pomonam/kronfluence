import argparse
import logging
import os

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from tqdm import tqdm

from examples.uci.pipeline import construct_regression_mlp, get_regression_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train regression models on UCI datasets.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="concrete",
        help="The name of the UCI regression dataset.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="A folder containing the UCI regression dataset.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
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
        default=1e-5,
        help="Weight decay to train the model.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=40,
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
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def train(
    dataset: data.Dataset,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    disable_tqdm: bool = False,
) -> nn.Module:
    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    model = construct_regression_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        with tqdm(train_dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                model.zero_grad()
                inputs, targets = batch
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=total_loss.item() / len(train_dataloader))
    return model


def evaluate(model: nn.Module, dataset: data.Dataset, batch_size: int) -> float:
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        with torch.no_grad():
            inputs, targets = batch
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets, reduction="sum")
            total_loss += loss.detach().float()

    return total_loss.item() / len(dataloader.dataset)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = get_regression_dataset(data_name=args.dataset_name, split="train", dataset_dir=args.dataset_dir)
    model = train(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    eval_train_dataset = get_regression_dataset(
        data_name=args.dataset_name, split="eval_train", dataset_dir=args.dataset_dir
    )
    train_loss = evaluate(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Train loss: {train_loss}")

    eval_dataset = get_regression_dataset(data_name=args.dataset_name, split="valid", dataset_dir=args.dataset_dir)
    eval_loss = evaluate(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Evaluation loss: {eval_loss}")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))


if __name__ == "__main__":
    main()
