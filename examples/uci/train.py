import argparse
import logging
import os
from torch.utils import data
import torch
import torch.nn.functional as F
from torch import nn
from accelerate.utils import set_seed
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


def train(dataset: data.Dataset, batch_size: int, num_train_epochs: int, learning_rate: float, weight_decay: float) -> nn.Module:
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
        total_loss = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, targets = batch
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tepoch.set_postfix(loss=total_loss.item() / len(train_dataloader))
    return model



def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = get_regression_dataset(data_name=args.dataset_name, split="train", data_path=args.dataset_dir)
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
    )
    model = construct_regression_mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    logger.info("Start training the model.")
    model.train()
    for epoch in range(args.num_train_epochs):
        total_loss = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs, targets = batch
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tepoch.set_postfix(loss=total_loss.item() / len(train_dataloader))

    logger.info("Start evaluating the model.")
    model.eval()
    train_eval_dataset = get_regression_dataset(
        data_name=args.dataset_name, split="eval_train", data_path=args.dataset_dir
    )
    train_eval_dataloader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    eval_dataset = get_regression_dataset(data_name=args.dataset_name, split="valid", data_path=args.dataset_dir)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )

    total_loss = 0
    for batch in train_eval_dataloader:
        with torch.no_grad():
            inputs, targets = batch
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets, reduction="sum")
            total_loss += loss.detach().float()
    logger.info(f"Train loss {total_loss.item() / len(train_eval_dataloader.dataset)}")

    total_loss = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            inputs, targets = batch
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets, reduction="sum")
            total_loss += loss.detach().float()
    logger.info(f"Evaluation loss {total_loss.item() / len(eval_dataloader.dataset)}")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))


if __name__ == "__main__":
    main()
