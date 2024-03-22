import argparse
import logging
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data

from examples.cifar.pipeline import construct_resnet9, get_cifar10_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-9 model on CIFAR-10 dataset.")

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
        "--train_batch_size",
        type=int,
        default=512,
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
        default=0.4,
        help="Initial learning rate to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay to train the model.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=25,
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


def train(
    dataset: data.Dataset,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> nn.Module:
    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = construct_resnet9().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    iters_per_epoch = len(train_dataloader)
    lr_peak_epoch = num_train_epochs // 5
    lr_schedule = np.interp(
        np.arange((num_train_epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, num_train_epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    start_time = time.time()
    model.train()
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            model.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach().float()
        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")
    return model


def evaluate(model: nn.Module, dataset: data.Dataset, batch_size: int) -> Tuple[float, float]:
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    total_loss, total_correct = 0.0, 0
    for batch in dataloader:
        with torch.no_grad():
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction="sum")
            total_loss += loss.detach().float()
            total_correct += outputs.detach().argmax(1).eq(labels).sum()

    return total_loss.item() / len(dataloader.dataset), total_correct.item() / len(dataloader.dataset)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = get_cifar10_dataset(
        split="train", corrupt_percentage=args.corrupt_percentage, dataset_dir=args.dataset_dir
    )
    model = train(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    eval_train_dataset = get_cifar10_dataset(split="eval_train", dataset_dir=args.dataset_dir)
    train_loss, train_acc = evaluate(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Train loss: {train_loss}, Train Accuracy: {train_acc}")

    eval_dataset = get_cifar10_dataset(split="valid", dataset_dir=args.dataset_dir)
    eval_loss, eval_acc = evaluate(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Evaluation loss: {eval_loss}, Evaluation Accuracy: {eval_acc}")

    if args.checkpoint_dir is not None:
        model_name = "model"
        if args.corrupt_percentage is not None:
            model_name += "_corrupt_" + str(args.corrupt_percentage)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{model_name}.pth"))


if __name__ == "__main__":
    main()
