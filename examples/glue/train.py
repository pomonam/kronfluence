import argparse
import logging
import os
import time
from typing import Tuple

import evaluate
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from transformers import default_data_collator

from examples.glue.pipeline import construct_bert, get_glue_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train text classification models on GLUE datasets.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sst2",
        help="A name of GLUE dataset.",
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
        default=32,
        help="Batch size for the evaluation dataloader.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-05,
        help="Fixed learning rate to train the model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
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
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
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
        collate_fn=default_data_collator,
    )

    model = construct_bert().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start_time = time.time()
    model.train()
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            loss = model(
                input_ids=batch["input_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
                token_type_ids=batch["token_type_ids"].to(device=DEVICE),
                labels=batch["labels"].to(device=DEVICE),
            ).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().float()
        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")
    return model


def evaluate_model(model: nn.Module, dataset: data.Dataset, batch_size: int) -> Tuple[float, float]:
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=default_data_collator
    )

    model.eval()
    metric = evaluate.load("glue", "sst2")
    total_loss = 0.0
    for batch in dataloader:
        with torch.no_grad():
            logits = model(
                input_ids=batch["input_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
                token_type_ids=batch["token_type_ids"].to(device=DEVICE),
            ).logits
            labels = batch["labels"].to(device=DEVICE)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").detach()
            predictions = logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=labels,
            )
    eval_metric = metric.compute()
    return total_loss.item() / len(dataloader.dataset), eval_metric["accuracy"]


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = get_glue_dataset(data_name=args.dataset_name, split="train")
    model = train(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    eval_train_dataset = get_glue_dataset(data_name=args.dataset_name, split="eval_train")
    train_loss, train_acc = evaluate_model(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Train loss: {train_loss}, Train Accuracy: {train_acc}")

    eval_dataset = get_glue_dataset(data_name=args.dataset_name, split="valid")
    eval_loss, eval_acc = evaluate_model(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Evaluation loss: {eval_loss}, Evaluation Accuracy: {eval_acc}")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))


if __name__ == "__main__":
    main()
