import argparse
import logging
import os
import math
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from tqdm import tqdm
from transformers import default_data_collator

from examples.wikitext.pipeline import construct_gpt2, get_wikitext_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train text classification models on GLUE datasets.")

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
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
        collate_fn=default_data_collator,
    )

    model = construct_gpt2().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.eval()
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        with tqdm(train_dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                model.zero_grad()
                lm_logits = model(
                    input_ids=batch["input_ids"].to(device=DEVICE),
                    attention_mask=batch["attention_mask"].to(device=DEVICE),
                ).logits
                labels = batch["labels"].to(device=DEVICE)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=total_loss.item() / len(train_dataloader))
    return model


def evaluate_model(model: nn.Module, dataset: data.Dataset, batch_size: int) -> float:
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=default_data_collator
    )

    model.eval()
    total_loss = 0.0
    total_num = 0
    for batch in dataloader:
        with torch.no_grad():
            lm_logits = model(
                input_ids=batch["input_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
            ).logits
            labels = batch["labels"].to(device=DEVICE)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            loss = (
                F.cross_entropy(reshaped_shift_logits, shift_labels.view(-1), reduction="sum").detach().float()
            )
            total_loss += loss
            total_num += reshaped_shift_logits.shape[0]
    return total_loss.item() / total_num


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = get_wikitext_dataset(split="train")
    model = train(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    eval_train_dataset = get_wikitext_dataset(split="eval_train")
    train_loss = evaluate_model(model=model, dataset=eval_train_dataset, batch_size=args.eval_batch_size)
    train_perplexity = math.exp(train_loss)
    logger.info(f"Train perplexity: {train_perplexity}")

    eval_dataset = get_wikitext_dataset(split="valid")
    eval_loss = evaluate_model(model=model, dataset=eval_dataset, batch_size=args.eval_batch_size)
    eval_perplexity = math.exp(eval_loss)
    logger.info(f"Evaluation perplexity: {eval_perplexity}")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))


if __name__ == "__main__":
    main()
