import argparse
import logging
import os
import time
from typing import Any, Dict

import evaluate
import nltk
import numpy as np
import torch
import torch.nn.functional as F
from accelerate.utils import send_to_device, set_seed
from filelock import FileLock
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils import data
from transformers import DataCollatorForSeq2Seq

from examples.dailymail.pipeline import (
    construct_t5,
    get_dailymail_dataset,
    get_tokenizer,
)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train seq2seq models on CNN/DailyMail dataset.")

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
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
        default=5e-05,
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
    tokenizer: Any,
    batch_size: int,
    num_train_epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> nn.Module:
    model = construct_t5().to(DEVICE)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )
    train_dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start_time = time.time()
    model.train()
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch = send_to_device(batch, device=DEVICE)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().float()
        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")
    return model


def evaluate_model(model: nn.Module, tokenizer: Any, dataset: data.Dataset, batch_size: int) -> Dict[str, Any]:
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=data_collator
    )
    model.eval()

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence.
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    gen_kwargs = {
        "max_length": 128,
    }
    metric = evaluate.load("rouge")
    loss_fn = CrossEntropyLoss(ignore_index=-100, reduction="mean")
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            logits = model(
                input_ids=batch["input_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
                decoder_input_ids=batch["decoder_input_ids"].to(device=DEVICE),
            ).logits
            labels = batch["labels"].to(device=DEVICE)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.detach().float().item()

            labels = labels.cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            generated_tokens = model.generate(
                batch["input_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
                **gen_kwargs,
            )
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
            )

    result = metric.compute(use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["loss"] = total_loss / len(dataloader)
    return result


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = get_tokenizer()
    train_dataset = get_dailymail_dataset(split="train")
    model = train(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    eval_train_dataset = get_dailymail_dataset(split="eval_train")
    results = evaluate_model(
        model=model, tokenizer=tokenizer, dataset=eval_train_dataset, batch_size=args.eval_batch_size
    )
    logger.info(f"Train evaluation results: {results}")

    eval_dataset = get_dailymail_dataset(split="valid")
    results = evaluate_model(model=model, tokenizer=tokenizer, dataset=eval_dataset, batch_size=args.eval_batch_size)
    logger.info(f"Valid evaluation results: {results}")

    if args.checkpoint_dir is not None:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))


if __name__ == "__main__":
    main()
