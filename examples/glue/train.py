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
from examples.mnist.pipeline import construct_mnist_mlp, get_mnist_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train classification models on MNIST datasets.")

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
    logger = logging.getLogger()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = get_glue_dataset(data_name=args.dataset_name, split="train", data_path=args.dataset_dir)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        drop_last=True,
    )
    model = construct_bert(args.data_name).to(device=device)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    # )
    #
    # logger.info("Start training the model.")
    # model.train()
    # for epoch in range(args.num_train_epochs):
    #
    #     total_loss = 0
    #
    #     with tqdm(train_dataloader, unit="batch") as tepoch:
    #
    #         for batch in tepoch:
    #             tepoch.set_description(f"Epoch {epoch}")
    #             inputs, labels = batch
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             logits = model(inputs)
    #             loss = F.cross_entropy(logits, labels)
    #             total_loss += loss.detach().float()
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             tepoch.set_postfix(loss=total_loss.item() / len(train_dataloader))
    #
    # logger.info("Start evaluating the model.")
    # model.eval()
    # train_eval_dataset = get_mnist_dataset(
    #     split="eval_train", data_path=args.dataset_dir
    # )
    # train_eval_dataloader = DataLoader(
    #     dataset=train_eval_dataset,
    #     batch_size=args.eval_batch_size,
    #     shuffle=False,
    #     drop_last=False,
    # )
    # eval_dataset = get_mnist_dataset(split="valid", data_path=args.dataset_dir)
    # eval_dataloader = DataLoader(
    #     dataset=eval_dataset,
    #     batch_size=args.eval_batch_size,
    #     shuffle=False,
    #     drop_last=False,
    # )
    #
    # total_loss = 0
    # correct = 0
    # for batch in train_eval_dataloader:
    #     with torch.no_grad():
    #         inputs, labels = batch
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         logits = model(inputs)
    #         loss = F.cross_entropy(logits, labels)
    #         preds = logits.argmax(dim=1, keepdim=True)
    #         correct += preds.eq(labels.view_as(preds)).sum().item()
    #         total_loss += loss.detach().float()
    #
    # logger.info(
    #     f"Train loss: {total_loss.item() / len(train_eval_dataloader.dataset)} | "
    #     f"Train Accuracy: {100 * correct / len(train_eval_dataloader.dataset)}"
    # )
    #
    # total_loss = 0
    # correct = 0
    # for batch in eval_dataloader:
    #     with torch.no_grad():
    #         inputs, labels = batch
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         logits = model(inputs)
    #         loss = F.cross_entropy(logits, labels)
    #         preds = logits.argmax(dim=1, keepdim=True)
    #         correct += preds.eq(labels.view_as(preds)).sum().item()
    #         total_loss += loss.detach().float()
    #
    # logger.info(
    #     f"Train loss: {total_loss.item() / len(eval_dataloader.dataset)} | "
    #     f"Train Accuracy: {100 * correct / len(eval_dataloader.dataset)}"
    # )
    #
    # if args.checkpoint_dir is not None:
    #     torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model.pth"))


if __name__ == "__main__":
    main()
