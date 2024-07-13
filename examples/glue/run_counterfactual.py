import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import set_seed
from torch.utils import data
from transformers import default_data_collator

from examples.glue.pipeline import get_glue_dataset
from examples.glue.train import train
from kronfluence import Analyzer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accuracy(model: nn.Module, dataset: data.Dataset) -> torch.Tensor:
    dataloader = data.DataLoader(
        dataset=dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=default_data_collator
    )

    model.eval()
    with torch.no_grad():
        acc_lst = []
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"].to(device=DEVICE),
                token_type_ids=batch["token_type_ids"].to(device=DEVICE),
                attention_mask=batch["attention_mask"].to(device=DEVICE),
            ).logits
            labels = batch["labels"].to(device=DEVICE)
            accs = (outputs.argmax(-1) == labels).float().cpu()
            acc_lst.append(accs)
        all_accs = torch.cat(acc_lst)
    return all_accs


def train_with_indices(dataset: data.Dataset, seed: int, indices_to_keep: Optional[List[int]] = None) -> nn.Module:
    if indices_to_keep is not None:
        dataset = dataset.select(indices_to_keep)

    set_seed(seed)
    model = train(dataset=dataset, batch_size=16, num_train_epochs=3, learning_rate=2e-05, weight_decay=0.01)
    return model


def train_with_configurations(
    dataset: data.Dataset,
    valid_dataset: data.Dataset,
    top_indices: List[int],
    interval: int,
    seed_ids: List[int],
) -> List[torch.Tensor]:
    num_train = len(dataset)
    indices_to_remove = top_indices[:interval]
    indices_to_keep = list(set(range(num_train)) - set(indices_to_remove))
    assert len(indices_to_keep) + len(indices_to_remove) == num_train

    valid_acc_lst = []
    for seed in seed_ids:
        model = train_with_indices(dataset=dataset, indices_to_keep=indices_to_keep, seed=seed + 2008)
        valid_results = get_accuracy(model, valid_dataset)
        valid_acc_lst.append(valid_results)
    return valid_acc_lst


def main():
    train_dataset = get_glue_dataset(
        data_name="rte",
        split="eval_train",
    )
    eval_dataset = get_glue_dataset(
        data_name="rte",
        split="valid",
    )
    num_target = 100
    assert num_target <= len(eval_dataset)

    remove_intervals = [20, 40, 60, 80, 100, 120]
    num_base_repeat = 5
    num_repeat = 3

    large_seed_ids = list(range(num_base_repeat))
    seed_ids = list(range(num_repeat))

    valid_acc_lst = []
    for seed in large_seed_ids:
        model = train_with_indices(dataset=train_dataset, seed=seed + 79, indices_to_keep=None)
        valid_results = get_accuracy(model, eval_dataset)
        valid_acc_lst.append(valid_results)

    # Selects validation data points that get correctly classified on all seeds.
    mask = np.array(valid_acc_lst).mean(0) >= 1.0
    print(f"Total target numbers: {mask.sum()}")

    # Get random baseline.
    start_time = time.time()
    random_results = []
    for valid_idx in range(num_target):
        print(f"{valid_idx}th validation data point.")
        if mask[valid_idx]:
            # Selects training data points with the same label.
            correct_label = eval_dataset[valid_idx]["label"]
            random_indices = list(
                np.random.permutation(
                    [
                        i
                        for i, x in enumerate([x["label"] for x in eval_dataset])
                        if x == correct_label and i < num_target
                    ]
                )
            )

            success_lst = []
            for interval in remove_intervals:
                results = train_with_configurations(
                    dataset=train_dataset,
                    top_indices=random_indices,
                    valid_dataset=eval_dataset.select([valid_idx]),
                    interval=interval,
                    seed_ids=seed_ids,
                )
                if np.array(results).mean() < 0.5:
                    success_lst.append(1)
                    break
                else:
                    success_lst.append(0)

            while len(success_lst) < len(remove_intervals):
                success_lst.append(1)

            random_results.append(success_lst)

    end_time = time.time()
    print(f"Took {end_time - start_time} seconds for the random baseline.")
    random_results = np.array(random_results).sum(0)
    print(f"Results: {random_results}")

    # Get EKFAC baseline.
    start_time = time.time()
    scores = Analyzer.load_file("influence_results/rte/scores_ekfac/pairwise_scores.safetensors")["all_modules"].to(
        dtype=torch.float32
    )
    ekfac_results = []
    for valid_idx in range(num_target):
        print(f"{valid_idx}th validation data point.")
        if mask[valid_idx]:
            top_indices = torch.argsort(scores[valid_idx], descending=True)
            top_indices = [idx.item() for idx in top_indices]

            success_lst = []
            for interval in remove_intervals:
                results = train_with_configurations(
                    dataset=train_dataset,
                    top_indices=top_indices,
                    valid_dataset=eval_dataset.select([valid_idx]),
                    interval=interval,
                    seed_ids=seed_ids,
                )
                if np.array(results).mean() < 0.5:
                    success_lst.append(1)
                    break
                else:
                    success_lst.append(0)

            while len(success_lst) < len(remove_intervals):
                success_lst.append(1)

            ekfac_results.append(success_lst)

    end_time = time.time()
    print(f"Took {end_time - start_time} seconds for the EKFAC baseline.")
    ekfac_results = np.array(ekfac_results).sum(0)
    print(f"Results: {ekfac_results}")

    # Get Identity baseline.
    start_time = time.time()
    scores = Analyzer.load_file("influence_results/rte/scores_identity/pairwise_scores.safetensors")["all_modules"].to(
        dtype=torch.float32
    )
    identity_results = []
    for valid_idx in range(num_target):
        print(f"{valid_idx}th validation data point.")
        if mask[valid_idx]:
            top_indices = torch.argsort(scores[valid_idx], descending=True)
            top_indices = [idx.item() for idx in top_indices]

            success_lst = []
            for interval in remove_intervals:
                results = train_with_configurations(
                    dataset=train_dataset,
                    top_indices=top_indices,
                    valid_dataset=eval_dataset.select([valid_idx]),
                    interval=interval,
                    seed_ids=seed_ids,
                )
                if np.array(results).mean() < 0.5:
                    success_lst.append(1)
                    break
                else:
                    success_lst.append(0)

            while len(success_lst) < len(remove_intervals):
                success_lst.append(1)

            identity_results.append(success_lst)

    end_time = time.time()
    print(f"Took {end_time - start_time} seconds for the identity baseline.")
    identity_results = np.array(identity_results).sum(0)
    print(f"Results: {identity_results}")

    print("final")
    print(f"Results: {random_results}")
    print(f"Results: {ekfac_results}")
    print(f"Results: {identity_results}")


if __name__ == "__main__":
    main()
