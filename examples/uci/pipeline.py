import os
from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset


def construct_regression_mlp() -> nn.Module:
    num_inputs = 8
    model = torch.nn.Sequential(
        nn.Linear(num_inputs, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 128, bias=True),
        nn.ReLU(),
        nn.Linear(128, 1, bias=True),
    )
    return model


class RegressionDataset(Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray) -> None:
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.data_x[index], self.data_y[index]

    def __len__(self) -> int:
        return self.data_x.shape[0]


def get_regression_dataset(
    data_name: str,
    split: str,
    indices: List[int] = None,
    dataset_dir: str = "data/",
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    # Load the dataset from the `.data` file.
    data = np.loadtxt(os.path.join(dataset_dir, data_name + ".data"), delimiter=None)
    data = data.astype(np.float32)

    # Shuffle the dataset.
    seed = np.random.RandomState(0)
    permutation = seed.choice(np.arange(data.shape[0]), data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0:size_train]
    index_val = permutation[size_train:]

    x_train, y_train = data[index_train, :-1], data[index_train, -1]
    x_val, y_val = data[index_val, :-1], data[index_val, -1]

    # Standardize the inputs.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    # Standardize the targets.
    scaler = StandardScaler()
    y_train = np.expand_dims(y_train, -1)
    y_val = np.expand_dims(y_val, -1)
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)

    if split in ["train", "eval_train"]:
        dataset = RegressionDataset(
            x_train_scaled.astype(np.float32),
            y_train_scaled.astype(np.float32),
        )
    else:
        dataset = RegressionDataset(x_val_scaled.astype(np.float32), y_val_scaled.astype(np.float32))

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset
