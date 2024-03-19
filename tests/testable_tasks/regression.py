# pylint: skip-file

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data

from kronfluence.task import Task

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


def make_mlp_model(bias: bool = True, seed: int = 0) -> nn.Module:
    set_seed(seed)
    return nn.Sequential(
        nn.Linear(10, 16, bias=bias),
        nn.ReLU(),
        nn.Linear(16, 16, bias=bias),
        nn.ReLU(),
        nn.Linear(16, 1, bias=bias),
    )


def make_repeated_mlp_model(bias: bool = True, seed: int = 0) -> nn.Module:
    set_seed(seed)

    shared_linear = nn.Linear(16, 16, bias=bias)
    return nn.Sequential(
        nn.Linear(10, 16, bias=bias),
        nn.ReLU(),
        shared_linear,
        nn.ReLU(),
        shared_linear,
        nn.ReLU(),
        nn.Linear(16, 1, bias=bias),
    )


def make_regression_dataset(num_data: int, seed: int = 0) -> data.Dataset:
    set_seed(seed)
    dataset = data.TensorDataset(
        torch.randn((num_data, 10), dtype=torch.float32),
        torch.randint(low=-5, high=5, size=(num_data, 1), dtype=torch.float32),
    )
    return dataset


class RegressionTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = model(inputs)
        if not sample:
            return F.mse_loss(outputs, targets, reduction="sum")
        with torch.no_grad():
            sampled_targets = torch.normal(outputs, std=math.sqrt(0.5))
        return F.mse_loss(outputs, sampled_targets.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model, sample=False)
