# pylint: skip-file

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from torch.utils.checkpoint import checkpoint_sequential

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


class RepeatedMLP(nn.Module):
    def __init__(self, bias: bool) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 16, bias=bias)
        self.shared_linear = nn.Linear(16, 16, bias=bias)
        self.linear2 = nn.Linear(16, 16, bias=bias)
        self.linear3 = nn.Linear(16, 1, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear1(inputs)
        x = torch.relu(x)
        x = self.shared_linear(x)
        x = torch.relu(x)
        x = self.shared_linear(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.shared_linear(x)
        x = torch.relu(x)
        return self.linear3(x)


def make_repeated_mlp_model(bias: bool = True, seed: int = 0) -> nn.Module:
    set_seed(seed)
    model = RepeatedMLP(bias=bias)
    return model


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

        if list(model.parameters())[1].dtype == torch.float64:
            inputs = inputs.to(dtype=torch.float64)
            targets = targets.to(dtype=torch.float64)
        outputs = model(inputs)

        if not sample:
            return F.mse_loss(outputs, targets, reduction="sum")
        with torch.no_grad():
            sampled_targets = torch.normal(outputs.detach(), std=math.sqrt(0.5))
        return F.mse_loss(outputs, sampled_targets, reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model, sample=False)


class GradientCheckpointRegressionTask(RegressionTask):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, targets = batch

        if list(model.parameters())[1].dtype == torch.float64:
            inputs = inputs.to(dtype=torch.float64)
            targets = targets.to(dtype=torch.float64)

        outputs = checkpoint_sequential(functions=model, segments=2, input=inputs, use_reentrant=False)

        if not sample:
            return F.mse_loss(outputs, targets, reduction="sum")
        with torch.no_grad():
            sampled_targets = torch.normal(outputs.detach(), std=math.sqrt(0.5))
        return F.mse_loss(outputs, sampled_targets, reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model, sample=False)
