# pylint: skip-file

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data

from kronfluence.task import Task

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


def make_conv_model(bias: bool = True, seed: int = 0) -> nn.Module:
    set_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, 1, bias=bias),
        nn.ReLU(),
        nn.Conv2d(4, 8, 3, 1, bias=bias),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1152, 5, bias=bias),
    )


def make_conv_bn_model(bias: bool = True, seed: int = 0) -> nn.Module:
    set_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, 1, bias=bias),
        nn.ReLU(),
        nn.BatchNorm2d(4),
        nn.Conv2d(4, 8, 3, 1, bias=bias),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Flatten(),
        nn.Linear(1152, 5, bias=bias),
    )


def make_classification_dataset(num_data: int, seed: int = 0) -> data.Dataset:
    set_seed(seed)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    return torchvision.datasets.FakeData(size=num_data, image_size=(3, 16, 16), num_classes=5, transform=transform)


class ClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()
