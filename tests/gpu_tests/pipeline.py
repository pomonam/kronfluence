# pylint: skip-file

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset

from kronfluence.task import Task

BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor]


class GpuTestTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs.double())
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
        logits = model(inputs.double())

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()


def construct_test_mlp() -> nn.Module:
    model = torch.nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1024, bias=True),
        nn.ReLU(),
        nn.Linear(1024, 1024, bias=True),
        nn.ReLU(),
        nn.Linear(1024, 1024, bias=True),
        nn.ReLU(),
        nn.Linear(1024, 10, bias=True),
    )
    return model


def get_mnist_dataset(
    split: str,
    indices: List[int] = None,
    data_path: str = "data/",
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = torchvision.datasets.MNIST(
        root=data_path,
        download=True,
        train=split in ["train", "eval_train"],
        transform=transforms,
    )

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset
