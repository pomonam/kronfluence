import copy
import math
from typing import List, Optional

import datasets
import numpy as np
import torch
import torchvision
from torch import nn


class Mul(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


def construct_resnet9() -> nn.Module:
    # ResNet-9 architecture from https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb.
    def conv_bn(
        channels_in: int,
        channels_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(),
        )

    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, 10, bias=False),
        Mul(0.2),
    )
    return model


def get_cifar10_dataset(
    split: str,
    indices: List[int] = None,
    corrupt_percentage: Optional[float] = None,
    dataset_dir: str = "data/",
) -> datasets.Dataset:
    assert split in ["train", "eval_train", "valid"]

    normalize = torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
    if split == "train":
        transform_config = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform_config = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

    dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        download=True,
        train=split in ["train", "eval_train"],
        transform=transform_config,
    )

    if corrupt_percentage is not None:
        if split == "valid":
            raise NotImplementedError("Performing corruption on the validation dataset is not supported.")
        assert 0.0 < corrupt_percentage <= 1.0
        num_corrupt = math.ceil(len(dataset) * corrupt_percentage)
        original_targets = np.array(copy.deepcopy(dataset.targets[:num_corrupt]))
        new_targets = torch.randint(
            0,
            10,
            size=original_targets[:num_corrupt].shape,
            generator=torch.Generator().manual_seed(0),
        ).numpy()
        offsets = torch.randint(
            1,
            9,
            size=new_targets[new_targets == original_targets].shape,
            generator=torch.Generator().manual_seed(0),
        ).numpy()
        new_targets[new_targets == original_targets] = (new_targets[new_targets == original_targets] + offsets) % 10
        assert (new_targets == original_targets).sum() == 0
        dataset.targets[:num_corrupt] = list(new_targets)

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset
