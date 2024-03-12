import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision


class Mul(torch.nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


def construct_resnet9() -> nn.Module:
    def conv_bn(
        channels_in: int,
        channels_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups=1,
    ) -> nn.Module:
        assert groups == 1
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


# def get_hyperparameters(data_name: str) -> Dict[str, float]:
#     wd = 0.001
#     if data_name == "cifar2":
#         lr = 0.5
#         epochs = 100
#     elif data_name == "cifar10":
#         lr = 0.4
#         epochs = 25
#     else:
#         raise NotImplementedError()
#     return {"lr": lr, "wd": wd, "epochs": epochs}


def get_cifar10_dataset(
    split: str,
    indices: List[int] = None,
    data_path: str = "data/",
):
    assert split in ["train", "eval_train", "valid"]

    normalize = torchvision.transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
    )

    if split in ["train", "eval_train"]:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    if split == "train":
        transform_config = [
            torchvision.transforms.RandomResizedCrop(
                size=224, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)
            ),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
        transform_config.extend([torchvision.transforms.ToTensor(), normalize])
        transform_config = torchvision.transforms.Compose(transform_config)

    else:
        transform_config = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=256),
                torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

    folder = "train" if split in ["train", "eval_train"] else "val"
    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_path, folder),
        transform=transform_config,
    )

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset


def get_cifar10_dataloader(
    batch_size: int,
    split: str = "train",
    indices: List[int] = None,
    do_corrupt: bool = False,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.247, 0.243, 0.261)

    if split in ["train", "eval_train"]:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=MEAN, std=STD),
            ]
        )

    dataset = torchvision.datasets.CIFAR10(
        root="/tmp/cifar/",
        download=True,
        train=split in ["train", "eval_train", "eval_train_with_aug"],
        transform=transforms,
    )

    if do_corrupt:
        if split == "valid":
            raise NotImplementedError(
                "Performing corruption on the validation dataset is not supported."
            )
        num_corrupt = math.ceil(len(dataset) * 0.1)
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
        new_targets[new_targets == original_targets] = (
            new_targets[new_targets == original_targets] + offsets
        ) % 10
        assert (new_targets == original_targets).sum() == 0
        dataset.targets[:num_corrupt] = list(new_targets)

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=split == "train",
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=split == "train",
        pin_memory=True,
    )
