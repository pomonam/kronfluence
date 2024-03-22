import os
from typing import List

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset


def construct_resnet50() -> nn.Module:
    return torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)


def get_imagenet_dataset(
    split: str,
    indices: List[int] = None,
    dataset_dir: str = "data/",
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    if split == "train":
        transform_config = [
            torchvision.transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
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
        root=os.path.join(dataset_dir, folder),
        transform=transform_config,
    )

    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset
