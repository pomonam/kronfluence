import numpy as np
import pytest
import torch
from torch.utils.data import DistributedSampler

from kronfluence.utils.dataset import (
    DistributedEvalSampler,
    DistributedSamplerWithStack,
)
from tests.utils import prepare_test


@pytest.mark.parametrize("dataset_size", [3, 105, 1027])
@pytest.mark.parametrize("num_replicas", [4, 105, 1027])
def test_eval_distributed_sampler(
    dataset_size: int,
    num_replicas: int,
):
    _, train_dataset, _, _, _ = prepare_test(
        test_name="mlp",
        train_size=dataset_size,
        seed=0,
    )

    indices = []
    for rank in range(num_replicas):
        sampler = DistributedEvalSampler(
            train_dataset, num_replicas=num_replicas, rank=rank
        )
        indices.append(np.array(list(iter(sampler))))

    assert len(np.hstack(indices)) == dataset_size
    # Make sure that there aren't any duplicates.
    assert len(np.unique(np.hstack(indices))) == dataset_size


@pytest.mark.parametrize("dataset_size", [3, 105, 1027])
@pytest.mark.parametrize("num_replicas", [4, 105, 1027])
def test_eval_distributed_sampler_with_stack(
    dataset_size: int,
    num_replicas: int,
):
    dataset_size = 1002
    _, train_dataset, _, _, _ = prepare_test(
        test_name="mlp",
        train_size=dataset_size,
        seed=0,
    )

    num_replicas = 4
    indices = []
    for rank in range(num_replicas):
        sampler = DistributedSamplerWithStack(
            train_dataset, num_replicas=num_replicas, rank=rank
        )
        indices.append(np.array(list(iter(sampler))))

    for i, sample_indices in enumerate(indices):
        if i != len(indices) - 1:
            assert np.all(np.sort(sample_indices) == sample_indices)
    assert len(np.unique(np.hstack(indices[:-1]))) == len(np.hstack(indices[:-1]))


def test_all_gather():
    dataset_size = 13
    num_replicas = 4
    _, train_dataset, _, _, _ = prepare_test(
        test_name="mlp",
        train_size=dataset_size,
        seed=0,
    )

    # Check aggregation for Distributed sampler.
    indices = []
    for rank in range(num_replicas):
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        indices.append(np.array(list(iter(sampler))))

    tensors = []
    for i, current_indices in enumerate(indices):
        added_tensor = torch.rand((len(current_indices), 6, 7), dtype=torch.float64)
        added_tensor[:, :, 0].fill_(i)
        tensors.append(added_tensor)

    unsqueeze_tensors = [t.unsqueeze(0) for t in tensors]
    aggregated_tensors = torch.cat(unsqueeze_tensors).transpose(0, 1).reshape(-1, 6, 7)
    assert aggregated_tensors.sum() == torch.cat(tensors).sum()

    # Check aggregation for stacked sampler.
    indices = []
    for rank in range(num_replicas):
        sampler = DistributedSamplerWithStack(
            train_dataset,
            num_replicas=num_replicas,
            rank=rank,
        )
        indices.append(np.array(list(iter(sampler))))

    tensors = []
    for i, current_indices in enumerate(indices):
        added_tensor = torch.rand((len(current_indices), 6, 7), dtype=torch.float64)
        added_tensor.fill_(i)
        tensors.append(added_tensor)

    unsqueeze_tensors = [t.unsqueeze(0) for t in tensors]
    aggregated_tensors = torch.cat(unsqueeze_tensors).transpose(0, 1).reshape(-1, 6, 7)
    assert aggregated_tensors.sum() == torch.cat(tensors).sum()
