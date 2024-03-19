# pylint: skip-file

import numpy as np
import pytest

from kronfluence.utils.dataset import (
    DistributedEvalSampler,
    DistributedSamplerWithStack,
    make_indices_partition,
)
from tests.utils import prepare_test


@pytest.mark.parametrize("dataset_size", [3, 105, 1027])
@pytest.mark.parametrize("num_replicas", [4, 105, 1027])
def test_eval_distributed_sampler(
    dataset_size: int,
    num_replicas: int,
) -> None:
    _, train_dataset, _, _, _ = prepare_test(
        test_name="mlp",
        train_size=dataset_size,
        seed=0,
    )

    indices = []
    for rank in range(num_replicas):
        sampler = DistributedEvalSampler(train_dataset, num_replicas=num_replicas, rank=rank)
        indices.append(np.array(list(iter(sampler))))

    assert len(np.hstack(indices)) == dataset_size
    # Make sure that there aren't any duplicates.
    assert len(np.unique(np.hstack(indices))) == dataset_size


@pytest.mark.parametrize("dataset_size", [3, 105, 1027])
@pytest.mark.parametrize("num_replicas", [4, 105, 1027])
def test_eval_distributed_sampler_with_stack(
    dataset_size: int,
    num_replicas: int,
) -> None:
    dataset_size = 1002
    _, train_dataset, _, _, _ = prepare_test(
        test_name="mlp",
        train_size=dataset_size,
        seed=0,
    )

    num_replicas = 4
    indices = []
    for rank in range(num_replicas):
        sampler = DistributedSamplerWithStack(train_dataset, num_replicas=num_replicas, rank=rank)
        indices.append(np.array(list(iter(sampler))))

    for i, sample_indices in enumerate(indices):
        if i != len(indices) - 1:
            assert np.all(np.sort(sample_indices) == sample_indices)
    assert len(np.unique(np.hstack(indices[:-1]))) == len(np.hstack(indices[:-1]))


@pytest.mark.parametrize("total_data_examples", [520, 1000, 8129])
@pytest.mark.parametrize("partition_size", [2, 270, 520])
def test_make_indices_partition(total_data_examples: int, partition_size: int) -> None:
    indices = make_indices_partition(total_data_examples=total_data_examples, partition_size=partition_size)
    assert len(indices) == partition_size
    reconstructions = []
    for start_index, end_index in indices:
        reconstructions.extend(list(range(start_index, end_index)))
    assert len(reconstructions) == total_data_examples
    assert len(set(reconstructions)) == len(reconstructions)
