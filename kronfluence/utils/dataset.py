import math
import multiprocessing
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import KwargsHandler
from accelerate.utils.memory import should_reduce_batch_size
from torch.utils import data
from torch.utils.data import Sampler

T_co = TypeVar("T_co", covariant=True)


@dataclass
class DataLoaderKwargs(KwargsHandler):
    """Customization options for DataLoader.

    This class encapsulates the arguments used to customize PyTorch's DataLoader. Default values are based on
    PyTorch version 2.3. For detailed information on each argument, refer to:
    https://pytorch.org/docs/stable/data.html.
    """

    num_workers: int = 0
    collate_fn: Optional[Callable] = None
    pin_memory: bool = False
    timeout: int = 0
    worker_init_fn: Optional[Callable] = None
    multiprocessing_context: Optional[multiprocessing.context.BaseContext] = None
    generator: torch.Generator = None
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""


def make_indices_partition(total_data_examples: int, partition_size: int) -> List[Tuple[int, int]]:
    """Partitions data indices into approximately equal-sized bins.

    Args:
        total_data_examples (int):
            Total number of data examples.
        partition_size (int):
            Number of partitions to create.

    Returns:
        List[Tuple[int, int]]:
            List of tuples, each containing start and end indices for a partition.

    Raises:
        ValueError: If `total_data_examples` is less than `partition_size`.
    """
    if total_data_examples < partition_size:
        raise ValueError("The total data examples must be equal to or greater than the partition size.")
    # See https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length.
    bins = list(map(len, np.array_split(range(total_data_examples), partition_size)))
    start_idx = 0
    indices_bin = []
    for i in range(partition_size):
        indices_bin.append((start_idx, start_idx + bins[i]))
        start_idx += bins[i]
    return indices_bin


def find_executable_batch_size(func: Callable, start_batch_size: int) -> int:
    """Finds the largest batch size that can be executed without OOM errors.

    This function progressively reduces the batch size until it finds a size that can be executed
    without running out of memory. The code is motivated from:
    https://github.com/huggingface/accelerate/blob/v0.27.2/src/accelerate/utils/memory.py#L83

    Args:
        func (Callable):
            Function to test with different batch sizes.
        start_batch_size (int):
            Initial batch size to try.

    Returns:
        int:
            The largest executable batch size.

    Raises:
        RuntimeError:
            If no executable batch size is found (reaches zero).
    """
    batch_size = start_batch_size

    while True:
        if batch_size == 0:
            raise RuntimeError("No executable batch size found, reached zero.")
        try:
            func(batch_size)

        except Exception as e:  # pylint: disable=broad-exception-caught
            if should_reduce_batch_size(exception=e):  # pylint: disable=no-else-continue
                batch_size //= 2
                continue
            else:
                raise
        return batch_size


class DistributedEvalSampler(Sampler[T_co]):
    """Sampler for distributed setting without adding extra samples.

    Unlike `DistributedSampler`, it does not add extra samples to make the dataset evenly divisible across processes.
    The code is adapted from https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        dataset: data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}].")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(self.dataset)
        indices = list(range(self.total_size))
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        self.seed = seed

    def __iter__(self) -> Iterable[int]:
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class DistributedSamplerWithStack(Sampler[T_co]):
    """Sampler that stacks the dataset for distributed setting.

    Instead of subsampling, this sampler stacks the dataset across processes. It ensures even distribution by
    adding padding samples if necessary.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        dataset: data.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}].")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterable[int]:
        indices = list(range(len(self.dataset)))

        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size

        # Instead of subsampling, divide the dataset by chunks.
        start_index = self.rank * (self.total_size // self.num_replicas)
        end_index = start_index + (self.total_size // self.num_replicas)
        indices = indices[start_index:end_index]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
