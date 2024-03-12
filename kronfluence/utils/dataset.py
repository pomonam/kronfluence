import math
import multiprocessing
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import KwargsHandler
from accelerate.utils.memory import should_reduce_batch_size
from torch.utils.data import Sampler


@dataclass
class DataLoaderKwargs(KwargsHandler):
    """The object used to customize DataLoader. Please refer to https://pytorch.org/docs/stable/data.html for
    detailed information of each argument. The default arguments are copied from PyTorch version 2.2.
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


def make_indices_partition(
    total_data_examples: int, partition_size: int
) -> List[Tuple[int, int]]:
    """Returns partitioned indices from the total data examples."""
    bins = list(map(len, np.array_split(range(total_data_examples), partition_size)))
    indices_bin = []
    start_idx = 0
    for i in range(partition_size):
        indices_bin.append((start_idx, start_idx + bins[i]))
        start_idx += bins[i]
    return indices_bin


def find_executable_batch_size(func: Callable, start_batch_size: int) -> int:
    """Finds executable batch size for calling the function that does not have OOM error. The code is motivated
    from https://github.com/huggingface/accelerate/blob/v0.27.2/src/accelerate/utils/memory.py#L83.
    """
    batch_size = start_batch_size

    while True:
        if batch_size == 0:
            raise RuntimeError("No executable batch size found, reached zero.")
        try:
            func(batch_size)

        except Exception as e:  # pylint: disable=broad-exception-caught
            if should_reduce_batch_size(e):  # pylint: disable=no-else-continue
                batch_size //= 2
                continue
            else:
                raise
        return batch_size


class DistributedEvalSampler(Sampler):
    """DistributedEvalSampler is different from DistributedSampler. It does NOT add extra samples to make
    it evenly divisible. DistributedEvalSampler should not be used for training. The distributed processes could
    hang forever. See this issue for details: https://github.com/pytorch/pytorch/issues/22584.

    The code is adapted from https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        dataset: torch.utils.data.Dataset,
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
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]."
            )

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


class DistributedSamplerWithStack(Sampler):
    """DistributedSampleWithStack is different from DistributedSampler. Instead of subsampling,
    it stacks the dataset."""

    def __init__(  # pylint: disable=super-init-not-called
        self,
        dataset: torch.utils.data.Dataset,
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
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]."
            )

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
