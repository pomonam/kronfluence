import contextlib
import gc
import os
from typing import Any, Callable, Dict

import torch
import torch.distributed as dist
from accelerate.state import SharedDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel


class State:
    """A singleton class to manage the process environment state, such as device and process count.

    This class is inspired by Accelerate's `PartialState`:
    https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py.

    The direct use of `PartialState` from Accelerate can be problematic, since the analysis
    (influence computation) environment may be different from the training environment.
    """

    _shared_state: Dict[str, Any] = SharedDict()

    def __init__(self, cpu: bool = False) -> None:
        """Initializes an instance of the State class.

        Args:
            cpu (bool):
                Specifies whether the analysis should be explicitly performed using the CPU.
                Defaults to False, utilizing GPU resources if available.
        """
        self.__dict__ = self._shared_state

        if not self.initialized:
            self.cpu = cpu

            if int(os.environ.get("LOCAL_RANK", -1)) != -1 and not cpu and torch.cuda.is_available():
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
                self.num_processes = dist.get_world_size()
                self.process_index = dist.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                self.device = torch.device("cuda", self.local_process_index)
                self.n_gpus = torch.cuda.device_count()
                torch.cuda.set_device(self.device)
            else:
                self.num_processes = 1
                self.process_index = self.local_process_index = 0
                self.n_gpus = torch.cuda.device_count()
                self.device = torch.device("cpu") if self.cpu else self.default_device

    def __repr__(self) -> str:
        return (
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
            f"Local process index: {self.local_process_index}\n"
            f"Device: {self.device}\n"
        )

    @staticmethod
    def _reset_state() -> None:
        """Resets `_shared_state`, is used internally and should not be called."""
        State._shared_state.clear()

    @property
    def initialized(self) -> bool:
        """Returns whether the `PartialState` has been initialized."""
        return self._shared_state != {}

    @property
    def use_distributed(self) -> bool:
        """Whether the State is configured for distributed training."""
        return self.num_processes > 1

    @property
    def is_main_process(self) -> bool:
        """Returns whether the current process is the main process."""
        return self.process_index == 0

    @property
    def is_local_main_process(self) -> bool:
        """Returns whether the current process is the main process on the local node."""
        return self.local_process_index == 0

    @property
    def is_last_process(self) -> bool:
        """Returns whether the current process is the last one."""
        return self.process_index == self.num_processes - 1

    def wait_for_everyone(self) -> None:
        """Will stop the execution of the current process until every other process has reached that point
        (so this does nothing when the script is only run in one process)."""
        if self.use_distributed:
            dist.barrier()

    @property
    def default_device(self) -> torch.device:
        """Finds the default device currently available."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def release_memory() -> None:
    """Releases the memory by calling `gc.collect()` and `torch.cuda.empty_cache()`."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def no_sync(model: nn.Module, state: State) -> Callable:
    """A context manager to avoid DDP synchronization. The code is adapted from
    https://github.com/huggingface/accelerate/blob/v0.27.2/src/accelerate/accelerator.py#L852."""
    context = contextlib.nullcontext

    # `no_sync()` for FSDP instance can result in higher memory usage, detailed in:
    # https://pytorch.org/docs/stable/fsdp.html.
    if state.use_distributed and not isinstance(model, FullyShardedDataParallel):
        context = getattr(model, "no_sync", context)

    with context():
        yield
