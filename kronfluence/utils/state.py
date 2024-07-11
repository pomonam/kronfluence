import contextlib
import gc
import os
from typing import Any, Callable, Dict, List

import torch
import torch.distributed as dist
from accelerate.state import SharedDict
from torch import nn


class State:
    """A singleton class to manage the process environment state, such as device and process count.

    This class is inspired by Accelerate's `PartialState`:
    https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py.

    The direct use of `PartialState` from Accelerate can be problematic, since the analysis
    (influence computation) environment may be different from the training environment.
    """

    _shared_state: Dict[str, Any] = SharedDict()

    def __init__(self, cpu: bool = False) -> None:
        """Initializes an instance of the `State` class.

        Args:
            cpu (bool):
                If `True`, forces the use of CPU even if GPUs are available. Defaults to `False`.
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
        """Provides a string representation of the `State` instance.

        Returns:
            str:
                A formatted string containing process and device information.
        """
        return (
            f"Num processes: {self.num_processes}\n"
            f"Process index: {self.process_index}\n"
            f"Local process index: {self.local_process_index}\n"
            f"Device: {self.device}\n"
        )

    @staticmethod
    def _reset_state() -> None:
        """Resets the shared state. For internal use only."""
        State._shared_state.clear()

    @property
    def initialized(self) -> bool:
        """Checks if the `State` has been initialized."""
        return self._shared_state != {}

    @property
    def use_distributed(self) -> bool:
        """Checks if the setup is configured for distributed setting."""
        return self.num_processes > 1

    @property
    def is_main_process(self) -> bool:
        """Checks if the current process is the main process."""
        return self.process_index == 0

    @property
    def is_local_main_process(self) -> bool:
        """Checks if the current process is the main process on the local node."""
        return self.local_process_index == 0

    @property
    def is_last_process(self) -> bool:
        """Checks if the current process is the last one."""
        return self.process_index == self.num_processes - 1

    def wait_for_everyone(self) -> None:
        """Synchronizes all processes.

        This method will pause the execution of the current process until all other processes
        reach this point. It has no effect in single-process execution.
        """
        if self.use_distributed:
            dist.barrier()

    @property
    def default_device(self) -> torch.device:
        """Determines the default device (CUDA if available, otherwise CPU).

        Returns:
            torch.device:
                The default device.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def release_memory() -> None:
    """Releases unused memory.

    This function calls Python's garbage collector and empties CUDA cache if CUDA is available.
    """
    gc.collect()
    torch.compiler.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_active_tensors() -> List[torch.Tensor]:
    """Gets a list of active tensors in memory.

    Returns:
        List[torch.Tensor]:
            A list of tuples containing tensor type and size.
    """
    tensor_lst = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            tensor_lst.append(type(obj), obj.size())
    return tensor_lst


@contextlib.contextmanager
def no_sync(model: nn.Module, state: State) -> Callable:
    """A context manager to temporarily disable gradient synchronization in distributed setting.

    Args:
        model (nn.Module):
            The PyTorch model.
        state (State):
            The current process state.

    Yields:
        A context where gradient synchronization is disabled (if applicable).

    Note:
        For FullyShardedDataParallel (FSDP) models, this may result in higher memory usage.
        See: https://pytorch.org/docs/stable/fsdp.html.
    """
    context = contextlib.nullcontext

    if state.use_distributed:
        context = getattr(model, "no_sync", context)

    with context():
        yield
