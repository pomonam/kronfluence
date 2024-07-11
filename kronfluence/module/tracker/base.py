from typing import List, Optional, Union

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class BaseTracker:
    """Base class for tracking module activations, gradients, and scores."""

    def __init__(self, module: nn.Module) -> None:
        """Initializes an instance of the `BaseTracker` class.

        Args:
            module (TrackedModule):
                The `TrackedModule` that wraps the original module.
        """
        self.module = module
        self.registered_hooks: List[RemovableHandle] = []
        self.cached_hooks: List[RemovableHandle] = []
        self.cached_activations: Optional[Union[List[torch.Tensor]], torch.Tensor] = None
        self.cached_per_sample_gradient: Optional[torch.Tensor] = None

    def release_hooks(self) -> None:
        """Removes all registered hooks."""
        self.clear_all_cache()
        while self.registered_hooks:
            handle = self.registered_hooks.pop()
            handle.remove()
        self.registered_hooks = []

    def clear_all_cache(self) -> None:
        """Clears all cached data and removes cached hooks."""
        del self.cached_activations, self.cached_per_sample_gradient
        self.cached_activations, self.cached_per_sample_gradient = None, None
        while self.cached_hooks:
            handle = self.cached_hooks.pop()
            handle.remove()
        self.cached_hooks = []

    def _raise_cache_not_found_exception(self) -> None:
        """Raises an exception when cached activations are not found."""
        raise RuntimeError(
            f"Module '{self.module.name}' has no cached activations. This can occur if:\n"
            f"1. The module was not used during the forward pass, or\n"
            f"2. The module was encountered multiple times in the forward pass.\n"
            f"For case 2, set 'has_shared_parameters=True' to enable parameter sharing."
        )

    def register_hooks(self) -> None:
        """Registers hooks for the module."""

    def finalize_iteration(self) -> None:
        """Finalizes statistics for the current iteration."""

    def exist(self) -> bool:
        """Checks if the desired statistics are available.

        Returns:
            bool:
                `True` if statistics exist, `False` otherwise.
        """
        return False

    def synchronize(self, num_processes: int) -> None:
        """Synchronizes statistics across multiple processes.

        Args:
            num_processes (int):
                The number of processes to synchronize across.
        """

    def truncate(self, keep_size: int) -> None:
        """Truncates stored statistics to a specified size.

        Args:
            keep_size (int):
                The number of dimensions to keep.
        """

    def accumulate_iterations(self) -> None:
        """Accumulates (or prepares to accumulate) statistics across multiple iterations."""

    def finalize_all_iterations(self) -> None:
        """Finalizes statistics after all iterations."""

    def release_memory(self) -> None:
        """Releases any memory held by the tracker."""
