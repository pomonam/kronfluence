from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn

from kronfluence.module.tracker.base import BaseTracker
from kronfluence.utils.constants import AGGREGATED_GRADIENT_NAME


class GradientTracker(BaseTracker):
    """Tracks and computes aggregated gradient for a given module."""

    def register_hooks(self) -> None:
        """Sets up hooks to compute and keep track of aggregated gradient."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach()
            device = "cpu" if self.module.score_args.offload_activations_to_cpu else cached_activation.device
            cached_activation = cached_activation.to(
                device=device,
                dtype=self.module.score_args.per_sample_gradient_dtype,
                copy=True,
            )
            if self.module.factor_args.has_shared_parameters:
                if self.cached_activations is None:
                    self.cached_activations = []
                self.cached_activations.append(cached_activation)
            else:
                self.cached_activations = cached_activation
            self.cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self.cached_activations is None:
                self._raise_cache_not_found_exception()
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.score_args.per_sample_gradient_dtype)
            if isinstance(self.cached_activations, list):
                cached_activation = self.cached_activations.pop()
            else:
                cached_activation = self.cached_activations
            if self.module.per_sample_gradient_process_fnc is None:
                summed_gradient = self.module.compute_summed_gradient(
                    input_activation=cached_activation.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
                self.clear_all_cache()
            else:
                summed_gradient = self.module.compute_per_sample_gradient(
                    input_activation=cached_activation.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                ).sum(dim=0, keepdim=True)
            if self.module.gradient_scale != 1.0:
                summed_gradient.mul_(self.module.gradient_scale)
            if self.module.storage[AGGREGATED_GRADIENT_NAME] is None:
                self.module.storage[AGGREGATED_GRADIENT_NAME] = torch.zeros_like(summed_gradient, requires_grad=False)
            self.module.storage[AGGREGATED_GRADIENT_NAME].add_(summed_gradient)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    def finalize_iteration(self):
        """Clears all cached data from memory."""
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if aggregated gradient is available."""
        return self.module.storage[AGGREGATED_GRADIENT_NAME] is not None

    def synchronize(self, num_processes: int = 1) -> None:
        """Aggregates summed gradient across multiple devices or nodes in a distributed setting."""
        del num_processes
        if dist.is_initialized() and torch.cuda.is_available():
            if self.module.storage[AGGREGATED_GRADIENT_NAME] is None:
                self.module.storage[AGGREGATED_GRADIENT_NAME] = torch.zeros(
                    size=(1,),
                    dtype=self.module.score_args.per_sample_gradient_dtype,
                    device="cuda",
                    requires_grad=False,
                )
            self.module.storage[AGGREGATED_GRADIENT_NAME] = self.module.storage[AGGREGATED_GRADIENT_NAME].contiguous()
            dist.all_reduce(
                tensor=self.module.storage[AGGREGATED_GRADIENT_NAME],
                op=dist.ReduceOp.SUM,
            )

    def release_memory(self) -> None:
        """Clears aggregated gradients from memory."""
        self.clear_all_cache()
        self.module.storage[AGGREGATED_GRADIENT_NAME] = None
