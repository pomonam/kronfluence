from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from kronfluence.factor.config import FactorConfig
from kronfluence.module.tracker.base import BaseTracker
from kronfluence.utils.constants import (
    ACCUMULATED_PRECONDITIONED_GRADIENT_NAME,
    AGGREGATED_GRADIENT_NAME,
    PRECONDITIONED_GRADIENT_NAME,
)


class GradientTracker(BaseTracker):
    """Tracks and computes summed gradient for a given module."""

    def register_hooks(self) -> None:
        """Sets up hooks to compute and keep track of summed gradient."""

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

            outputs.register_hook(
                shared_backward_hook if self.module.factor_args.has_shared_parameters else backward_hook
            )

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self.cached_activations is None:
                self._raise_cache_not_found_exception()

            output_gradient = self._scale_output_gradient(
                output_gradient=output_gradient, target_dtype=self.module.score_args.per_sample_gradient_dtype
            )
            if self.module.per_sample_gradient_process_fnc is None:
                summed_gradient = self.module.compute_summed_gradient(
                    input_activation=self.cached_activations.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
            else:
                summed_gradient = self.module.compute_per_sample_gradient(
                    input_activation=self.cached_activations.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                ).sum(dim=0, keepdim=True)
            self.clear_all_cache()

            if self.module.storage[AGGREGATED_GRADIENT_NAME] is None:
                self.module.storage[AGGREGATED_GRADIENT_NAME] = torch.zeros_like(summed_gradient, requires_grad=False)
            self.module.storage[AGGREGATED_GRADIENT_NAME].add_(summed_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            output_gradient = self._scale_output_gradient(
                output_gradient=output_gradient, target_dtype=self.module.score_args.per_sample_gradient_dtype
            )
            cached_activation = self.cached_activations.pop()
            if self.module.per_sample_gradient_process_fnc is None:
                summed_gradient = self.module.compute_summed_gradient(
                    input_activation=cached_activation.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
            else:
                summed_gradient = self.module.comute_per_sample_gradient(
                    input_activation=cached_activation.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                ).sum(dim=0, keepdim=True)

            if self.cached_per_sample_gradient is None:
                self.cached_per_sample_gradient = torch.zeros_like(summed_gradient, requires_grad=False)
            self.cached_per_sample_gradient.add_(summed_gradient)

        self.registered_hooks.append(self.module.original_module.register_forward_hook(forward_hook))

    def exist(self) -> bool:
        return self.module.storage[AGGREGATED_GRADIENT_NAME] is not None

    @torch.no_grad()
    def finalize_iteration(self):
        """Computes preconditioned gradient using cached per-sample gradients."""
        if not self.module.factor_args.has_shared_parameters:
            return
        if self.module.storage[AGGREGATED_GRADIENT_NAME] is None:
            self.module.storage[AGGREGATED_GRADIENT_NAME] = torch.zeros_like(
                self.cached_per_sample_gradient, requires_grad=False
            )
        self.module.storage[AGGREGATED_GRADIENT_NAME].add_(self.cached_per_sample_gradient)
        self.clear_all_cache()

    def release_memory(self) -> None:
        """Clears summed gradients from memory."""
        del self.module.storage[AGGREGATED_GRADIENT_NAME]
        self.module.storage[AGGREGATED_GRADIENT_NAME] = None
        self.clear_all_cache()

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
            dist.all_reduce(
                tensor=self.module.storage[AGGREGATED_GRADIENT_NAME],
                op=dist.ReduceOp.SUM,
            )
