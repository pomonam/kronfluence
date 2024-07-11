from typing import Tuple

import torch
from torch import nn

from kronfluence.factor.config import STORAGE_TYPE, FactorConfig
from kronfluence.module.tracker.base import BaseTracker
from kronfluence.utils.constants import (
    PRECONDITIONED_GRADIENT_NAME,
    SELF_SCORE_VECTOR_NAME,
)


def move_storage_to_device(storage: STORAGE_TYPE, target_device: torch.device) -> None:
    """Moves all stored factors in the storage dictionary to the specified target device.

    Args:
        storage (STORAGE_TYPE):
            A dictionary containing stored factors.
        target_device (torch.device):
            The target device to move the factors to.
    """
    for name, factor in storage.items():
        if factor is not None:
            if isinstance(factor, list):
                for i in range(len(storage[name])):
                    storage[name][i] = factor[i].to(device=target_device)
            if isinstance(factor, torch.Tensor):
                storage[name] = factor.to(device=target_device)


class SelfScoreTracker(BaseTracker):
    """Computes self-influence scores for a given module."""

    storage_at_device: bool = False

    def _compute_self_score(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes self-influence scores using per-sample gradients.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample gradient tensor for the given batch.
        """
        if not self.storage_at_device:
            move_storage_to_device(
                storage=self.module.storage,
                target_device=per_sample_gradient.device,
            )
            self.storage_at_device = True

        preconditioned_gradient = (
            FactorConfig.CONFIGS[self.module.factor_args.strategy]
            .precondition_gradient(
                gradient=per_sample_gradient,
                storage=self.module.storage,
            )
            .to(dtype=self.module.score_args.score_dtype)
        )
        per_sample_gradient = per_sample_gradient.to(dtype=self.module.score_args.score_dtype)
        preconditioned_gradient.mul_(per_sample_gradient)
        self.module.storage[SELF_SCORE_VECTOR_NAME] = preconditioned_gradient.sum(dim=(1, 2))

    def register_hooks(self) -> None:
        """Sets up hooks to compute self-influence scores."""

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
            self.cached_hooks.append(
                outputs.register_hook(
                    shared_backward_hook if self.module.factor_args.has_shared_parameters else backward_hook
                )
            )

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self.cached_activations is None:
                self._raise_cache_not_found_exception()
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.score_args.per_sample_gradient_dtype)
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=self.cached_activations.to(device=output_gradient.device),
                output_gradient=output_gradient,
            ).to(dtype=self.module.score_args.precondition_dtype)
            self.clear_all_cache()
            del output_gradient
            if self.module.gradient_scale != 1.0:
                per_sample_gradient.mul_(self.module.gradient_scale)
            self._compute_self_score(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.score_args.per_sample_gradient_dtype)
            cached_activation = self.cached_activations.pop()
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient,
            )
            if self.cached_per_sample_gradient is None:
                self.cached_per_sample_gradient = torch.zeros_like(per_sample_gradient, requires_grad=False)
            self.cached_per_sample_gradient.add_(per_sample_gradient)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    @torch.no_grad()
    def finalize_iteration(self) -> None:
        """Computes self-influence scores using cached per-sample gradients."""
        if self.module.factor_args.has_shared_parameters:
            self.cached_per_sample_gradient = self.cached_per_sample_gradient.to(
                dtype=self.module.score_args.precondition_dtype
            )
            if self.module.gradient_scale != 1.0:
                self.cached_per_sample_gradient.mul_(self.module.gradient_scale)
            self._compute_self_score(per_sample_gradient=self.cached_per_sample_gradient)
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if self-influence score is available."""
        return self.module.storage[SELF_SCORE_VECTOR_NAME] is not None

    def accumulate_iterations(self) -> None:
        """Removes self-influence scores from memory after a single iteration."""
        self.release_memory()

    def release_memory(self) -> None:
        """Releases self-influence scores from memory."""
        self.clear_all_cache()
        if self.storage_at_device:
            move_storage_to_device(storage=self.module.storage, target_device=torch.device("cpu"))
        self.storage_at_device = False
        del self.module.storage[SELF_SCORE_VECTOR_NAME]
        self.module.storage[SELF_SCORE_VECTOR_NAME] = None


class SelfScoreWithMeasurementTracker(BaseTracker):
    """Computes self-influence scores with measurement for a given module."""

    storage_at_device: bool = False

    def _compute_self_measurement_score_with_gradient(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes self-influence scores with measurement using per-sample-gradients.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample-gradient tensor for the given batch.
        """
        scores = per_sample_gradient.mul_(self.module.storage[PRECONDITIONED_GRADIENT_NAME]).sum(dim=(1, 2))
        self.module.storage[PRECONDITIONED_GRADIENT_NAME] = None
        if self.module.storage[SELF_SCORE_VECTOR_NAME] is None:
            self.module.storage[SELF_SCORE_VECTOR_NAME] = scores
        else:
            self.module.storage[SELF_SCORE_VECTOR_NAME].add_(scores)

    def register_hooks(self) -> None:
        """Sets up hooks to compute self-influence scores with measurement."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach()
            device = "cpu" if self.module.score_args.offload_activations_to_cpu else cached_activation.device
            cached_activation = cached_activation.to(
                device=device,
                dtype=self.module.score_args.score_dtype,
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

            if not self.storage_at_device:
                move_storage_to_device(
                    storage=self.module.storage,
                    target_device=output_gradient.device,
                )
                self.storage_at_device = True

            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.score_args.score_dtype)
            if isinstance(self.cached_activations, list):
                cached_activation = self.cached_activations.pop()
            else:
                cached_activation = self.cached_activations
            if self.module.per_sample_gradient_process_fnc is None:
                scores = self.module.compute_self_measurement_score(
                    preconditioned_gradient=self.module.storage[PRECONDITIONED_GRADIENT_NAME],
                    input_activation=cached_activation.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
                self.module.storage[PRECONDITIONED_GRADIENT_NAME] = None
                self.clear_all_cache()
                if self.module.gradient_scale != 1.0:
                    scores.mul_(self.module.gradient_scale)
                if self.module.storage[SELF_SCORE_VECTOR_NAME] is None:
                    self.module.storage[SELF_SCORE_VECTOR_NAME] = scores
                else:
                    self.module.storage[SELF_SCORE_VECTOR_NAME].add_(scores)
            else:
                per_sample_gradient = self.module.compute_per_sample_gradient(
                    input_activation=cached_activation.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
                del cached_activation, output_gradient
                if self.module.gradient_scale != 1.0:
                    per_sample_gradient.mul_(self.module.gradient_scale)
                self._compute_self_measurement_score_with_gradient(per_sample_gradient=per_sample_gradient)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    def finalize_iteration(self) -> None:
        """Clears all cached data from memory."""
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if self-influence score is available."""
        return self.module.storage[SELF_SCORE_VECTOR_NAME] is not None

    def accumulate_iterations(self) -> None:
        """Removes self-influence scores from memory after a single iteration."""
        self.release_memory()

    def release_memory(self) -> None:
        """Releases self-influence scores from memory."""
        self.clear_all_cache()
        if self.storage_at_device:
            move_storage_to_device(storage=self.module.storage, target_device=torch.device("cpu"))
        self.storage_at_device = False
        del self.module.storage[SELF_SCORE_VECTOR_NAME]
        self.module.storage[SELF_SCORE_VECTOR_NAME] = None
