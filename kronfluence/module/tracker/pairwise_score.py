from typing import List, Tuple

import torch
import torch.nn as nn
from opt_einsum import DynamicProgramming, contract_expression

from kronfluence.module.tracker.base import BaseTracker
from kronfluence.utils.constants import (
    ACCUMULATED_PRECONDITIONED_GRADIENT_NAME,
    AGGREGATED_GRADIENT_NAME,
    PAIRWISE_SCORE_MATRIX_NAME,
    PRECONDITIONED_GRADIENT_NAME,
)


class PairwiseScoreTracker(BaseTracker):
    """Computes pairwise influence scores for a given module."""

    def _compute_pairwise_score_with_gradient(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes pairwise influence scores using per-sample-gradient.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample-gradient tensor for the given batch.
        """
        if isinstance(self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME], list):
            left_mat, right_mat = self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME]
            if self.module.einsum_expression is None:
                self.module.einsum_expression = contract_expression(
                    "qki,toi,qok->qt",
                    right_mat.shape,
                    per_sample_gradient.shape,
                    left_mat.shape,
                    optimize=DynamicProgramming(
                        search_outer=True, minimize="size" if self.module.score_args.einsum_minimize_size else "flops"
                    ),
                )
            self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] = self.module.einsum_expression(
                right_mat, per_sample_gradient, left_mat
            )
        else:
            self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] = torch.einsum(
                "qio,tio->qt",
                self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME],
                per_sample_gradient,
            )

    def register_hooks(self) -> None:
        """Sets up hooks to compute pairwise influence scores."""

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

            outputs.register_hook(
                shared_backward_hook if self.module.factor_args.has_shared_parameters else backward_hook
            )

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self.cached_activations is None:
                self._raise_cache_not_found_exception()

            output_gradient = self._scale_output_gradient(
                output_gradient=output_gradient, target_dtype=self.module.score_args.score_dtype
            )
            if self.module.per_sample_gradient_process_fnc is None:
                self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] = self.module.compute_pairwise_score(
                    preconditioned_gradient=self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME],
                    input_activation=self.cached_activations.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
                self.clear_all_cache()
            else:
                per_sample_gradient = self.module.compute_per_sample_gradient(
                    input_activation=self.cached_activations.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
                self.clear_all_cache()
                self._compute_pairwise_score_with_gradient(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            output_gradient = self._scale_output_gradient(
                output_gradient=output_gradient, target_dtype=self.module.score_args.score_dtype
            )
            cached_activation = self.cached_activations.pop()
            per_sample_gradient = self.module.compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient,
            )
            if self.cached_per_sample_gradient is None:
                self.cached_per_sample_gradient = torch.zeros_like(per_sample_gradient, requires_grad=False)
            self.cached_per_sample_gradient.add_(per_sample_gradient)

        self.registered_hooks.append(self.module.original_module.register_forward_hook(forward_hook))

    @torch.no_grad()
    def finalize_iteration(self) -> None:
        """Computes pairwise influence scores using cached per-sample gradients."""
        if self.module.factor_args.has_shared_parameters:
            self._compute_pairwise_score_with_gradient(per_sample_gradient=self.cached_per_sample_gradient)
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if pairwise score is available."""
        return self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] is not None

    def accumulate_iterations(self) -> None:
        """Removes pairwise scores from memory after a single iteration."""
        self.release_memory()

    def finalize_all_iterations(self) -> None:
        """Removes cached preconditioned gradient from memory. Additionally, if aggregated gradients are available,
        computes the pairwise score using them."""
        if self.module.storage[AGGREGATED_GRADIENT_NAME] is not None:
            self.module.storage[AGGREGATED_GRADIENT_NAME] = self.module.storage[AGGREGATED_GRADIENT_NAME].to(
                dtype=self.module.score_args.precondition_dtype
            )
            self._compute_pairwise_score_with_gradient(
                per_sample_gradient=self.module.storage[AGGREGATED_GRADIENT_NAME]
            )
        del (
            self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME],
            self.module.storage[PRECONDITIONED_GRADIENT_NAME],
        )
        self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = None
        self.module.storage[PRECONDITIONED_GRADIENT_NAME] = None
        self.clear_all_cache()

    def release_memory(self) -> None:
        """Releases pairwise scores from memory."""
        self.clear_all_cache()
        del self.module.storage[PAIRWISE_SCORE_MATRIX_NAME]
        self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] = None
