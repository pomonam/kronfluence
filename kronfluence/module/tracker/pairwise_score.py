from typing import Tuple

import torch
from opt_einsum import DynamicProgramming, contract_path
from torch import _VF, nn

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
        precondition_name = ACCUMULATED_PRECONDITIONED_GRADIENT_NAME
        if isinstance(self.module.storage[precondition_name], list):
            left_mat, right_mat = self.module.storage[precondition_name]
            expr = "qki,toi,qok->qt"
            if self.module.einsum_path is None:
                path = contract_path(
                    expr,
                    right_mat,
                    per_sample_gradient,
                    left_mat,
                    optimize=DynamicProgramming(search_outer=True, minimize="flops"),
                )[0]
                self.module.einsum_path = [item for pair in path for item in pair]
            scores = _VF.einsum(expr, (right_mat, per_sample_gradient, left_mat), path=self.module.einsum_path)  # pylint: disable=no-member
        else:
            scores = torch.einsum(
                "qio,tio->qt",
                self.module.storage[precondition_name],
                per_sample_gradient,
            )

        if self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] is not None:
            self.module.storage[PAIRWISE_SCORE_MATRIX_NAME].add_(scores)
        else:
            self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] = scores

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
            self.cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self.cached_activations is None:
                self._raise_cache_not_found_exception()
            handle = self.cached_hooks.pop()
            handle.remove()
            output_gradient = output_gradient.detach().to(dtype=self.module.score_args.score_dtype)
            if isinstance(self.cached_activations, list):
                cached_activation = self.cached_activations.pop()
            else:
                cached_activation = self.cached_activations
            # Computes pairwise influence scores during backward pass.
            if self.module.per_sample_gradient_process_fnc is None:
                self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] = self.module.compute_pairwise_score(
                    preconditioned_gradient=self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME],
                    input_activation=cached_activation.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
                if self.module.gradient_scale != 1.0:
                    self.module.storage[PAIRWISE_SCORE_MATRIX_NAME].mul_(self.module.gradient_scale)
                del cached_activation, output_gradient
                self.clear_all_cache()
            else:
                per_sample_gradient = self.module.compute_per_sample_gradient(
                    input_activation=cached_activation.to(device=output_gradient.device),
                    output_gradient=output_gradient,
                )
                del cached_activation, output_gradient
                if self.module.gradient_scale != 1.0:
                    per_sample_gradient.mul_(self.module.gradient_scale)
                self._compute_pairwise_score_with_gradient(per_sample_gradient=per_sample_gradient)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    def finalize_iteration(self) -> None:
        """Clears all cached data from memory."""
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if pairwise score is available."""
        return self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] is not None

    def accumulate_iterations(self) -> None:
        """Removes pairwise scores from memory after a single iteration."""
        self.release_memory()

    @torch.no_grad()
    def finalize_all_iterations(self) -> None:
        """Removes cached preconditioned gradient from memory. Additionally, if aggregated gradients are available,
        computes the pairwise score using them."""
        if self.module.storage[AGGREGATED_GRADIENT_NAME] is not None:
            self.module.storage[AGGREGATED_GRADIENT_NAME] = self.module.storage[AGGREGATED_GRADIENT_NAME].to(
                dtype=self.module.score_args.score_dtype
            )
            self._compute_pairwise_score_with_gradient(
                per_sample_gradient=self.module.storage[AGGREGATED_GRADIENT_NAME]
            )
        self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = None
        self.module.storage[PRECONDITIONED_GRADIENT_NAME] = None
        self.clear_all_cache()

    def release_memory(self) -> None:
        """Releases pairwise scores from memory."""
        self.clear_all_cache()
        self.module.storage[PAIRWISE_SCORE_MATRIX_NAME] = None
