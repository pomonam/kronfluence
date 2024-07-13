from typing import List, Tuple

import torch
import torch.distributed as dist
from torch import nn

from kronfluence.factor.config import FactorConfig
from kronfluence.module.tracker.base import BaseTracker
from kronfluence.utils.constants import (
    ACCUMULATED_PRECONDITIONED_GRADIENT_NAME,
    AGGREGATED_GRADIENT_NAME,
    PRECONDITIONED_GRADIENT_NAME,
)


class PreconditionTracker(BaseTracker):
    """Computes preconditioned gradient for a given module."""

    def _compute_low_rank_preconditioned_gradient(
        self,
        preconditioned_gradient: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        """Performs low-rank approximation of the preconditioned gradient.

        Args:
            preconditioned_gradient (torch.Tensor):
                The preconditioned per-sample gradient tensor to be low-rank approximated.
            target_dtype (torch.dtype):
                The desired dtype for the output.

        Returns:
            List[torch.Tensor, torch.Tensor]:
                Low-rank matrices approximating the original preconditioned gradient.
        """
        rank = self.module.score_args.query_gradient_low_rank
        if self.module.score_args.use_full_svd:
            U, S, V = torch.linalg.svd(  # pylint: disable=not-callable
                preconditioned_gradient,
                full_matrices=False,
            )
            U_k = U[:, :, :rank]
            S_k = S[:, :rank]
            # Avoid holding the full memory of the original tensor before indexing.
            V_k = V[:, :rank, :].to(dtype=target_dtype, copy=True)
            left_mat = torch.matmul(U_k, torch.diag_embed(S_k)).to(dtype=target_dtype)
            return [left_mat, V_k]

        U, S, V = torch.svd_lowrank(preconditioned_gradient, q=rank)
        left_mat = torch.matmul(U, torch.diag_embed(S)).to(dtype=target_dtype)
        V = V.transpose(1, 2).to(dtype=target_dtype)
        return [left_mat, V]

    def _process_preconditioned_gradient(self, preconditioned_gradient: torch.Tensor) -> None:
        """Processes the preconditioned per-sample gradient.

        Args:
            preconditioned_gradient (torch.Tensor):
                The preconditioned per-sample gradient tensor for the given batch.
        """
        if (
            self.module.score_args.query_gradient_low_rank is not None
            and min(preconditioned_gradient.size()[1:]) > self.module.score_args.query_gradient_low_rank
        ):
            # Apply low-rank approximation to the preconditioned gradient.
            preconditioned_gradient = preconditioned_gradient.to(
                dtype=self.module.score_args.query_gradient_svd_dtype
            ).contiguous()
            preconditioned_gradient = self._compute_low_rank_preconditioned_gradient(
                preconditioned_gradient=preconditioned_gradient,
                target_dtype=self.module.score_args.score_dtype,
            )
        else:
            preconditioned_gradient = preconditioned_gradient.to(dtype=self.module.score_args.score_dtype)
        self.module.storage[PRECONDITIONED_GRADIENT_NAME] = preconditioned_gradient

    def register_hooks(self) -> None:
        """Sets up hooks to compute preconditioned per-sample gradient."""

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
            # Computes preconditioned per-sample gradient during backward pass.
            preconditioned_gradient = FactorConfig.CONFIGS[self.module.factor_args.strategy].precondition_gradient(
                gradient=per_sample_gradient,
                storage=self.module.storage,
            )
            if self.module.gradient_scale != 1.0:
                preconditioned_gradient.mul_(self.module.gradient_scale)
            del per_sample_gradient
            self._process_preconditioned_gradient(preconditioned_gradient=preconditioned_gradient)

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
            # Aggregates per-sample gradients during backward pass.
            self.cached_per_sample_gradient.add_(per_sample_gradient)

        self.registered_hooks.append(self.module.register_forward_hook(forward_hook))

    @torch.no_grad()
    def finalize_iteration(self) -> None:
        """Computes preconditioned gradient using cached per-sample gradients."""
        if self.module.factor_args.has_shared_parameters:
            self.cached_per_sample_gradient = self.cached_per_sample_gradient.to(
                dtype=self.module.score_args.precondition_dtype
            )
            preconditioned_gradient = FactorConfig.CONFIGS[self.module.factor_args.strategy].precondition_gradient(
                gradient=self.cached_per_sample_gradient,
                storage=self.module.storage,
            )
            self.cached_per_sample_gradient = None
            if self.module.gradient_scale != 1.0:
                preconditioned_gradient.mul_(self.module.gradient_scale)
            self._process_preconditioned_gradient(preconditioned_gradient=preconditioned_gradient)
        self.clear_all_cache()

    def exist(self) -> bool:
        """Checks if preconditioned gradient is available."""
        return (
            self.module.storage[PRECONDITIONED_GRADIENT_NAME] is not None
            or self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] is not None
        )

    def synchronize(self, num_processes: int = 1) -> None:
        """Stacks preconditioned gradient across multiple devices or nodes in a distributed setting."""
        if (
            dist.is_initialized()
            and torch.cuda.is_available()
            and self.module.storage[PRECONDITIONED_GRADIENT_NAME] is not None
        ):
            if isinstance(self.module.storage[PRECONDITIONED_GRADIENT_NAME], list):
                for i in range(len(self.module.storage[PRECONDITIONED_GRADIENT_NAME])):
                    size = self.module.storage[PRECONDITIONED_GRADIENT_NAME][i].size()
                    stacked_matrix = torch.empty(
                        size=(num_processes, size[0], size[1], size[2]),
                        dtype=self.module.storage[PRECONDITIONED_GRADIENT_NAME][i].dtype,
                        device=self.module.storage[PRECONDITIONED_GRADIENT_NAME][i].device,
                    )
                    torch.distributed.all_gather_into_tensor(
                        output_tensor=stacked_matrix,
                        input_tensor=self.module.storage[PRECONDITIONED_GRADIENT_NAME][i].contiguous(),
                    )
                    self.module.storage[PRECONDITIONED_GRADIENT_NAME][i] = stacked_matrix.transpose(0, 1).reshape(
                        num_processes * size[0], size[1], size[2]
                    )
            else:
                size = self.module.storage[PRECONDITIONED_GRADIENT_NAME].size()
                stacked_preconditioned_gradient = torch.empty(
                    size=(num_processes, size[0], size[1], size[2]),
                    dtype=self.module.storage[PRECONDITIONED_GRADIENT_NAME].dtype,
                    device=self.module.storage[PRECONDITIONED_GRADIENT_NAME].device,
                )
                torch.distributed.all_gather_into_tensor(
                    output_tensor=stacked_preconditioned_gradient,
                    input_tensor=self.module.storage[PRECONDITIONED_GRADIENT_NAME].contiguous(),
                )
                self.module.storage[PRECONDITIONED_GRADIENT_NAME] = stacked_preconditioned_gradient.transpose(
                    0, 1
                ).reshape(num_processes * size[0], size[1], size[2])

    def truncate(self, keep_size: int) -> None:
        """Truncates preconditioned gradient to appropriate dimension."""
        if isinstance(self.module.storage[PRECONDITIONED_GRADIENT_NAME], list):
            assert len(self.module.storage[PRECONDITIONED_GRADIENT_NAME]) == 2
            self.module.storage[PRECONDITIONED_GRADIENT_NAME] = [
                self.module.storage[PRECONDITIONED_GRADIENT_NAME][0][:keep_size].clone(),
                self.module.storage[PRECONDITIONED_GRADIENT_NAME][1][:keep_size].clone(),
            ]
        else:
            self.module.storage[PRECONDITIONED_GRADIENT_NAME] = self.module.storage[PRECONDITIONED_GRADIENT_NAME][
                :keep_size
            ].clone()

    def accumulate_iterations(self) -> None:
        """Accumulates preconditioned gradient across multiple iterations."""
        accumulated_gradient = self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME]
        gradient = self.module.storage[PRECONDITIONED_GRADIENT_NAME]

        if self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] is None:
            if isinstance(self.module.storage[PRECONDITIONED_GRADIENT_NAME], list):
                self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = [
                    tensor.contiguous() for tensor in gradient
                ]
            else:
                self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = gradient.contiguous()

        else:
            if isinstance(self.module.storage[PRECONDITIONED_GRADIENT_NAME], list):
                self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = [
                    torch.cat((accumulated_gradient[0], gradient[0]), dim=0).contiguous(),
                    torch.cat((accumulated_gradient[1], gradient[1]), dim=0).contiguous(),
                ]
            else:
                self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = torch.cat(
                    (accumulated_gradient, gradient), dim=0
                ).contiguous()
        del self.module.storage[PRECONDITIONED_GRADIENT_NAME], gradient
        self.module.storage[PRECONDITIONED_GRADIENT_NAME] = None

    @torch.no_grad()
    def finalize_all_iterations(self) -> None:
        """Preconditions aggregated gradient if it exists in storage."""
        if self.module.storage[AGGREGATED_GRADIENT_NAME] is not None:
            self.module.storage[AGGREGATED_GRADIENT_NAME] = self.module.storage[AGGREGATED_GRADIENT_NAME].to(
                dtype=self.module.score_args.precondition_dtype
            )
            preconditioned_gradient = FactorConfig.CONFIGS[self.module.factor_args.strategy].precondition_gradient(
                gradient=self.module.storage[AGGREGATED_GRADIENT_NAME],
                storage=self.module.storage,
            )
            self.module.storage[AGGREGATED_GRADIENT_NAME] = None
            self._process_preconditioned_gradient(preconditioned_gradient=preconditioned_gradient)
            self.accumulate_iterations()

    def release_memory(self) -> None:
        """Clears preconditioned gradients from memory."""
        self.module.storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = None
        self.module.storage[PRECONDITIONED_GRADIENT_NAME] = None
        self.clear_all_cache()
