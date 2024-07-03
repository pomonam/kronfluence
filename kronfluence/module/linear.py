from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from opt_einsum import DynamicProgramming, contract, contract_expression
from torch import nn

from kronfluence.factor.config import FactorConfig
from kronfluence.module.tracked_module import TrackedModule
from kronfluence.utils.constants import (
    ACCUMULATED_PRECONDITIONED_GRADIENT_NAME,
    PAIRWISE_SCORE_MATRIX_NAME,
    SELF_SCORE_VECTOR_NAME,
)


class TrackedLinear(TrackedModule, module_type=nn.Linear):
    """A tracking wrapper for `nn.Linear` modules."""

    @property
    def in_features(self) -> int:  # pylint: disable=missing-function-docstring
        return self.original_module.in_features

    @property
    def out_features(self) -> int:  # pylint: disable=missing-function-docstring
        return self.original_module.out_features

    @property
    def weight(self) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return self.original_module.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:  # pylint: disable=missing-function-docstring
        return self.original_module.bias

    def _get_flattened_activation(
        self, input_activation: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        flattened_activation = rearrange(tensor=input_activation, pattern="b ... d_in -> (b ...) d_in")

        flattened_attention_mask = None
        if self._attention_mask is not None and flattened_activation.size(0) == self._attention_mask.numel():
            # If the binary attention mask is provided, zero-out appropriate activations.
            flattened_attention_mask = rearrange(tensor=self._attention_mask, pattern="b ... -> (b ...) 1")
            flattened_activation.mul_(flattened_attention_mask)

        if self.original_module.bias is not None:
            append_term = flattened_activation.new_ones((flattened_activation.size(0), 1), requires_grad=False)
            if flattened_attention_mask is not None:
                append_term.mul_(flattened_attention_mask)
            flattened_activation = torch.cat([flattened_activation, append_term], dim=-1)

        count = flattened_activation.size(0) if flattened_attention_mask is None else flattened_attention_mask.sum()
        return flattened_activation, count

    def _get_flattened_gradient(self, output_gradient: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        flattened_gradient = rearrange(tensor=output_gradient, pattern="b ... d_out -> (b ...) d_out")
        if self._attention_mask is not None and flattened_gradient.size(0) == self._attention_mask.numel():
            count = self._attention_mask.sum()
        else:
            count = flattened_gradient.size(0)
        return flattened_gradient, count

    @torch.no_grad()
    def _compute_per_sample_gradient(
        self, input_activation: torch.Tensor, output_gradient: torch.Tensor
    ) -> torch.Tensor:
        if self.original_module.bias is not None:
            shape = list(input_activation.size()[:-1]) + [1]
            append_term = input_activation.new_ones(shape, requires_grad=False)
            input_activation = torch.cat([input_activation, append_term], dim=-1)
        return torch.einsum("b...i,b...o->bio", output_gradient, input_activation)

    @torch.no_grad()
    def _compute_pairwise_score(self, input_activation: torch.Tensor, output_gradient: torch.Tensor) -> None:
        if self.original_module.bias is not None:
            shape = list(input_activation.size()[:-1]) + [1]
            append_term = input_activation.new_ones(shape, requires_grad=False)
            input_activation = torch.cat([input_activation, append_term], dim=-1)

        if isinstance(self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME], list):
            left_mat, right_mat = self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME]

            if self._opt_einsum_expression is None:
                if self.score_args.compute_per_token_scores and len(input_activation.shape) == 3:
                    self._opt_einsum_expression = contract_expression(
                        "qik,qko,bti,bto->qbt",
                        left_mat.shape,
                        right_mat.shape,
                        output_gradient.shape,
                        input_activation.shape,
                        optimize=DynamicProgramming(
                            search_outer=True, minimize="size" if self.score_args.einsum_minimize_size else "flops"
                        ),
                    )
                else:
                    self._opt_einsum_expression = contract_expression(
                        "qik,qko,b...i,b...o->qb",
                        left_mat.shape,
                        right_mat.shape,
                        output_gradient.shape,
                        input_activation.shape,
                        optimize=DynamicProgramming(
                            search_outer=True, minimize="size" if self.score_args.einsum_minimize_size else "flops"
                        ),
                    )
            self._storage[PAIRWISE_SCORE_MATRIX_NAME] = self._opt_einsum_expression(
                left_mat, right_mat, output_gradient, input_activation
            )

        else:
            if self._opt_einsum_expression is None:
                if self.score_args.compute_per_token_scores and len(input_activation.shape) == 3:
                    self._opt_einsum_expression = contract_expression(
                        "qio,bti,bto->qbt",
                        self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME].shape,
                        output_gradient.shape,
                        input_activation.shape,
                        optimize=DynamicProgramming(
                            search_outer=True, minimize="size" if self.score_args.einsum_minimize_size else "flops"
                        ),
                    )
                else:
                    self._opt_einsum_expression = contract_expression(
                        "qio,b...i,b...o->qb",
                        self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME].shape,
                        output_gradient.shape,
                        input_activation.shape,
                        optimize=DynamicProgramming(
                            search_outer=True, minimize="size" if self.score_args.einsum_minimize_size else "flops"
                        ),
                    )
            self._storage[PAIRWISE_SCORE_MATRIX_NAME] = self._opt_einsum_expression(
                self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME], output_gradient, input_activation
            )
