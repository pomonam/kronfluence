from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from opt_einsum import DynamicProgramming, contract_path
from torch import _VF, nn

from kronfluence.module.tracked_module import TrackedModule


class TrackedLinear(TrackedModule, module_type=nn.Linear):
    """A wrapper for `nn.Linear` modules."""

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

    def get_flattened_activation(self, input_activation: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        flattened_activation = rearrange(tensor=input_activation, pattern="b ... d_in -> (b ...) d_in")

        flattened_attention_mask = None
        if self.attention_mask is not None and flattened_activation.size(0) == self.attention_mask.numel():
            # If the binary attention mask is provided, zero-out appropriate activations.
            flattened_attention_mask = rearrange(tensor=self.attention_mask, pattern="b ... -> (b ...) 1")
            flattened_activation.mul_(flattened_attention_mask)

        if self.original_module.bias is not None:
            append_term = flattened_activation.new_ones((flattened_activation.size(0), 1), requires_grad=False)
            if flattened_attention_mask is not None:
                append_term.mul_(flattened_attention_mask)
            flattened_activation = torch.cat([flattened_activation, append_term], dim=-1)

        count = flattened_activation.size(0) if flattened_attention_mask is None else flattened_attention_mask.sum()
        return flattened_activation, count

    def get_flattened_gradient(self, output_gradient: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        flattened_gradient = rearrange(tensor=output_gradient, pattern="b ... d_out -> (b ...) d_out")
        if self.attention_mask is not None and flattened_gradient.size(0) == self.attention_mask.numel():
            count = self.attention_mask.sum()
        else:
            count = flattened_gradient.size(0)
        return flattened_gradient, count

    def _flatten_input_activation(self, input_activation: torch.Tensor) -> torch.Tensor:
        if self.original_module.bias is not None:
            shape = list(input_activation.size()[:-1]) + [1]
            append_term = input_activation.new_ones(shape, requires_grad=False)
            input_activation = torch.cat([input_activation, append_term], dim=-1)
        return input_activation

    def compute_summed_gradient(self, input_activation: torch.Tensor, output_gradient: torch.Tensor) -> torch.Tensor:
        input_activation = self._flatten_input_activation(input_activation=input_activation)
        summed_gradient = torch.einsum("b...i,b...o->io", output_gradient, input_activation).unsqueeze_(dim=0)
        return summed_gradient

    def compute_per_sample_gradient(
        self, input_activation: torch.Tensor, output_gradient: torch.Tensor
    ) -> torch.Tensor:
        input_activation = self._flatten_input_activation(input_activation=input_activation)
        per_sample_gradient = torch.einsum("b...i,b...o->bio", output_gradient, input_activation)
        if self.per_sample_gradient_process_fnc is not None:
            per_sample_gradient = self.per_sample_gradient_process_fnc(
                module_name=self.name, gradient=per_sample_gradient
            )
        return per_sample_gradient

    def compute_pairwise_score(
        self, preconditioned_gradient: torch.Tensor, input_activation: torch.Tensor, output_gradient: torch.Tensor
    ) -> torch.Tensor:
        input_activation = self._flatten_input_activation(input_activation=input_activation)
        if isinstance(preconditioned_gradient, list):
            left_mat, right_mat = preconditioned_gradient
            if self.score_args.compute_per_token_scores and len(input_activation.shape) == 3:
                expr = "qik,qko,bti,bto->qbt"
            else:
                expr = "qik,qko,b...i,b...o->qb"
            if self.einsum_path is None:
                path = contract_path(
                    expr,
                    left_mat,
                    right_mat,
                    output_gradient,
                    input_activation,
                    optimize=DynamicProgramming(search_outer=True, minimize="flops"),
                )[0]
                self.einsum_path = [item for pair in path for item in pair]
            return _VF.einsum(expr, (left_mat, right_mat, output_gradient, input_activation), path=self.einsum_path)  # pylint: disable=no-member
        if self.score_args.compute_per_token_scores and len(input_activation.shape) == 3:
            expr = "qio,bti,bto->qbt"
            if self.einsum_path is None:
                path = contract_path(
                    expr,
                    preconditioned_gradient,
                    output_gradient,
                    input_activation,
                    optimize=DynamicProgramming(search_outer=True, minimize="flops"),
                )[0]
                self.einsum_path = [item for pair in path for item in pair]
            return _VF.einsum(expr, (preconditioned_gradient, output_gradient, input_activation), path=self.einsum_path)  # pylint: disable=no-member
        expr = "qio,b...i,b...o->qb"
        if self.einsum_path is None:
            path = contract_path(
                expr,
                preconditioned_gradient,
                output_gradient,
                input_activation,
                optimize=DynamicProgramming(search_outer=True, minimize="flops"),
            )[0]
            self.einsum_path = [item for pair in path for item in pair]
        return _VF.einsum(expr, (preconditioned_gradient, output_gradient, input_activation), path=self.einsum_path)  # pylint: disable=no-member

    def compute_self_measurement_score(
        self, preconditioned_gradient: torch.Tensor, input_activation: torch.Tensor, output_gradient: torch.Tensor
    ) -> torch.Tensor:
        input_activation = self._flatten_input_activation(input_activation=input_activation)
        expr = "bio,b...i,b...o->b"
        if self.einsum_path is None:
            path = contract_path(
                expr,
                preconditioned_gradient,
                output_gradient,
                input_activation,
                optimize=DynamicProgramming(search_outer=True, minimize="flops"),
            )[0]
            self.einsum_path = [item for pair in path for item in pair]
        return _VF.einsum(expr, (preconditioned_gradient, output_gradient, input_activation), path=self.einsum_path)  # pylint: disable=no-member
