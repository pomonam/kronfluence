from typing import Tuple, Union

import torch
from einops import rearrange
from opt_einsum import contract
from torch import nn

from kronfluence.module.tracked_module import TrackedModule


class TrackedLinear(TrackedModule, module_type=nn.Linear):
    """A tracking wrapper for `nn.Linear` modules."""

    def _get_flattened_activation(
        self, input_activation: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """Returns the flattened activation tensor and the number of stacked activations.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The flattened activation tensor and the number of stacked activations. The flattened
                activation is a 2-dimensional matrix with dimension `activation_num x activation_dim`.
        """
        flattened_activation = rearrange(tensor=input_activation, pattern="b ... d_in -> (b ...) d_in")

        flattened_attention_mask = None
        if self._attention_mask is not None and flattened_activation.size(0) == self._attention_mask.numel():
            # If the binary attention mask is provided, zero-out appropriate activations.
            flattened_attention_mask = rearrange(tensor=self._attention_mask, pattern="b ... -> (b ...) 1")
            # Make sure in-place operation does not change the activation during the forward pass.
            flattened_activation = flattened_activation.clone()
            flattened_activation.mul_(flattened_attention_mask)

        if self.original_module.bias is not None and not self.factor_args.ignore_bias:
            append_term = flattened_activation.new_ones((flattened_activation.size(0), 1), requires_grad=False)
            if flattened_attention_mask is not None:
                append_term.mul_(flattened_attention_mask)
            flattened_activation = torch.cat([flattened_activation, append_term], dim=-1)

        count = flattened_activation.size(0) if flattened_attention_mask is None else flattened_attention_mask.sum()
        return flattened_activation, count

    def _get_flattened_gradient(self, output_gradient: torch.Tensor) -> torch.Tensor:
        """Returns the flattened gradient tensor.

        Args:
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the
                PyTorch's backward hook.

        Returns:
            torch.Tensor:
                The flattened output gradient tensor. The flattened gradient is a 2-dimensional matrix
                with dimension `gradient_num x gradient_dim`.
        """
        return rearrange(tensor=output_gradient, pattern="b ... d_out -> (b ...) d_out")

    def _compute_per_sample_gradient(
        self, input_activation: torch.Tensor, output_gradient: torch.Tensor
    ) -> torch.Tensor:
        """Returns the flattened per-sample-gradient tensor.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the
                PyTorch's backward hook.

        Returns:
            torch.Tensor:
                The per-sample-gradient tensor. The per-sample-gradient is a 3-dimensional matrix
                with dimension `batch_size x gradient_dim x activation_dim`. An additional dimension is added
                when the bias term is used.
        """
        if self.original_module.bias is not None and not self.factor_args.ignore_bias:
            shape = list(input_activation.size()[:-1]) + [1]
            append_term = input_activation.new_ones(shape, requires_grad=False)
            input_activation = torch.cat([input_activation, append_term], dim=-1)
        return contract("b...i,b...o->bio", output_gradient, input_activation)
