from typing import Tuple, Union

import torch
import torch.nn.functional as F
from einconv.utils import get_conv_paddings
from einops import rearrange, reduce
from opt_einsum import contract
from torch import nn
from torch.nn.modules.utils import _pair

from kronfluence.module.tracked_module import TrackedModule
from kronfluence.utils.exceptions import UnsupportableModuleError


def extract_patches(
    inputs: torch.Tensor,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int],
    padding: Union[Tuple[int, int], int, str],
    dilation: Union[Tuple[int, int], int],
    groups: int,
) -> torch.Tensor:
    """Extracts patches from the inputs of the `nn.Conv2d` module. This implementation is
    heavily based on https://github.com/f-dangel/singd.

    Args:
        inputs (torch.Tensor):
            The inputs tensor to the `nn.Conv2d` module.
        kernel_size (tuple, int):
            Size of the convolutional kernel.
        stride (tuple, int):
            Stride of the convolution.
        padding (int, tuple, str):
            Padding added to all four sides of the input.
        dilation (tuple, int):
            Spacing between kernel elements.
        groups (int):
            Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor:
            Extracted patches of shape `batch_size x (O1 * O2) x C_in // groups * K1 * K2`,
            where each column `[b, O1 * O2, :]` contains the flattened patch of sample `b` used
            for output location `(O1, O2)`, averaged over channel groups.
    """
    if isinstance(padding, str):
        padding_as_int = []
        for k, s, d in zip(_pair(kernel_size), _pair(stride), _pair(dilation)):
            p_left, p_right = get_conv_paddings(k, s, padding, d)
            if p_left != p_right:
                raise UnsupportableModuleError("Unequal padding not supported in unfold.")
            padding_as_int.append(p_left)
        padding = tuple(padding_as_int)

    inputs = rearrange(tensor=inputs, pattern="b (g c_in) i1 i2 -> b g c_in i1 i2", g=groups)
    inputs = reduce(tensor=inputs, pattern="b g c_in i1 i2 -> b c_in i1 i2", reduction="mean")
    inputs_unfold = F.unfold(
        input=inputs,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    return rearrange(tensor=inputs_unfold, pattern="b c_in_k1_k2 o1_o2 -> b o1_o2 c_in_k1_k2")


class TrackedConv2d(TrackedModule, module_type=nn.Conv2d):
    """A tracking wrapper for `nn.Conv2D` modules."""

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
        input_activation = extract_patches(
            inputs=input_activation,
            kernel_size=self.original_module.kernel_size,
            stride=self.original_module.stride,
            padding=self.original_module.padding,
            dilation=self.original_module.dilation,
            groups=self.original_module.groups,
        )
        input_activation = rearrange(
            tensor=input_activation,
            pattern="b o1_o2 c_in_k1_k2 -> (b o1_o2) c_in_k1_k2",
        )

        if self.original_module.bias is not None and not self.factor_args.ignore_bias:
            input_activation = torch.cat(
                [
                    input_activation,
                    input_activation.new_ones((input_activation.size(0), 1), requires_grad=False),
                ],
                dim=-1,
            )
        count = input_activation.size(0)
        return input_activation, count

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
        return rearrange(output_gradient, "b c o1 o2 -> (b o1 o2) c")

    def _compute_per_sample_gradient(
        self,
        input_activation: torch.Tensor,
        output_gradient: torch.Tensor,
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
        input_activation = extract_patches(
            inputs=input_activation,
            kernel_size=self.original_module.kernel_size,
            stride=self.original_module.stride,
            padding=self.original_module.padding,
            dilation=self.original_module.dilation,
            groups=self.original_module.groups,
        )
        input_activation = rearrange(
            tensor=input_activation,
            pattern="b o1_o2 c_in_k1_k2 -> (b o1_o2) c_in_k1_k2",
        )

        if self.original_module.bias is not None and not self.factor_args.ignore_bias:
            input_activation = torch.cat(
                [
                    input_activation,
                    input_activation.new_ones((input_activation.size(0), 1), requires_grad=False),
                ],
                dim=-1,
            )
        input_activation = input_activation.view(output_gradient.size(0), -1, input_activation.size(-1))
        output_gradient = rearrange(tensor=output_gradient, pattern="b o i1 i2 -> b (i1 i2) o")
        return contract("abm,abn->amn", output_gradient, input_activation)
