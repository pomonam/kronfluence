from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einconv.utils import get_conv_paddings
from einops import rearrange, reduce
from opt_einsum import DynamicProgramming, contract_path
from torch import _VF, nn
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
    """A wrapper for `nn.Conv2d` modules."""

    @property
    def in_channels(self) -> int:  # pylint: disable=missing-function-docstring
        return self.original_module.in_channels

    @property
    def out_channels(self) -> int:  # pylint: disable=missing-function-docstring
        return self.original_module.out_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:  # pylint: disable=missing-function-docstring
        return self.original_module.kernel_size

    @property
    def padding(self) -> Tuple[int, int]:  # pylint: disable=missing-function-docstring
        return self.original_module.padding

    @property
    def dilation(self) -> Tuple[int, int]:  # pylint: disable=missing-function-docstring
        return self.original_module.dilation

    @property
    def groups(self) -> int:  # pylint: disable=missing-function-docstring
        return self.original_module.groups

    @property
    def padding_mode(self) -> str:  # pylint: disable=missing-function-docstring
        return self.original_module.padding_mode

    @property
    def weight(self) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return self.original_module.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:  # pylint: disable=missing-function-docstring
        return self.original_module.bias

    def get_flattened_activation(self, input_activation: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
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
        if self.original_module.bias is not None:
            input_activation = torch.cat(
                [
                    input_activation,
                    input_activation.new_ones((input_activation.size(0), 1), requires_grad=False),
                ],
                dim=-1,
            )
        count = input_activation.size(0)
        return input_activation, count

    def get_flattened_gradient(self, output_gradient: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        output_gradient = rearrange(output_gradient, "b c o1 o2 -> (b o1 o2) c")
        return output_gradient, output_gradient.size(0)

    def _flatten_input_activation(self, input_activation: torch.Tensor) -> torch.Tensor:
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
        if self.original_module.bias is not None:
            input_activation = torch.cat(
                [
                    input_activation,
                    input_activation.new_ones((input_activation.size(0), 1), requires_grad=False),
                ],
                dim=-1,
            )
        return input_activation

    def compute_summed_gradient(self, input_activation: torch.Tensor, output_gradient: torch.Tensor) -> torch.Tensor:
        input_activation = self._flatten_input_activation(input_activation=input_activation)
        input_activation = input_activation.view(output_gradient.size(0), -1, input_activation.size(-1))
        output_gradient = rearrange(tensor=output_gradient, pattern="b o i1 i2 -> b (i1 i2) o")
        summed_gradient = torch.einsum("bci,bco->io", output_gradient, input_activation).unsqueeze_(dim=0)
        return summed_gradient

    def compute_per_sample_gradient(
        self,
        input_activation: torch.Tensor,
        output_gradient: torch.Tensor,
    ) -> torch.Tensor:
        input_activation = self._flatten_input_activation(input_activation=input_activation)
        input_activation = input_activation.view(output_gradient.size(0), -1, input_activation.size(-1))
        output_gradient = rearrange(tensor=output_gradient, pattern="b o i1 i2 -> b (i1 i2) o")
        per_sample_gradient = torch.einsum("bci,bco->bio", output_gradient, input_activation)
        if self.per_sample_gradient_process_fnc is not None:
            per_sample_gradient = self.per_sample_gradient_process_fnc(
                module_name=self.name, gradient=per_sample_gradient
            )
        return per_sample_gradient

    def compute_pairwise_score(
        self, preconditioned_gradient: torch.Tensor, input_activation: torch.Tensor, output_gradient: torch.Tensor
    ) -> torch.Tensor:
        input_activation = self._flatten_input_activation(input_activation=input_activation)
        input_activation = input_activation.view(output_gradient.size(0), -1, input_activation.size(-1))
        output_gradient = rearrange(tensor=output_gradient, pattern="b o i1 i2 -> b (i1 i2) o")
        if isinstance(preconditioned_gradient, list):
            left_mat, right_mat = preconditioned_gradient
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
        expr = "qio,bti,bto->qb"
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
        input_activation = input_activation.view(output_gradient.size(0), -1, input_activation.size(-1))
        output_gradient = rearrange(tensor=output_gradient, pattern="b o i1 i2 -> b (i1 i2) o")
        expr = "bio,bci,bco->b"
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
