from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
from accelerate.utils.dataclasses import BaseEnum
from opt_einsum import contract
from torch import nn
from torch.utils.hooks import RemovableHandle

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.factor.config import FactorConfig
from kronfluence.utils.constants import (
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    ACTIVATION_EIGENVECTORS_NAME,
    COVARIANCE_FACTOR_NAMES,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_FACTOR_NAMES,
    LAMBDA_MATRIX_NAME,
    NUM_COVARIANCE_PROCESSED,
    NUM_LAMBDA_PROCESSED,
    PAIRWISE_SCORE_MATRIX_NAME,
    PRECONDITIONED_GRADIENT_NAME,
    SELF_SCORE_VECTOR_NAME,
)
from kronfluence.utils.exceptions import FactorsNotFoundError


class ModuleMode(str, BaseEnum):
    """Enum to represent a module's mode, indicating which factors need to be computed
    during forward and backward passes."""

    DEFAULT = "default"
    COVARIANCE = "covariance"
    LAMBDA = "lambda"
    PRECONDITION_GRADIENT = "precondition_gradient"
    PAIRWISE_SCORE = "pairwise_score"
    SELF_SCORE = "self_score"


@torch.no_grad()
def full_backward_gradient_removal_hook(
    module: nn.Module,
    grad_inputs: Tuple[torch.Tensor],
    grad_outputs: Tuple[torch.Tensor],
) -> None:
    """Removes all saved `.grad` computed by Autograd from model's parameters."""
    del grad_inputs, grad_outputs
    for parameter in module.parameters():
        parameter.grad = None


class TrackedModule(nn.Module):
    """A wrapper class for PyTorch modules to compute preconditioning factors and influence scores."""

    SUPPORTED_MODULES: Dict[Type[nn.Module], Any] = {}

    def __init_subclass__(cls, module_type: Type[nn.Module] = None, **kwargs) -> None:
        """Automatically registers subclasses as supported modules."""
        super().__init_subclass__(**kwargs)
        if module_type is not None:
            cls.SUPPORTED_MODULES[module_type] = cls

    def __init__(
        self,
        name: str,
        original_module: nn.Module,
        factor_args: Optional[FactorArguments] = None,
        score_args: Optional[ScoreArguments] = None,
    ) -> None:
        """Initializes an instance of the TrackedModule class.

        Args:
            name (str):
                The original name of the module.
            original_module (nn.Module):
                The original module that will be wrapped with this module.
            factor_args (FactorArguments, optional):
                Arguments related to computing the influence factors.
            score_args (ScoreArguments, optional):
                Arguments related to computing the influence scores.
        """
        super().__init__()

        self.name = name
        self.original_module = original_module

        if factor_args is None:
            factor_args = FactorArguments()
        self.factor_args = factor_args
        if score_args is None:
            score_args = ScoreArguments()
        self.score_args = score_args

        self._mode: ModuleMode = ModuleMode.DEFAULT
        self._cached_activations: List[torch.Tensor] = []
        self._cached_per_sample_gradient: Optional[torch.Tensor] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._registered_hooks: List[RemovableHandle] = []
        self._cached_hooks: List[RemovableHandle] = []
        self._storage: Dict[str, Optional[Any]] = {}
        self._storge_at_current_device: bool = False

        # Storage for activation and pseudo-gradient covariance matrices. #
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            self._storage[covariance_factor_name] = None

        # Storage for eigenvectors and eigenvalues. #
        for eigen_factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
            self._storage[eigen_factor_name] = None

        # Storage for lambda matrices. #
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            self._storage[lambda_factor_name] = None

        # Storage for preconditioned query gradients and influence scores. #
        self._storage[PRECONDITIONED_GRADIENT_NAME] = None
        self._storage[PAIRWISE_SCORE_MATRIX_NAME] = None
        self._storage[SELF_SCORE_VECTOR_NAME] = None

    def update_factor_args(self, factor_args: FactorArguments) -> None:
        """Updates the factor arguments."""
        self.factor_args = factor_args

    def update_score_args(self, score_args: ScoreArguments) -> None:
        """Updates the score arguments."""
        self.score_args = score_args

    def get_factor(self, factor_name: str) -> Optional[torch.Tensor]:
        """Returns the factor with the given name."""
        if factor_name not in self._storage:
            return None
        return self._storage[factor_name]

    def set_factor(self, factor_name: str, factor: Any) -> None:
        """Sets the factor with the given name."""
        if factor_name in self._storage:
            self._storage[factor_name] = factor

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> Any:
        """A forward pass of the tracked module. This should have identical behavior to
        the original module."""
        return self.original_module(inputs, *args, **kwargs)

    def set_mode(self, mode: ModuleMode, keep_factors: bool = True) -> None:
        """Sets the module mode of all `TrackedModule` instances within a model."""
        current_mode = self._mode
        self.remove_registered_hooks()

        if current_mode == ModuleMode.COVARIANCE and not keep_factors:
            self._release_covariance_matrices()

        if current_mode == ModuleMode.LAMBDA and not keep_factors:
            self._release_eigendecomposition_results()
            self._release_lambda_matrix()

        if (
            current_mode
            in [
                ModuleMode.PRECONDITION_GRADIENT,
                ModuleMode.PAIRWISE_SCORE,
                ModuleMode.SELF_SCORE,
            ]
            and not keep_factors
        ):
            self._release_preconditioned_gradient()
            self.release_scores()

        if mode == ModuleMode.DEFAULT and not keep_factors:
            # Releases all factors when the mode is set to default.
            self.remove_attention_mask()
            self._release_covariance_matrices()
            self._release_eigendecomposition_results()
            self._release_lambda_matrix()
            self._release_preconditioned_gradient()
            self.release_scores()

        if mode == ModuleMode.COVARIANCE:
            self._register_covariance_hooks()

        if mode == ModuleMode.LAMBDA:
            self._register_lambda_hooks()

        if mode == ModuleMode.PRECONDITION_GRADIENT:
            self._register_precondition_gradient_hooks()

        if mode == ModuleMode.PAIRWISE_SCORE:
            self._register_pairwise_score_hooks()

        if mode == ModuleMode.SELF_SCORE:
            self._register_self_score_hooks()

        self._mode = mode

    def remove_registered_hooks(self) -> None:
        """Removes all registered hooks within the module."""
        while self._registered_hooks:
            handle = self._registered_hooks.pop()
            handle.remove()
        self._registered_hooks = []
        while self._cached_hooks:
            handle = self._cached_hooks.pop()
            handle.remove()
        self._cached_hooks = []

    ##############################################
    # Methods for computing covariance matrices. #
    ##############################################
    def set_attention_mask(self, attention_mask: Optional[torch.Tensor] = None) -> None:
        """Sets the attention mask for the activation covariance computation."""
        self._attention_mask = attention_mask

    def remove_attention_mask(self) -> None:
        """Removes the currently set attention mask."""
        self._attention_mask = None

    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement the `_get_flattened_activation` method.")

    @torch.no_grad()
    def _update_activation_covariance_matrix(self, input_activation: torch.Tensor) -> None:
        """Updates the activation covariance matrix.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.
        """
        input_activation = input_activation.to(dtype=self.factor_args.activation_covariance_dtype)
        flattened_activation, count = self._get_flattened_activation(input_activation)

        if self._storage[ACTIVATION_COVARIANCE_MATRIX_NAME] is None:
            dimension = flattened_activation.size(1)
            self._storage[ACTIVATION_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=flattened_activation.dtype,
                device=flattened_activation.device,
                requires_grad=False,
            )
        # Add the current batch's activation covariance to the stored activation covariance matrix.
        self._storage[ACTIVATION_COVARIANCE_MATRIX_NAME].addmm_(flattened_activation.t(), flattened_activation)

        if self._storage[NUM_COVARIANCE_PROCESSED] is None:
            device = None
            if isinstance(count, torch.Tensor):
                # When using attention masks, `count` can be tensor.
                device = count.device
            self._storage[NUM_COVARIANCE_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                device=device,
                requires_grad=False,
            )
        # Keep track of total number of elements used to aggregate covariance matrices.
        self._storage[NUM_COVARIANCE_PROCESSED].add_(count)

    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement the `_get_flattened_gradient` method.")

    @torch.no_grad()
    def _update_gradient_covariance_matrix(self, output_gradient: torch.Tensor) -> None:
        """Updates the pseudo-gradient covariance matrix.

        Args:
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the
                PyTorch's backward hook.
        """
        output_gradient = output_gradient.to(dtype=self.factor_args.gradient_covariance_dtype)
        flattened_gradient = self._get_flattened_gradient(output_gradient)

        if self._storage[GRADIENT_COVARIANCE_MATRIX_NAME] is None:
            # Initialize pseudo-gradient covariance matrix if it does not exist.
            dimension = flattened_gradient.size(1)
            self._storage[GRADIENT_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=flattened_gradient.dtype,
                device=flattened_gradient.device,
                requires_grad=False,
            )
        # Add the current batch's pseudo-gradient covariance to the stored pseudo-gradient covariance matrix.
        self._storage[GRADIENT_COVARIANCE_MATRIX_NAME].addmm_(flattened_gradient.t(), flattened_gradient)

    def _register_covariance_hooks(self) -> None:
        """Installs forward and backward hooks for computation of the covariance matrices."""

        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> None:
            del module
            with torch.no_grad():
                # Compute and update activation covariance matrix in the forward pass.
                self._update_activation_covariance_matrix(inputs[0].detach())
            # Register backward hook to obtain gradient with respect to the output.
            self._cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self._cached_hooks.pop()
            handle.remove()
            # Compute and update pseudo-gradient covariance matrix in the backward pass.
            self._update_gradient_covariance_matrix(output_gradient.detach())

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

        if self.factor_args.immediate_gradient_removal:
            self._registered_hooks.append(
                self.original_module.register_full_backward_hook(full_backward_gradient_removal_hook)
            )

    def _release_covariance_matrices(self) -> None:
        """Clears the stored activation and pseudo-gradient covariance matrices from memory."""
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            del self._storage[covariance_factor_name]
            self._storage[covariance_factor_name] = None

    def _covariance_matrices_available(self) -> bool:
        """Checks if the covariance matrices are currently stored in the storage."""
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            if self._storage[covariance_factor_name] is None:
                return False
        return True

    @torch.no_grad()
    def synchronize_covariance_matrices(self) -> None:
        """Aggregates covariance matrices across multiple devices or nodes in a distributed setting."""
        if dist.is_initialized() and torch.cuda.is_available() and self._covariance_matrices_available():
            # Note that only the main process holds the aggregated covariance matrix.
            for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
                self._storage[covariance_factor_name] = self._storage[covariance_factor_name].cuda()
                dist.reduce(
                    tensor=self._storage[covariance_factor_name],
                    op=dist.ReduceOp.SUM,
                    dst=0,
                )

    ##########################################
    # Methods for computing Lambda matrices. #
    ##########################################
    def _release_eigendecomposition_results(self) -> None:
        """Clears the stored eigenvectors and eigenvalues from memory."""
        for eigen_factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
            del self._storage[eigen_factor_name]
            self._storage[eigen_factor_name] = None

    def _eigendecomposition_results_available(self) -> bool:
        """Checks if the eigendecomposition results are currently stored in storage."""
        for eigen_factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
            if self._storage[eigen_factor_name] is None:
                return False
        return True

    @abstractmethod
    def _compute_per_sample_gradient(
        self, input_activation: torch.Tensor, output_gradient: torch.Tensor
    ) -> torch.Tensor:
        """Returns the flattened per-sample-gradient tensor. For the brief introduction to
        per-sample-gradients, see https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html.

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
        raise NotImplementedError("Subclasses must implement the `_compute_per_sample_gradient` method.")

    @torch.no_grad()
    def _update_lambda_matrix(self, per_sample_gradient: torch.Tensor) -> None:
        """Updates the Lambda matrix using the per-sample-gradient.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample-gradient tensor for the given batch.
        """
        batch_size = per_sample_gradient.size(0)

        if self._storage[LAMBDA_MATRIX_NAME] is None:
            # Initialize Lambda matrix if it does not exist.
            self._storage[LAMBDA_MATRIX_NAME] = torch.zeros(
                size=(per_sample_gradient.size(1), per_sample_gradient.size(2)),
                dtype=per_sample_gradient.dtype,
                device=per_sample_gradient.device,
                requires_grad=False,
            )

            if FactorConfig.CONFIGS[self.factor_args.strategy].requires_eigendecomposition_for_lambda:
                if not self._eigendecomposition_results_available():
                    error_msg = (
                        f"The strategy {self.factor_args.strategy} requires Eigendecomposition "
                        f"results to be loaded for Lambda computations. However, Eigendecomposition "
                        f"results are not found."
                    )
                    raise FactorsNotFoundError(error_msg)

                # Move activation and pseudo-gradient eigenvectors to appropriate devices.
                self._storage[ACTIVATION_EIGENVECTORS_NAME] = self._storage[ACTIVATION_EIGENVECTORS_NAME].to(
                    dtype=self.factor_args.lambda_dtype,
                    device=per_sample_gradient.device,
                )
                self._storage[GRADIENT_EIGENVECTORS_NAME] = self._storage[GRADIENT_EIGENVECTORS_NAME].to(
                    dtype=self.factor_args.lambda_dtype,
                    device=per_sample_gradient.device,
                )

        if self._storage[NUM_LAMBDA_PROCESSED] is None:
            self._storage[NUM_LAMBDA_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                requires_grad=False,
            )

        if FactorConfig.CONFIGS[self.factor_args.strategy].requires_eigendecomposition_for_lambda:
            if self.factor_args.lambda_iterative_aggregate:
                # This batch-wise iterative update can be useful when the GPU memory is limited.
                per_sample_gradient = torch.matmul(
                    per_sample_gradient,
                    self._storage[ACTIVATION_EIGENVECTORS_NAME],
                )
                for i in range(batch_size):
                    sqrt_lambda = torch.matmul(
                        self._storage[GRADIENT_EIGENVECTORS_NAME].t(),
                        per_sample_gradient[i],
                    )
                    self._storage[LAMBDA_MATRIX_NAME].add_(sqrt_lambda.square_())
            else:
                per_sample_gradient = torch.matmul(
                    self._storage[GRADIENT_EIGENVECTORS_NAME].t(),
                    torch.matmul(per_sample_gradient, self._storage[ACTIVATION_EIGENVECTORS_NAME]),
                )
                self._storage[LAMBDA_MATRIX_NAME].add_(per_sample_gradient.square_().sum(dim=0))
        else:
            # Assume that the eigenbasis is identity.
            self._storage[LAMBDA_MATRIX_NAME].add_(per_sample_gradient.square_().sum(dim=0))

        self._storage[NUM_LAMBDA_PROCESSED].add_(batch_size)

    def _register_lambda_hooks(self) -> None:
        """Installs forward and backward hooks for computation of the Lambda matrices."""

        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> None:
            del module
            with torch.no_grad():
                cached_activation = inputs[0].detach().to(dtype=self.factor_args.lambda_dtype)
                if self.factor_args.cached_activation_cpu_offload:
                    self._cached_activations.append(cached_activation.cpu())
                else:
                    self._cached_activations.append(cached_activation)
            # Register backward hook to obtain gradient with respect to the output.
            self._cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self._cached_hooks.pop()
            handle.remove()
            cached_activation = self._cached_activations.pop()
            if self.factor_args.cached_activation_cpu_offload:
                cached_activation = cached_activation.to(device=output_gradient.device)

            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation,
                output_gradient=output_gradient.detach().to(dtype=self.factor_args.lambda_dtype),
            )
            del cached_activation

            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)
                del per_sample_gradient

            if len(self._cached_activations) == 0:
                self._update_lambda_matrix(per_sample_gradient=self._cached_per_sample_gradient)
                del self._cached_per_sample_gradient
                self._cached_per_sample_gradient = None

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

        if self.factor_args.immediate_gradient_removal:
            self._registered_hooks.append(
                self.original_module.register_full_backward_hook(full_backward_gradient_removal_hook)
            )

    def _release_lambda_matrix(self) -> None:
        """Clears the stored Lambda matrix from memory."""
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            del self._storage[lambda_factor_name]
            self._storage[lambda_factor_name] = None
        self._cached_activations = []
        del self._cached_per_sample_gradient
        self._cached_per_sample_gradient = None

    def _lambda_matrix_available(self) -> bool:
        """Checks if the Lamda matrix is currently stored in storage."""
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            if self._storage[lambda_factor_name] is None:
                return False
        return True

    @torch.no_grad()
    def synchronize_lambda_matrices(self) -> None:
        """Aggregates Lambda matrices across multiple devices or nodes in a distributed setting."""
        if dist.is_initialized() and torch.cuda.is_available() and self._lambda_matrix_available():
            # Note that only the main process holds the aggregated Lambda matrix.
            for lambda_factor_name in LAMBDA_FACTOR_NAMES:
                self._storage[lambda_factor_name] = self._storage[lambda_factor_name].cuda()
                torch.distributed.reduce(
                    tensor=self._storage[lambda_factor_name],
                    op=dist.ReduceOp.SUM,
                    dst=0,
                )

    ##################################################
    # Methods for computing preconditioned gradient. #
    ##################################################
    @torch.no_grad()
    def _compute_low_rank_preconditioned_gradient(
        self,
        preconditioned_gradient: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Performs low-rank approximation of the preconditioned gradient with SVD.

        Args:
            preconditioned_gradient (torch.Tensor):
                The preconditioned per-sample-gradient matrix to be low-rank approximated.

        Returns:
            List[torch.Tensor, torch.Tensor]:
                Low-rank matrices that approximate the original preconditioned gradient.
        """
        U, S, V = torch.linalg.svd(  # pylint: disable=not-callable
            preconditioned_gradient.contiguous().to(dtype=self.score_args.query_gradient_svd_dtype),
            full_matrices=False,
        )
        rank = self.score_args.query_gradient_rank
        U_k = U[:, :, :rank]
        S_k = S[:, :rank]
        # Avoid holding the full memory of the original tensor before indexing.
        V_k = V[:, :rank, :].clone()
        return [
            torch.matmul(U_k, torch.diag_embed(S_k)).to(dtype=self.score_args.score_dtype).contiguous(),
            V_k.to(dtype=self.score_args.score_dtype).contiguous(),
        ]

    def _register_precondition_gradient_hooks(self) -> None:
        """Installs forward and backward hooks for computation of preconditioned per-sample-gradient."""

        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> None:
            del module
            with torch.no_grad():
                cached_activation = inputs[0].detach().to(dtype=self.score_args.per_sample_gradient_dtype)
                if self.score_args.cached_activation_cpu_offload:
                    self._cached_activations.append(cached_activation.cpu())
                else:
                    self._cached_activations.append(cached_activation)
            # Register backward hook to obtain gradient with respect to the output.
            self._cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self._cached_hooks.pop()
            handle.remove()
            cached_activation = self._cached_activations.pop()
            if self.score_args.cached_activation_cpu_offload:
                cached_activation = cached_activation.to(device=output_gradient.device)

            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation,
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            )
            del cached_activation

            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)
                del per_sample_gradient

            if len(self._cached_activations) == 0:
                preconditioned_gradient = FactorConfig.CONFIGS[self.factor_args.strategy].precondition_gradient(
                    gradient=self._cached_per_sample_gradient.to(dtype=self.score_args.precondition_dtype),
                    storage=self._storage,
                    damping=self.score_args.damping,
                )
                del self._cached_per_sample_gradient
                self._cached_per_sample_gradient = None

                if (
                    self.score_args.query_gradient_rank is not None
                    and min(preconditioned_gradient.size()[1:]) > self.score_args.query_gradient_rank
                ):
                    # Apply low-rank approximation to the preconditioned gradient.
                    preconditioned_gradient = self._compute_low_rank_preconditioned_gradient(
                        preconditioned_gradient=preconditioned_gradient
                    )
                    self._storage[PRECONDITIONED_GRADIENT_NAME] = preconditioned_gradient
                else:
                    self._storage[PRECONDITIONED_GRADIENT_NAME] = preconditioned_gradient.to(
                        dtype=self.score_args.score_dtype
                    )

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

        if self.factor_args.immediate_gradient_removal:
            self._registered_hooks.append(
                self.original_module.register_full_backward_hook(full_backward_gradient_removal_hook)
            )

    def _release_preconditioned_gradient(self) -> None:
        """Clears the preconditioned per-sample-gradient from memory."""
        del self._storage[PRECONDITIONED_GRADIENT_NAME]
        self._storage[PRECONDITIONED_GRADIENT_NAME] = None
        self._cached_activations = []
        del self._cached_per_sample_gradient
        self._cached_per_sample_gradient = None

    def get_preconditioned_gradient_batch_size(self) -> Optional[int]:
        """Returns the saved batch dimension for the preconditioned gradient."""
        if self._storage[PRECONDITIONED_GRADIENT_NAME] is not None:
            if isinstance(self._storage[PRECONDITIONED_GRADIENT_NAME], list):
                return self._storage[PRECONDITIONED_GRADIENT_NAME][0].size(0)
            return self._storage[PRECONDITIONED_GRADIENT_NAME].size(0)
        return None

    @torch.no_grad()
    def truncate_preconditioned_gradient(self, keep_size: int) -> None:
        """Truncates and keeps only the first keep_size dimension for the preconditioned gradient."""
        if self._storage[PRECONDITIONED_GRADIENT_NAME] is not None:
            if isinstance(self._storage[PRECONDITIONED_GRADIENT_NAME], list):
                assert len(self._storage[PRECONDITIONED_GRADIENT_NAME]) == 2
                self._storage[PRECONDITIONED_GRADIENT_NAME] = [
                    self._storage[PRECONDITIONED_GRADIENT_NAME][0][:keep_size].clone(),
                    self._storage[PRECONDITIONED_GRADIENT_NAME][1][:keep_size].clone(),
                ]
            else:
                self._storage[PRECONDITIONED_GRADIENT_NAME] = self._storage[PRECONDITIONED_GRADIENT_NAME][
                    :keep_size
                ].clone()

    def _preconditioned_gradient_available(self) -> bool:
        """Checks if the preconditioned matrices are currently stored in the storage."""
        return self._storage[PRECONDITIONED_GRADIENT_NAME] is not None

    @torch.no_grad()
    def synchronize_preconditioned_gradient(self, num_processes: int) -> None:
        """Stacks preconditioned gradient across multiple devices or nodes in a distributed setting."""
        if dist.is_initialized() and torch.cuda.is_available() and self._preconditioned_gradient_available():
            if isinstance(self._storage[PRECONDITIONED_GRADIENT_NAME], list):
                assert len(self._storage[PRECONDITIONED_GRADIENT_NAME]) == 2
                for i in range(len(self._storage[PRECONDITIONED_GRADIENT_NAME])):
                    size = self._storage[PRECONDITIONED_GRADIENT_NAME][i].size()
                    stacked_matrix = torch.empty(
                        size=(num_processes, size[0], size[1], size[2]),
                        dtype=self._storage[PRECONDITIONED_GRADIENT_NAME][i].dtype,
                        device=self._storage[PRECONDITIONED_GRADIENT_NAME][i].device,
                    )
                    torch.distributed.all_gather_into_tensor(
                        output_tensor=stacked_matrix,
                        input_tensor=self._storage[PRECONDITIONED_GRADIENT_NAME][i].contiguous(),
                    )
                    self._storage[PRECONDITIONED_GRADIENT_NAME][i] = (
                        stacked_matrix.transpose(0, 1).reshape(num_processes * size[0], size[1], size[2]).contiguous()
                    )
            else:
                size = self._storage[PRECONDITIONED_GRADIENT_NAME].size()
                stacked_preconditioned_gradient = torch.empty(
                    size=(num_processes, size[0], size[1], size[2]),
                    dtype=self._storage[PRECONDITIONED_GRADIENT_NAME].dtype,
                    device=self._storage[PRECONDITIONED_GRADIENT_NAME].device,
                )
                torch.distributed.all_gather_into_tensor(
                    output_tensor=stacked_preconditioned_gradient,
                    input_tensor=self._storage[PRECONDITIONED_GRADIENT_NAME].contiguous(),
                )
                self._storage[PRECONDITIONED_GRADIENT_NAME] = (
                    stacked_preconditioned_gradient.transpose(0, 1)
                    .reshape(num_processes * size[0], size[1], size[2])
                    .contiguous()
                )

    ###########################################
    # Methods for computing influence scores. #
    ###########################################
    def _register_pairwise_score_hooks(self) -> None:
        """Installs forward and backward hooks for computation of pairwise influence scores."""

        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> None:
            del module
            with torch.no_grad():
                cached_activation = inputs[0].detach().to(dtype=self.score_args.per_sample_gradient_dtype)
                if self.score_args.cached_activation_cpu_offload:
                    self._cached_activations.append(cached_activation.cpu())
                else:
                    self._cached_activations.append(cached_activation)
            # Register backward hook to obtain gradient with respect to the output.
            self._cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self._cached_hooks.pop()
            handle.remove()
            cached_activation = self._cached_activations.pop()
            if self.score_args.cached_activation_cpu_offload:
                cached_activation = cached_activation.to(device=output_gradient.device)

            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation,
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            ).to(dtype=self.score_args.score_dtype)
            del cached_activation

            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)
                del per_sample_gradient

            # If the module was used multiple times throughout the forward pass,
            # only compute scores after aggregating all per-sample-gradients.
            if len(self._cached_activations) == 0:
                if isinstance(self._storage[PRECONDITIONED_GRADIENT_NAME], list):
                    # The preconditioned gradient is stored as a low-rank approximation.
                    left_mat, right_mat = self._storage[PRECONDITIONED_GRADIENT_NAME]

                    input_dim = right_mat.size(2)
                    output_dim = left_mat.size(1)
                    query_batch_size = left_mat.size(0)
                    train_batch_size = self._cached_per_sample_gradient.size(0)
                    rank = self.score_args.query_gradient_rank
                    if (
                        train_batch_size * query_batch_size * rank * min((input_dim, output_dim))
                        > query_batch_size * input_dim * output_dim
                    ):
                        # If reconstructing the gradient is more memory efficient, reconstruct and compute the score.
                        self._storage[PAIRWISE_SCORE_MATRIX_NAME] = contract(
                            "qki,toi,qok->qt",
                            right_mat,
                            self._cached_per_sample_gradient,
                            left_mat,
                        )
                    # Otherwise, try to avoid reconstructing the full per-sample-gradient.
                    elif output_dim >= input_dim:
                        intermediate = contract("toi,qok->qtik", self._cached_per_sample_gradient, left_mat)
                        self._storage[PAIRWISE_SCORE_MATRIX_NAME] = contract("qki,qtik->qt", right_mat, intermediate)
                    else:
                        intermediate = contract("qki,toi->qtko", right_mat, self._cached_per_sample_gradient)
                        self._storage[PAIRWISE_SCORE_MATRIX_NAME] = contract("qtko,qok->qt", intermediate, left_mat)
                else:
                    self._storage[PAIRWISE_SCORE_MATRIX_NAME] = contract(
                        "qio,tio->qt",
                        self._storage[PRECONDITIONED_GRADIENT_NAME],
                        self._cached_per_sample_gradient,
                    )
                del self._cached_per_sample_gradient
                self._cached_per_sample_gradient = None

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

        if self.factor_args.immediate_gradient_removal:
            self._registered_hooks.append(
                self.original_module.register_full_backward_hook(full_backward_gradient_removal_hook)
            )

    def _register_self_score_hooks(self) -> None:
        """Installs forward and backward hooks for computation of self-influence scores."""

        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]) -> None:
            del module
            with torch.no_grad():
                cached_activation = inputs[0].detach().to(dtype=self.score_args.per_sample_gradient_dtype)
                if self.score_args.cached_activation_cpu_offload:
                    self._cached_activations.append(cached_activation.cpu())
                else:
                    self._cached_activations.append(cached_activation)
            # Register backward hook to obtain gradient with respect to the output.
            self._cached_hooks.append(outputs.register_hook(backward_hook))

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            handle = self._cached_hooks.pop()
            handle.remove()
            cached_activation = self._cached_activations.pop()
            if self.score_args.cached_activation_cpu_offload:
                cached_activation = cached_activation.to(device=output_gradient.device)

            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation,
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            )
            del cached_activation

            # The preconditioning factors need to be loaded to appropriate device as they will be
            # used at each iteration.
            if not self._storge_at_current_device:
                for name, factor in self._storage.items():
                    if factor is not None:
                        if isinstance(factor, torch.Tensor):
                            self._storage[name] = factor.to(
                                device=per_sample_gradient.device,
                                dtype=self.score_args.precondition_dtype,
                            )
                        elif isinstance(factor, list):
                            for i in range(len(self._storage[name])):
                                self._storage[name][i] = factor[i].to(
                                    device=per_sample_gradient.device,
                                    dtype=self.score_args.precondition_dtype,
                                )
                        else:
                            raise RuntimeError(f"`{name}` in `TrackedModule` storage does not have a valid type.")
                self._storge_at_current_device = True

            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)
                del per_sample_gradient

            # If the module was used multiple times throughout the forward pass,
            # only compute scores after aggregating all per-sample-gradients.
            if len(self._cached_activations) == 0:
                preconditioned_gradient = (
                    FactorConfig.CONFIGS[self.factor_args.strategy]
                    .precondition_gradient(
                        gradient=self._cached_per_sample_gradient.to(dtype=self.score_args.precondition_dtype),
                        storage=self._storage,
                        damping=self.score_args.damping,
                    )
                    .to(dtype=self.score_args.score_dtype)
                )
                self._cached_per_sample_gradient = self._cached_per_sample_gradient.to(
                    dtype=self.score_args.score_dtype
                )
                preconditioned_gradient.mul_(self._cached_per_sample_gradient)
                self._storage[SELF_SCORE_VECTOR_NAME] = preconditioned_gradient.sum(dim=(1, 2))
                self._cached_per_sample_gradient = None

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

        if self.factor_args.immediate_gradient_removal:
            self._registered_hooks.append(
                self.original_module.register_full_backward_hook(full_backward_gradient_removal_hook)
            )

    def release_scores(self) -> None:
        """Clears the influence scores from memory."""
        del self._storage[PAIRWISE_SCORE_MATRIX_NAME]
        self._storage[PAIRWISE_SCORE_MATRIX_NAME] = None
        del self._storage[SELF_SCORE_VECTOR_NAME]
        self._storage[SELF_SCORE_VECTOR_NAME] = None
        self._cached_activations = []
        del self._cached_per_sample_gradient
        self._cached_per_sample_gradient = None
        self._storge_at_current_device = False
