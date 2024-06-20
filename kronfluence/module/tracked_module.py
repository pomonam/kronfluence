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
    AGGREGATED_PRECONDITIONED_GRADIENT_NAME,
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
    SELF_MEASUREMENT_SCORE = "self_measurement_score"


def do_nothing(_: Any) -> None:
    """Does not perform any operations."""
    pass


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

        # A way to avoid Autograd computing the gradient with respect to the model parameters.
        self._constant: torch.Tensor = nn.Parameter(
            torch.zeros(
                1,
                requires_grad=True,
                device=self.original_module.weight.device,
                dtype=torch.float16,
            )
        )
        # Operations that will be performed before and after a forward pass.
        self._pre_forward = do_nothing
        self._post_forward = do_nothing
        self._num_forward_passes = torch.zeros(
                1,
                requires_grad=False,
                dtype=torch.int64,
            )
        self._num_backward_passes = torch.zeros(
                1,
                requires_grad=False,
                dtype=torch.int64,
            )

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
        self._gradient_scale: float = 1.0
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
        self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME] = None
        self._storage[PAIRWISE_SCORE_MATRIX_NAME] = None
        self._storage[SELF_SCORE_VECTOR_NAME] = None

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """A forward pass of the tracked module. This should have identical behavior to the original module."""
        with torch.no_grad():
            self._pre_forward(inputs)
        outputs = self.original_module(inputs + self._constant, *args, **kwargs)
        with torch.no_grad():
            self._post_forward(outputs)
        return outputs

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

    def set_gradient_scale(self, scale: float = 1.0) -> None:
        """Sets the scale of the gradient obtained from `GradScaler`."""
        self._gradient_scale = scale

    def remove_gradient_scale(self) -> None:
        """Resets the scale of the gradient."""
        self._gradient_scale = 1.0

    def set_mode(self, mode: ModuleMode, keep_factors: bool = True) -> None:
        """Sets the module mode of all `TrackedModule` instances within a model."""
        self.remove_attention_mask()
        self.remove_gradient_scale()
        self._num_forward_passes = torch.zeros(
                1,
                requires_grad=False,
                dtype=torch.int64,
            )
        self._num_backward_passes = torch.zeros(
                1,
                requires_grad=False,
                dtype=torch.int64,
            )

        if not keep_factors:
            self._release_covariance_matrices()
            self._release_eigendecomposition_results()
            self._release_lambda_matrix()
            self._release_preconditioned_gradient()
            self.release_scores()

        if mode == ModuleMode.DEFAULT:
            self._pre_forward = do_nothing
            self._post_forward = do_nothing

        elif mode == ModuleMode.COVARIANCE:
            self._pre_forward = self._covariance_pre_forward
            self._post_forward = self._covariance_post_forward

        elif mode == ModuleMode.LAMBDA:
            self._pre_forward = self._lambda_pre_forward
            self._post_forward = self._lambda_post_forward

        elif mode == ModuleMode.PRECONDITION_GRADIENT:
            self._pre_forward = self._activation_cache_pre_forward
            self._post_forward = self._precondition_post_forward

        elif mode == ModuleMode.PAIRWISE_SCORE:
            self._pre_forward = self._activation_cache_pre_forward
            self._post_forward = self._pairwise_post_forward

        elif mode == ModuleMode.SELF_SCORE:
            self._pre_forward = self._activation_cache_pre_forward
            self._post_forward = self._self_post_forward

        elif mode == ModuleMode.SELF_MEASUREMENT_SCORE:
            self._pre_forward = self._activation_cache_pre_forward
            self._post_forward = self._self_measurement_post_forward

        else:
            raise RuntimeError(f"Unknown module mode {mode}.")

        self._mode = mode

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
                The input tensor to the module.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The flattened activation tensor and the number of stacked activations. The flattened
                activation is a 2-dimensional matrix with dimension `activation_num x activation_dim`.
        """
        raise NotImplementedError("Subclasses must implement the `_get_flattened_activation` method.")

    def _update_activation_covariance_matrix(self, input_activation: torch.Tensor) -> None:
        """Updates the activation covariance matrix.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module.
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
        # Adds the current batch's activation covariance to the stored activation covariance matrix.
        self._storage[ACTIVATION_COVARIANCE_MATRIX_NAME].addmm_(flattened_activation.t(), flattened_activation)

        if self._storage[NUM_COVARIANCE_PROCESSED] is None:
            device = None
            if isinstance(count, torch.Tensor):
                # When using attention masks, `count` can be a tensor.
                device = count.device
            self._storage[NUM_COVARIANCE_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                device=device,
                requires_grad=False,
            )
        # Keeps track of total number of elements used to aggregate covariance matrices.
        self._storage[NUM_COVARIANCE_PROCESSED].add_(count)
        self._num_forward_passes.add_(1)

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

    def _update_gradient_covariance_matrix(self, output_gradient: torch.Tensor) -> None:
        """Updates the pseudo-gradient covariance matrix.

        Args:
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the
                PyTorch's backward hook.
        """
        output_gradient = output_gradient.to(dtype=self.factor_args.gradient_covariance_dtype)
        flattened_gradient = self._get_flattened_gradient(output_gradient)
        if self._gradient_scale != 1.0:
            # Avoiding in-place operation here.
            flattened_gradient = self._gradient_scale * flattened_gradient

        if self._storage[GRADIENT_COVARIANCE_MATRIX_NAME] is None:
            # Initializes pseudo-gradient covariance matrix if it does not exist.
            dimension = flattened_gradient.size(1)
            self._storage[GRADIENT_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=flattened_gradient.dtype,
                device=flattened_gradient.device,
                requires_grad=False,
            )
        # Adds the current batch's pseudo-gradient covariance to the stored pseudo-gradient covariance matrix.
        self._storage[GRADIENT_COVARIANCE_MATRIX_NAME].addmm_(flattened_gradient.t(), flattened_gradient)
        self._num_backward_passes.add_(1)

    def _covariance_pre_forward(self, inputs: torch.Tensor) -> None:
        """Computes and updates activation covariance matrix in the forward pass."""
        self._update_activation_covariance_matrix(inputs.detach())

    def _covariance_post_forward(self, outputs: torch.Tensor) -> None:
        """Computes and updates pseudo-gradient covariance matrix in the backward pass."""
        def backward_hook(output_gradient: torch.Tensor) -> None:
            self._update_gradient_covariance_matrix(output_gradient.detach())
        # Registers backward hook to obtain gradient with respect to the output.
        outputs.register_hook(backward_hook)

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

    def finalize_covariance_matrices(self) -> None:
        """Rescales the activation covariance matrix if the number of forward and backward passes do not match. This
        could happen when using gradient checkpointing or torch.compile."""
        if self._num_forward_passes == self._num_backward_passes:
            return
        assert self._num_forward_passes % self._num_backward_passes == 0
        mismatch_ratio = self._num_forward_passes // self._num_backward_passes
        self._storage[ACTIVATION_COVARIANCE_MATRIX_NAME].div_(mismatch_ratio)
        self._storage[NUM_COVARIANCE_PROCESSED].div_(mismatch_ratio)

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
        """Returns the flattened per-sample-gradient tensor. For a brief introduction to
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

        if self._gradient_scale != 1.0:
            per_sample_gradient.mul_(self._gradient_scale)

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

    @torch.no_grad()
    def _lambda_pre_forward(self, inputs: Any) -> Any:
        cached_activation = inputs.detach().to(dtype=self.factor_args.lambda_dtype)
        if self.factor_args.cached_activation_cpu_offload:
            self._cached_activations.append(cached_activation.cpu())
        else:
            self._cached_activations.append(cached_activation)

    @torch.no_grad()
    def _lambda_post_forward(self, outputs: torch.Tensor) -> Any:
        def backward_hook(output_gradient: torch.Tensor) -> None:
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

        # Register backward hook to obtain gradient with respect to the output.
        outputs.register_hook(backward_hook)

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
        # Avoid holding the full memory of the original tensor before indexing.
        U_k = U[:, :, :rank]
        S_k = S[:, :rank]
        V_k = V[:, :rank, :].contiguous().clone()
        return [
            torch.matmul(U_k, torch.diag_embed(S_k)).to(dtype=self.score_args.score_dtype).contiguous().clone(),
            V_k.to(dtype=self.score_args.score_dtype),
        ]

    @torch.no_grad()
    def _activation_cache_pre_forward(self, inputs: Any) -> Any:
        cached_activation = inputs.detach().to(dtype=self.score_args.per_sample_gradient_dtype)
        if self.factor_args.cached_activation_cpu_offload:
            self._cached_activations.append(cached_activation.cpu())
        else:
            self._cached_activations.append(cached_activation)

    @torch.no_grad()
    def _precondition_post_forward(self, outputs: torch.Tensor) -> Any:
        def backward_hook(output_gradient: torch.Tensor) -> None:
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
                if self._gradient_scale != 1.0:
                    self._cached_per_sample_gradient.mul_(self._gradient_scale)

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

        # Register backward hook to obtain gradient with respect to the output.
        outputs.register_hook(backward_hook)

    @torch.no_grad()
    def aggregate_preconditioned_gradient(self):
        """Aggregates the preconditioned per-sample-gradients."""
        if self._storage[PRECONDITIONED_GRADIENT_NAME] is None:
            return

        if isinstance(self._storage[PRECONDITIONED_GRADIENT_NAME], list):
            if self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME] is not None:
                self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME] = [
                    torch.cat(
                        (
                            self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME][0],
                            self._storage[PRECONDITIONED_GRADIENT_NAME][0],
                        ),
                        dim=0,
                    ),
                    torch.cat(
                        (
                            self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME][1],
                            self._storage[PRECONDITIONED_GRADIENT_NAME][1],
                        ),
                        dim=0,
                    ),
                ]
            else:
                self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME] = self._storage[PRECONDITIONED_GRADIENT_NAME]
        else:
            if self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME] is not None:
                self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME] = torch.cat(
                    (
                        self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME],
                        self._storage[PRECONDITIONED_GRADIENT_NAME],
                    ),
                    dim=0,
                )
                del self._storage[PRECONDITIONED_GRADIENT_NAME]
                self._storage[PRECONDITIONED_GRADIENT_NAME] = None
            else:
                self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME] = self._storage[PRECONDITIONED_GRADIENT_NAME]

        del self._storage[PRECONDITIONED_GRADIENT_NAME]
        self._storage[PRECONDITIONED_GRADIENT_NAME] = None

    def _release_preconditioned_gradient(self) -> None:
        """Clears the preconditioned per-sample-gradient from memory."""
        del self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME]
        self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME] = None
        del self._storage[PRECONDITIONED_GRADIENT_NAME]
        self._storage[PRECONDITIONED_GRADIENT_NAME] = None
        self._cached_activations = []
        del self._cached_per_sample_gradient
        self._cached_per_sample_gradient = None

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
                        stacked_matrix.transpose(0, 1)
                        .reshape(num_processes * size[0], size[1], size[2])
                        .contiguous()
                        .clone()
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
                    .clone()
                )

    ###########################################
    # Methods for computing influence scores. #
    ###########################################
    @torch.no_grad()
    def _pairwise_post_forward(self, outputs: torch.Tensor) -> Any:
        def backward_hook(output_gradient: torch.Tensor) -> None:
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
                if self._gradient_scale != 1.0:
                    self._cached_per_sample_gradient.mul_(self._gradient_scale)

                if isinstance(self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME], list):
                    # The preconditioned gradient is stored as a low-rank approximation.
                    left_mat, right_mat = self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME]

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
                        self._storage[AGGREGATED_PRECONDITIONED_GRADIENT_NAME],
                        self._cached_per_sample_gradient,
                    )
                del self._cached_per_sample_gradient
                self._cached_per_sample_gradient = None

        # Register backward hook to obtain gradient with respect to the output.
        outputs.register_hook(backward_hook)

    @torch.no_grad()
    def _self_post_forward(self, outputs: torch.Tensor) -> Any:
        def backward_hook(output_gradient: torch.Tensor) -> None:
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
                self._move_storage_to_device(target_device=per_sample_gradient.device)
                self._storge_at_current_device = True

            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)
                del per_sample_gradient

            # If the module was used multiple times throughout the forward pass,
            # only compute scores after aggregating all per-sample-gradients.
            if len(self._cached_activations) == 0:
                if self._gradient_scale != 1.0:
                    self._cached_per_sample_gradient.mul_(self._gradient_scale)

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
        # Register backward hook to obtain gradient with respect to the output.
        outputs.register_hook(backward_hook)

    @torch.no_grad()
    def _self_measurement_post_forward(self, outputs: torch.Tensor) -> Any:
        def backward_hook(output_gradient: torch.Tensor) -> None:
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
                self._move_storage_to_device(target_device=per_sample_gradient.device)
                self._storge_at_current_device = True

            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)
                del per_sample_gradient

            # If the module was used multiple times throughout the forward pass,
            # only compute scores after aggregating all per-sample-gradients.
            if len(self._cached_activations) == 0:
                if self._gradient_scale != 1.0:
                    self._cached_per_sample_gradient.mul_(self._gradient_scale)

                self._cached_per_sample_gradient = self._cached_per_sample_gradient.to(
                    dtype=self.score_args.score_dtype
                )
                self._storage[SELF_SCORE_VECTOR_NAME] = self._cached_per_sample_gradient.mul_(
                    self._storage[PRECONDITIONED_GRADIENT_NAME]
                ).sum(dim=(1, 2))
                self._cached_per_sample_gradient = None
                del self._storage[PRECONDITIONED_GRADIENT_NAME]
                self._storage[PRECONDITIONED_GRADIENT_NAME] = None
        # Register backward hook to obtain gradient with respect to the output.
        outputs.register_hook(backward_hook)

    def _move_storage_to_device(self, target_device: Union[torch.device, str]) -> None:
        """Moves stored factors into the target device."""
        for name, factor in self._storage.items():
            if factor is not None:
                if isinstance(factor, torch.Tensor):
                    self._storage[name] = factor.to(
                        device=target_device,
                        dtype=self.score_args.precondition_dtype,
                    )
                elif isinstance(factor, list):
                    for i in range(len(self._storage[name])):
                        self._storage[name][i] = factor[i].to(
                            device=target_device,
                            dtype=self.score_args.precondition_dtype,
                        )
                else:
                    raise RuntimeError(f"`{name}` in `TrackedModule` storage does not have a valid type.")

    def release_scores(self) -> None:
        """Clears the influence scores from memory."""
        del self._storage[PAIRWISE_SCORE_MATRIX_NAME]
        self._storage[PAIRWISE_SCORE_MATRIX_NAME] = None
        del self._storage[SELF_SCORE_VECTOR_NAME]
        self._storage[SELF_SCORE_VECTOR_NAME] = None
        self._cached_activations = []
        del self._cached_per_sample_gradient
        self._cached_per_sample_gradient = None
