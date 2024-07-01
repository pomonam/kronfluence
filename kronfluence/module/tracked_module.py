from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
from accelerate.utils.dataclasses import BaseEnum
from opt_einsum import contract
from torch import nn
from torch.utils.hooks import RemovableHandle

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.factor.config import FactorConfig
from kronfluence.utils.constants import (
    ACCUMULATED_PRECONDITIONED_GRADIENT_NAME,
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    ACTIVATION_EIGENVECTORS_NAME,
    COVARIANCE_FACTOR_NAMES,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_FACTOR_NAMES,
    LAMBDA_MATRIX_NAME,
    NUM_ACTIVATION_COVARIANCE_PROCESSED,
    NUM_GRADIENT_COVARIANCE_PROCESSED,
    NUM_LAMBDA_PROCESSED,
    PAIRWISE_SCORE_MATRIX_NAME,
    PRECONDITIONED_GRADIENT_NAME,
    SELF_SCORE_VECTOR_NAME,
)
from kronfluence.utils.exceptions import FactorsNotFoundError


class ModuleMode(str, BaseEnum):
    """Enum to represent a module's mode, indicating which factors and scores need to be computed
    during forward and backward passes."""

    DEFAULT = "default"
    COVARIANCE = "covariance"
    LAMBDA = "lambda"
    PRECONDITION_GRADIENT = "precondition_gradient"
    PAIRWISE_SCORE = "pairwise_score"
    SELF_SCORE = "self_score"
    SELF_MEASUREMENT_SCORE = "self_measurement_score"


class TrackedModule(nn.Module):
    """A wrapper class for PyTorch modules to compute influence factors and scores."""

    SUPPORTED_MODULES: Dict[Type[nn.Module], Any] = {}

    def __init_subclass__(cls, module_type: Type[nn.Module] = None, **kwargs: Any) -> None:
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
        per_sample_gradient_process_fnc: Optional[Callable] = None,
    ) -> None:
        """Initializes an instance of the TrackedModule class.

        Args:
            name (str):
                The original name of the module.
            original_module (nn.Module):
                The original module to be wrapped.
            factor_args (FactorArguments, optional):
                Arguments for computing influence factors.
            score_args (ScoreArguments, optional):
                Arguments for computing influence scores.
            per_sample_gradient_process_fnc (Callable, optional):
                An optional function to post process per-sample-gradient.
        """
        super().__init__()

        self.name = name
        self.original_module = original_module
        # A way to avoid Autograd computing the gradient with respect to the model parameters.
        self._constant: torch.Tensor = nn.Parameter(
            torch.zeros(
                1,
                requires_grad=True,
                dtype=torch.float16,
            )
        )
        self.factor_args = FactorArguments() if factor_args is None else factor_args
        self.score_args = ScoreArguments() if score_args is None else score_args
        self.per_sample_gradient_process_fnc = per_sample_gradient_process_fnc

        self._cached_activations: Optional[Union[List[torch.Tensor]], torch.Tensor] = None
        self._cached_per_sample_gradient: Optional[torch.Tensor] = None
        self._attention_mask: Optional[torch.Tensor] = None
        self._gradient_scale: float = 1.0
        self._registered_hooks: List[RemovableHandle] = []
        self._storage: Dict[str, Optional[Union[torch.Tensor, List[torch.Tensor]]]] = {}
        self._storage_at_device: bool = False

        # Storage for activation and pseudo-gradient covariance matrices. #
        for covariance_factor_name in COVARIANCE_FACTOR_NAMES:
            self._storage[covariance_factor_name]: Optional[torch.Tensor] = None

        # Storage for eigenvectors and eigenvalues. #
        for eigen_factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
            self._storage[eigen_factor_name]: Optional[torch.Tensor] = None

        # Storage for lambda matrices. #
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            self._storage[lambda_factor_name]: Optional[torch.Tensor] = None

        # Storage for preconditioned query gradients and influence scores. #
        self._storage[PRECONDITIONED_GRADIENT_NAME]: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None
        self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME]: Optional[Union[torch.Tensor, List[torch.Tensor]]] = (
            None
        )
        self._storage[PAIRWISE_SCORE_MATRIX_NAME]: Optional[torch.Tensor] = None
        self._storage[SELF_SCORE_VECTOR_NAME]: Optional[torch.Tensor] = None

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

    def forward(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """A forward pass of the tracked module. This should have identical behavior to that of the original module."""
        return self.original_module(inputs + self._constant, *args, **kwargs)

    def set_mode(self, mode: ModuleMode, keep_factors: bool = True) -> None:
        """Sets the module mode of the current `TrackedModule` instance."""
        self.set_attention_mask(attention_mask=None)
        self._remove_registered_hooks()

        if not keep_factors:
            self._release_covariance_matrices()
            self._release_eigendecomposition_results()
            self._release_lambda_matrix()
            self.release_preconditioned_gradient()
            self._storage_at_device = False
            self.release_scores()

        if mode == ModuleMode.DEFAULT:
            pass
        elif mode == ModuleMode.COVARIANCE:
            self._register_covariance_hooks()
        elif mode == ModuleMode.LAMBDA:
            self._register_lambda_hooks()
        elif mode == ModuleMode.PRECONDITION_GRADIENT:
            self._register_precondition_gradient_hooks()
        elif mode == ModuleMode.PAIRWISE_SCORE:
            self._register_pairwise_score_hooks()
        elif mode == ModuleMode.SELF_SCORE:
            self._register_self_score_hooks()
        elif mode == ModuleMode.SELF_MEASUREMENT_SCORE:
            self._register_self_measurement_score_hooks()
        else:
            raise RuntimeError(f"Unknown module mode {mode}.")

    def _remove_registered_hooks(self) -> None:
        """Removes all registered hooks within the module."""
        while self._registered_hooks:
            handle = self._registered_hooks.pop()
            handle.remove()
        self._registered_hooks = []

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor] = None) -> None:
        """Sets the attention mask for activation covariance computations."""
        self._attention_mask = attention_mask

    def set_gradient_scale(self, scale: float = 1.0) -> None:
        """Sets the scale of the gradient obtained from `GradScaler`."""
        self._gradient_scale = scale

    ##############################################
    # Methods for computing covariance matrices. #
    ##############################################
    @abstractmethod
    def _get_flattened_activation(
        self, input_activation: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """Returns the flattened activation tensor and the number of stacked activations.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, int]]:
                The flattened activation tensor and the number of stacked activations. The flattened
                activation is a 2-dimensional matrix with dimension `activation_num x activation_dim`.
        """
        raise NotImplementedError("Subclasses must implement the `_get_flattened_activation` method.")

    def _update_activation_covariance_matrix(self, input_activation: torch.Tensor) -> None:
        """Computes and updates the activation covariance matrix.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.
        """
        input_activation = input_activation.to(dtype=self.factor_args.activation_covariance_dtype)
        flattened_activation, count = self._get_flattened_activation(input_activation=input_activation)

        if self._storage[ACTIVATION_COVARIANCE_MATRIX_NAME] is None:
            dimension = flattened_activation.size(1)
            self._storage[ACTIVATION_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=flattened_activation.dtype,
                device=flattened_activation.device,
                requires_grad=False,
            )
        self._storage[ACTIVATION_COVARIANCE_MATRIX_NAME].addmm_(flattened_activation.t(), flattened_activation)

        if self._storage[NUM_ACTIVATION_COVARIANCE_PROCESSED] is None:
            device = None
            if isinstance(count, torch.Tensor):
                # When using attention masks, `count` can be a tensor.
                device = count.device
            self._storage[NUM_ACTIVATION_COVARIANCE_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                device=device,
                requires_grad=False,
            )
        self._storage[NUM_ACTIVATION_COVARIANCE_PROCESSED].add_(count)

    @abstractmethod
    def _get_flattened_gradient(self, output_gradient: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """Returns the flattened output gradient tensor.

        Args:
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the
                PyTorch's backward hook.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, int]]:
                The flattened output gradient tensor and the number of stacked gradients. The flattened
                gradient is a 2-dimensional matrix  with dimension `gradient_num x gradient_dim`.
        """
        raise NotImplementedError("Subclasses must implement the `_get_flattened_gradient` method.")

    def _update_gradient_covariance_matrix(self, output_gradient: torch.Tensor) -> None:
        """Computes and updates the pseudo-gradient covariance matrix.

        Args:
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the
                PyTorch's backward hook.
        """
        output_gradient = output_gradient.to(dtype=self.factor_args.gradient_covariance_dtype)
        flattened_gradient, count = self._get_flattened_gradient(output_gradient=output_gradient)
        if self._gradient_scale != 1.0:
            # Avoids in-place operation here.
            flattened_gradient = flattened_gradient * self._gradient_scale

        if self._storage[GRADIENT_COVARIANCE_MATRIX_NAME] is None:
            dimension = flattened_gradient.size(1)
            self._storage[GRADIENT_COVARIANCE_MATRIX_NAME] = torch.zeros(
                size=(dimension, dimension),
                dtype=flattened_gradient.dtype,
                device=flattened_gradient.device,
                requires_grad=False,
            )
        self._storage[GRADIENT_COVARIANCE_MATRIX_NAME].addmm_(flattened_gradient.t(), flattened_gradient)

        # This is not necessary in most cases as `NUM_GRADIENT_COVARIANCE_PROCESSED` should be typically identical to
        # `NUM_ACTIVATION_COVARIANCE_PROCESSED`. However, they can be different when using gradient checkpointing
        # or torch compile (`torch.compile`).
        if self._storage[NUM_GRADIENT_COVARIANCE_PROCESSED] is None:
            self._storage[NUM_GRADIENT_COVARIANCE_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                device=count.device if isinstance(count, torch.Tensor) else None,
                requires_grad=False,
            )
        self._storage[NUM_GRADIENT_COVARIANCE_PROCESSED].add_(count)

    def _register_covariance_hooks(self) -> None:
        """Installs forward and backward hooks for computation of the covariance matrices."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            # Computes and updates activation covariance in the forward pass.
            self._update_activation_covariance_matrix(inputs[0].detach().clone())
            # Registers backward hook to obtain gradient with respect to the output.
            outputs.register_hook(backward_hook)

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            # Computes and updates pseudo-gradient covariance in the backward pass.
            self._update_gradient_covariance_matrix(output_gradient.detach())

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

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
        """Returns the flattened per-sample-gradient tensor. For a brief introduction to
        per-sample-gradient, see https://pytorch.org/functorch/stable/notebooks/per_sample_grads.html.

        Args:
            input_activation (torch.Tensor):
                The input tensor to the module, provided by the PyTorch's forward hook.
            output_gradient (torch.Tensor):
                The gradient tensor with respect to the output of the module, provided by the PyTorch's backward hook.

        Returns:
            torch.Tensor:
                The per-sample-gradient tensor. The per-sample-gradient is a 3-dimensional matrix
                with dimension `batch_size x gradient_dim x activation_dim`.
        """
        raise NotImplementedError("Subclasses must implement the `_compute_per_sample_gradient` method.")

    def _update_lambda_matrix(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes and updates the Lambda matrix using the provided per-sample-gradient.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample-gradient tensor for the given batch.
        """
        per_sample_gradient = per_sample_gradient.to(self.factor_args.lambda_dtype)
        batch_size = per_sample_gradient.size(0)
        if self._gradient_scale != 1.0:
            per_sample_gradient.mul_(self._gradient_scale)

        if self._storage[LAMBDA_MATRIX_NAME] is None:
            # Initializes Lambda matrix if it does not exist.
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
                # Moves activation and pseudo-gradient eigenvectors to appropriate devices.
                self._storage[ACTIVATION_EIGENVECTORS_NAME] = self._storage[ACTIVATION_EIGENVECTORS_NAME].to(
                    dtype=self.factor_args.lambda_dtype,
                    device=per_sample_gradient.device,
                )
                self._storage[GRADIENT_EIGENVECTORS_NAME] = self._storage[GRADIENT_EIGENVECTORS_NAME].to(
                    dtype=self.factor_args.lambda_dtype,
                    device=per_sample_gradient.device,
                )

        if FactorConfig.CONFIGS[self.factor_args.strategy].requires_eigendecomposition_for_lambda:
            if self.factor_args.use_iterative_lambda_aggregation:
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
            # Approximates the eigenbasis as identity.
            self._storage[LAMBDA_MATRIX_NAME].add_(per_sample_gradient.square_().sum(dim=0))

        if self._storage[NUM_LAMBDA_PROCESSED] is None:
            self._storage[NUM_LAMBDA_PROCESSED] = torch.zeros(
                size=(1,),
                dtype=torch.int64,
                requires_grad=False,
            )
        self._storage[NUM_LAMBDA_PROCESSED].add_(batch_size)

    def _register_lambda_hooks(self) -> None:
        """Installs forward and backward hooks for computation of Lambda matrices."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach().clone().to(dtype=self.factor_args.per_sample_gradient_dtype)
            if self.factor_args.offload_activations_to_cpu:
                cached_activation = cached_activation.cpu()

            if self.factor_args.has_shared_parameters:
                if self._cached_activations is None:
                    self._cached_activations = []
                self._cached_activations.append(cached_activation)
            else:
                self._cached_activations = cached_activation

            # Registers backward hook to obtain gradient with respect to the output.
            outputs.register_hook(shared_backward_hook if self.factor_args.has_shared_parameters else backward_hook)

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self._cached_activations is None:
                raise RuntimeError(
                    f"The module {self.name} is used several times during a forward pass. "
                    "Set `has_shared_parameters=True` to avoid this error."
                )
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=self._cached_activations.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.factor_args.per_sample_gradient_dtype),
            ).to(dtype=self.factor_args.lambda_dtype)
            del self._cached_activations
            self._cached_activations = None
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            self._update_lambda_matrix(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            cached_activation = self._cached_activations.pop()
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.factor_args.per_sample_gradient_dtype),
            )
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

    def _clear_per_sample_gradient_cache(self) -> None:
        """Clears all caches from per-sample-gradient computations."""
        del self._cached_per_sample_gradient
        self._cached_per_sample_gradient = None
        del self._cached_activations
        self._cached_activations = None

    @torch.no_grad()
    def finalize_lambda_matrix(self) -> None:
        """Computes and updates the Lambda matrix using the cached per-sample-gradient."""
        self._update_lambda_matrix(
            per_sample_gradient=self._cached_per_sample_gradient.to(dtype=self.factor_args.lambda_dtype)
        )
        self._clear_per_sample_gradient_cache()

    def _release_lambda_matrix(self) -> None:
        """Clears the stored Lambda matrix from memory."""
        for lambda_factor_name in LAMBDA_FACTOR_NAMES:
            del self._storage[lambda_factor_name]
            self._storage[lambda_factor_name] = None
        self._clear_per_sample_gradient_cache()

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
                Low-rank matrices that approximate the original preconditioned query gradient.
        """
        rank = self.score_args.query_gradient_rank
        if self.score_args.use_full_svd:
            U, S, V = torch.linalg.svd(  # pylint: disable=not-callable
                preconditioned_gradient.contiguous().to(dtype=self.score_args.query_gradient_svd_dtype),
                full_matrices=False,
            )
            U_k = U[:, :, :rank]
            S_k = S[:, :rank]
            # Avoids holding the full memory of the original tensor before indexing.
            V_k = V[:, :rank, :].contiguous().clone()
            return [
                torch.matmul(U_k, torch.diag_embed(S_k)).to(dtype=self.score_args.score_dtype).contiguous().clone(),
                V_k.to(dtype=self.score_args.score_dtype),
            ]
        U, S, V = torch.svd_lowrank(
            preconditioned_gradient.contiguous().to(dtype=self.score_args.query_gradient_svd_dtype),
            q=rank,
        )
        return [
            torch.matmul(U, torch.diag_embed(S)).to(dtype=self.score_args.score_dtype).contiguous().clone(),
            V.transpose(1, 2).contiguous().to(dtype=self.score_args.score_dtype),
        ]

    def _compute_preconditioned_gradient(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes the preconditioned per-sample-gradient.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample-gradient tensor for the given batch.
        """
        per_sample_gradient = per_sample_gradient.to(dtype=self.score_args.precondition_dtype)
        if self._gradient_scale != 1.0:
            per_sample_gradient.mul_(self._gradient_scale)

        preconditioned_gradient = FactorConfig.CONFIGS[self.factor_args.strategy].precondition_gradient(
            gradient=per_sample_gradient,
            storage=self._storage,
            damping=self.score_args.damping,
        )
        del per_sample_gradient

        if (
            self.score_args.query_gradient_rank is not None
            and min(preconditioned_gradient.size()[1:]) > self.score_args.query_gradient_rank
        ):
            # Applies low-rank approximation to the preconditioned gradient.
            preconditioned_gradient = self._compute_low_rank_preconditioned_gradient(
                preconditioned_gradient=preconditioned_gradient
            )
            self._storage[PRECONDITIONED_GRADIENT_NAME] = preconditioned_gradient
        else:
            self._storage[PRECONDITIONED_GRADIENT_NAME] = preconditioned_gradient.to(dtype=self.score_args.score_dtype)

    def _register_precondition_gradient_hooks(self) -> None:
        """Installs forward and backward hooks for computation of preconditioned per-sample-gradient."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach().clone().to(dtype=self.score_args.per_sample_gradient_dtype)
            if self.score_args.cached_activation_cpu_offload:
                cached_activation = cached_activation.cpu()

            if self.factor_args.has_shared_parameters:
                if self._cached_activations is None:
                    self._cached_activations = []
                self._cached_activations.append(cached_activation)
            else:
                self._cached_activations = cached_activation

            outputs.register_hook(shared_backward_hook if self.factor_args.has_shared_parameters else backward_hook)

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self._cached_activations is None:
                raise RuntimeError(
                    f"The module {self.name} is used several times during a forward pass. "
                    "Set `has_shared_parameters=True` to avoid this error."
                )
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=self._cached_activations.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            ).to(dtype=self.score_args.precondition_dtype)
            del self._cached_activations
            self._cached_activations = None
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            self._compute_preconditioned_gradient(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            cached_activation = self._cached_activations.pop()
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            )
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

    @torch.no_grad()
    def finalize_preconditioned_gradient(self) -> None:
        """Computes the aggregated preconditioned gradient using the cached per-sample-gradient."""
        self._compute_preconditioned_gradient(
            per_sample_gradient=self._cached_per_sample_gradient.to(dtype=self.score_args.precondition_dtype)
        )
        self._clear_per_sample_gradient_cache()

    @torch.no_grad()
    def accumulate_preconditioned_gradient(self) -> None:
        """Accumulates the preconditioned per-sample-gradients computed over different batches."""
        if self._storage[PRECONDITIONED_GRADIENT_NAME] is None:
            return

        accumulated_gradient = self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME]
        gradient = self._storage[PRECONDITIONED_GRADIENT_NAME]

        if self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] is None:
            self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = gradient
        else:
            if isinstance(self._storage[PRECONDITIONED_GRADIENT_NAME], list):
                self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = [
                    torch.cat((accumulated_gradient[0], gradient[0]), dim=0).contiguous(),
                    torch.cat((accumulated_gradient[1], gradient[1]), dim=0).contiguous(),
                ]
            else:
                self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = torch.cat(
                    (accumulated_gradient, gradient), dim=0
                ).contiguous()
        del self._storage[PRECONDITIONED_GRADIENT_NAME]
        self._storage[PRECONDITIONED_GRADIENT_NAME] = None

    def release_preconditioned_gradient(self) -> None:
        """Clears the preconditioned per-sample-gradient from memory."""
        del self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME]
        self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME] = None
        del self._storage[PRECONDITIONED_GRADIENT_NAME]
        self._storage[PRECONDITIONED_GRADIENT_NAME] = None
        self._clear_per_sample_gradient_cache()

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

    @torch.no_grad()
    def synchronize_preconditioned_gradient(self, num_processes: int) -> None:
        """Stacks preconditioned gradient across multiple devices or nodes in a distributed setting."""
        if (
            dist.is_initialized()
            and torch.cuda.is_available()
            and self._storage[PRECONDITIONED_GRADIENT_NAME] is not None
        ):
            if isinstance(self._storage[PRECONDITIONED_GRADIENT_NAME], list):
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
    def _compute_pairwise_score(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes the pairwise influence scores.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample-gradient tensor for the given batch.
        """
        per_sample_gradient = per_sample_gradient.to(dtype=self.score_args.score_dtype)
        if self._gradient_scale != 1.0:
            per_sample_gradient.mul_(self._gradient_scale)

        if isinstance(self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME], list):
            # The preconditioned gradient is stored as a low-rank approximation.
            left_mat, right_mat = self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME]
            input_dim = right_mat.size(2)
            output_dim = left_mat.size(1)
            query_batch_size = left_mat.size(0)
            train_batch_size = per_sample_gradient.size(0)
            rank = self.score_args.query_gradient_rank
            if (
                train_batch_size * query_batch_size * rank * min((input_dim, output_dim))
                > query_batch_size * input_dim * output_dim
            ):
                # If reconstructing the gradient is more memory efficient, reconstructs and computes the score.
                self._storage[PAIRWISE_SCORE_MATRIX_NAME] = contract(
                    "qki,toi,qok->qt",
                    right_mat,
                    per_sample_gradient,
                    left_mat,
                )
            # Otherwise, tries to avoid reconstructing the full per-sample-gradient.
            elif output_dim >= input_dim:
                self._storage[PAIRWISE_SCORE_MATRIX_NAME] = contract(
                    "qki,qtik->qt", right_mat, contract("toi,qok->qtik", per_sample_gradient, left_mat)
                )
            else:
                self._storage[PAIRWISE_SCORE_MATRIX_NAME] = contract(
                    "qtko,qok->qt", contract("qki,toi->qtko", right_mat, per_sample_gradient), left_mat
                )
        else:
            self._storage[PAIRWISE_SCORE_MATRIX_NAME] = contract(
                "qio,tio->qt",
                self._storage[ACCUMULATED_PRECONDITIONED_GRADIENT_NAME],
                per_sample_gradient,
            )

    def _register_pairwise_score_hooks(self) -> None:
        """Installs forward and backward hooks for computation of pairwise influence scores."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach().clone().to(dtype=self.score_args.per_sample_gradient_dtype)
            if self.score_args.cached_activation_cpu_offload:
                cached_activation = cached_activation.cpu()

            if self.factor_args.has_shared_parameters:
                if self._cached_activations is None:
                    self._cached_activations = []
                self._cached_activations.append(cached_activation)
            else:
                self._cached_activations = cached_activation

            outputs.register_hook(shared_backward_hook if self.factor_args.has_shared_parameters else backward_hook)

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self._cached_activations is None:
                raise RuntimeError(
                    f"The module {self.name} is used several times during a forward pass. "
                    "Set `has_shared_parameters=True` to avoid the error."
                )
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=self._cached_activations.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            ).to(dtype=self.score_args.score_dtype)
            del self._cached_activations
            self._cached_activations = None
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            self._compute_pairwise_score(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            cached_activation = self._cached_activations.pop()
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            )
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

    @torch.no_grad()
    def finalize_pairwise_score(self) -> None:
        """Computes the pairwise influence scores using the cached per-sample-gradient."""
        self._compute_pairwise_score(
            per_sample_gradient=self._cached_per_sample_gradient.to(dtype=self.score_args.score_dtype)
        )
        self._clear_per_sample_gradient_cache()

    def _compute_self_score(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes the self-influence scores.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample-gradient tensor for the given batch.
        """
        if self._gradient_scale != 1.0:
            per_sample_gradient.mul_(self._gradient_scale)

        if not self._storage_at_device:
            self._move_storage_to_device(
                target_device=per_sample_gradient.device, target_dtype=self.score_args.precondition_dtype
            )
            self._storage_at_device = True
        preconditioned_gradient = (
            FactorConfig.CONFIGS[self.factor_args.strategy]
            .precondition_gradient(
                gradient=per_sample_gradient.to(dtype=self.score_args.precondition_dtype),
                storage=self._storage,
                damping=self.score_args.damping,
            )
            .to(dtype=self.score_args.score_dtype)
        )
        preconditioned_gradient.mul_(per_sample_gradient.to(dtype=self.score_args.score_dtype))
        self._storage[SELF_SCORE_VECTOR_NAME] = preconditioned_gradient.sum(dim=(1, 2))

    def _register_self_score_hooks(self) -> None:
        """Installs forward and backward hooks for computation of self-influence scores."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach().clone().to(dtype=self.score_args.per_sample_gradient_dtype)
            if self.score_args.cached_activation_cpu_offload:
                cached_activation = cached_activation.cpu()

            if self.factor_args.has_shared_parameters:
                if self._cached_activations is None:
                    self._cached_activations = []
                self._cached_activations.append(cached_activation)
            else:
                self._cached_activations = cached_activation

            outputs.register_hook(shared_backward_hook if self.factor_args.has_shared_parameters else backward_hook)

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self._cached_activations is None:
                raise RuntimeError(
                    f"The module {self.name} is used several times during a forward pass. "
                    "Set `has_shared_parameters=True` to avoid this error."
                )
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=self._cached_activations.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            ).to(dtype=self.score_args.precondition_dtype)
            del self._cached_activations
            self._cached_activations = None
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            self._compute_self_score(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            cached_activation = self._cached_activations.pop()
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            )
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

    def finalize_self_score(self) -> None:
        """Computes the self-influence scores using the cached per-sample-gradient."""
        self._compute_self_score(per_sample_gradient=self._cached_per_sample_gradient)
        self._clear_per_sample_gradient_cache()

    def _compute_self_measurement_score(self, per_sample_gradient: torch.Tensor) -> None:
        """Computes the self-influence scores with measurement.

        Args:
            per_sample_gradient (torch.Tensor):
                The per-sample-gradient tensor for the given batch.
        """
        per_sample_gradient = per_sample_gradient.to(dtype=self.score_args.score_dtype)
        if self._gradient_scale != 1.0:
            per_sample_gradient.mul_(self._gradient_scale)
        if not self._storage_at_device:
            self._move_storage_to_device(
                target_device=per_sample_gradient.device, target_dtype=self.score_args.precondition_dtype
            )
            self._storage_at_device = True
        self._storage[SELF_SCORE_VECTOR_NAME] = per_sample_gradient.mul_(
            self._storage[PRECONDITIONED_GRADIENT_NAME]
        ).sum(dim=(1, 2))
        del self._storage[PRECONDITIONED_GRADIENT_NAME]
        self._storage[PRECONDITIONED_GRADIENT_NAME] = None

    def _register_self_measurement_score_hooks(self) -> None:
        """Installs forward and backward hooks for computation of self-influence scores."""

        @torch.no_grad()
        def forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor], outputs: torch.Tensor) -> None:
            del module
            cached_activation = inputs[0].detach().clone().to(dtype=self.score_args.per_sample_gradient_dtype)
            if self.score_args.cached_activation_cpu_offload:
                cached_activation = cached_activation.cpu()

            if self.factor_args.has_shared_parameters:
                if self._cached_activations is None:
                    self._cached_activations = []
                self._cached_activations.append(cached_activation)
            else:
                self._cached_activations = cached_activation

            outputs.register_hook(shared_backward_hook if self.factor_args.has_shared_parameters else backward_hook)

        @torch.no_grad()
        def backward_hook(output_gradient: torch.Tensor) -> None:
            if self._cached_activations is None:
                raise RuntimeError(
                    f"The module {self.name} is used several times during a forward pass. "
                    "Set `has_shared_parameters=True` to avoid this error."
                )
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=self._cached_activations.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            ).to(dtype=self.score_args.score_dtype)
            del self._cached_activations
            self._cached_activations = None
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            self._compute_self_measurement_score(per_sample_gradient=per_sample_gradient)

        @torch.no_grad()
        def shared_backward_hook(output_gradient: torch.Tensor) -> None:
            cached_activation = self._cached_activations.pop()
            per_sample_gradient = self._compute_per_sample_gradient(
                input_activation=cached_activation.to(device=output_gradient.device),
                output_gradient=output_gradient.detach().to(dtype=self.score_args.per_sample_gradient_dtype),
            )
            if self.per_sample_gradient_process_fnc is not None:
                per_sample_gradient = self.per_sample_gradient_process_fnc(
                    module_name=self.name, gradient=per_sample_gradient
                )
            if self._cached_per_sample_gradient is None:
                self._cached_per_sample_gradient = per_sample_gradient
            else:
                self._cached_per_sample_gradient.add_(per_sample_gradient)

        self._registered_hooks.append(self.original_module.register_forward_hook(forward_hook))

    @torch.no_grad()
    def finalize_self_measurement_score(self) -> None:
        """Computes the self-influence scores with measurement using the cached per-sample-gradient."""
        self._compute_self_measurement_score(
            per_sample_gradient=self._cached_per_sample_gradient.to(dtype=self.score_args.score_dtype)
        )
        self._clear_per_sample_gradient_cache()

    def _move_storage_to_device(self, target_device: torch.device, target_dtype: torch.dtype) -> None:
        """Moves stored factors into the target device."""
        for name, factor in self._storage.items():
            if factor is not None:
                if isinstance(factor, list):
                    for i in range(len(self._storage[name])):
                        self._storage[name][i] = factor[i].to(
                            device=target_device,
                            dtype=target_dtype,
                        )
                else:
                    self._storage[name] = factor.to(device=target_device, dtype=target_dtype)

    def release_scores(self) -> None:
        """Clears the influence scores from memory."""
        del self._storage[PAIRWISE_SCORE_MATRIX_NAME]
        self._storage[PAIRWISE_SCORE_MATRIX_NAME] = None
        del self._storage[SELF_SCORE_VECTOR_NAME]
        self._storage[SELF_SCORE_VECTOR_NAME] = None
        self._clear_per_sample_gradient_cache()
