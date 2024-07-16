from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

import torch
from accelerate.utils.dataclasses import BaseEnum

from kronfluence.utils.constants import (
    ACTIVATION_EIGENVALUES_NAME,
    ACTIVATION_EIGENVECTORS_NAME,
    GRADIENT_EIGENVALUES_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    HEURISTIC_DAMPING_SCALE,
    LAMBDA_DTYPE,
    LAMBDA_MATRIX_NAME,
    NUM_LAMBDA_PROCESSED,
)

STORAGE_TYPE = Dict[str, Any]


class FactorStrategy(str, BaseEnum):
    """Strategies for computing preconditioning factors."""

    IDENTITY = "identity"
    DIAGONAL = "diagonal"
    KFAC = "kfac"
    EKFAC = "ekfac"


class FactorConfig(metaclass=ABCMeta):
    """Configurations for each supported factor strategy."""

    CONFIGS: Dict[FactorStrategy, Any] = {}

    def __init_subclass__(cls, factor_strategy: Optional[FactorStrategy] = None, **kwargs) -> None:
        """Registers all subclasses of `FactorConfig`."""
        super().__init_subclass__(**kwargs)
        if factor_strategy is not None:
            assert factor_strategy in [strategy.value for strategy in FactorStrategy]
            cls.CONFIGS[factor_strategy] = cls()

    @property
    @abstractmethod
    def requires_covariance_matrices(self) -> bool:
        """Returns `True` if the strategy requires computing covariance matrices."""
        raise NotImplementedError("Subclasses must implement the `requires_covariance_matrices` property.")

    @property
    @abstractmethod
    def requires_eigendecomposition(self) -> bool:
        """Returns `True` if the strategy requires performing Eigendecomposition."""
        raise NotImplementedError("Subclasses must implement the `requires_eigendecomposition` property.")

    @property
    @abstractmethod
    def requires_lambda_matrices(self) -> bool:
        """Returns `True` if the strategy requires computing Lambda matrices."""
        raise NotImplementedError("Subclasses must implement the `requires_lambda_matrices` property.")

    @property
    @abstractmethod
    def requires_eigendecomposition_for_lambda(self) -> bool:
        """Returns `True` if the strategy requires loading Eigendecomposition results, before computing
        Lambda matrices."""
        raise NotImplementedError("Subclasses must implement the `requires_eigendecomposition_for_lambda` property.")

    @property
    @abstractmethod
    def requires_covariance_matrices_for_precondition(self) -> bool:
        """Returns `True` if the strategy requires loading covariance matrices, before computing
        preconditioned gradient."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_covariance_matrices_for_precondition` property."
        )

    @property
    @abstractmethod
    def requires_eigendecomposition_for_precondition(self) -> bool:
        """Returns `True` if the strategy requires loading Eigendecomposition results, before computing
        preconditioned gradient."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_eigendecomposition_for_precondition` property."
        )

    @property
    @abstractmethod
    def requires_lambda_matrices_for_precondition(self) -> bool:
        """Returns `True` if the strategy requires loading Lambda matrices, before computing
        the preconditioned gradient."""
        raise NotImplementedError("Subclasses must implement the `requires_lambda_matrices_for_precondition` property.")

    def prepare(self, storage: STORAGE_TYPE, score_args: Any, device: torch.device) -> None:
        """Performs necessary operations before computing the preconditioned gradient.

        Args:
            storage (STORAGE_TYPE):
                A dictionary containing various factors required to compute the preconditioned gradient.
                See `.storage` in `TrackedModule` for details.
            score_args (ScoreArguments):
                Arguments for computing the preconditioned gradient.
            device (torch.device):
                Device used for computing the preconditioned gradient.
        """

    @abstractmethod
    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: STORAGE_TYPE,
    ) -> torch.Tensor:
        """Preconditions the per-sample gradient. The per-sample gradient is a 3-dimensional
        tensor with shape `batch_size x output_dim x input_dim`.

        Args:
            gradient (torch.Tensor):
                The per-sample gradient tensor.
            storage (STORAGE_TYPE):
                A dictionary containing various factors required to compute the preconditioned gradient.
                See `.storage` in `TrackedModule` for details.

        Returns:
            torch.Tensor:
                The preconditioned per-sample gradient tensor.
        """
        raise NotImplementedError("Subclasses must implement the `precondition_gradient` method.")


class Identity(FactorConfig, factor_strategy=FactorStrategy.IDENTITY):
    """Applies no preconditioning to the gradient."""

    @property
    def requires_covariance_matrices(self) -> bool:
        return False

    @property
    def requires_eigendecomposition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_lambda(self) -> bool:
        return False

    @property
    def requires_lambda_matrices(self) -> bool:
        return False

    @property
    def requires_covariance_matrices_for_precondition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_precondition(self) -> bool:
        return False

    @property
    def requires_lambda_matrices_for_precondition(self) -> bool:
        return False

    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: STORAGE_TYPE,
    ) -> torch.Tensor:
        del storage
        return gradient


class Diagonal(FactorConfig, factor_strategy=FactorStrategy.DIAGONAL):
    """Applies diagonal preconditioning to the gradient."""

    @property
    def requires_covariance_matrices(self) -> bool:
        return False

    @property
    def requires_eigendecomposition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_lambda(self) -> bool:
        return False

    @property
    def requires_lambda_matrices(self) -> bool:
        return True

    @property
    def requires_covariance_matrices_for_precondition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_precondition(self) -> bool:
        return False

    @property
    def requires_lambda_matrices_for_precondition(self) -> bool:
        return True

    def prepare(self, storage: STORAGE_TYPE, score_args: Any, device: torch.device) -> None:
        lambda_matrix = storage[LAMBDA_MATRIX_NAME].to(dtype=LAMBDA_DTYPE, device=device)
        lambda_matrix.div_(storage[NUM_LAMBDA_PROCESSED].to(device=device))
        damping_factor = score_args.damping_factor
        if damping_factor is None:
            damping_factor = HEURISTIC_DAMPING_SCALE * torch.mean(lambda_matrix)
        lambda_matrix.add_(damping_factor)
        lambda_matrix.reciprocal_()
        storage[LAMBDA_MATRIX_NAME] = lambda_matrix.to(dtype=score_args.precondition_dtype, device="cpu").contiguous()
        storage[NUM_LAMBDA_PROCESSED] = None

    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: STORAGE_TYPE,
    ) -> torch.Tensor:
        lambda_matrix = storage[LAMBDA_MATRIX_NAME].to(device=gradient.device)
        return gradient * lambda_matrix


class Kfac(FactorConfig, factor_strategy=FactorStrategy.KFAC):
    """Applies KFAC preconditioning to the gradient.

    See https://arxiv.org/pdf/1503.05671.pdf for details.
    """

    @property
    def requires_covariance_matrices(self) -> bool:
        return True

    @property
    def requires_eigendecomposition(self) -> bool:
        return True

    @property
    def requires_eigendecomposition_for_lambda(self) -> bool:
        return False

    @property
    def requires_lambda_matrices(self) -> bool:
        return False

    @property
    def requires_covariance_matrices_for_precondition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_precondition(self) -> bool:
        return True

    @property
    def requires_lambda_matrices_for_precondition(self) -> bool:
        return False

    def prepare(self, storage: STORAGE_TYPE, score_args: Any, device: torch.device) -> None:
        storage[ACTIVATION_EIGENVECTORS_NAME] = (
            storage[ACTIVATION_EIGENVECTORS_NAME].to(dtype=score_args.precondition_dtype).contiguous()
        )
        storage[GRADIENT_EIGENVECTORS_NAME] = (
            storage[GRADIENT_EIGENVECTORS_NAME].to(dtype=score_args.precondition_dtype).contiguous()
        )
        activation_eigenvalues = storage[ACTIVATION_EIGENVALUES_NAME].to(dtype=LAMBDA_DTYPE, device=device)
        gradient_eigenvalues = storage[GRADIENT_EIGENVALUES_NAME].to(dtype=LAMBDA_DTYPE, device=device)
        lambda_matrix = torch.kron(activation_eigenvalues.unsqueeze(0), gradient_eigenvalues.unsqueeze(-1)).unsqueeze(0)
        damping_factor = score_args.damping_factor
        if damping_factor is None:
            damping_factor = HEURISTIC_DAMPING_SCALE * torch.mean(lambda_matrix)
        lambda_matrix.add_(damping_factor)
        lambda_matrix.reciprocal_()
        storage[LAMBDA_MATRIX_NAME] = lambda_matrix.to(dtype=score_args.precondition_dtype, device="cpu").contiguous()
        storage[NUM_LAMBDA_PROCESSED] = None
        storage[ACTIVATION_EIGENVALUES_NAME] = None
        storage[GRADIENT_EIGENVALUES_NAME] = None

    @torch.no_grad()
    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: STORAGE_TYPE,
    ) -> torch.Tensor:
        activation_eigenvectors = storage[ACTIVATION_EIGENVECTORS_NAME].to(device=gradient.device)
        gradient_eigenvectors = storage[GRADIENT_EIGENVECTORS_NAME].to(device=gradient.device)
        lambda_matrix = storage[LAMBDA_MATRIX_NAME].to(device=gradient.device)
        gradient = torch.matmul(gradient_eigenvectors.t(), torch.matmul(gradient, activation_eigenvectors))
        gradient.mul_(lambda_matrix)
        gradient = torch.matmul(gradient_eigenvectors, torch.matmul(gradient, activation_eigenvectors.t()))
        return gradient


class Ekfac(FactorConfig, factor_strategy=FactorStrategy.EKFAC):
    """Applies EKFAC preconditioning to the gradient.

    See https://arxiv.org/pdf/1806.03884.pdf for details.
    """

    @property
    def requires_covariance_matrices(self) -> bool:
        return True

    @property
    def requires_eigendecomposition(self) -> bool:
        return True

    @property
    def requires_eigendecomposition_for_lambda(self) -> bool:
        return True

    @property
    def requires_lambda_matrices(self) -> bool:
        return True

    @property
    def requires_covariance_matrices_for_precondition(self) -> bool:
        return False

    @property
    def requires_eigendecomposition_for_precondition(self) -> bool:
        return True

    @property
    def requires_lambda_matrices_for_precondition(self) -> bool:
        return True

    def prepare(self, storage: STORAGE_TYPE, score_args: Any, device: torch.device) -> None:
        storage[ACTIVATION_EIGENVECTORS_NAME] = (
            storage[ACTIVATION_EIGENVECTORS_NAME].to(dtype=score_args.precondition_dtype).contiguous()
        )
        storage[GRADIENT_EIGENVECTORS_NAME] = (
            storage[GRADIENT_EIGENVECTORS_NAME].to(dtype=score_args.precondition_dtype).contiguous()
        )
        storage[ACTIVATION_EIGENVALUES_NAME] = None
        storage[GRADIENT_EIGENVALUES_NAME] = None
        lambda_matrix = storage[LAMBDA_MATRIX_NAME].to(dtype=LAMBDA_DTYPE, device=device)
        lambda_matrix.div_(storage[NUM_LAMBDA_PROCESSED].to(device=device))
        damping_factor = score_args.damping_factor
        if damping_factor is None:
            damping_factor = HEURISTIC_DAMPING_SCALE * torch.mean(lambda_matrix)
        lambda_matrix.add_(damping_factor)
        lambda_matrix.reciprocal_()
        storage[LAMBDA_MATRIX_NAME] = lambda_matrix.to(dtype=score_args.precondition_dtype, device="cpu").contiguous()
        storage[NUM_LAMBDA_PROCESSED] = None

    @torch.no_grad()
    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: STORAGE_TYPE,
    ) -> torch.Tensor:
        activation_eigenvectors = storage[ACTIVATION_EIGENVECTORS_NAME].to(device=gradient.device)
        gradient_eigenvectors = storage[GRADIENT_EIGENVECTORS_NAME].to(device=gradient.device)
        lambda_matrix = storage[LAMBDA_MATRIX_NAME].to(device=gradient.device)
        gradient = torch.matmul(gradient_eigenvectors.t(), torch.matmul(gradient, activation_eigenvectors))
        gradient.mul_(lambda_matrix)
        gradient = torch.matmul(gradient_eigenvectors, torch.matmul(gradient, activation_eigenvectors.t()))
        return gradient
