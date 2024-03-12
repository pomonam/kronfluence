from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional

import torch
from accelerate.utils.dataclasses import BaseEnum

from kronfluence.module.constants import (
    ACTIVATION_EIGENVALUES_NAME,
    ACTIVATION_EIGENVECTORS_NAME,
    GRADIENT_EIGENVALUES_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_MATRIX_NAME,
    NUM_LAMBDA_PROCESSED,
)


class FactorStrategy(str, BaseEnum):
    """A strategy for computing preconditioning factor."""

    IDENTITY = "identity"
    DIAGONAL = "diag"
    KFAC = "kfac"
    EKFAC = "ekfac"


class FactorConfig(metaclass=ABCMeta):
    """Configuration for each factor strategy."""

    CONFIGS: Dict[FactorStrategy, Any] = {}

    def __init_subclass__(
        cls, factor_strategy: Optional[FactorStrategy] = None, **kwargs
    ) -> None:
        """Registers all subclasses of `FactorConfig`."""
        super().__init_subclass__(**kwargs)
        if factor_strategy is not None:
            assert factor_strategy in [strategy.value for strategy in FactorStrategy]
            cls.CONFIGS[factor_strategy] = cls()

    @property
    @abstractmethod
    def requires_covariance_matrices(self) -> bool:
        """Returns True if the given strategy requires computing covariance matrices."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_covariance_matrices` property."
        )

    @property
    @abstractmethod
    def requires_eigendecomposition(self) -> bool:
        """Returns True if the given strategy requires performing Eigendecomposition."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_eigendecomposition` property."
        )

    @property
    @abstractmethod
    def requires_lambda_matrices(self) -> bool:
        """Returns True if the given strategy requires computing Lambda matrices."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_lambda_matrices` property."
        )

    @property
    @abstractmethod
    def requires_eigendecomposition_for_lambda(self) -> bool:
        """Returns True if the given strategy requires loading Eigendecomposition results, before
        computing Lambda matrices."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_eigendecomposition_for_lambda` property."
        )

    @property
    @abstractmethod
    def requires_covariance_matrices_for_precondition(self) -> bool:
        """Returns True if the given strategy requires loading covariance matrices, before
        computing preconditioned gradient."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_covariance_matrices_for_precondition` property."
        )

    @property
    @abstractmethod
    def requires_eigendecomposition_for_precondition(self) -> bool:
        """Returns True if the given strategy requires loading Eigendecomposition results, before
        computing preconditioned gradient."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_eigendecomposition_for_precondition` property."
        )

    @property
    @abstractmethod
    def requires_lambda_matrices_for_precondition(self) -> bool:
        """Returns True if the given strategy requires loading Lambda matrices, before
        computing the preconditioned gradient."""
        raise NotImplementedError(
            "Subclasses must implement the `requires_lambda_matrices_for_precondition` property."
        )

    @abstractmethod
    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: Dict[str, torch.Tensor],
        damping: float,
    ) -> torch.Tensor:
        """Preconditions the per-sample-gradient with the appropriate strategy. The per-sample-gradient
        is a 3-dimensional tensor with shape `batch_size x input_dim x output_dim`.

        Args:
            gradient (torch.Tensor):
                The per-sample-gradient tensor.
            storage (Dict[str, Any]):
                A dictionary containing various factors to help perform preconditioning.
            damping (float):
                The damping factor when computing the preconditioned gradient.
        """
        raise NotImplementedError(
            "Subclasses must implement the `precondition_gradient` property."
        )


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
        storage: Dict[str, torch.Tensor],
        damping: Optional[int],
    ) -> torch.Tensor:
        del storage, damping
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

    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: Dict[str, torch.Tensor],
        damping: Optional[int],
    ) -> torch.Tensor:
        lambda_matrix = storage[LAMBDA_MATRIX_NAME].to(
            dtype=gradient.dtype, device=gradient.device
        )
        num_lambda_processed = storage[NUM_LAMBDA_PROCESSED].to(device=gradient.device)
        if damping is None:
            damping = 0.1 * torch.mean(lambda_matrix)
        return num_lambda_processed * gradient / (lambda_matrix + damping)


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

    @torch.no_grad()
    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: Dict[str, torch.Tensor],
        damping: Optional[int],
    ) -> torch.Tensor:
        activation_eigenvectors = storage[ACTIVATION_EIGENVECTORS_NAME].to(
            dtype=gradient.dtype, device=gradient.device
        )
        gradient_eigenvectors = storage[GRADIENT_EIGENVECTORS_NAME].to(
            dtype=gradient.dtype, device=gradient.device
        )
        activation_eigenvalues = storage[ACTIVATION_EIGENVALUES_NAME].to(
            dtype=gradient.dtype, device=gradient.device
        )
        gradient_eigenvalues = storage[GRADIENT_EIGENVALUES_NAME].to(
            dtype=gradient.dtype, device=gradient.device
        )
        lambda_matrix = torch.kron(
            activation_eigenvalues.unsqueeze(0), gradient_eigenvalues.unsqueeze(-1)
        ).unsqueeze(0)

        rotated_gradient = torch.einsum(
            "ij,bjl,lk->bik",
            (
                gradient_eigenvectors.t(),
                gradient,
                activation_eigenvectors,
            ),
        )

        if damping is None:
            damping = 0.1 * torch.mean(lambda_matrix)

        rotated_gradient.div_(lambda_matrix + damping)
        return torch.einsum(
            "ij,bjl,lk->bik",
            (gradient_eigenvectors, rotated_gradient, activation_eigenvectors.t()),
        )


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

    @torch.no_grad()
    def precondition_gradient(
        self,
        gradient: torch.Tensor,
        storage: Dict[str, torch.Tensor],
        damping: Optional[int],
    ) -> torch.Tensor:
        activation_eigenvectors = storage[ACTIVATION_EIGENVECTORS_NAME].to(
            dtype=gradient.dtype, device=gradient.device
        )
        gradient_eigenvectors = storage[GRADIENT_EIGENVECTORS_NAME].to(
            dtype=gradient.dtype, device=gradient.device
        )
        lambda_matrix = storage[LAMBDA_MATRIX_NAME].to(
            dtype=gradient.dtype, device=gradient.device
        )
        num_lambda_processed = storage[NUM_LAMBDA_PROCESSED].to(device=gradient.device)

        rotated_gradient = torch.einsum(
            "ij,bjl,lk->bik",
            (
                gradient_eigenvectors.t(),
                gradient,
                activation_eigenvectors,
            ),
        )

        if damping is None:
            damping = 0.1 * torch.mean(lambda_matrix)

        rotated_gradient.div_(lambda_matrix + damping)
        return num_lambda_processed * torch.einsum(
            "ij,bjl,lk->bik",
            (gradient_eigenvectors, rotated_gradient, activation_eigenvectors.t()),
        )
