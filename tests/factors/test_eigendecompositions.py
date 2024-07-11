# pylint: skip-file

import pytest
import torch

from kronfluence.arguments import FactorArguments
from kronfluence.utils.constants import (
    ACTIVATION_EIGENVECTORS_NAME,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    GRADIENT_EIGENVECTORS_NAME,
)
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import DEFAULT_FACTORS_NAME, prepare_model_and_analyzer, prepare_test


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "bert",
    ],
)
@pytest.mark.parametrize("eigendecomposition_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("train_size", [1, 30])
@pytest.mark.parametrize("seed", [0])
def test_perform_eigendecomposition(
    test_name: str,
    eigendecomposition_dtype: torch.dtype,
    train_size: int,
    seed: int,
) -> None:
    # Makes sure that the Eigendecomposition computations are working properly.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    factor_args = FactorArguments(
        eigendecomposition_dtype=eigendecomposition_dtype,
    )
    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        factor_args=factor_args,
        dataset=train_dataset,
        per_device_batch_size=None,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    analyzer.perform_eigendecomposition(
        factors_name=DEFAULT_FACTORS_NAME,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    eigen_factors = analyzer.load_eigendecomposition(factors_name=DEFAULT_FACTORS_NAME)
    assert set(eigen_factors.keys()) == set(EIGENDECOMPOSITION_FACTOR_NAMES)
    assert len(eigen_factors[ACTIVATION_EIGENVECTORS_NAME]) > 0
    assert len(eigen_factors[GRADIENT_EIGENVECTORS_NAME]) > 0
    for module_name in eigen_factors[ACTIVATION_EIGENVECTORS_NAME]:
        assert eigen_factors[ACTIVATION_EIGENVECTORS_NAME][module_name] is not None
        assert eigen_factors[GRADIENT_EIGENVECTORS_NAME][module_name] is not None
