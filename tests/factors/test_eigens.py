# pylint: skip-file

from typing import Tuple

import pytest
import torch
from torch import nn

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.task import Task
from kronfluence.utils.constants import (
    ACTIVATION_EIGENVECTORS_NAME,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_FACTOR_NAMES,
    LAMBDA_MATRIX_NAME,
    NUM_LAMBDA_PROCESSED,
)
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import ATOL, RTOL, check_tensor_dict_equivalence, prepare_test


def prepare_model_and_analyzer(model: nn.Module, task: Task) -> Tuple[nn.Module, Analyzer]:
    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name=f"pytest_{__name__}",
        model=model,
        task=task,
        disable_model_save=True,
        cpu=True,
    )
    return model, analyzer


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "repeated_mlp",
        "conv",
        "conv_bn",
        "bert",
        "gpt",
    ],
)
@pytest.mark.parametrize("eigendecomposition_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("train_size", [16])
@pytest.mark.parametrize("seed", [0])
def test_perform_eigendecomposition(
    test_name: str,
    eigendecomposition_dtype: torch.dtype,
    train_size: int,
    seed: int,
) -> None:
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
    factors_name = f"pytest_{test_name}_{test_perform_eigendecomposition.__name__}"
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=4,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    factor_args = FactorArguments(
        eigendecomposition_dtype=eigendecomposition_dtype,
    )
    analyzer.perform_eigendecomposition(
        factors_name=factors_name,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    eigen_factors = analyzer.load_eigendecomposition(factors_name=factors_name)
    assert set(eigen_factors.keys()) == set(EIGENDECOMPOSITION_FACTOR_NAMES)
    assert len(eigen_factors[ACTIVATION_EIGENVECTORS_NAME]) > 0
    for module_name in eigen_factors[ACTIVATION_EIGENVECTORS_NAME]:
        assert eigen_factors[ACTIVATION_EIGENVECTORS_NAME][module_name] is not None
        assert eigen_factors[GRADIENT_EIGENVECTORS_NAME][module_name] is not None


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "conv_bn",
        "gpt",
        "bert",
    ],
)
@pytest.mark.parametrize("lambda_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_size", [16])
@pytest.mark.parametrize("seed", [0])
def test_fit_lambda_matrices(
    test_name: str,
    lambda_dtype: torch.dtype,
    train_size: int,
    seed: int,
) -> None:
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
        lambda_dtype=lambda_dtype,
    )
    factors_name = f"pytest_{test_name}_{test_fit_lambda_matrices.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=4,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )

    lambda_factors = analyzer.load_lambda_matrices(factors_name=factors_name)
    assert set(lambda_factors.keys()) == set(LAMBDA_FACTOR_NAMES)
    assert len(lambda_factors[LAMBDA_MATRIX_NAME]) > 0
    for module_name in lambda_factors[LAMBDA_MATRIX_NAME]:
        assert lambda_factors[LAMBDA_MATRIX_NAME][module_name].dtype == lambda_dtype


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "gpt",
    ],
)
@pytest.mark.parametrize("strategy", ["diagonal", "ekfac"])
@pytest.mark.parametrize("train_size", [50])
@pytest.mark.parametrize("seed", [1])
def test_lambda_matrices_batch_size_equivalence(
    test_name: str,
    strategy: str,
    train_size: int,
    seed: int,
) -> None:
    # Lambda matrices should be identical regardless of the batch size used.
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
        strategy=strategy,
        use_empirical_fisher=True,
    )
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}_{strategy}_bs1",
        dataset=train_dataset,
        per_device_batch_size=1,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    bs1_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_name}_{strategy}_bs1",
    )

    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}_{strategy}_bs8",
        dataset=train_dataset,
        per_device_batch_size=8,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    bs8_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_name}_{strategy}_bs8",
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(bs1_lambda_factors[name], bs8_lambda_factors[name], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "gpt",
    ],
)
@pytest.mark.parametrize("strategy", ["diagonal", "ekfac"])
@pytest.mark.parametrize("data_partition_size", [1, 4])
@pytest.mark.parametrize("module_partition_size", [1, 3])
@pytest.mark.parametrize("train_size", [81])
@pytest.mark.parametrize("seed", [2])
def test_lambda_matrices_partition_equivalence(
    test_name: str,
    strategy: str,
    data_partition_size: int,
    module_partition_size: int,
    train_size: int,
    seed: int,
) -> None:
    # Covariance matrices should be identical regardless of the partition used.
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
        strategy=strategy,
        use_empirical_fisher=True,
    )
    factors_name = f"pytest_{test_name}_{strategy}_{test_lambda_matrices_partition_equivalence.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=factors_name,
    )

    factor_args = FactorArguments(
        strategy=strategy,
        use_empirical_fisher=True,
        lambda_data_partition_size=data_partition_size,
        lambda_module_partition_size=module_partition_size,
    )
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}_{strategy}_{data_partition_size}_{module_partition_size}",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=6,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    partitioned_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_name}_{strategy}_{data_partition_size}_{module_partition_size}",
    )
    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            lambda_factors[name], partitioned_lambda_factors[name], atol=ATOL, rtol=RTOL
        )


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "gpt",
    ],
)
@pytest.mark.parametrize("train_size", [63])
@pytest.mark.parametrize("seed", [3])
def test_lambda_matrices_iterative_aggregate(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Make sure aggregated lambda computation is working and the results are identical.
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

    factors_name = f"pytest_{test_name}_{test_lambda_matrices_iterative_aggregate.__name__}"
    factor_args = FactorArguments(
        use_empirical_fisher=True,
        lambda_iterative_aggregate=False,
    )
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=factors_name,
    )

    factor_args = FactorArguments(
        use_empirical_fisher=True,
        lambda_iterative_aggregate=True,
    )
    analyzer.fit_all_factors(
        factors_name=factors_name + "_iterative",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=4,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    iterative_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=factors_name + "_iterative",
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(lambda_factors[name], iterative_lambda_factors[name], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
    ],
)
@pytest.mark.parametrize("data_partition_size", [1, 4])
@pytest.mark.parametrize("train_size", [82])
@pytest.mark.parametrize("seed", [4])
def test_lambda_matrices_max_examples(
    test_name: str,
    data_partition_size: int,
    train_size: int,
    seed: int,
) -> None:
    # Make sure the max Lambda data selection is working.
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

    MAX_EXAMPLES = 28
    factor_args = FactorArguments(
        use_empirical_fisher=True, lambda_max_examples=MAX_EXAMPLES, lambda_data_partition_size=data_partition_size
    )
    factors_name = f"pytest_{test_name}_{test_lambda_matrices_max_examples.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=factors_name,
    )
    for num_examples in lambda_factors[NUM_LAMBDA_PROCESSED].values():
        assert num_examples == MAX_EXAMPLES
