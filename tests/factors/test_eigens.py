# pylint: skip-file

import pytest
import torch

from kronfluence.arguments import FactorArguments
from kronfluence.utils.common.factor_arguments import test_factor_arguments
from kronfluence.utils.constants import (
    ACTIVATION_EIGENVECTORS_NAME,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_FACTOR_NAMES,
    LAMBDA_MATRIX_NAME,
    NUM_LAMBDA_PROCESSED,
)
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import (
    ATOL,
    RTOL,
    check_tensor_dict_equivalence,
    prepare_model_and_analyzer,
    prepare_test,
)


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "repeated_mlp",
        "mlp_checkpoint",
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
    factors_name = f"pytest_{test_name}_{test_perform_eigendecomposition.__name__}"
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        factor_args=factor_args,
        dataset=train_dataset,
        per_device_batch_size=4,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
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
        "repeated_mlp",
        "mlp_checkpoint",
        "conv",
        "conv_bn",
        "bert",
        "gpt",
    ],
)
@pytest.mark.parametrize("per_sample_gradient_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lambda_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_size", [16])
@pytest.mark.parametrize("seed", [0])
def test_fit_lambda_matrices(
    test_name: str,
    per_sample_gradient_dtype: torch.dtype,
    lambda_dtype: torch.dtype,
    train_size: int,
    seed: int,
) -> None:
    # Makes sure that the Lambda computations are working properly.
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
        per_sample_gradient_dtype=per_sample_gradient_dtype,
    )
    if test_name == "repeated_mlp":
        factor_args.shared_parameters_exist = True

    factors_name = f"pytest_{test_name}_{test_fit_lambda_matrices.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        per_device_batch_size=train_size // 4,
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
    # Lambda matrices should be identical regardless of what batch size used.
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

    factor_args = test_factor_arguments(strategy=strategy)
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}_{test_lambda_matrices_batch_size_equivalence.__name__}_{strategy}_bs1",
        dataset=train_dataset,
        per_device_batch_size=1,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    bs1_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_name}_{test_lambda_matrices_batch_size_equivalence.__name__}_{strategy}_bs1",
    )

    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}_{test_lambda_matrices_batch_size_equivalence.__name__}_{strategy}_bs8",
        dataset=train_dataset,
        per_device_batch_size=8,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    bs8_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_name}_{test_lambda_matrices_batch_size_equivalence.__name__}_{strategy}_bs8",
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(bs1_lambda_factors[name], bs8_lambda_factors[name], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
    ],
)
@pytest.mark.parametrize("strategy", ["diagonal", "ekfac"])
@pytest.mark.parametrize("data_partition_size", [4])
@pytest.mark.parametrize("module_partition_size", [3])
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
    # Lambda matrices should be identical regardless of what the partition used.
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

    factor_args = test_factor_arguments(strategy=strategy)
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

    factor_args.lambda_data_partition_size = data_partition_size
    factor_args.lambda_module_partition_size = module_partition_size
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
        "bert",
    ],
)
@pytest.mark.parametrize("train_size", [63])
@pytest.mark.parametrize("seed", [3])
def test_lambda_matrices_iterative_aggregate(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Makes sure iterative lambda computation is working properly.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    model = model.to(dtype=torch.float64)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factors_name = f"pytest_{test_name}_{test_lambda_matrices_iterative_aggregate.__name__}"
    factor_args = test_factor_arguments()
    factor_args.lambda_iterative_aggregate = False
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

    factor_args.lambda_iterative_aggregate = True
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
    ["mlp", "conv"],
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
    # Makes sure the max Lambda data selection is working properly.
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

    MAX_EXAMPLES = 33
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


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
    ],
)
@pytest.mark.parametrize("train_size", [100])
@pytest.mark.parametrize("seed", [8])
def test_lambda_matrices_amp(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Lambda matrices should be similar when AMP is enabled.
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

    factor_args = test_factor_arguments()
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}_{test_lambda_matrices_amp.__name__}",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_name}_{test_lambda_matrices_amp.__name__}"
    )

    factor_args.amp_dtype = torch.float16
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}_{test_lambda_matrices_amp.__name__}_amp",
        dataset=train_dataset,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    amp_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_name}_{test_lambda_matrices_amp.__name__}_amp",
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(lambda_factors[name], amp_lambda_factors[name], atol=1e-01, rtol=1e-02)


@pytest.mark.parametrize("train_size", [105])
@pytest.mark.parametrize("seed", [12])
def test_lambda_matrices_gradient_checkpoint(
    train_size: int,
    seed: int,
) -> None:
    # Lambda matrices should be the same even when gradient checkpointing is used.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name="mlp",
        train_size=train_size,
        seed=seed,
    )
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factor_args = test_factor_arguments()
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_lambda_matrices_gradient_checkpoint.__name__}",
        dataset=train_dataset,
        per_device_batch_size=5,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_lambda_matrices_gradient_checkpoint.__name__}",
    )

    model, _, _, _, task = prepare_test(
        test_name="mlp_checkpoint",
        train_size=train_size,
        seed=seed,
    )
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_lambda_matrices_gradient_checkpoint.__name__}_cp",
        dataset=train_dataset,
        per_device_batch_size=6,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    checkpoint_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_lambda_matrices_gradient_checkpoint.__name__}_cp",
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            lambda_factors[name], checkpoint_lambda_factors[name], atol=ATOL, rtol=RTOL
        )


@pytest.mark.parametrize("train_size", [105])
@pytest.mark.parametrize("seed", [12])
def test_lambda_matrices_shared_parameters(
    train_size: int,
    seed: int,
) -> None:
    # When there are no shared parameters, results with and without `shared_parameters_exist` should
    # produce the same results.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name="mlp",
        train_size=train_size,
        seed=seed,
    )
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factor_args = test_factor_arguments()
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_lambda_matrices_shared_parameters.__name__}",
        dataset=train_dataset,
        per_device_batch_size=5,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_lambda_matrices_shared_parameters.__name__}",
    )

    model, train_dataset, _, _, task = prepare_test(
        test_name="mlp",
        train_size=train_size,
        seed=seed,
    )
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factor_args.shared_parameters_exist = True
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_lambda_matrices_shared_parameters.__name__}_shared",
        dataset=train_dataset,
        per_device_batch_size=6,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    checkpoint_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=f"pytest_{test_lambda_matrices_shared_parameters.__name__}_shared",
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            lambda_factors[name], checkpoint_lambda_factors[name], atol=ATOL, rtol=RTOL
        )
