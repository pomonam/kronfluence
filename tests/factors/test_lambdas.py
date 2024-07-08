# pylint: skip-file

import pytest
import torch

from kronfluence.arguments import FactorArguments
from kronfluence.utils.common.factor_arguments import pytest_factor_arguments
from kronfluence.utils.constants import (
    LAMBDA_FACTOR_NAMES,
    LAMBDA_MATRIX_NAME,
    NUM_LAMBDA_PROCESSED,
)
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import (
    ATOL,
    DEFAULT_FACTORS_NAME,
    RTOL,
    check_tensor_dict_equivalence,
    custom_factors_name,
    prepare_model_and_analyzer,
    prepare_test,
)


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "repeated_mlp",
        "conv",
        "bert",
        "roberta",
        "gpt",
        "gpt_checkpoint",
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
        factor_args.has_shared_parameters = True

    analyzer.fit_all_factors(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=train_size // 4,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )

    lambda_factors = analyzer.load_lambda_matrices(factors_name=DEFAULT_FACTORS_NAME)
    assert set(lambda_factors.keys()) == set(LAMBDA_FACTOR_NAMES)
    assert len(lambda_factors[LAMBDA_MATRIX_NAME]) > 0
    for module_name in lambda_factors[LAMBDA_MATRIX_NAME]:
        assert lambda_factors[LAMBDA_MATRIX_NAME][module_name].dtype == lambda_dtype


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "roberta",
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
    model = model.to(dtype=torch.float64)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factor_args = pytest_factor_arguments(strategy=strategy)
    analyzer.fit_all_factors(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=1,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    bs1_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )

    analyzer.fit_all_factors(
        factors_name=custom_factors_name("bs8"),
        dataset=train_dataset,
        per_device_batch_size=8,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    bs8_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=custom_factors_name("bs8"),
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(bs1_lambda_factors[name], bs8_lambda_factors[name], atol=ATOL, rtol=RTOL)

    analyzer.fit_all_factors(
        factors_name=custom_factors_name("auto"),
        dataset=train_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    auto_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=custom_factors_name("auto"),
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(bs1_lambda_factors[name], auto_lambda_factors[name], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("test_name", ["mlp", "conv_bn"])
@pytest.mark.parametrize("strategy", ["diagonal", "ekfac"])
@pytest.mark.parametrize("data_partitions", [2, 4])
@pytest.mark.parametrize("module_partitions", [2, 3])
@pytest.mark.parametrize("train_size", [81])
@pytest.mark.parametrize("seed", [2])
def test_lambda_matrices_partition_equivalence(
    test_name: str,
    strategy: str,
    data_partitions: int,
    module_partitions: int,
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

    factor_args = pytest_factor_arguments(strategy=strategy)
    analyzer.fit_all_factors(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )

    factor_args.lambda_data_partitions = data_partitions
    factor_args.lambda_module_partitions = module_partitions
    analyzer.fit_all_factors(
        factors_name=custom_factors_name(f"{data_partitions}_{module_partitions}"),
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=6,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    partitioned_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=custom_factors_name(f"{data_partitions}_{module_partitions}"),
    )
    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            lambda_factors[name], partitioned_lambda_factors[name], atol=ATOL, rtol=RTOL
        )


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv_bn",
        "bert",
        "gpt",
    ],
)
@pytest.mark.parametrize("train_size", [63, 121])
@pytest.mark.parametrize("seed", [3])
def test_lambda_matrices_iterative_lambda_aggregation(
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

    factor_args = pytest_factor_arguments()
    factor_args.use_iterative_lambda_aggregation = False
    analyzer.fit_all_factors(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )

    factor_args.use_iterative_lambda_aggregation = True
    analyzer.fit_all_factors(
        factors_name=custom_factors_name("iterative"),
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=16,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    iterative_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=custom_factors_name("iterative"),
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(lambda_factors[name], iterative_lambda_factors[name], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "test_name",
    ["conv_bn", "gpt"],
)
@pytest.mark.parametrize("max_examples", [4, 31])
@pytest.mark.parametrize("data_partitions", [1, 3])
@pytest.mark.parametrize("module_partitions", [1, 2])
@pytest.mark.parametrize("train_size", [82])
@pytest.mark.parametrize("seed", [4])
def test_lambda_matrices_max_examples(
    test_name: str,
    max_examples: int,
    data_partitions: int,
    module_partitions: int,
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

    factor_args = FactorArguments(
        lambda_max_examples=max_examples,
        lambda_data_partitions=data_partitions,
        lambda_module_partitions=module_partitions,
    )
    analyzer.fit_all_factors(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )
    for num_examples in lambda_factors[NUM_LAMBDA_PROCESSED].values():
        assert num_examples == max_examples


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "gpt",
    ],
)
@pytest.mark.parametrize("train_size", [105])
@pytest.mark.parametrize("seed", [6])
def test_lambda_matrices_gradient_checkpoint(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Lambda matrices should be the same even when gradient checkpointing is used.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    model = model.to(dtype=torch.float64)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)

    factor_args = pytest_factor_arguments()
    analyzer.fit_all_factors(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=5,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )

    model, _, _, _, task = prepare_test(
        test_name=test_name + "_checkpoint",
        train_size=train_size,
        seed=seed,
    )
    model = model.to(dtype=torch.float64)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    analyzer.fit_all_factors(
        factors_name=custom_factors_name("cp"),
        dataset=train_dataset,
        per_device_batch_size=6,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    checkpoint_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=custom_factors_name("cp"),
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            lambda_factors[name], checkpoint_lambda_factors[name], atol=ATOL, rtol=RTOL
        )


@pytest.mark.parametrize(
    "test_name",
    ["mlp", "conv", "gpt"],
)
@pytest.mark.parametrize("train_size", [105])
@pytest.mark.parametrize("seed", [7])
def test_lambda_matrices_shared_parameters(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # When there are no shared parameters, results with and without `has_shared_parameters` should
    # produce the same results.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    model = model.to(dtype=torch.float64)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)

    factor_args = pytest_factor_arguments()
    analyzer.fit_all_factors(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=5,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )

    factor_args.has_shared_parameters = True
    analyzer.fit_all_factors(
        factors_name=custom_factors_name("shared"),
        dataset=train_dataset,
        per_device_batch_size=6,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    shared_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=custom_factors_name("shared"),
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(lambda_factors[name], shared_lambda_factors[name], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("train_size", [121])
@pytest.mark.parametrize("seed", [8])
def test_lambda_matrices_inplace(
    train_size: int,
    seed: int,
) -> None:
    # Lambda matrices should be the identical for with and without in-place ReLU.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name="conv",
        train_size=train_size,
        seed=seed,
    )
    model = model.to(dtype=torch.float64)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)

    factor_args = pytest_factor_arguments()
    analyzer.fit_all_factors(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=5,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    lambda_factors = analyzer.load_lambda_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )

    model, _, _, _, task = prepare_test(
        test_name="conv_inplace",
        train_size=train_size,
        seed=seed,
    )
    model = model.to(dtype=torch.float64)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    analyzer.fit_all_factors(
        factors_name=custom_factors_name("inplace"),
        dataset=train_dataset,
        per_device_batch_size=6,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    inplace_lambda_factors = analyzer.load_lambda_matrices(
        factors_name=custom_factors_name("inplace"),
    )

    for name in LAMBDA_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(lambda_factors[name], inplace_lambda_factors[name], atol=ATOL, rtol=RTOL)
