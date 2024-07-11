# pylint: skip-file

import pytest
import torch

from kronfluence.utils.common.factor_arguments import (
    default_factor_arguments,
    pytest_factor_arguments,
)
from kronfluence.utils.constants import (
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    COVARIANCE_FACTOR_NAMES,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    NUM_ACTIVATION_COVARIANCE_PROCESSED,
    NUM_GRADIENT_COVARIANCE_PROCESSED,
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
@pytest.mark.parametrize("activation_covariance_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("gradient_covariance_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("train_size", [16])
@pytest.mark.parametrize("seed", [0])
def test_fit_covariance_matrices(
    test_name: str,
    activation_covariance_dtype: torch.dtype,
    gradient_covariance_dtype: torch.dtype,
    train_size: int,
    seed: int,
) -> None:
    # Makes sure that the covariance computations are working properly.
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
    factor_args = default_factor_arguments()
    factor_args.activation_covariance_dtype = activation_covariance_dtype
    factor_args.gradient_covariance_dtype = gradient_covariance_dtype
    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        factor_args=factor_args,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=train_size // 4,
        overwrite_output_dir=True,
    )
    covariance_factors = analyzer.load_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )
    assert set(covariance_factors.keys()) == set(COVARIANCE_FACTOR_NAMES)
    assert len(covariance_factors[ACTIVATION_COVARIANCE_MATRIX_NAME]) > 0
    assert len(covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME]) > 0
    for module_name in covariance_factors[ACTIVATION_COVARIANCE_MATRIX_NAME]:
        assert covariance_factors[ACTIVATION_COVARIANCE_MATRIX_NAME][module_name].dtype == activation_covariance_dtype
        assert covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME][module_name].dtype == gradient_covariance_dtype


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "gpt",
    ],
)
@pytest.mark.parametrize("train_size", [100])
@pytest.mark.parametrize("seed", [1])
def test_covariance_matrices_batch_size_equivalence(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Covariance matrices should be identical regardless of what batch size used.
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
    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=1,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    bs1_covariance_factors = analyzer.load_covariance_matrices(factors_name=DEFAULT_FACTORS_NAME)

    analyzer.fit_covariance_matrices(
        factors_name=custom_factors_name(name="bs8"),
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    bs8_covariance_factors = analyzer.load_covariance_matrices(factors_name=custom_factors_name(name="bs8"))

    for name in COVARIANCE_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            bs1_covariance_factors[name],
            bs8_covariance_factors[name],
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv_bn",
        "bert",
    ],
)
@pytest.mark.parametrize("data_partitions", [2, 4])
@pytest.mark.parametrize("module_partitions", [2, 3])
@pytest.mark.parametrize("train_size", [62])
@pytest.mark.parametrize("seed", [2])
def test_covariance_matrices_partition_equivalence(
    test_name: str,
    data_partitions: int,
    module_partitions: int,
    train_size: int,
    seed: int,
) -> None:
    # Covariance matrices should be identical regardless of what the partition used.
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
    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(factors_name=DEFAULT_FACTORS_NAME)

    factor_args.covariance_data_partitions = data_partitions
    factor_args.covariance_module_partitions = module_partitions
    analyzer.fit_covariance_matrices(
        factors_name=custom_factors_name(f"{data_partitions}_{module_partitions}"),
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=7,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    partitioned_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=custom_factors_name(f"{data_partitions}_{module_partitions}"),
    )

    for name in COVARIANCE_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            covariance_factors[name],
            partitioned_covariance_factors[name],
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize("test_name", ["bert", "wrong_bert", "roberta"])
@pytest.mark.parametrize("train_size", [213])
@pytest.mark.parametrize("seed", [3])
def test_covariance_matrices_attention_mask(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Makes sure the attention mask is correctly implemented by comparing with the results
    # without any padding applied (and batch size of 1).
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
    )
    _, no_padded_train_dataset, _, _, _ = prepare_test(
        test_name=test_name,
        train_size=train_size,
        seed=seed,
        do_not_pad=True,
    )

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
    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=train_size // 4,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
    )

    analyzer.fit_covariance_matrices(
        factors_name=custom_factors_name("no_pad"),
        dataset=no_padded_train_dataset,
        factor_args=factor_args,
        per_device_batch_size=1,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    no_padded_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=custom_factors_name("no_pad"),
    )

    for name in COVARIANCE_FACTOR_NAMES:
        if "wrong" in test_name and name in [
            ACTIVATION_COVARIANCE_MATRIX_NAME,
            NUM_ACTIVATION_COVARIANCE_PROCESSED,
            NUM_GRADIENT_COVARIANCE_PROCESSED,
        ]:
            assert not check_tensor_dict_equivalence(
                covariance_factors[name],
                no_padded_covariance_factors[name],
                atol=ATOL,
                rtol=RTOL,
            )
        else:
            assert check_tensor_dict_equivalence(
                covariance_factors[name],
                no_padded_covariance_factors[name],
                atol=ATOL,
                rtol=RTOL,
            )


@pytest.mark.parametrize("test_name", ["mlp"])
@pytest.mark.parametrize("train_size", [62])
@pytest.mark.parametrize("seed", [4])
def test_covariance_matrices_automatic_batch_size(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Makes sure the automatic batch size search feature is working properly.
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
    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(factors_name=DEFAULT_FACTORS_NAME)

    analyzer.fit_covariance_matrices(
        factors_name=custom_factors_name("auto"),
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=None,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    auto_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=custom_factors_name("auto"),
    )

    for name in COVARIANCE_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            covariance_factors[name],
            auto_covariance_factors[name],
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize("test_name", ["mlp"])
@pytest.mark.parametrize("max_examples", [4, 26])
@pytest.mark.parametrize("data_partitions", [1, 4])
@pytest.mark.parametrize("module_partitions", [1, 2])
@pytest.mark.parametrize("train_size", [80])
@pytest.mark.parametrize("seed", [5])
def test_covariance_matrices_max_examples(
    test_name: str,
    max_examples: int,
    data_partitions: int,
    module_partitions: int,
    train_size: int,
    seed: int,
) -> None:
    # Makes sure the max covariance data selection is working properly.
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

    factor_args = pytest_factor_arguments()
    factor_args.covariance_max_examples = max_examples
    factor_args.covariance_data_partitions = data_partitions
    factor_args.covariance_module_partitions = module_partitions

    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=32,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(factors_name=DEFAULT_FACTORS_NAME)

    for num_examples in covariance_factors[NUM_ACTIVATION_COVARIANCE_PROCESSED].values():
        assert num_examples == max_examples

    for num_examples in covariance_factors[NUM_GRADIENT_COVARIANCE_PROCESSED].values():
        assert num_examples == max_examples


@pytest.mark.parametrize("test_name", ["mlp", "gpt"])
@pytest.mark.parametrize("train_size", [100])
@pytest.mark.parametrize("seed", [6])
def test_covariance_matrices_gradient_checkpoint(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Covariance matrices should be the same even when gradient checkpointing is used.
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
    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
        factor_args=factor_args,
    )
    covariance_factors = analyzer.load_covariance_matrices(
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
    analyzer.fit_covariance_matrices(
        factors_name=custom_factors_name("cp"),
        dataset=train_dataset,
        per_device_batch_size=4,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    checkpoint_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=custom_factors_name("cp"),
    )

    assert check_tensor_dict_equivalence(
        covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME],
        checkpoint_covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME],
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("train_size", [100])
@pytest.mark.parametrize("seed", [7, 8])
def test_covariance_matrices_inplace(
    train_size: int,
    seed: int,
) -> None:
    # Covariance matrices should be the identical for with and without in-place ReLU.
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

    factor_args = pytest_factor_arguments()
    analyzer.fit_covariance_matrices(
        factors_name=DEFAULT_FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    covariance_factors = analyzer.load_covariance_matrices(
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
    analyzer.fit_covariance_matrices(
        factors_name=custom_factors_name("inplace"),
        dataset=train_dataset,
        per_device_batch_size=4,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    inplace_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=custom_factors_name("inplace"),
    )

    assert check_tensor_dict_equivalence(
        covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME],
        inplace_covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME],
        atol=ATOL,
        rtol=RTOL,
    )
