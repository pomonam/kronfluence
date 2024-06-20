# pylint: skip-file

import pytest
import torch

from kronfluence.arguments import FactorArguments
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
@pytest.mark.parametrize("activation_covariance_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gradient_covariance_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_size", [16])
@pytest.mark.parametrize("seed", [0])
def test_fit_covariance_matrices(
    test_name: str,
    activation_covariance_dtype: torch.dtype,
    gradient_covariance_dtype: torch.dtype,
    train_size: int,
    seed: int,
) -> None:
    # Make sure that the covariance computations are working properly.
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
        activation_covariance_dtype=activation_covariance_dtype,
        gradient_covariance_dtype=gradient_covariance_dtype,
    )
    factors_name = f"pytest_{test_name}_{test_fit_covariance_matrices.__name__}"
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        factor_args=factor_args,
        per_device_batch_size=train_size // 4,
        overwrite_output_dir=True,
    )
    covariance_factors = analyzer.load_covariance_matrices(
        factors_name=factors_name,
    )
    assert set(covariance_factors.keys()) == set(COVARIANCE_FACTOR_NAMES)
    assert len(covariance_factors[ACTIVATION_COVARIANCE_MATRIX_NAME]) > 0
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
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factor_args = FactorArguments(
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
    )
    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_name}_{test_covariance_matrices_batch_size_equivalence.__name__}_bs1",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=1,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    bs1_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=f"pytest_{test_name}_{test_covariance_matrices_batch_size_equivalence.__name__}_bs1"
    )

    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_name}_{test_covariance_matrices_batch_size_equivalence.__name__}_bs8",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    bs8_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=f"pytest_{test_name}_{test_covariance_matrices_batch_size_equivalence.__name__}_bs8"
    )

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
        "conv",
        "gpt",
    ],
)
@pytest.mark.parametrize("data_partition_size", [1, 4])
@pytest.mark.parametrize("module_partition_size", [1, 3])
@pytest.mark.parametrize("train_size", [62])
@pytest.mark.parametrize("seed", [2])
def test_covariance_matrices_partition_equivalence(
    test_name: str,
    data_partition_size: int,
    module_partition_size: int,
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
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factor_args = FactorArguments(
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
    )
    factors_name = f"pytest_{test_name}_{test_covariance_matrices_partition_equivalence.__name__}"
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(factors_name=factors_name)

    factor_args = FactorArguments(
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
        covariance_data_partition_size=data_partition_size,
        covariance_module_partition_size=module_partition_size,
    )
    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_name}_partitioned_{data_partition_size}_{module_partition_size}",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=7,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    partitioned_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=f"pytest_{test_name}_partitioned_{data_partition_size}_{module_partition_size}",
    )

    for name in COVARIANCE_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            covariance_factors[name],
            partitioned_covariance_factors[name],
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize("test_name", ["bert", "wrong_bert", "gpt"])
@pytest.mark.parametrize("train_size", [213])
@pytest.mark.parametrize("seed", [3])
def test_covariance_matrices_attention_mask(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Make sure the attention mask is correctly implemented by comparing with the results
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

    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factor_args = FactorArguments(
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
    )
    factors_name = f"pytest_{test_name}_{test_covariance_matrices_attention_mask.__name__}"
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=train_size // 4,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(
        factors_name=factors_name,
    )

    analyzer.fit_covariance_matrices(
        factors_name=factors_name + "_no_pad",
        dataset=no_padded_train_dataset,
        factor_args=factor_args,
        per_device_batch_size=1,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    no_padded_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=factors_name + "_no_pad",
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
    # Make sure the automatic batch size search feature is working properly.
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
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
    )
    factors_name = f"pytest_{test_name}_{test_covariance_matrices_automatic_batch_size.__name__}"
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(factors_name=factors_name)

    analyzer.fit_covariance_matrices(
        factors_name=factors_name + "_auto",
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=None,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    auto_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=factors_name + "_auto",
    )

    for name in COVARIANCE_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            covariance_factors[name],
            auto_covariance_factors[name],
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize("test_name", ["mlp"])
@pytest.mark.parametrize("data_partition_size", [1, 4])
@pytest.mark.parametrize("train_size", [80])
@pytest.mark.parametrize("seed", [5])
def test_covariance_matrices_max_examples(
    test_name: str,
    data_partition_size: int,
    train_size: int,
    seed: int,
) -> None:
    # Make sure the max covariance data selection is working properly.
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

    MAX_EXAMPLES = 26
    factor_args = FactorArguments(
        use_empirical_fisher=True,
        covariance_max_examples=MAX_EXAMPLES,
        covariance_data_partition_size=data_partition_size,
    )
    factors_name = f"pytest_{test_name}_{test_covariance_matrices_max_examples.__name__}"
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=train_dataset,
        factor_args=factor_args,
        per_device_batch_size=32,
        overwrite_output_dir=True,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(factors_name=factors_name)

    for num_examples in covariance_factors[NUM_ACTIVATION_COVARIANCE_PROCESSED].values():
        assert num_examples == MAX_EXAMPLES

    for num_examples in covariance_factors[NUM_GRADIENT_COVARIANCE_PROCESSED].values():
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
def test_covariance_matrices_amp(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Covariance matrices should be similar when AMP is enabled.
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
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
    )
    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_name}_{test_covariance_matrices_amp.__name__}",
        dataset=train_dataset,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    covariance_factors = analyzer.load_covariance_matrices(
        factors_name=f"pytest_{test_name}_{test_covariance_matrices_amp.__name__}"
    )

    factor_args = FactorArguments(
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
        amp_dtype=torch.float16,
    )
    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_name}_{test_covariance_matrices_amp.__name__}_amp",
        dataset=train_dataset,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        factor_args=factor_args,
        dataloader_kwargs=kwargs,
    )
    amp_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=f"pytest_{test_name}_{test_covariance_matrices_amp.__name__}_amp"
    )

    for name in COVARIANCE_FACTOR_NAMES:
        assert check_tensor_dict_equivalence(
            covariance_factors[name],
            amp_covariance_factors[name],
            atol=1e-01,
            rtol=1e-02,
        )


@pytest.mark.parametrize("train_size", [100])
@pytest.mark.parametrize("seed", [12])
def test_covariance_matrices_gradient_checkpoint(
    train_size: int,
    seed: int,
) -> None:
    # Covariance matrices should be the same even when gradient checkpointing is used.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name="mlp",
        train_size=train_size,
        seed=seed,
    )
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factor_args = FactorArguments(
        use_empirical_fisher=True,
        activation_covariance_dtype=torch.float64,
        gradient_covariance_dtype=torch.float64,
    )
    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_covariance_matrices_gradient_checkpoint.__name__}",
        dataset=train_dataset,
        per_device_batch_size=8,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    covariance_factors = analyzer.load_covariance_matrices(
        factors_name=f"pytest_{test_covariance_matrices_gradient_checkpoint.__name__}",
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
    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_covariance_matrices_gradient_checkpoint.__name__}_cp",
        dataset=train_dataset,
        per_device_batch_size=4,
        overwrite_output_dir=True,
        factor_args=factor_args,
    )
    checkpoint_covariance_factors = analyzer.load_covariance_matrices(
        factors_name=f"pytest_{test_covariance_matrices_gradient_checkpoint.__name__}_cp",
    )

    assert check_tensor_dict_equivalence(
        covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME],
        checkpoint_covariance_factors[GRADIENT_COVARIANCE_MATRIX_NAME],
        atol=ATOL,
        rtol=RTOL,
    )
