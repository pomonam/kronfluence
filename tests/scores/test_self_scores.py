# pylint: skip-file

import pytest
import torch

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.constants import ALL_MODULE_NAME
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
        "conv",
        "conv_bn",
        "bert",
        "gpt",
    ],
)
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_size", [22])
@pytest.mark.parametrize("seed", [0])
def test_compute_self_scores(
    test_name: str,
    score_dtype: torch.dtype,
    train_size: int,
    seed: int,
) -> None:
    # Make sure that the self-influence computations are working properly.
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
    factors_name = f"pytest_{test_name}_{test_compute_self_scores.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(
        score_dtype=score_dtype,
    )
    scores_name = f"pytest_{test_name}_{test_compute_self_scores.__name__}_scores"
    analyzer.compute_self_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=4,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    self_scores = analyzer.load_self_scores(scores_name=scores_name)
    assert self_scores[ALL_MODULE_NAME].size(0) == train_size
    assert len(self_scores[ALL_MODULE_NAME].shape) == 1
    assert self_scores[ALL_MODULE_NAME].dtype == score_dtype


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
    ],
)
@pytest.mark.parametrize("per_sample_gradient_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("precondition_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [6])
def test_compute_self_scores_dtype(
    test_name: str,
    per_sample_gradient_dtype: torch.dtype,
    precondition_dtype: torch.dtype,
    score_dtype: torch.dtype,
    train_size: int,
    seed: int,
) -> None:
    # Make sure that the self-influence computations are working properly with different dtypes.
    model, train_dataset, _, data_collator, task = prepare_test(
        test_name=test_name,
        query_size=10,
        train_size=train_size,
        seed=seed,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    factors_name = f"pytest_{test_name}_{test_compute_self_scores_dtype.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(
        score_dtype=score_dtype,
        per_sample_gradient_dtype=per_sample_gradient_dtype,
        precondition_dtype=precondition_dtype,
    )
    scores_name = f"pytest_{test_name}_{test_compute_self_scores_dtype.__name__}_scores"
    analyzer.compute_self_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    self_scores = analyzer.load_self_scores(scores_name=scores_name)
    assert self_scores[ALL_MODULE_NAME].size(0) == train_size
    assert len(self_scores[ALL_MODULE_NAME].shape) == 1
    assert self_scores[ALL_MODULE_NAME].dtype == score_dtype


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "gpt",
    ],
)
@pytest.mark.parametrize("strategy", ["identity", "diagonal", "kfac", "ekfac"])
@pytest.mark.parametrize("train_size", [50])
@pytest.mark.parametrize("seed", [1])
def test_self_scores_batch_size_equivalence(
    test_name: str,
    strategy: str,
    train_size: int,
    seed: int,
) -> None:
    # Self-influence scores should be identical regardless of what batch size used.
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
        strategy=strategy,
    )
    factors_name = f"pytest_{test_name}_{test_self_scores_batch_size_equivalence.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=4,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(
        per_module_score=False,
        score_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        precondition_dtype=torch.float64,
    )
    analyzer.compute_self_scores(
        scores_name=f"pytest_{test_name}_{test_self_scores_batch_size_equivalence.__name__}_{strategy}_score_bs1",
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=1,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    bs1_scores = analyzer.load_self_scores(
        scores_name=f"pytest_{test_name}_{test_self_scores_batch_size_equivalence.__name__}_{strategy}_score_bs1",
    )

    analyzer.compute_self_scores(
        scores_name=f"pytest_{test_name}_{test_self_scores_batch_size_equivalence.__name__}_{strategy}_score_bs8",
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    bs8_scores = analyzer.load_self_scores(
        scores_name=f"pytest_{test_name}_{test_self_scores_batch_size_equivalence.__name__}_{strategy}_score_bs8",
    )

    assert check_tensor_dict_equivalence(
        bs1_scores,
        bs8_scores,
        atol=ATOL,
        rtol=RTOL,
    )

    analyzer.compute_self_scores(
        scores_name=f"pytest_{test_name}_{test_self_scores_batch_size_equivalence.__name__}_{strategy}_score_auto",
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=None,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    bs_auto_scores = analyzer.load_self_scores(
        scores_name=f"pytest_{test_name}_{test_self_scores_batch_size_equivalence.__name__}_{strategy}_score_auto",
    )

    assert check_tensor_dict_equivalence(
        bs1_scores,
        bs_auto_scores,
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
@pytest.mark.parametrize("train_size", [64])
@pytest.mark.parametrize("seed", [2])
def test_self_scores_partition_equivalence(
    test_name: str,
    data_partition_size: int,
    module_partition_size: int,
    train_size: int,
    seed: int,
) -> None:
    # Influence scores should be identical regardless of what the partition used.
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

    factors_name = f"pytest_{test_name}_{test_self_scores_partition_equivalence.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=8,
        overwrite_output_dir=True,
    )

    scores_name = f"pytest_{test_name}_{test_self_scores_partition_equivalence.__name__}_scores"
    score_args = ScoreArguments(
        score_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        precondition_dtype=torch.float64,
    )
    analyzer.compute_self_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_self_scores(scores_name=scores_name)

    score_args = ScoreArguments(
        data_partition_size=data_partition_size,
        module_partition_size=module_partition_size,
        score_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        precondition_dtype=torch.float64,
    )
    analyzer.compute_self_scores(
        scores_name=f"pytest_{test_name}_partition_{data_partition_size}_{module_partition_size}",
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=5,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    partitioned_scores = analyzer.load_self_scores(
        scores_name=f"pytest_{test_name}_partition_{data_partition_size}_{module_partition_size}",
    )

    assert check_tensor_dict_equivalence(
        scores,
        partitioned_scores,
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
@pytest.mark.parametrize("train_size", [64])
@pytest.mark.parametrize("seed", [4])
def test_per_module_scores_equivalence(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Influence scores should be identical with and without per module score computations.
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

    factors_name = f"pytest_{test_name}_{test_per_module_scores_equivalence.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=8,
        overwrite_output_dir=True,
    )

    scores_name = f"pytest_{test_name}_{test_per_module_scores_equivalence.__name__}_scores"
    analyzer.compute_self_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_self_scores(scores_name=scores_name)

    score_args = ScoreArguments(per_module_score=True)
    analyzer.compute_self_scores(
        scores_name=scores_name + "_per_module",
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    per_module_scores = analyzer.load_self_scores(scores_name=scores_name + "_per_module")

    total_scores = None
    for module_name in per_module_scores:
        if total_scores is None:
            total_scores = per_module_scores[module_name]
        else:
            total_scores.add_(per_module_scores[module_name])

    assert torch.allclose(total_scores, scores[ALL_MODULE_NAME], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv_bn",
        "gpt",
    ],
)
@pytest.mark.parametrize("train_size", [60])
@pytest.mark.parametrize("seed", [7])
def test_compute_self_scores_with_indices(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Make sure the indices selection is correctly implemented.
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
    factors_name = f"pytest_{test_name}_{test_compute_self_scores_with_indices.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(data_partition_size=2)
    scores_name = f"pytest_{test_name}_{test_compute_self_scores_with_indices.__name__}_scores"
    analyzer.compute_self_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        train_indices=list(range(48)),
        per_device_train_batch_size=8,
        score_args=score_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )

    self_scores = analyzer.load_self_scores(scores_name=scores_name)
    assert self_scores[ALL_MODULE_NAME].size(0) == 48


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
    ],
)
@pytest.mark.parametrize("train_size", [60])
@pytest.mark.parametrize("seed", [1])
def test_compute_self_scores_with_diagonal_pairwise_equivalence(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Self-influence scores should be identical to the diagonal entries of pairwise influence scores.
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
    factors_name = f"pytest_{test_name}_{test_compute_self_scores_with_diagonal_pairwise_equivalence.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        overwrite_output_dir=True,
    )

    scores_name = f"pytest_{test_name}_{test_compute_self_scores_with_diagonal_pairwise_equivalence.__name__}_scores"
    score_args = ScoreArguments(
        use_measurement_for_self_influence=False,
        score_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        precondition_dtype=torch.float64,
    )
    analyzer.compute_self_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    self_scores = analyzer.load_self_scores(scores_name=scores_name)

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        query_dataset=train_dataset,
        per_device_query_batch_size=6,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    pairwise_scores = analyzer.load_pairwise_scores(scores_name=scores_name)

    assert torch.allclose(
        torch.diag(pairwise_scores[ALL_MODULE_NAME]),
        self_scores[ALL_MODULE_NAME],
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "test_name",
    ["mlp", "conv", "conv_bn", "wrong_conv"],
)
@pytest.mark.parametrize("train_size", [24])
@pytest.mark.parametrize("seed", [7])
def test_compute_self_measurement_scores_with_diagonal_pairwise_equivalence(
    test_name: str,
    train_size: int,
    seed: int,
) -> None:
    # Self-influence scores should be identical to the diagonal entries of pairwise influence scores.
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
    factors_name = (
        f"pytest_{test_name}_{test_compute_self_measurement_scores_with_diagonal_pairwise_equivalence.__name__}"
    )
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        overwrite_output_dir=True,
    )

    scores_name = (
        f"pytest_{test_name}_{test_compute_self_measurement_scores_with_diagonal_pairwise_equivalence.__name__}_scores"
    )
    score_args = ScoreArguments(
        use_measurement_for_self_influence=True,
        score_dtype=torch.float64,
        per_sample_gradient_dtype=torch.float64,
        precondition_dtype=torch.float64,
    )
    analyzer.compute_self_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    self_scores = analyzer.load_self_scores(scores_name=scores_name)

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        query_dataset=train_dataset,
        per_device_query_batch_size=6,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    pairwise_scores = analyzer.load_pairwise_scores(scores_name=scores_name)

    assert torch.allclose(
        torch.diag(pairwise_scores[ALL_MODULE_NAME]),
        self_scores[ALL_MODULE_NAME],
        atol=ATOL,
        rtol=RTOL,
    )
