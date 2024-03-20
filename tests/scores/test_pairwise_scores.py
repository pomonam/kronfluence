# pylint: skip-file

from typing import Optional, Tuple

import pytest
import torch
from torch import nn

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task
from kronfluence.utils.constants import ALL_MODULE_NAME
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
        "gpt",
    ],
)
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("query_gradient_rank", [None, 16])
@pytest.mark.parametrize("query_size", [16])
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [0])
def test_compute_pairwise_scores(
    test_name: str,
    score_dtype: torch.dtype,
    query_gradient_rank: Optional[int],
    query_size: int,
    train_size: int,
    seed: int,
) -> None:
    model, train_dataset, test_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        query_size=query_size,
        train_size=train_size,
        seed=seed,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    factors_name = f"pytest_{test_name}_{test_compute_pairwise_scores.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(
        score_dtype=score_dtype,
        query_gradient_rank=query_gradient_rank,
    )
    scores_name = f"pytest_{test_name}_{test_compute_pairwise_scores.__name__}_{query_gradient_rank}_scores"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    pairwise_scores = analyzer.load_pairwise_scores(scores_name=scores_name)
    assert pairwise_scores[ALL_MODULE_NAME].size(0) == query_size
    assert pairwise_scores[ALL_MODULE_NAME].size(1) == train_size
    assert pairwise_scores[ALL_MODULE_NAME].dtype == score_dtype


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
    ],
)
@pytest.mark.parametrize("per_sample_gradient_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("precondition_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("query_gradient_rank", [None, 16])
@pytest.mark.parametrize("query_size", [16])
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [6])
def test_compute_pairwise_scores_dtype(
    test_name: str,
    per_sample_gradient_dtype: torch.dtype,
    precondition_dtype: torch.dtype,
    score_dtype: torch.dtype,
    query_gradient_rank: Optional[int],
    query_size: int,
    train_size: int,
    seed: int,
) -> None:
    model, train_dataset, test_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        query_size=query_size,
        train_size=train_size,
        seed=seed,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    factors_name = f"pytest_{test_name}_{test_compute_pairwise_scores_dtype.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(
        score_dtype=score_dtype,
        query_gradient_rank=query_gradient_rank,
        per_sample_gradient_dtype=per_sample_gradient_dtype,
        precondition_dtype=precondition_dtype,
    )
    scores_name = f"pytest_{test_name}_{test_compute_pairwise_scores_dtype.__name__}_{query_gradient_rank}_scores"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    pairwise_scores = analyzer.load_pairwise_scores(scores_name=scores_name)
    assert pairwise_scores[ALL_MODULE_NAME].size(0) == query_size
    assert pairwise_scores[ALL_MODULE_NAME].size(1) == train_size
    assert pairwise_scores[ALL_MODULE_NAME].dtype == score_dtype


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "conv",
        "gpt",
    ],
)
@pytest.mark.parametrize("strategy", ["identity", "diagonal", "kfac", "ekfac"])
@pytest.mark.parametrize("query_size", [20])
@pytest.mark.parametrize("train_size", [50])
@pytest.mark.parametrize("seed", [1])
def test_pairwise_scores_batch_size_equivalence(
    test_name: str,
    strategy: str,
    query_size: int,
    train_size: int,
    seed: int,
) -> None:
    model, train_dataset, test_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        query_size=query_size,
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
    )
    factors_name = f"pytest_{test_name}_{test_pairwise_scores_batch_size_equivalence.__name__}"
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
    )
    analyzer.compute_pairwise_scores(
        scores_name=f"pytest_{test_name}_{strategy}_score_bs1",
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        per_device_train_batch_size=1,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    bs1_scores = analyzer.load_pairwise_scores(
        scores_name=f"pytest_{test_name}_{strategy}_score_bs1",
    )

    analyzer.compute_pairwise_scores(
        scores_name=f"pytest_{test_name}_{strategy}_score_bs8",
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=3,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    bs8_scores = analyzer.load_pairwise_scores(
        scores_name=f"pytest_{test_name}_{strategy}_score_bs8",
    )

    assert check_tensor_dict_equivalence(
        bs1_scores,
        bs8_scores,
        atol=ATOL,
        rtol=RTOL,
    )

    analyzer.compute_pairwise_scores(
        scores_name=f"pytest_{test_name}_{strategy}_score_bs_auto",
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=10,
        train_dataset=train_dataset,
        per_device_train_batch_size=None,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    bs_auto_scores = analyzer.load_pairwise_scores(
        scores_name=f"pytest_{test_name}_{strategy}_score_bs_auto",
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
@pytest.mark.parametrize("per_module_score", [True, False])
@pytest.mark.parametrize("query_size", [32])
@pytest.mark.parametrize("train_size", [64])
@pytest.mark.parametrize("seed", [2])
def test_pairwise_scores_partition_equivalence(
    test_name: str,
    data_partition_size: int,
    module_partition_size: int,
    per_module_score: bool,
    query_size: int,
    train_size: int,
    seed: int,
) -> None:
    model, train_dataset, test_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        query_size=query_size,
        train_size=train_size,
        seed=seed,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )

    factors_name = f"pytest_{test_name}_{test_pairwise_scores_partition_equivalence.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=8,
        overwrite_output_dir=True,
    )

    scores_name = f"pytest_{test_name}_{test_pairwise_scores_partition_equivalence.__name__}_scores"
    score_args = ScoreArguments(
        per_module_score=per_module_score,
    )
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(scores_name=scores_name)

    score_args = ScoreArguments(
        data_partition_size=data_partition_size,
        module_partition_size=module_partition_size,
        per_module_score=per_module_score,
    )
    analyzer.compute_pairwise_scores(
        scores_name=f"pytest_{test_name}_partition_{data_partition_size}_{module_partition_size}",
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=10,
        train_dataset=train_dataset,
        per_device_train_batch_size=5,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    partitioned_scores = analyzer.load_pairwise_scores(
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
@pytest.mark.parametrize("query_size", [32])
@pytest.mark.parametrize("train_size", [64])
@pytest.mark.parametrize("seed", [0])
def test_per_module_scores_equivalence(
    test_name: str,
    query_size: int,
    train_size: int,
    seed: int,
) -> None:
    model, train_dataset, test_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        query_size=query_size,
        train_size=train_size,
        seed=seed,
    )
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
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(scores_name=scores_name)

    score_args = ScoreArguments(per_module_score=True)
    analyzer.compute_pairwise_scores(
        scores_name=scores_name + "_per_module",
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    per_module_scores = analyzer.load_pairwise_scores(scores_name=scores_name + "_per_module")

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
@pytest.mark.parametrize("query_size", [60])
@pytest.mark.parametrize("train_size", [60])
@pytest.mark.parametrize("seed", [0])
def test_compute_pairwise_scores_with_indices(
    test_name: str,
    query_size: int,
    train_size: int,
    seed: int,
) -> None:
    model, train_dataset, test_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        query_size=query_size,
        seed=seed,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    model, analyzer = prepare_model_and_analyzer(
        model=model,
        task=task,
    )
    factors_name = f"pytest_{test_name}_{test_compute_pairwise_scores_with_indices.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(data_partition_size=2)
    scores_name = f"pytest_{test_name}_{test_compute_pairwise_scores_with_indices.__name__}_scores"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=test_dataset,
        query_indices=list(range(30)),
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        train_indices=list(range(50)),
        per_device_train_batch_size=8,
        score_args=score_args,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )

    pairwise_scores = analyzer.load_pairwise_scores(scores_name=scores_name)
    assert pairwise_scores[ALL_MODULE_NAME].size(0) == 30
    assert pairwise_scores[ALL_MODULE_NAME].size(1) == 50
