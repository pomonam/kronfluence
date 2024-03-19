# pylint: skip-file

import pytest
import torch

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import prepare_test


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
@pytest.mark.parametrize(
    "strategy",
    [
        "identity",
        "diagonal",
        "kfac",
        "ekfac",
    ],
)
@pytest.mark.parametrize("seed", [0])
def test_analyzer(
    test_name: str,
    strategy: str,
    seed: int,
    train_size: int = 32,
    query_size: int = 4,
) -> None:
    model, train_dataset, query_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        query_size=query_size,
        seed=seed,
    )
    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name=f"pytest_{test_name}",
        model=model,
        task=task,
        disable_model_save=True,
        cpu=True,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    factor_args = FactorArguments(strategy=strategy)
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_name}",
        dataset=train_dataset,
        per_device_batch_size=16,
        dataloader_kwargs=kwargs,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    analyzer.compute_pairwise_scores(
        scores_name="pairwise",
        factors_name=f"pytest_{test_name}",
        query_dataset=query_dataset,
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    analyzer.compute_self_scores(
        scores_name="self",
        factors_name=f"pytest_{test_name}",
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )


def test_default_factor_arguments() -> None:
    factor_args = FactorArguments()

    assert factor_args.strategy == "ekfac"
    assert factor_args.use_empirical_fisher is False
    assert factor_args.immediate_gradient_removal is False
    assert factor_args.ignore_bias is False

    assert factor_args.covariance_max_examples == 100_000
    assert factor_args.covariance_data_partition_size == 1
    assert factor_args.covariance_module_partition_size == 1
    assert factor_args.activation_covariance_dtype == torch.float32
    assert factor_args.gradient_covariance_dtype == torch.float32
    assert factor_args.eigendecomposition_dtype == torch.float64

    assert factor_args.lambda_max_examples == 100_000
    assert factor_args.lambda_data_partition_size == 1
    assert factor_args.lambda_module_partition_size == 1
    assert factor_args.lambda_iterative_aggregate is False
    assert factor_args.cached_activation_cpu_offload is False
    assert factor_args.lambda_dtype == torch.float32


def test_default_score_arguments() -> None:
    factor_args = ScoreArguments()

    assert factor_args.damping is None
    assert factor_args.immediate_gradient_removal is False
    assert factor_args.cached_activation_cpu_offload is False

    assert factor_args.data_partition_size == 1
    assert factor_args.module_partition_size == 1
    assert factor_args.per_module_score is False

    assert factor_args.query_gradient_rank is None
    assert factor_args.query_gradient_svd_dtype == torch.float64

    assert factor_args.score_dtype == torch.float32
    assert factor_args.per_sample_gradient_dtype == torch.float32
    assert factor_args.precondition_dtype == torch.float32
