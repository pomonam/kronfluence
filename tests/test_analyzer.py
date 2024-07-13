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
        "mlp_checkpoint",
        "repeated_mlp",
        "conv_bn",
        "bert",
        "roberta",
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
        disable_tqdm=True,
        disable_model_save=True,
        cpu=True,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)

    factor_args = FactorArguments(strategy=strategy)
    if test_name == "repeated_mlp":
        factor_args.has_shared_parameters = True
    analyzer.fit_all_factors(
        factors_name=f"pytest_{test_analyzer.__name__}_{test_name}",
        dataset=train_dataset,
        per_device_batch_size=16,
        dataloader_kwargs=kwargs,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    analyzer.compute_pairwise_scores(
        scores_name="pairwise",
        factors_name=f"pytest_{test_analyzer.__name__}_{test_name}",
        query_dataset=query_dataset,
        per_device_query_batch_size=4,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    score_args = ScoreArguments()
    analyzer.compute_self_scores(
        scores_name="self",
        factors_name=f"pytest_{test_analyzer.__name__}_{test_name}",
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    score_args.use_measurement_for_self_influence = True
    analyzer.compute_self_scores(
        scores_name="self",
        factors_name=f"pytest_{test_analyzer.__name__}_{test_name}",
        train_dataset=train_dataset,
        per_device_train_batch_size=6,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )


def test_default_factor_arguments() -> None:
    factor_args = FactorArguments()

    assert factor_args.strategy == "ekfac"
    assert factor_args.use_empirical_fisher is False
    assert factor_args.amp_dtype is None
    assert factor_args.amp_scale == 2.0**16
    assert factor_args.has_shared_parameters is False

    assert factor_args.covariance_max_examples == 100_000
    assert factor_args.covariance_data_partitions == 1
    assert factor_args.covariance_module_partitions == 1
    assert factor_args.activation_covariance_dtype == torch.float32
    assert factor_args.gradient_covariance_dtype == torch.float32

    assert factor_args.eigendecomposition_dtype == torch.float64

    assert factor_args.lambda_max_examples == 100_000
    assert factor_args.lambda_data_partitions == 1
    assert factor_args.lambda_module_partitions == 1
    assert factor_args.use_iterative_lambda_aggregation is False
    assert factor_args.offload_activations_to_cpu is False
    assert factor_args.per_sample_gradient_dtype == torch.float32
    assert factor_args.lambda_dtype == torch.float32


def test_default_score_arguments() -> None:
    score_args = ScoreArguments()

    assert score_args.damping_factor == 1e-08
    assert score_args.amp_dtype is None
    assert score_args.offload_activations_to_cpu is False

    assert score_args.data_partitions == 1
    assert score_args.module_partitions == 1

    assert score_args.compute_per_module_scores is False
    assert score_args.compute_per_token_scores is False

    assert score_args.query_gradient_accumulation_steps == 1
    assert score_args.query_gradient_low_rank is None
    assert score_args.use_full_svd is False
    assert score_args.aggregate_query_gradients is False
    assert score_args.aggregate_train_gradients is False

    assert score_args.use_measurement_for_self_influence is False

    assert score_args.query_gradient_svd_dtype == torch.float32
    assert score_args.per_sample_gradient_dtype == torch.float32
    assert score_args.precondition_dtype == torch.float32
    assert score_args.score_dtype == torch.float32
