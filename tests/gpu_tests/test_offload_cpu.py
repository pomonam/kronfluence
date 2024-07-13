# pylint: skip-file


import pytest
import torch

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.common.factor_arguments import pytest_factor_arguments
from kronfluence.utils.common.score_arguments import pytest_score_arguments
from kronfluence.utils.constants import ALL_MODULE_NAME
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import ATOL, RTOL, check_tensor_dict_equivalence, prepare_test


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
@pytest.mark.parametrize("offload_activations_to_cpu", [True, False])
@pytest.mark.parametrize("query_size", [16])
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [1])
def test_cpu_offloads(
    test_name: str,
    offload_activations_to_cpu: bool,
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
    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name=f"pytest_{__name__}",
        model=model,
        task=task,
        disable_model_save=True,
        disable_tqdm=True,
    )
    factor_args = FactorArguments(
        offload_activations_to_cpu=offload_activations_to_cpu,
    )
    if test_name == "repeated_mlp":
        factor_args.has_shared_parameters = True
    factors_name = f"pytest_{test_name}_{test_cpu_offloads.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )

    score_args = ScoreArguments(
        offload_activations_to_cpu=offload_activations_to_cpu,
    )
    scores_name = f"pytest_{test_name}_{test_cpu_offloads.__name__}_scores"
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


@pytest.mark.parametrize(
    "test_name",
    [
        "mlp",
        "repeated_mlp",
        "mlp_checkpoint",
        "conv",
    ],
)
@pytest.mark.parametrize("per_module_score", [False, True])
@pytest.mark.parametrize("query_size", [50])
@pytest.mark.parametrize("train_size", [102])
@pytest.mark.parametrize("seed", [1])
def test_cpu_offloads_identical(
    test_name: str,
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
    model = model.to(dtype=torch.float64)
    model = prepare_model(model=model, task=task)

    analyzer = Analyzer(
        analysis_name=f"pytest_{test_cpu_offloads_identical}_{__name__}",
        model=model,
        task=task,
        disable_model_save=True,
        disable_tqdm=True,
    )
    factor_args = pytest_factor_arguments()
    if test_name == "repeated_mlp":
        factor_args.has_shared_parameters = True
    factors_name = f"pytest_{test_name}_{test_cpu_offloads_identical.__name__}"
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=32,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    score_args = pytest_score_arguments()
    scores_name = f"pytest_{test_name}_{test_cpu_offloads_identical.__name__}_scores"
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

    factors_name = f"pytest_{test_name}_{test_cpu_offloads_identical.__name__}_cached"
    factor_args.offload_activations_to_cpu = True
    analyzer.fit_all_factors(
        factors_name=factors_name,
        dataset=train_dataset,
        dataloader_kwargs=kwargs,
        per_device_batch_size=16,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    score_args.offload_activations_to_cpu = True
    scores_name = f"pytest_{test_name}_{test_cpu_offloads_identical.__name__}_cached_scores"
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=test_dataset,
        per_device_query_batch_size=6,
        train_dataset=train_dataset,
        per_device_train_batch_size=8,
        dataloader_kwargs=kwargs,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    cached_pairwise_scores = analyzer.load_pairwise_scores(scores_name=scores_name)

    assert check_tensor_dict_equivalence(pairwise_scores, cached_pairwise_scores, atol=ATOL, rtol=RTOL)
