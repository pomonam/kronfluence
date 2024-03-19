# pylint: skip-file

from typing import Tuple

import pytest
from task import Task
from torch import nn

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.utils.constants import ALL_MODULE_NAME
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import prepare_test


def prepare_model_and_analyzer(model: nn.Module, task: Task) -> Tuple[nn.Module, Analyzer]:
    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name=f"pytest_{__name__}",
        model=model,
        task=task,
        disable_model_save=True,
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
@pytest.mark.parametrize("cached_activation_cpu_offload", [True, False])
@pytest.mark.parametrize("query_size", [16])
@pytest.mark.parametrize("train_size", [32])
@pytest.mark.parametrize("seed", [0])
def test_cpu_offloads(
    test_name: str,
    cached_activation_cpu_offload: bool,
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
        cached_activation_cpu_offload=cached_activation_cpu_offload,
    )
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
        cached_activation_cpu_offload=cached_activation_cpu_offload,
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
