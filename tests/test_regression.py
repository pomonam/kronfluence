# pylint: skip-file

import torch

from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs
from tests.utils import prepare_test


def test_mlp_regression(
    test_name: str = "mlp",
    strategy: str = "ekfac",
    seed: int = 0,
    train_size: int = 32,
    query_size: int = 16,
) -> None:
    model, train_dataset, query_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        query_size=query_size,
        seed=seed,
    )
    assert round(list(model.named_parameters())[0][1].sum().item(), 2) == -2.45

    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name=f"pytest_regression_{test_name}",
        model=model,
        task=task,
        disable_model_save=True,
        cpu=True,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    factor_args = FactorArguments(strategy=strategy, use_empirical_fisher=True)
    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_name}",
        dataset=train_dataset,
        per_device_batch_size=1,
        dataloader_kwargs=kwargs,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    covariance_matrices = analyzer.load_covariance_matrices(f"pytest_{test_name}")
    assert round(torch.sum(covariance_matrices["activation_covariance"]["0"] / train_size).item(), 2) == 7.81

    analyzer.perform_eigendecomposition(
        factors_name=f"pytest_{test_name}",
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    eigen_factors = analyzer.load_eigendecomposition(f"pytest_{test_name}")
    assert round(eigen_factors["activation_eigenvectors"]["0"].sum().item(), 2) == 2.64

    analyzer.fit_lambda_matrices(
        factors_name=f"pytest_{test_name}",
        dataset=train_dataset,
        per_device_batch_size=1,
        dataloader_kwargs=kwargs,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    lambda_matrices = analyzer.load_lambda_matrices(f"pytest_{test_name}")
    assert round((lambda_matrices["lambda_matrix"]["0"] / train_size).sum().item(), 2) == 15.14

    analyzer.compute_pairwise_scores(
        scores_name="pairwise",
        factors_name=f"pytest_{test_name}",
        query_dataset=query_dataset,
        per_device_query_batch_size=1,
        train_dataset=train_dataset,
        per_device_train_batch_size=1,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores("pairwise")
    assert round(scores["all_modules"].sum().item(), 2) == 145.53


def test_conv_regression(
    test_name: str = "conv",
    strategy: str = "ekfac",
    seed: int = 0,
    train_size: int = 32,
    query_size: int = 16,
) -> None:
    model, train_dataset, query_dataset, data_collator, task = prepare_test(
        test_name=test_name,
        train_size=train_size,
        query_size=query_size,
        seed=seed,
    )
    assert round(list(model.named_parameters())[0][1].sum().item(), 2) == -0.75

    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(
        analysis_name=f"pytest_regression_{test_name}",
        model=model,
        task=task,
        disable_model_save=True,
        cpu=True,
    )
    kwargs = DataLoaderKwargs(collate_fn=data_collator)
    factor_args = FactorArguments(strategy=strategy, use_empirical_fisher=True)
    analyzer.fit_covariance_matrices(
        factors_name=f"pytest_{test_name}",
        dataset=train_dataset,
        per_device_batch_size=1,
        dataloader_kwargs=kwargs,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    covariance_matrices = analyzer.load_covariance_matrices(f"pytest_{test_name}")
    assert round(torch.sum(covariance_matrices["activation_covariance"]["0"] / train_size).item(), 2) == 42299.42

    analyzer.perform_eigendecomposition(
        factors_name=f"pytest_{test_name}",
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    eigen_factors = analyzer.load_eigendecomposition(f"pytest_{test_name}")
    assert round(eigen_factors["activation_eigenvectors"]["0"].sum().item(), 2) == 4.34

    analyzer.fit_lambda_matrices(
        factors_name=f"pytest_{test_name}",
        dataset=train_dataset,
        per_device_batch_size=1,
        dataloader_kwargs=kwargs,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    lambda_matrices = analyzer.load_lambda_matrices(f"pytest_{test_name}")
    assert round((lambda_matrices["lambda_matrix"]["0"] / train_size).sum().item(), 2) == 0.18

    analyzer.compute_pairwise_scores(
        scores_name="pairwise",
        factors_name=f"pytest_{test_name}",
        query_dataset=query_dataset,
        per_device_query_batch_size=1,
        train_dataset=train_dataset,
        per_device_train_batch_size=1,
        dataloader_kwargs=kwargs,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores("pairwise")
    assert round(scores["all_modules"].sum().item(), 2) == 6268.84
