import torch

from kronfluence import FactorArguments


def default_factor_arguments(strategy: str = "ekfac") -> FactorArguments:
    """Creates default factor arguments"""
    factor_args = FactorArguments(strategy=strategy)
    return factor_args


def pytest_factor_arguments(strategy: str = "ekfac") -> FactorArguments:
    """Creates factor arguments for unit tests"""
    factor_args = FactorArguments(strategy=strategy)
    factor_args.use_empirical_fisher = True
    factor_args.activation_covariance_dtype = torch.float64
    factor_args.gradient_covariance_dtype = torch.float64
    factor_args.per_sample_gradient_dtype = torch.float64
    factor_args.lambda_dtype = torch.float64
    return factor_args


def smart_low_precision_factor_arguments(
    strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16
) -> FactorArguments:
    """Creates factor arguments with low precision, except for Lambda computations."""
    factor_args = FactorArguments(strategy=strategy)
    factor_args.amp_dtype = dtype
    factor_args.activation_covariance_dtype = dtype
    factor_args.gradient_covariance_dtype = dtype
    factor_args.per_sample_gradient_dtype = dtype
    factor_args.lambda_dtype = torch.float32
    return factor_args


def all_low_precision_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    """Creates factor arguments with low precision for all computations."""
    factor_args = FactorArguments(strategy=strategy)
    factor_args.amp_dtype = dtype
    factor_args.activation_covariance_dtype = dtype
    factor_args.gradient_covariance_dtype = dtype
    factor_args.per_sample_gradient_dtype = dtype
    factor_args.lambda_dtype = dtype
    return factor_args


def reduce_memory_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    """Creates factor arguments with low precision and iterative lambda aggregations."""
    factor_args = all_low_precision_factor_arguments(strategy=strategy, dtype=dtype)
    factor_args.use_iterative_lambda_aggregation = True
    return factor_args


def extreme_reduce_memory_factor_arguments(
    strategy: str = "ekfac", module_partitions: int = 1, dtype: torch.dtype = torch.bfloat16
) -> FactorArguments:
    """Creates factor arguments for models that are difficult to fit on a single GPU."""
    factor_args = reduce_memory_factor_arguments(strategy=strategy, dtype=dtype)
    factor_args.offload_activations_to_cpu = True
    factor_args.covariance_module_partitions = module_partitions
    factor_args.lambda_module_partitions = module_partitions
    return factor_args
