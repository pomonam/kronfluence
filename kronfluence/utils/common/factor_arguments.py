import torch

from kronfluence import FactorArguments


def default_factor_arguments(strategy: str = "ekfac") -> FactorArguments:
    """Default factor arguments."""
    factor_args = FactorArguments(strategy=strategy)
    return factor_args


def pytest_factor_arguments(strategy: str = "ekfac") -> FactorArguments:
    """Factor arguments used for unit tests."""
    factor_args = FactorArguments(strategy=strategy)
    # Makes the computations deterministic.
    factor_args.use_empirical_fisher = True
    factor_args.activation_covariance_dtype = torch.float64
    factor_args.gradient_covariance_dtype = torch.float64
    factor_args.per_sample_gradient_dtype = torch.float64
    factor_args.lambda_dtype = torch.float64
    return factor_args


def smart_low_precision_factor_arguments(
    strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16
) -> FactorArguments:
    """Factor arguments with low precision, except for the lambda computations."""
    factor_args = FactorArguments(strategy=strategy)
    factor_args.amp_dtype = dtype
    factor_args.activation_covariance_dtype = dtype
    factor_args.gradient_covariance_dtype = dtype
    factor_args.per_sample_gradient_dtype = dtype
    factor_args.lambda_dtype = torch.float32
    return factor_args


def all_low_precision_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    """Factor arguments with low precision for all computations."""
    factor_args = FactorArguments(strategy=strategy)
    factor_args.amp_dtype = dtype
    factor_args.activation_covariance_dtype = dtype
    factor_args.gradient_covariance_dtype = dtype
    factor_args.per_sample_gradient_dtype = dtype
    factor_args.lambda_dtype = dtype
    return factor_args


def reduce_memory_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    """Factor arguments with low precision and iterative lambda aggregations."""
    factor_args = all_low_precision_factor_arguments(strategy=strategy, dtype=dtype)
    factor_args.use_iterative_lambda_aggregation = True
    return factor_args


def extreme_reduce_memory_factor_arguments(
    strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16
) -> FactorArguments:
    """Factor arguments for models that is difficult to fit with a single GPU."""
    factor_args = reduce_memory_factor_arguments(strategy=strategy, dtype=dtype)
    factor_args.offload_activations_to_cpu = True
    factor_args.covariance_module_partitions = 4
    factor_args.lambda_module_partitions = 4
    return factor_args
