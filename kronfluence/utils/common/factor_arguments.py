import torch

from kronfluence import FactorArguments


def default_factor_arguments(strategy: str = "ekfac") -> FactorArguments:
    factor_args = FactorArguments(strategy=strategy)
    return factor_args


def test_factor_arguments(strategy: str = "ekfac") -> FactorArguments:
    factor_args = FactorArguments(strategy=strategy)
    factor_args.use_empirical_fisher = True
    factor_args.activation_covariance_dtype = torch.float64
    factor_args.gradient_covariance_dtype = torch.float64
    factor_args.per_sample_gradient_dtype = torch.float64
    factor_args.lambda_dtype = torch.float32
    return factor_args


def smart_low_precision_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    factor_args = FactorArguments(strategy=strategy)
    factor_args.amp_dtype = dtype
    factor_args.activation_covariance_dtype = dtype
    factor_args.gradient_covariance_dtype = dtype
    factor_args.per_sample_gradient_dtype = dtype
    factor_args.lambda_dtype = torch.float32
    return factor_args


def all_low_precision_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    factor_args = FactorArguments(strategy=strategy)
    factor_args.amp_dtype = dtype
    factor_args.activation_covariance_dtype = dtype
    factor_args.gradient_covariance_dtype = dtype
    factor_args.per_sample_gradient_dtype = dtype
    factor_args.lambda_dtype = dtype
    return factor_args


def reduce_memory_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    factor_args = all_low_precision_factor_arguments(strategy=strategy, dtype=dtype)
    factor_args.lambda_iterative_aggregate = True
    return factor_args


def extreme_reduce_memory_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    factor_args = all_low_precision_factor_arguments(strategy=strategy, dtype=dtype)
    factor_args.lambda_iterative_aggregate = True
    factor_args.cached_activation_cpu_offload = True
    factor_args.covariance_module_partition_size = 2
    factor_args.lambda_module_partition_size = 2
    return factor_args


def large_dataset_factor_arguments(strategy: str = "ekfac", dtype: torch.dtype = torch.bfloat16) -> FactorArguments:
    factor_args = smart_low_precision_factor_arguments(strategy=strategy, dtype=dtype)
    factor_args.covariance_data_partition_size = 4
    factor_args.lambda_data_partition_size = 4
    return factor_args
