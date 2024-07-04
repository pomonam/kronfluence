from typing import Optional

import torch

from kronfluence import ScoreArguments


def default_score_arguments(
    damping_factor: Optional[float] = 1e-08, query_gradient_low_rank: Optional[int] = None
) -> ScoreArguments:
    """Creates default score arguments"""
    score_args = ScoreArguments(damping_factor=damping_factor, query_gradient_low_rank=query_gradient_low_rank)
    if score_args.query_gradient_low_rank is not None:
        score_args.query_gradient_accumulation_steps = 10
    return score_args


def pytest_score_arguments(
    damping_factor: Optional[float] = 1e-08, query_gradient_low_rank: Optional[int] = None
) -> ScoreArguments:
    """Creates score arguments for unit tests"""
    score_args = ScoreArguments(damping_factor=damping_factor, query_gradient_low_rank=query_gradient_low_rank)
    score_args.query_gradient_svd_dtype = torch.float64
    score_args.score_dtype = torch.float64
    score_args.per_sample_gradient_dtype = torch.float64
    score_args.precondition_dtype = torch.float64
    return score_args


def smart_low_precision_score_arguments(
    damping_factor: Optional[float] = 1e-08,
    query_gradient_low_rank: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ScoreArguments:
    """Creates score arguments with low precision, except for preconditioning computations."""
    score_args = default_score_arguments(damping_factor=damping_factor, query_gradient_low_rank=query_gradient_low_rank)
    score_args.amp_dtype = dtype
    score_args.score_dtype = dtype
    score_args.per_sample_gradient_dtype = dtype
    score_args.query_gradient_svd_dtype = torch.float32
    score_args.precondition_dtype = torch.float32
    return score_args


def all_low_precision_score_arguments(
    damping_factor: Optional[float] = 1e-08,
    query_gradient_low_rank: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ScoreArguments:
    """Creates score arguments with low precision for all computations."""
    score_args = default_score_arguments(damping_factor=damping_factor, query_gradient_low_rank=query_gradient_low_rank)
    score_args.amp_dtype = dtype
    score_args.score_dtype = dtype
    score_args.per_sample_gradient_dtype = dtype
    score_args.precondition_dtype = dtype
    score_args.query_gradient_svd_dtype = torch.float32
    return score_args


def reduce_memory_score_arguments(
    damping_factor: Optional[float] = 1e-08,
    query_gradient_low_rank: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ScoreArguments:
    """Creates score arguments with low precision and CPU offloading."""
    score_args = all_low_precision_score_arguments(
        damping_factor=damping_factor, query_gradient_low_rank=query_gradient_low_rank, dtype=dtype
    )
    score_args.offload_activations_to_cpu = True
    return score_args


def extreme_reduce_memory_score_arguments(
    damping_factor: Optional[float] = 1e-08,
    module_partitions: int = 4,
    query_gradient_low_rank: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ScoreArguments:
    """Creates score arguments for models that are difficult to fit on a single GPU."""
    score_args = reduce_memory_score_arguments(
        damping_factor=damping_factor, query_gradient_low_rank=query_gradient_low_rank, dtype=dtype
    )
    score_args.module_partitions = module_partitions
    return score_args
