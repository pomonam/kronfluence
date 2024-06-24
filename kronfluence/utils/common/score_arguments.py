from typing import Optional

import torch

from kronfluence import ScoreArguments


def default_score_arguments(
    damping: Optional[float] = 1e-08, query_gradient_rank: Optional[int] = None
) -> ScoreArguments:
    """Default score arguments."""
    score_args = ScoreArguments(damping=damping)
    score_args.query_gradient_rank = query_gradient_rank
    if score_args.query_gradient_rank is not None:
        score_args.num_query_gradient_accumulations = 10
    return score_args


def test_score_arguments(damping: Optional[float] = 1e-08, query_gradient_rank: Optional[int] = None) -> ScoreArguments:
    """Score arguments used for unit tests."""
    score_args = ScoreArguments(damping=damping)
    score_args.query_gradient_svd_dtype = torch.float64
    score_args.score_dtype = torch.float64
    score_args.per_sample_gradient_dtype = torch.float64
    score_args.precondition_dtype = torch.float64
    score_args.query_gradient_rank = query_gradient_rank
    return score_args


def smart_low_precision_score_arguments(
    damping: Optional[float] = 1e-08, query_gradient_rank: Optional[int] = None, dtype: torch.dtype = torch.bfloat16
) -> ScoreArguments:
    """Score arguments with low precision, except for the preconditioning computations."""
    score_args = ScoreArguments(damping=damping)
    score_args.amp_dtype = dtype
    score_args.query_gradient_svd_dtype = torch.float32
    score_args.score_dtype = dtype
    score_args.per_sample_gradient_dtype = dtype
    score_args.precondition_dtype = torch.float32
    score_args.query_gradient_rank = query_gradient_rank
    if score_args.query_gradient_rank is not None:
        score_args.num_query_gradient_accumulations = 10
    return score_args


def all_low_precision_score_arguments(
    damping: Optional[float] = 1e-08, query_gradient_rank: Optional[int] = None, dtype: torch.dtype = torch.bfloat16
) -> ScoreArguments:
    """Score arguments with low precision."""
    score_args = ScoreArguments(damping=damping)
    score_args.amp_dtype = dtype
    score_args.query_gradient_svd_dtype = torch.float32
    score_args.score_dtype = dtype
    score_args.per_sample_gradient_dtype = dtype
    score_args.precondition_dtype = dtype
    score_args.query_gradient_rank = query_gradient_rank
    if score_args.query_gradient_rank is not None:
        score_args.num_query_gradient_accumulations = 10
    return score_args


def reduce_memory_score_arguments(
    damping: Optional[float] = 1e-08, query_gradient_rank: Optional[int] = None, dtype: torch.dtype = torch.bfloat16
) -> ScoreArguments:
    """Score arguments with low precision + CPU offload."""
    score_args = all_low_precision_score_arguments(damping=damping, dtype=dtype)
    score_args.cached_activation_cpu_offload = True
    score_args.query_gradient_rank = query_gradient_rank
    if score_args.query_gradient_rank is not None:
        score_args.num_query_gradient_accumulations = 10
    return score_args


def extreme_reduce_memory_score_arguments(
    damping: Optional[float] = 1e-08, query_gradient_rank: Optional[int] = None, dtype: torch.dtype = torch.bfloat16
) -> ScoreArguments:
    """Score arguments for models that is difficult to fit in a single GPU."""
    score_args = all_low_precision_score_arguments(damping=damping, dtype=dtype)
    score_args.cached_activation_cpu_offload = True
    score_args.query_gradient_rank = query_gradient_rank
    score_args.module_partition_size = 4
    if score_args.query_gradient_rank is not None:
        score_args.num_query_gradient_accumulations = 10
    return score_args


def large_dataset_score_arguments(
    damping: Optional[float] = 1e-08, query_gradient_rank: Optional[int] = None, dtype: torch.dtype = torch.bfloat16
) -> ScoreArguments:
    """Score arguments for large models and datasets."""
    score_args = smart_low_precision_score_arguments(damping=damping, dtype=dtype)
    score_args.data_partition_size = 4
    score_args.query_gradient_rank = query_gradient_rank
    if score_args.query_gradient_rank is not None:
        score_args.num_query_gradient_accumulations = 10
    return score_args
