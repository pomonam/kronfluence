import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class Arguments:
    """Base class for specifying arguments for computing factors and influence scores."""

    def to_dict(self) -> Dict[str, Any]:
        """Converts the arguments to a dictionary."""
        config = copy.deepcopy(self.__dict__)
        for key, value in config.items():
            if isinstance(value, torch.dtype):
                config[key] = str(value)
        return config

    def to_str_dict(self) -> Dict[str, str]:
        """Converts the arguments to a dictionary, where all values are converted to strings."""
        config = copy.deepcopy(self.__dict__)
        for key, value in config.items():
            config[key] = str(value)
        return config


@dataclass
class FactorArguments(Arguments):
    """Arguments for computing influence factors."""

    # General configuration. #
    strategy: str = field(
        default="ekfac",
        metadata={"help": "Strategy for computing preconditioning factors."},
    )
    use_empirical_fisher: bool = field(
        default=False,
        metadata={
            "help": "Whether to use empirical fisher (using labels from batch) instead of "
            "true Fisher (using sampled labels)."
        },
    )
    immediate_gradient_removal: bool = field(
        default=False,
        metadata={"help": "Whether to immediately remove computed `.grad` by Autograd within the backward hook."},
    )
    ignore_bias: bool = field(
        default=False,
        metadata={"help": "Whether to ignore factor computations on bias parameters. Defaults to False."},
    )
    distributed_sync_steps: int = field(
        default=1_000,
        metadata={
            "help": "Specifies the total iteration step to synchronize the process when using distributed setting."
        },
    )

    # Configuration for fitting covariance matrices. #
    covariance_max_examples: Optional[int] = field(
        default=100_000,
        metadata={
            "help": "Maximum number of examples for fitting covariance matrices. "
            "Uses all data examples for the given dataset if None."
        },
    )
    covariance_data_partition_size: int = field(
        default=1,
        metadata={
            "help": "Number of data partitions for computing covariance matrices. "
            "For example, when `covariance_data_partition_size = 2`, the dataset is split "
            "into 2 chunks and covariance matrices are separately computed for each chunk. "
            "This is useful with GPU preemption as intermediate covariance matrices are saved "
            "in disk."
        },
    )
    covariance_module_partition_size: int = field(
        default=1,
        metadata={
            "help": "Number of module partitions for computing covariance matrices. "
            "For example, when `covariance_module_partition_size = 2`, the module is split "
            "into 2 chunks and covariance matrices are separately computed for each chunk. "
            "This is useful when the available GPU memory is limited; the total covariance matrices cannot "
            "fit into memory."
        },
    )
    activation_covariance_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Dtype for computing activation covariance matrices."},
    )
    gradient_covariance_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Dtype for computing pseudo-gradient covariance matrices."},
    )

    # Configuration for performing eigendecomposition. #
    eigendecomposition_dtype: torch.dtype = field(
        default=torch.float64,
        metadata={"help": "Dtype for performing Eigendecomposition. Recommended to use `torch.float64."},
    )

    # Configuration for fitting Lambda matrices. #
    lambda_max_examples: Optional[int] = field(
        default=100_000,
        metadata={
            "help": "Maximum number of examples for fitting Lambda matrices. "
            "Uses all data examples for the given dataset if None."
        },
    )
    lambda_data_partition_size: int = field(
        default=1,
        metadata={
            "help": "Number of data partitions for computing Lambda matrices. "
            "For example, when `lambda_data_partition_size = 2`, the dataset is split "
            "into 2 chunks and Lambda matrices are separately computed for each chunk. "
            "This is useful with GPU preemption as intermediate Lambda matrices are saved "
            "in disk."
        },
    )
    lambda_module_partition_size: int = field(
        default=1,
        metadata={
            "help": "Number of module partitions for computing Lambda matrices. "
            "For example, when `lambda_module_partition_size = 2`, the module is split "
            "into 2 chunks and Lambda matrices are separately computed for each chunk. "
            "This is useful when the available GPU memory is limited; the total Lambda matrices cannot "
            "fit into memory."
        },
    )
    lambda_iterative_aggregate: bool = field(
        default=False,
        metadata={
            "help": "Whether to aggregate squared sum of projected per-sample-gradient with for-loop "
            "iterations. This is helpful when the available GPU memory is limited."
        },
    )
    cached_activation_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload cached activation to CPU for computing the "
            "per-sample-gradient. This is helpful when the available GPU memory is limited."
        },
    )
    lambda_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Dtype for computing Lambda matrices. Recommended to use `torch.float32`."},
    )


@dataclass
class ScoreArguments(Arguments):
    """Arguments for computing influence scores."""

    # General configuration. #
    damping: Optional[float] = field(
        default=None,
        metadata={
            "help": "A damping factor for the damped matrix-vector product. "
            "Uses a heuristic based on mean eigenvalues (0.1 x mean eigenvalues) if None."
        },
    )
    immediate_gradient_removal: bool = field(
        default=False,
        metadata={"help": "Whether to immediately remove computed `.grad` by Autograd within the backward hook."},
    )
    cached_activation_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload cached activation to CPU for computing the "
            "per-sample-gradient. This is helpful when the available GPU memory is limited."
        },
    )
    distributed_sync_steps: int = field(
        default=1_000,
        metadata={
            "help": "Specifies the total iteration step to synchronize the process when using distributed setting."
        },
    )

    # Partition configuration. #
    data_partition_size: int = field(
        default=1,
        metadata={
            "help": "Number of data partitions for computing influence scores. For example, when "
            "`data_partition_size = 2`, the dataset is split into 2 chunks and scores are separately "
            "computed for each chunk."
        },
    )
    module_partition_size: int = field(
        default=1,
        metadata={
            "help": "Number of module partitions for computing influence scores. For example, when "
            "`module_partition_size = 2`, the module is split into 2 modules and scores are separately computed "
            "for each chunk."
        },
    )

    # Score configuration. #
    per_module_score: bool = field(
        default=False,
        metadata={
            "help": "Whether to obtain per-module scores instead of the summed scores across all modules. "
            "This is useful when performing layer-wise influence analysis."
        },
    )
    query_gradient_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Rank for the query gradient. Does not apply low-rank approximation if None."},
    )

    # Dtype configuration. #
    query_gradient_svd_dtype: torch.dtype = field(
        default=torch.float64,
        metadata={"help": "Dtype for performing singular value decomposition (SVD) on the query gradient."},
    )
    score_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Dtype for computing and storing influence scores."},
    )
    per_sample_gradient_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Dtype for computing per-sample-gradients."},
    )
    precondition_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Dtype for computing the preconditioned gradient. Recommended to use `torch.float32`."},
    )
