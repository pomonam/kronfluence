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
        metadata={
            "help": "Specifies the algorithm for computing influence factors. Default is 'ekfac' "
            "(Eigenvalue-corrected Kronecker-factored Approximate Curvature)."
        },
    )
    use_empirical_fisher: bool = field(
        default=False,
        metadata={
            "help": "Determines whether to approximate empirical Fisher (using true labels) or "
            "true Fisher (using sampled labels)."
        },
    )
    distributed_sync_interval: int = field(
        default=1_000,
        metadata={"help": "Number of iterations between synchronization steps in distributed computing settings."},
    )
    amp_dtype: Optional[torch.dtype] = field(
        default=None,
        metadata={"help": "Data type for automatic mixed precision (AMP). If `None`, AMP is disabled."},
    )
    has_shared_parameters: bool = field(
        default=False,
        metadata={"help": "Indicates whether shared parameters are present in the model's forward pass."},
    )

    # Configuration for fitting covariance matrices. #
    covariance_max_examples: Optional[int] = field(
        default=100_000,
        metadata={
            "help": "Maximum number of examples to use when fitting covariance matrices. "
            "Uses entire dataset if `None`."
        },
    )
    covariance_data_partitions: int = field(
        default=1,
        metadata={"help": "Number of partitions to divide the dataset into for covariance matrix computation."},
    )
    covariance_module_partitions: int = field(
        default=1,
        metadata={
            "help": "Number of partitions to divide the model's modules (layers) into for "
            "covariance matrix computation."
        },
    )
    activation_covariance_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for activation covariance computations."},
    )
    gradient_covariance_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for pseudo-gradient covariance computations."},
    )

    # Configuration for performing eigendecomposition. #
    eigendecomposition_dtype: torch.dtype = field(
        default=torch.float64,
        metadata={
            "help": "Data type for eigendecomposition computations. Double precision (`torch.float64`) is "
            "recommended for numerical stability."
        },
    )

    # Configuration for fitting Lambda matrices. #
    lambda_max_examples: Optional[int] = field(
        default=100_000,
        metadata={
            "help": "Maximum number of examples to use when fitting Lambda matrices. Uses entire dataset if `None`."
        },
    )
    lambda_data_partitions: int = field(
        default=1,
        metadata={"help": "Number of partitions to divide the dataset into for Lambda matrix computation."},
    )
    lambda_module_partitions: int = field(
        default=1,
        metadata={
            "help": "Number of partitions to divide the model's modules (layers) into for Lambda matrix computation."
        },
    )
    use_iterative_lambda_aggregation: bool = field(
        default=False,
        metadata={
            "help": "If True, aggregates the squared sum of projected per-sample gradients "
            "iteratively to reduce GPU memory usage."
        },
    )
    offload_activations_to_cpu: bool = field(
        default=False,
        metadata={
            "help": "If True, offloads cached activations to CPU memory when computing "
            "per-sample gradients, reducing GPU memory usage."
        },
    )
    per_sample_gradient_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for per-sample-gradient computations."},
    )
    lambda_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for Lambda matrix computations."},
    )


@dataclass
class ScoreArguments(Arguments):
    """Arguments for computing influence scores."""

    # General configuration. #
    damping: Optional[float] = field(
        default=1e-08,
        metadata={
            "help": "A damping factor for the damped inverse Hessian-vector product (iHVP). "
            "Uses a heuristic based on mean eigenvalues (0.1 x mean eigenvalues) if set to None."
        },
    )
    cached_activation_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload cached activation to CPU for computing the per-sample-gradient."},
    )
    distributed_sync_steps: int = field(
        default=1_000,
        metadata={
            "help": "Specifies the total iteration step to synchronize the process when using distributed setting."
        },
    )
    amp_dtype: Optional[torch.dtype] = field(
        default=None,
        metadata={"help": "Dtype for automatic mixed precision (AMP). Disables AMP if None."},
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
            "`module_partition_size = 2`, the module (layers) are split into 2 modules and scores are separately "
            "computed for each chunk."
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
    num_query_gradient_accumulations: int = field(
        default=1,
        metadata={"help": "Number of query batches to accumulate over before iterating over training examples."},
    )
    query_gradient_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Rank for the query gradient. Does not apply low-rank approximation if None."},
    )
    use_full_svd: bool = field(
        default=True,
        metadata={
            "help": "Whether to perform to use `torch.linalg.svd` instead of `torch.svd_lowrank` for "
            "query batching. `torch.svd_lowrank` can result in a more inaccurate low-rank approximations."
        },
    )
    use_measurement_for_self_influence: bool = field(
        default=False,
        metadata={"help": "Whether to use the measurement (instead of the loss) for computing self-influence scores."},
    )

    # Dtype configuration. #
    query_gradient_svd_dtype: torch.dtype = field(
        default=torch.float32,
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
