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
            "help": "If `True`, aggregates the squared sum of projected per-sample gradients "
            "iteratively (with for-loop) to reduce GPU memory usage."
        },
    )
    offload_activations_to_cpu: bool = field(
        default=False,
        metadata={
            "help": "If `True`, offloads cached activations to CPU memory when computing "
            "per-sample gradients, reducing GPU memory usage."
        },
    )
    per_sample_gradient_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for per-sample gradient computations."},
    )
    lambda_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for Lambda matrix computations."},
    )


@dataclass
class ScoreArguments(Arguments):
    """Arguments for computing influence scores."""

    # General configuration. #
    damping_factor: Optional[float] = field(
        default=1e-08,
        metadata={
            "help": "Damping factor for the inverse Hessian-vector product (iHVP). "
            "If `None`, uses a heuristic of 0.1 times the mean eigenvalue."
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
    offload_activations_to_cpu: bool = field(
        default=False,
        metadata={
            "help": "If `True`, offloads cached activations to CPU memory when computing "
            "per-sample gradients, reducing GPU memory usage."
        },
    )
    einsum_minimize_size: bool = field(
        default=False,
        metadata={
            "help": "If `True`, einsum operations find the contraction that minimizes the size of the "
            "largest intermediate tensor."
        },
    )

    # Partition configuration. #
    data_partitions: int = field(
        default=1,
        metadata={"help": "Number of partitions to divide the dataset into for influence score computation."},
    )
    module_partitions: int = field(
        default=1,
        metadata={
            "help": "Number of partitions to divide the model's modules (layers) into for influence score computation."
        },
    )

    # General score configuration. #
    compute_per_module_scores: bool = field(
        default=False,
        metadata={"help": "If `True`, computes separate scores for each module instead of summing across all modules."},
    )
    compute_per_token_scores: bool = field(
        default=False,
        metadata={
            "help": "If `True`, computes separate scores for each token instead of summing across all tokens. "
            "Only applicable to transformer-based models."
        },
    )

    # Pairwise influence score configuration. #
    query_gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of query batches to accumulate before processing training examples."},
    )
    query_gradient_low_rank: Optional[int] = field(
        default=None,
        metadata={
            "help": "Rank for the low-rank approximation of the query gradient. "
            "If `None`, no low-rank approximation is applied."
        },
    )
    use_full_svd: bool = field(
        default=False,
        metadata={
            "help": "If `True`, uses `torch.linalg.svd` instead of `torch.svd_lowrank` for query batching. "
            "Full SVD is more accurate but slower and more memory-intensive."
        },
    )
    aggregate_query_gradients: bool = field(
        default=False,
        metadata={
            "help": "If `True`, uses the summed query gradient instead of per-sample query gradients "
            "for pairwise influence computation."
        },
    )
    aggregate_train_gradients: bool = field(
        default=False,
        metadata={
            "help": "If `True`, uses the summed train gradient instead of per-sample train gradients "
            "for pairwise influence computation."
        },
    )

    # Self-influence score configuration. #
    use_measurement_for_self_influence: bool = field(
        default=False,
        metadata={"help": "If `True`, uses the measurement (instead of the loss) for computing self-influence scores."},
    )

    # Precision configuration. #
    query_gradient_svd_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for singular value decomposition (SVD) of query gradient."},
    )
    score_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for computing and storing influence scores."},
    )
    per_sample_gradient_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for computing per-sample gradients."},
    )
    precondition_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for computing the preconditioned gradient."},
    )
