import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class Arguments:
    """Base class for specifying arguments for computing factors and influence scores."""

    def to_dict(self) -> Dict[str, Any]:
        """Converts the arguments to a dictionary.

        Returns:
            Dict[str, Any]:
                A dictionary representation of the arguments, with `torch.dtype` values converted to strings.
        """
        config = copy.deepcopy(self.__dict__)
        for key, value in config.items():
            if isinstance(value, torch.dtype):
                config[key] = str(value)
        return config

    def to_str_dict(self) -> Dict[str, str]:
        """Converts the arguments to a dictionary with all values as strings.

        Returns:
            Dict[str, str]:
                A dictionary representation of the arguments, with all values converted to strings.
        """
        config = copy.deepcopy(self.__dict__)
        for key, value in config.items():
            config[key] = str(value)
        return config


@dataclass
class FactorArguments(Arguments):
    """Arguments for computing influence factors."""

    # General configuration #
    strategy: str = field(
        default="ekfac",
        metadata={"help": "Specifies the algorithm for computing influence factors. Default is 'ekfac'."},
    )
    use_empirical_fisher: bool = field(
        default=False,
        metadata={
            "help": "If `True`, approximates empirical Fisher (using true labels) instead of "
            "true Fisher (using sampled labels from model's outputs)."
        },
    )
    amp_dtype: Optional[torch.dtype] = field(
        default=None,
        metadata={"help": "Data type for automatic mixed precision (AMP). If `None`, AMP is disabled."},
    )
    amp_scale: float = field(
        default=2.0**16,
        metadata={"help": "Scale factor for AMP (only applicable when `amp_dtype=torch.float16`)."},
    )
    has_shared_parameters: bool = field(
        default=False,
        metadata={"help": "Indicates whether shared parameters are present in the model's forward pass."},
    )

    # Configuration for fitting covariance matrices. #
    covariance_max_examples: Optional[int] = field(
        default=100_000,
        metadata={"help": "Maximum number of examples for fitting covariance matrices. Uses entire dataset if `None`."},
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
        metadata={"help": "Data type for activation covariance computation."},
    )
    gradient_covariance_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for pseudo-gradient covariance computation."},
    )

    # Configuration for performing eigendecomposition #
    eigendecomposition_dtype: torch.dtype = field(
        default=torch.float64,
        metadata={
            "help": "Data type for eigendecomposition. Double precision (`torch.float64`) is recommended "
            "for numerical stability."
        },
    )

    # Configuration for fitting Lambda matrices #
    lambda_max_examples: Optional[int] = field(
        default=100_000,
        metadata={"help": "Maximum number of examples for fitting Lambda matrices. Uses entire dataset if `None`."},
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
        metadata={"help": "If `True`, offloads cached activations to CPU memory when computing per-sample gradients."},
    )
    per_sample_gradient_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for per-sample gradient computation."},
    )
    lambda_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for Lambda matrix computation."},
    )

    def __post_init__(self) -> None:
        if self.covariance_max_examples is not None and self.covariance_max_examples <= 0:
            raise ValueError("`covariance_max_examples` must be `None` or positive.")

        if self.lambda_max_examples is not None and self.lambda_max_examples <= 0:
            raise ValueError("`lambda_max_examples` must be `None` or positive.")

        if any(
            partition <= 0
            for partition in [
                self.covariance_data_partitions,
                self.covariance_module_partitions,
                self.lambda_data_partitions,
                self.lambda_module_partitions,
            ]
        ):
            raise ValueError("All data and module partitions must be positive.")

        # For backward compatibility:
        if not hasattr(self, "amp_scale"):
            self.amp_scale = 2.0**16


@dataclass
class ScoreArguments(Arguments):
    """Arguments for computing influence scores."""

    # General configuration #
    damping_factor: Optional[float] = field(
        default=1e-08,
        metadata={
            "help": "Damping factor for inverse Hessian-vector product (iHVP). "
            "If `None`, uses a heuristic of 0.1 times the mean eigenvalue."
        },
    )
    amp_dtype: Optional[torch.dtype] = field(
        default=None,
        metadata={"help": "Data type for automatic mixed precision (AMP). If `None`, AMP is disabled."},
    )
    offload_activations_to_cpu: bool = field(
        default=False,
        metadata={"help": "If `True`, offloads cached activations to CPU memory when computing per-sample gradients."},
    )

    # Partition configuration #
    data_partitions: int = field(
        default=1,
        metadata={"help": "Number of partitions to divide the dataset for influence score computation."},
    )
    module_partitions: int = field(
        default=1,
        metadata={
            "help": "Number of partitions to divide the model's modules (layers) into for influence score computation."
        },
    )

    # General score configuration #
    compute_per_module_scores: bool = field(
        default=False,
        metadata={"help": "If `True`, computes separate scores for each module instead of summing across all."},
    )
    compute_per_token_scores: bool = field(
        default=False,
        metadata={
            "help": "If `True`, computes separate scores for each token instead of summing across all. "
            "Only applicable to transformer-based models."
        },
    )

    # Pairwise influence score configuration #
    query_gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of query batches to accumulate before processing training examples."},
    )
    query_gradient_low_rank: Optional[int] = field(
        default=None,
        metadata={
            "help": "Rank for the low-rank approximation of the query gradient (query batching). "
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
            "help": "If `True`, uses the summed training gradient instead of per-sample training gradients "
            "for pairwise influence computation."
        },
    )

    # Self-influence score configuration #
    use_measurement_for_self_influence: bool = field(
        default=False,
        metadata={"help": "If `True`, uses the measurement (instead of the loss) for computing self-influence scores."},
    )

    # Precision configuration #
    query_gradient_svd_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for singular value decomposition (SVD) of query gradient."},
    )
    per_sample_gradient_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for query per-sample gradient computation."},
    )
    precondition_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for preconditioned gradient computation."},
    )
    score_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "Data type for influence score computation."},
    )

    def __post_init__(self) -> None:
        if self.damping_factor is not None and self.damping_factor < 0:
            raise ValueError("`damping_factor` must be `None` or positive.")

        if any(partition <= 0 for partition in [self.data_partitions, self.module_partitions]):
            raise ValueError("Both data and module partitions must be positive.")

        if self.query_gradient_accumulation_steps <= 0:
            raise ValueError("`query_gradient_accumulation_steps` must be positive.")

        if self.query_gradient_low_rank is not None and self.query_gradient_low_rank <= 0:
            raise ValueError("`query_gradient_low_rank` must be `None` or positive.")
