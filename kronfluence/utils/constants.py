"""A collection of constants."""

from typing import Dict, List, Optional, Tuple, Union

import torch

FACTOR_TYPE = Dict[str, Dict[str, torch.Tensor]]
PARTITION_TYPE = Tuple[int, int]
SCORE_TYPE = Dict[str, torch.Tensor]
PRECONDITIONED_GRADIENT_TYPE = Optional[Union[torch.Tensor, List[torch.Tensor]]]

# Constants for file naming conventions.
FACTOR_SAVE_PREFIX = "factors_"
SCORE_SAVE_PREFIX = "scores_"
FACTOR_ARGUMENTS_NAME = "factor"
SCORE_ARGUMENTS_NAME = "score"

# The total iteration step to synchronize the process when using distributed setting.
DISTRIBUTED_SYNC_INTERVAL = 1_000

# The scale for the heuristic damping term.
HEURISTIC_DAMPING_SCALE = 0.1

# Activation covariance matrix.
ACTIVATION_COVARIANCE_MATRIX_NAME = "activation_covariance"
# Pseudo-gradient covariance matrix.
GRADIENT_COVARIANCE_MATRIX_NAME = "gradient_covariance"
# Number of elements used to aggregate activation and gradient covariance.
NUM_ACTIVATION_COVARIANCE_PROCESSED = "num_activation_covariance_processed"
NUM_GRADIENT_COVARIANCE_PROCESSED = "num_gradient_covariance_processed"


# A list of factors to keep track of when computing covariance matrices.
COVARIANCE_FACTOR_NAMES = [
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    NUM_ACTIVATION_COVARIANCE_PROCESSED,
    NUM_GRADIENT_COVARIANCE_PROCESSED,
]


# Eigenvectors for the activation covariance matrix.
ACTIVATION_EIGENVECTORS_NAME = "activation_eigenvectors"
# Eigenvalues for the activation covariance matrix.
ACTIVATION_EIGENVALUES_NAME = "activation_eigenvalues"
# Eigenvectors for the pseudo-gradient covariance matrix.
GRADIENT_EIGENVECTORS_NAME = "gradient_eigenvectors"
# Eigenvalues for the pseudo-gradient covariance matrix.
GRADIENT_EIGENVALUES_NAME = "gradient_eigenvalues"

# A list of factors to keep track of when performing Eigendecomposition.
EIGENDECOMPOSITION_FACTOR_NAMES = [
    ACTIVATION_EIGENVECTORS_NAME,
    ACTIVATION_EIGENVALUES_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    GRADIENT_EIGENVALUES_NAME,
]

# Lambda matrix (e.g., corrected-eigenvalues for EK-FAC).
LAMBDA_MATRIX_NAME = "lambda_matrix"
# Number of data points used to computed Lambda matrix.
NUM_LAMBDA_PROCESSED = "num_lambda_processed"

# A list of factors to keep track of when computing Lambda matrices.
LAMBDA_FACTOR_NAMES = [LAMBDA_MATRIX_NAME, NUM_LAMBDA_PROCESSED]

# Preconditioned per-sample gradient.
PRECONDITIONED_GRADIENT_NAME = "preconditioned_gradient"
# Accumulated preconditioned per-sample gradient.
ACCUMULATED_PRECONDITIONED_GRADIENT_NAME = "accumulated_preconditioned_gradient"
# Aggregated gradient.
AGGREGATED_GRADIENT_NAME = "aggregated_gradient"
# Pairwise influence scores.
PAIRWISE_SCORE_MATRIX_NAME = "pairwise_score_matrix"
# Self-influence scores.
SELF_SCORE_VECTOR_NAME = "self_score_vector"

# The dictionary key for storing summed scores.
ALL_MODULE_NAME = "all_modules"

# Data type when computing the reciprocal.
LAMBDA_DTYPE = torch.float64
