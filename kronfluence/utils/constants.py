"""A collection of constants."""

from typing import Dict, Tuple

import torch

FACTOR_TYPE = Dict[str, Dict[str, torch.Tensor]]
PARTITION_TYPE = Tuple[int, int]
SCORE_TYPE = Dict[str, torch.Tensor]

# Activation covariance matrix.
ACTIVATION_COVARIANCE_MATRIX_NAME = "activation_covariance"
# Pseudo-gradient covariance matrix.
GRADIENT_COVARIANCE_MATRIX_NAME = "gradient_covariance"
# Number of elements used to aggregate activation and gradient covariance.
NUM_COVARIANCE_PROCESSED = "num_covariance_processed"

# A list of factors to keep track of when computing covariance matrices.
COVARIANCE_FACTOR_NAMES = [
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    NUM_COVARIANCE_PROCESSED,
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
# Pairwise influence scores.
PAIRWISE_SCORE_MATRIX_NAME = "pairwise_score_matrix"
# Self-influence scores.
SELF_SCORE_VECTOR_NAME = "self_score_vector"

# The dictionary key for storing scores for all modules.
ALL_MODULE_NAME = "all_modules"
