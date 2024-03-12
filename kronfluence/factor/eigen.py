from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import find_batch_size, send_to_device
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments
from kronfluence.module.constants import (
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    ACTIVATION_EIGENVALUES_NAME,
    ACTIVATION_EIGENVECTORS_NAME,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    FACTOR_TYPE,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    GRADIENT_EIGENVALUES_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_FACTOR_NAMES,
    NUM_COVARIANCE_PROCESSED,
)
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    get_tracked_module_names,
    load_factors,
    set_factors,
    set_mode,
    synchronize_lambda_matrices,
    update_factor_args,
)
from kronfluence.task import Task
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync


def eigendecomposition_save_path(
    output_dir: Path,
    eigen_factor_name: str,
) -> Path:
    """Generates the path for saving/loading Eigendecomposition results."""
    assert eigen_factor_name in EIGENDECOMPOSITION_FACTOR_NAMES
    return output_dir / f"{eigen_factor_name}_eigendecomposition.safetensors"


def save_eigendecomposition(
    output_dir: Path,
    eigen_factors: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    """Saves Eigendecomposition results to disk."""
    assert set(eigen_factors.keys()) == set(EIGENDECOMPOSITION_FACTOR_NAMES)
    for name in eigen_factors:
        save_path = eigendecomposition_save_path(
            output_dir=output_dir,
            eigen_factor_name=name,
        )
        save_file(tensors=eigen_factors[name], filename=save_path)


def load_eigendecomposition(
    output_dir: Path,
) -> FACTOR_TYPE:
    """Loads Eigendecomposition results from disk."""
    eigen_factors = {}
    for name in EIGENDECOMPOSITION_FACTOR_NAMES:
        save_path = eigendecomposition_save_path(
            output_dir=output_dir,
            eigen_factor_name=name,
        )
        eigen_factors[name] = load_file(filename=save_path)
    return eigen_factors


def eigendecomposition_exist(
    output_dir: Path,
) -> bool:
    """Checks if Eigendecomposition results exist at specified path."""
    for name in EIGENDECOMPOSITION_FACTOR_NAMES:
        save_path = eigendecomposition_save_path(
            output_dir=output_dir,
            eigen_factor_name=name,
        )
        if not save_path.exists():
            return False
    return True


@torch.no_grad()
def perform_eigendecomposition(
    covariance_factors: FACTOR_TYPE,
    model: nn.Module,
    state: State,
    factor_args: FactorArguments,
) -> FACTOR_TYPE:
    """Performs Eigendecomposition on activation and pseudo-gradient covariance matrices.

    Args:
        model (nn.Module):
            The model which contains modules which Eigendecomposition will be performed.
        state (State):
            The current process's information (e.g., device being used).
        covariance_factors (FACTOR_TYPE):
            The covariance matrices to load from.
        factor_args (FactorArguments):
            Arguments related to performing Eigendecomposition.

    Returns:
        FACTOR_TYPE:
            The Eigendecomposition results in CPU. The Eigendecomposition results are organized in
            nested dictionaries, where the first key in the name of the Eigendecomposition factor (e.g.,
            eigenvector), and the second key is the module name.
    """
    eigen_factors: FACTOR_TYPE = {}
    for factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
        eigen_factors[factor_name] = {}
    tracked_module_names = get_tracked_module_names(model=model)

    with tqdm(
        total=len(tracked_module_names),
        desc="Performing Eigendecomposition",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process,
    ) as pbar:
        for module_name in tracked_module_names:
            for covariance_name, eigenvectors_name, eigenvalues_name in [
                (
                    ACTIVATION_COVARIANCE_MATRIX_NAME,
                    ACTIVATION_EIGENVECTORS_NAME,
                    ACTIVATION_EIGENVALUES_NAME,
                ),
                (
                    GRADIENT_COVARIANCE_MATRIX_NAME,
                    GRADIENT_EIGENVECTORS_NAME,
                    GRADIENT_EIGENVALUES_NAME,
                ),
            ]:
                original_dtype = covariance_factors[covariance_name][module_name].dtype
                covariance_factors[covariance_name][module_name].div_(
                    covariance_factors[NUM_COVARIANCE_PROCESSED][module_name]
                )
                covariance_matrix = covariance_factors[covariance_name][module_name].to(
                    device=state.device,
                    dtype=factor_args.eigendecomposition_dtype,
                )
                # In case covariance matrices are not symmetric due to numerical issues.
                covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.t())
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                eigen_factors[eigenvectors_name][module_name] = (
                    eigenvectors.to(dtype=original_dtype).contiguous().cpu()
                )
                eigen_factors[eigenvalues_name][module_name] = eigenvalues.to(
                    dtype=original_dtype
                ).cpu()
                del eigenvectors, eigenvalues
            pbar.update(1)
    return eigen_factors


def lambda_matrices_save_path(
    output_dir: Path,
    lambda_factor_name: str,
    partition: Optional[Tuple[int, int]] = None,
) -> Path:
    """Generates the path for saving/loading Lambda matrices."""
    assert lambda_factor_name in LAMBDA_FACTOR_NAMES
    if partition is not None:
        data_partition, module_partition = partition
        return output_dir / (
            f"{lambda_factor_name}_lambda_data_partition{data_partition}"
            f"_module_partition{module_partition}.safetensors"
        )
    return output_dir / f"{lambda_factor_name}_lambda.safetensors"


def save_lambda_matrices(
    output_dir: Path,
    lambda_factors: Dict[str, Dict[str, torch.Tensor]],
    partition: Optional[Tuple[int, int]] = None,
) -> None:
    """Saves Lambda matrices to disk."""
    assert set(lambda_factors.keys()) == set(LAMBDA_FACTOR_NAMES)
    for name in lambda_factors:
        save_path = lambda_matrices_save_path(
            output_dir=output_dir,
            lambda_factor_name=name,
            partition=partition,
        )
        save_file(tensors=lambda_factors[name], filename=save_path)


def load_lambda_matrices(
    output_dir: Path,
    partition: Optional[Tuple[int, int]] = None,
) -> FACTOR_TYPE:
    """Loads Lambda matrices from disk."""
    lambda_factors = {}
    for name in LAMBDA_FACTOR_NAMES:
        save_path = lambda_matrices_save_path(
            output_dir=output_dir,
            lambda_factor_name=name,
            partition=partition,
        )
        lambda_factors[name] = load_file(filename=save_path)
    return lambda_factors


def lambda_matrices_exist(
    output_dir: Path,
    partition: Optional[Tuple[int, int]] = None,
) -> bool:
    """Check if Lambda matrices exist at specified path."""
    for name in LAMBDA_FACTOR_NAMES:
        save_path = lambda_matrices_save_path(
            output_dir=output_dir,
            lambda_factor_name=name,
            partition=partition,
        )
        if not save_path.exists():
            return False
    return True


def fit_lambda_matrices_with_loader(
    model: nn.Module,
    state: State,
    task: Task,
    eigen_factors: Optional[FACTOR_TYPE],
    loader: data.DataLoader,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, FACTOR_TYPE]:
    """Computes Lambda matrices for a given model and task.

    Args:
        model (nn.Module):
            The model that Lambda matrices will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        eigen_factors (FACTOR_TYPE, optional):
            The eigendecomposition results to load from, before computing the Lambda matrices.
        loader (data.DataLoader):
            The data loader that will be used to compute Lambda matrices.
        factor_args (FactorArguments):
            Arguments related to computing Lambda matrices.
        tracked_module_names (List[str], optional):
            A list of module names that Lambda matrices will be computed. If not specified, Lambda
            matrices will be computed for all available tracked modules.

    Returns:
        Tuple[torch.Tensor, FACTOR_TYPE]:
            A tuple containing the number of data points processed, and computed Lambda matrices in CPU.
            The Lambda matrices are organized in nested dictionaries, where the first key in the name of
            the Lambda matrix and the second key is the module name.
    """
    with torch.no_grad():
        update_factor_args(model=model, factor_args=factor_args)
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)
        set_mode(
            model=model,
            tracked_module_names=tracked_module_names,
            mode=ModuleMode.LAMBDA,
        )
        if eigen_factors is not None:
            for name in eigen_factors:
                set_factors(model=model, factor_name=name, factors=eigen_factors[name])
    num_data_processed = torch.zeros(
        (1,), dtype=torch.int64, device=state.device, requires_grad=False
    )

    with tqdm(
        total=len(loader),
        desc="Fitting Lambda matrices",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process,
    ) as pbar:
        for batch in loader:
            batch = send_to_device(batch, device=state.device)

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                loss = task.compute_train_loss(
                    batch=batch,
                    model=model,
                    sample=not factor_args.use_empirical_fisher,
                )
                loss.backward()
            num_data_processed += find_batch_size(batch)
            pbar.update(1)

    if state.use_distributed:
        # Aggregate Lambda matrices across multiple devices or nodes.
        synchronize_lambda_matrices(model=model)
        dist.all_reduce(tensor=num_data_processed, op=torch.distributed.ReduceOp.SUM)

    with torch.no_grad():
        saved_factors: FACTOR_TYPE = {}
        for covariance_factor_name in LAMBDA_FACTOR_NAMES:
            saved_factors[covariance_factor_name] = load_factors(
                model=model, factor_name=covariance_factor_name
            )
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)
    return num_data_processed, saved_factors
