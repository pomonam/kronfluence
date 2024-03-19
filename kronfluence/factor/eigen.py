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
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    get_tracked_module_names,
    load_factors,
    remove_attention_mask,
    set_factors,
    set_mode,
    synchronize_lambda_matrices,
    update_factor_args,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
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
    PARTITION_TYPE,
)
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync


def eigendecomposition_save_path(
    output_dir: Path,
    factor_name: str,
) -> Path:
    """Generates the path for saving/loading Eigendecomposition results."""
    assert factor_name in EIGENDECOMPOSITION_FACTOR_NAMES
    return output_dir / f"{factor_name}.safetensors"


def save_eigendecomposition(output_dir: Path, factors: FACTOR_TYPE, metadata: Optional[Dict[str, str]] = None) -> None:
    """Saves Eigendecomposition results to disk."""
    assert set(factors.keys()) == set(EIGENDECOMPOSITION_FACTOR_NAMES)
    for factor_name in factors:
        save_path = eigendecomposition_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
        )
        save_file(tensors=factors[factor_name], filename=save_path, metadata=metadata)


def load_eigendecomposition(
    output_dir: Path,
) -> FACTOR_TYPE:
    """Loads Eigendecomposition results from disk."""
    eigen_factors = {}
    for factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
        save_path = eigendecomposition_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
        )
        eigen_factors[factor_name] = load_file(filename=save_path)
    return eigen_factors


def eigendecomposition_exist(
    output_dir: Path,
) -> bool:
    """Checks if Eigendecomposition results exist at specified path."""
    for factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
        save_path = eigendecomposition_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
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
        covariance_factors (FACTOR_TYPE):
            The covariance matrices to perform Eigendecomposition on.
        model (nn.Module):
            The model which contains modules which Eigendecomposition will be performed.
        state (State):
            The current process's information (e.g., device being used).
        factor_args (FactorArguments):
            Arguments related to performing Eigendecomposition.

    Returns:
        FACTOR_TYPE:
            The Eigendecomposition results in CPU. The Eigendecomposition results are organized in
            nested dictionaries, where the first key in the name of the Eigendecomposition factor (e.g.,
            activation eigenvector), and the second key is the module name.
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
                covariance_matrix = covariance_factors[covariance_name][module_name].to(
                    device=state.device,
                    dtype=factor_args.eigendecomposition_dtype,
                )
                # Normalize covariance matrices.
                covariance_matrix.div_(
                    covariance_factors[NUM_COVARIANCE_PROCESSED][module_name].to(device=state.device)
                )
                # In cases where covariance matrices are not exactly symmetric due to numerical issues.
                covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.t())
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                eigen_factors[eigenvalues_name][module_name] = eigenvalues.to(dtype=original_dtype).contiguous().cpu()
                eigen_factors[eigenvectors_name][module_name] = eigenvectors.to(dtype=original_dtype).contiguous().cpu()
                del eigenvalues, eigenvectors
            pbar.update(1)
    return eigen_factors


def lambda_matrices_save_path(
    output_dir: Path,
    factor_name: str,
    partition: Optional[PARTITION_TYPE] = None,
) -> Path:
    """Generates the path for saving/loading Lambda matrices."""
    assert factor_name in LAMBDA_FACTOR_NAMES
    if partition is not None:
        data_partition, module_partition = partition
        return output_dir / (
            f"{factor_name}_data_partition{data_partition}_module_partition{module_partition}.safetensors"
        )
    return output_dir / f"{factor_name}.safetensors"


def save_lambda_matrices(
    output_dir: Path,
    factors: FACTOR_TYPE,
    partition: Optional[PARTITION_TYPE] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves Lambda matrices to disk."""
    assert set(factors.keys()) == set(LAMBDA_FACTOR_NAMES)
    for factor_name in factors:
        save_path = lambda_matrices_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
            partition=partition,
        )
        save_file(tensors=factors[factor_name], filename=save_path, metadata=metadata)


def load_lambda_matrices(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> FACTOR_TYPE:
    """Loads Lambda matrices from disk."""
    lambda_factors = {}
    for factor_name in LAMBDA_FACTOR_NAMES:
        save_path = lambda_matrices_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
            partition=partition,
        )
        lambda_factors[factor_name] = load_file(filename=save_path)
    return lambda_factors


def lambda_matrices_exist(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> bool:
    """Check if Lambda matrices exist at specified path."""
    for factor_name in LAMBDA_FACTOR_NAMES:
        save_path = lambda_matrices_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
            partition=partition,
        )
        if not save_path.exists():
            return False
    return True


def fit_lambda_matrices_with_loader(
    model: nn.Module,
    state: State,
    task: Task,
    loader: data.DataLoader,
    factor_args: FactorArguments,
    eigen_factors: Optional[FACTOR_TYPE] = None,
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
        loader (data.DataLoader):
            The data loader that will be used to compute Lambda matrices.
        factor_args (FactorArguments):
            Arguments for computing Lambda matrices.
        eigen_factors (FACTOR_TYPE, optional):
            The eigendecomposition results to use for computing Lambda matrices.
        tracked_module_names (List[str], optional):
            A list of module names that Lambda matrices will be computed. If not specified, Lambda
            matrices will be computed for all available tracked modules.

    Returns:
        Tuple[torch.Tensor, FACTOR_TYPE]:
            A tuple containing the number of data points processed, and computed Lambda matrices in CPU.
            The Lambda matrices are organized in nested dictionaries, where the first key in the name of
            the computed variable and the second key is the module name.
    """
    with torch.no_grad():
        update_factor_args(model=model, factor_args=factor_args)
        remove_attention_mask(model=model)
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)
        set_mode(
            model=model,
            tracked_module_names=tracked_module_names,
            mode=ModuleMode.LAMBDA,
        )
        if eigen_factors is not None:
            for name in eigen_factors:
                set_factors(model=model, factor_name=name, factors=eigen_factors[name])
    total_steps = 0
    num_data_processed = torch.zeros((1,), dtype=torch.int64, requires_grad=False)

    with tqdm(
        total=len(loader),
        desc="Fitting Lambda matrices",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process,
    ) as pbar:
        for index, batch in enumerate(loader):
            batch = send_to_device(tensor=batch, device=state.device)

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                loss = task.compute_train_loss(
                    batch=batch,
                    model=model,
                    sample=not factor_args.use_empirical_fisher,
                )
                loss.backward()
            num_data_processed += find_batch_size(data=batch)
            total_steps += 1

            if (
                state.use_distributed
                and total_steps % factor_args.distributed_sync_steps == 0
                and index not in [len(loader) - 1, len(loader) - 2]
            ):
                # Periodically synchronize all processes to avoid timeout at the final Lambda synchronization.
                state.wait_for_everyone()

            pbar.update(1)

    with torch.no_grad():
        if state.use_distributed:
            # Aggregate Lambda matrices across multiple devices or nodes.
            synchronize_lambda_matrices(model=model)
            num_data_processed = num_data_processed.to(device=state.device)
            dist.all_reduce(tensor=num_data_processed, op=torch.distributed.ReduceOp.SUM)

        saved_factors: FACTOR_TYPE = {}
        if state.is_main_process:
            for factor_name in LAMBDA_FACTOR_NAMES:
                saved_factors[factor_name] = load_factors(model=model, factor_name=factor_name)
        state.wait_for_everyone()
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)
    return num_data_processed, saved_factors
