from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import find_batch_size, send_to_device
from accelerate.utils.memory import should_reduce_batch_size
from safetensors.torch import load_file, save_file
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    finalize_iteration,
    get_tracked_module_names,
    load_factors,
    set_factors,
    set_gradient_scale,
    set_mode,
    synchronize_modules,
    update_factor_args,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
    ACTIVATION_COVARIANCE_MATRIX_NAME,
    ACTIVATION_EIGENVALUES_NAME,
    ACTIVATION_EIGENVECTORS_NAME,
    DISTRIBUTED_SYNC_INTERVAL,
    EIGENDECOMPOSITION_FACTOR_NAMES,
    FACTOR_TYPE,
    GRADIENT_COVARIANCE_MATRIX_NAME,
    GRADIENT_EIGENVALUES_NAME,
    GRADIENT_EIGENVECTORS_NAME,
    LAMBDA_FACTOR_NAMES,
    NUM_ACTIVATION_COVARIANCE_PROCESSED,
    NUM_GRADIENT_COVARIANCE_PROCESSED,
    PARTITION_TYPE,
)
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync, release_memory


def eigendecomposition_save_path(
    output_dir: Path,
    factor_name: str,
) -> Path:
    """Generates the path for saving or loading eigendecomposition results.

    Args:
        output_dir (Path):
            Directory to save or load eigenvectors and eigenvalues.
        factor_name (str):
            Name of the factor (must be in `EIGENDECOMPOSITION_FACTOR_NAMES`).

    Returns:
        Path:
            The full path for the eigendecomposition file.

    Raises:
        AssertionError:
            If `factor_name` is not in `EIGENDECOMPOSITION_FACTOR_NAMES`.
    """
    assert factor_name in EIGENDECOMPOSITION_FACTOR_NAMES
    return output_dir / f"{factor_name}.safetensors"


def save_eigendecomposition(output_dir: Path, factors: FACTOR_TYPE, metadata: Optional[Dict[str, str]] = None) -> None:
    """Saves eigendecomposition results to disk.

    Args:
        output_dir (Path):
            Directory to save the eigenvectors and eigenvalues.
        factors (FACTOR_TYPE):
            Dictionary of factors to save.
        metadata (Dict[str, str], optional):
            Additional metadata to save with the factors.

    Raises:
        AssertionError:
            If factors keys don't match `EIGENDECOMPOSITION_FACTOR_NAMES`.
    """
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
    """Loads eigendecomposition results from disk.

    Args:
        output_dir (Path):
            Directory to load the results from.

    Returns:
        FACTOR_TYPE:
            Dictionary of loaded eigendecomposition results.
    """
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
    """Checks if eigendecomposition results exist at the specified directory.

    Args:
        output_dir (Path):
            Directory to check for results.

    Returns:
        bool:
            `True` if all eigendecomposition results exist, `False` otherwise.
    """
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
    disable_tqdm: bool = False,
) -> FACTOR_TYPE:
    """Performs eigendecomposition on activation and pseudo-gradient covariance matrices.

    Args:
        covariance_factors (FACTOR_TYPE):
            Computed covariance factors.
        model (nn.Module):
            The model used to compute covariance matrices.
        state (State):
            The current process's information (e.g., device being used).
        factor_args (FactorArguments):
            Arguments for performing eigendecomposition.
        disable_tqdm (bool, optional):
            Whether to disable the progress bar. Defaults to `False`.

    Returns:
        FACTOR_TYPE:
            The results are organized in nested dictionaries, where the first key is the name of the factor
            (e.g., activation eigenvector), and the second key is the module name.
    """
    eigen_factors: FACTOR_TYPE = {}
    for factor_name in EIGENDECOMPOSITION_FACTOR_NAMES:
        eigen_factors[factor_name] = {}
    tracked_module_names = get_tracked_module_names(model=model)

    with tqdm(
        total=len(tracked_module_names),
        desc="Performing Eigendecomposition",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for module_name in tracked_module_names:
            for covariance_name, num_processed_name, eigenvectors_name, eigenvalues_name in [
                (
                    ACTIVATION_COVARIANCE_MATRIX_NAME,
                    NUM_ACTIVATION_COVARIANCE_PROCESSED,
                    ACTIVATION_EIGENVECTORS_NAME,
                    ACTIVATION_EIGENVALUES_NAME,
                ),
                (
                    GRADIENT_COVARIANCE_MATRIX_NAME,
                    NUM_GRADIENT_COVARIANCE_PROCESSED,
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
                covariance_matrix.div_(covariance_factors[num_processed_name][module_name].to(device=state.device))
                # In cases where covariance matrices are not exactly symmetric due to numerical issues.
                covariance_matrix = covariance_matrix + covariance_matrix.t()
                covariance_matrix.mul_(0.5)

                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    # If we get OOM error, release the memory and try again.
                    if should_reduce_batch_size(exception=e):  # pylint: disable=no-else-continue
                        release_memory()
                        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
                    else:
                        raise
                del covariance_matrix
                eigen_factors[eigenvalues_name][module_name] = eigenvalues.contiguous().to(
                    dtype=original_dtype, device="cpu"
                )
                eigen_factors[eigenvectors_name][module_name] = eigenvectors.contiguous().to(
                    dtype=original_dtype, device="cpu"
                )
                del eigenvalues, eigenvectors

            pbar.update(1)

    return eigen_factors


def lambda_matrices_save_path(
    output_dir: Path,
    factor_name: str,
    partition: Optional[PARTITION_TYPE] = None,
) -> Path:
    """Generates the path for saving or loading Lambda matrices.

    Args:
        output_dir (Path):
            Directory to save or load the matrices.
        factor_name (str):
            Name of the factor (must be in `LAMBDA_FACTOR_NAMES`).
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        Path:
            The full path for the Lambda matrix file.

    Raises:
        AssertionError:
            If `factor_name` is not in `LAMBDA_FACTOR_NAMES`.
    """
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
    """Saves Lambda matrices to disk.

    Args:
        output_dir (Path):
            Directory to save the matrices.
        factors (FACTOR_TYPE):
            Dictionary of factors to save.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.
        metadata (Dict[str, str], optional):
            Additional metadata to save with the factors.

    Raises:
        AssertionError:
            If factors keys don't match `LAMBDA_FACTOR_NAMES`.
    """
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
    """Loads Lambda matrices from disk.

    Args:
        output_dir (Path):
            Directory to load the matrices from.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        FACTOR_TYPE:
            Dictionary of loaded Lambda factors.
    """
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
    """Checks if Lambda matrices exist at the specified directory.

    Args:
        output_dir (Path):
            Directory to check for matrices.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        bool:
            `True` if all Lambda matrices exist, `False` otherwise.
    """
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
    disable_tqdm: bool = False,
) -> Tuple[torch.Tensor, FACTOR_TYPE]:
    """Computes Lambda matrices for a given model and task.

    Args:
        model (nn.Module):
            The model for which Lambda matrices will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        loader (data.DataLoader):
            The data loader that will be used to compute Lambda matrices.
        factor_args (FactorArguments):
            Arguments for computing Lambda matrices.
        eigen_factors (FACTOR_TYPE, optional):
            Computed eigendecomposition results.
        tracked_module_names (List[str], optional):
            A list of module names for which Lambda matrices will be computed. If not specified,
            Lambda matrices will be computed for all tracked modules.
        disable_tqdm (bool, optional):
            Whether to disable the progress bar. Defaults to `False`.

    Returns:
        Tuple[torch.Tensor, FACTOR_TYPE]:
            - Number of data points processed.
            - Computed Lambda matrices (nested dict: factor_name -> module_name -> tensor).
    """
    update_factor_args(model=model, factor_args=factor_args)
    if tracked_module_names is None:
        tracked_module_names = get_tracked_module_names(model=model)
    set_mode(
        model=model,
        tracked_module_names=tracked_module_names,
        mode=ModuleMode.LAMBDA,
        release_memory=True,
    )
    if eigen_factors is not None:
        for name in eigen_factors:
            set_factors(model=model, factor_name=name, factors=eigen_factors[name], clone=True)

    total_steps = 0
    num_data_processed = torch.zeros((1,), dtype=torch.int64, requires_grad=False)
    enable_amp = factor_args.amp_dtype is not None
    enable_grad_scaler = enable_amp and factor_args.amp_dtype == torch.float16
    scaler = GradScaler(init_scale=factor_args.amp_scale, enabled=enable_grad_scaler)
    if enable_grad_scaler:
        gradient_scale = 1.0 / scaler.get_scale()
        set_gradient_scale(model=model, gradient_scale=gradient_scale)

    with tqdm(
        total=len(loader),
        desc="Fitting Lambda matrices",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for index, batch in enumerate(loader):
            batch = send_to_device(tensor=batch, device=state.device)

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                with autocast(device_type=state.device.type, enabled=enable_amp, dtype=factor_args.amp_dtype):
                    loss = task.compute_train_loss(
                        batch=batch,
                        model=model,
                        sample=not factor_args.use_empirical_fisher,
                    )
                scaler.scale(loss).backward()

            if factor_args.has_shared_parameters:
                finalize_iteration(model=model, tracked_module_names=tracked_module_names)

            if (
                state.use_distributed
                and total_steps % DISTRIBUTED_SYNC_INTERVAL == 0
                and index not in [len(loader) - 1, len(loader) - 2]
            ):
                state.wait_for_everyone()

            num_data_processed.add_(find_batch_size(data=batch))
            del loss
            total_steps += 1
            pbar.update(1)

    if state.use_distributed:
        synchronize_modules(model=model, tracked_module_names=tracked_module_names)
        num_data_processed = num_data_processed.to(device=state.device)
        dist.all_reduce(tensor=num_data_processed, op=torch.distributed.ReduceOp.SUM)
        num_data_processed = num_data_processed.cpu()

    saved_factors: FACTOR_TYPE = {}
    if state.is_main_process:
        for factor_name in LAMBDA_FACTOR_NAMES:
            factor = load_factors(
                model=model,
                factor_name=factor_name,
                tracked_module_names=tracked_module_names,
                cpu=True,
            )
            if len(factor) == 0:
                raise ValueError(f"Factor `{factor_name}` has not been computed.")
            saved_factors[factor_name] = factor

    model.zero_grad(set_to_none=True)
    if enable_grad_scaler:
        set_gradient_scale(model=model, gradient_scale=1.0)
    set_mode(model=model, mode=ModuleMode.DEFAULT, release_memory=True)
    state.wait_for_everyone()

    return num_data_processed, saved_factors
