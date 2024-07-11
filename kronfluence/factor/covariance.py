from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import find_batch_size, send_to_device
from safetensors.torch import load_file, save_file
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    get_tracked_module_names,
    load_factors,
    set_attention_mask,
    set_gradient_scale,
    set_mode,
    synchronize_modules,
    update_factor_args,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
    COVARIANCE_FACTOR_NAMES,
    DISTRIBUTED_SYNC_INTERVAL,
    FACTOR_TYPE,
    PARTITION_TYPE,
)
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync


def covariance_matrices_save_path(
    output_dir: Path,
    factor_name: str,
    partition: Optional[PARTITION_TYPE] = None,
) -> Path:
    """Generates the path for saving or loading covariance matrices.

    Args:
        output_dir (Path):
            Directory to save or load the matrices.
        factor_name (str):
            Name of the factor (must be in `COVARIANCE_FACTOR_NAMES`).
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        Path:
            The full path for the covariance matrix file.

    Raises:
        AssertionError:
            If `factor_name` is not in `COVARIANCE_FACTOR_NAMES`.
    """
    assert factor_name in COVARIANCE_FACTOR_NAMES
    if partition is not None:
        data_partition, module_partition = partition
        return output_dir / (
            f"{factor_name}_data_partition{data_partition}_module_partition{module_partition}.safetensors"
        )
    return output_dir / f"{factor_name}.safetensors"


def save_covariance_matrices(
    output_dir: Path,
    factors: FACTOR_TYPE,
    partition: Optional[PARTITION_TYPE] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves covariance matrices to disk.

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
            If factors keys don't match `COVARIANCE_FACTOR_NAMES`.
    """
    assert set(factors.keys()) == set(COVARIANCE_FACTOR_NAMES)
    for factor_name in factors:
        save_path = covariance_matrices_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
            partition=partition,
        )
        save_file(tensors=factors[factor_name], filename=save_path, metadata=metadata)


def load_covariance_matrices(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> FACTOR_TYPE:
    """Loads covariance matrices from disk.

    Args:
        output_dir (Path):
            Directory to load the matrices from.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        FACTOR_TYPE:
            Dictionary of loaded covariance factors.
    """
    covariance_factors = {}
    for factor_name in COVARIANCE_FACTOR_NAMES:
        save_path = covariance_matrices_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
            partition=partition,
        )
        covariance_factors[factor_name] = load_file(filename=save_path)
    return covariance_factors


def covariance_matrices_exist(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> bool:
    """Checks if covariance matrices exist at the specified directory.

    Args:
        output_dir (Path):
            Directory to check for matrices.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        bool:
            `True` if all covariance matrices exist, `False` otherwise.
    """
    for factor_name in COVARIANCE_FACTOR_NAMES:
        save_path = covariance_matrices_save_path(
            output_dir=output_dir,
            factor_name=factor_name,
            partition=partition,
        )
        if not save_path.exists():
            return False
    return True


def fit_covariance_matrices_with_loader(
    model: nn.Module,
    state: State,
    task: Task,
    loader: data.DataLoader,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]] = None,
    disable_tqdm: bool = False,
) -> Tuple[torch.Tensor, FACTOR_TYPE]:
    """Computes activation and pseudo-gradient covariance matrices for a given model and task.

    Args:
        model (nn.Module):
            The model for which covariance matrices will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        loader (data.DataLoader):
            The data loader that will be used to fit covariance matrices.
        factor_args (FactorArguments):
            Arguments for computing covariance matrices.
        tracked_module_names (List[str], optional):
            A list of module names for which covariance matrices will be computed. If not specified,
            covariance matrices will be computed for all tracked modules.
        disable_tqdm (bool, optional):
            Whether to disable the progress bar. Defaults to `False`.

    Returns:
        Tuple[torch.Tensor, FACTOR_TYPE]:
            - Number of data points processed.
            - Computed covariance matrices (nested dict: factor_name -> module_name -> tensor).
    """
    update_factor_args(model=model, factor_args=factor_args)
    if tracked_module_names is None:
        tracked_module_names = get_tracked_module_names(model=model)
    set_mode(
        model=model,
        tracked_module_names=tracked_module_names,
        mode=ModuleMode.COVARIANCE,
        release_memory=True,
    )

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
        desc="Fitting covariance matrices",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for index, batch in enumerate(loader):
            batch = send_to_device(batch, device=state.device)

            attention_mask = task.get_attention_mask(batch=batch)
            if attention_mask is not None:
                set_attention_mask(model=model, attention_mask=attention_mask)

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                with autocast(device_type=state.device.type, enabled=enable_amp, dtype=factor_args.amp_dtype):
                    loss = task.compute_train_loss(
                        batch=batch,
                        model=model,
                        sample=not factor_args.use_empirical_fisher,
                    )
                scaler.scale(loss).backward()

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
        for factor_name in COVARIANCE_FACTOR_NAMES:
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
    set_attention_mask(model=model, attention_mask=None)
    if enable_grad_scaler:
        set_gradient_scale(model=model, gradient_scale=1.0)
    set_mode(model=model, mode=ModuleMode.DEFAULT, release_memory=True)
    state.wait_for_everyone()

    return num_data_processed, saved_factors
