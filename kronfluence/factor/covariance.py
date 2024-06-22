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
    remove_gradient_scale,
    set_attention_mask,
    set_gradient_scale,
    set_mode,
    synchronize_covariance_matrices,
    update_factor_args,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
    COVARIANCE_FACTOR_NAMES,
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
    """Generates the path for saving/loading covariance matrices."""
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
    """Saves covariance matrices to disk."""
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
    """Loads covariance matrices from disk."""
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
    """Checks if covariance matrices exist at the specified directory."""
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
            Disables TQDM progress bars. Defaults to False.

    Returns:
        Tuple[torch.Tensor, FACTOR_TYPE]:
            A tuple containing the number of data points processed and computed covariance matrices.
            The covariance matrices are organized in nested dictionaries, where the first key is the name of the
            covariance matrix (e.g., activation covariance and pseudo-gradient covariance) and the second key is
            the module name.
    """
    with torch.no_grad():
        update_factor_args(model=model, factor_args=factor_args)
        if tracked_module_names is None:
            tracked_module_names = get_tracked_module_names(model=model)
        set_mode(
            model=model,
            tracked_module_names=tracked_module_names,
            mode=ModuleMode.COVARIANCE,
            keep_factors=False,
        )

    total_steps = 0
    num_data_processed = torch.zeros((1,), dtype=torch.int64, requires_grad=False)
    enable_amp = factor_args.amp_dtype is not None
    scaler = GradScaler(enabled=enable_amp)
    if enable_amp:
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
            with torch.no_grad():
                attention_mask = task.get_attention_mask(batch=batch)
                if attention_mask is not None:
                    set_attention_mask(model=model, attention_mask=attention_mask)

            model.zero_grad(set_to_none=True)
            with no_sync(model=model, state=state):
                with autocast(device_type=state.device.type, enabled=enable_amp, dtype=factor_args.amp_dtype):
                    loss = task.compute_train_loss(
                        batch=batch,
                        model=model,
                        sample=not factor_args.use_empirical_fisher,
                    )
                scaler.scale(loss).backward()

            num_data_processed += find_batch_size(data=batch)
            total_steps += 1

            if (
                state.use_distributed
                and total_steps % factor_args.distributed_sync_steps == 0
                and index not in [len(loader) - 1, len(loader) - 2]
            ):
                # Periodically synchronizes all processes to avoid timeout at the final synchronization.
                state.wait_for_everyone()

            pbar.update(1)

    with torch.no_grad():
        if state.use_distributed:
            # Aggregates covariance matrices across multiple devices or nodes.
            synchronize_covariance_matrices(model=model)
            num_data_processed = num_data_processed.to(device=state.device)
            dist.all_reduce(tensor=num_data_processed, op=torch.distributed.ReduceOp.SUM)

        saved_factors: FACTOR_TYPE = {}
        for factor_name in COVARIANCE_FACTOR_NAMES:
            saved_factors[factor_name] = load_factors(model=model, factor_name=factor_name)
        state.wait_for_everyone()

        # Clean up the memory.
        model.zero_grad(set_to_none=True)
        remove_gradient_scale(model=model)
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)

    return num_data_processed, saved_factors
