from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from safetensors.torch import load_file, save_file
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.module import TrackedModule
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    accumulate_iterations,
    finalize_iteration,
    get_tracked_module_names,
    prepare_modules,
    set_factors,
    set_gradient_scale,
    set_mode,
    update_factor_args,
    update_score_args,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
    ALL_MODULE_NAME,
    DISTRIBUTED_SYNC_INTERVAL,
    FACTOR_TYPE,
    PARTITION_TYPE,
    SCORE_TYPE,
    SELF_SCORE_VECTOR_NAME,
)
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync, release_memory


def self_scores_save_path(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> Path:
    """Generates the path for saving or loading self-influence scores.

    Args:
        output_dir (Path):
            Directory to save or load the matrices.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        Path:
            The full path for the score file.
    """
    if partition is not None:
        data_partition, module_partition = partition
        return output_dir / (
            f"self_scores_data_partition{data_partition}_module_partition{module_partition}.safetensors"
        )
    return output_dir / "self_scores.safetensors"


def save_self_scores(
    output_dir: Path,
    scores: SCORE_TYPE,
    partition: Optional[PARTITION_TYPE] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves self-influence scores to disk.

    Args:
        output_dir (Path):
            Directory to save the scores.
        scores (SCORE_TYPE):
            Dictionary of scores to save.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.
        metadata (Dict[str, str], optional):
            Additional metadata to save with the scores.
    """
    save_path = self_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    save_file(tensors=scores, filename=save_path, metadata=metadata)


def load_self_scores(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> SCORE_TYPE:
    """Loads self-influence scores from disk.

    Args:
        output_dir (Path):
            Directory to load the scores from.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        SCORE_TYPE:
            Dictionary of loaded scores.
    """
    save_path = self_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return load_file(filename=save_path)


def self_scores_exist(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> bool:
    """Checks if self-influence scores exist at the specified directory.

    Args:
        output_dir (Path):
            Directory to check for scores.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        bool:
            `True` if scores exist, `False` otherwise.
    """
    save_path = self_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return save_path.exists()


def compute_self_scores_with_loaders(
    loaded_factors: FACTOR_TYPE,
    model: nn.Module,
    state: State,
    task: Task,
    train_loader: data.DataLoader,
    score_args: ScoreArguments,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]],
    disable_tqdm: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes self-influence scores for a given model and task.

    Args:
        loaded_factors (FACTOR_TYPE):
            Computed factors.
        model (nn.Module):
            The model for which self-influence scores will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        train_loader (data.DataLoader):
            The data loader that will be used to compute training gradients.
        score_args (ScoreArguments):
            Arguments for computing self-influence scores.
        factor_args (FactorArguments):
            Arguments used to compute factors.
        tracked_module_names (List[str], optional):
            A list of module names that self-influence scores will be computed. If not specified, scores
            will be computed for all available tracked modules.
        disable_tqdm (bool, optional):
            Whether to disable the progress bar. Defaults to `False`.

    Returns:
        SCORE_TYPE:
            A dictionary containing the module name and its self-influence scores.
    """
    update_factor_args(model=model, factor_args=factor_args)
    update_score_args(model=model, score_args=score_args)
    if tracked_module_names is None:
        tracked_module_names = get_tracked_module_names(model=model)
    set_mode(
        model=model,
        mode=ModuleMode.SELF_SCORE,
        tracked_module_names=tracked_module_names,
        release_memory=True,
    )
    if len(loaded_factors) > 0:
        for name in loaded_factors:
            set_factors(model=model, factor_name=name, factors=loaded_factors[name], clone=True)
    prepare_modules(model=model, tracked_module_names=tracked_module_names, device=state.device)

    dataset_size = len(train_loader.dataset)
    score_chunks: Dict[str, List[torch.Tensor]] = {}
    if score_args.compute_per_module_scores:
        for module in model.modules():
            if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                score_chunks[module.name] = []
    else:
        score_chunks[ALL_MODULE_NAME] = []

    cached_module_lst = []
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            cached_module_lst.append(module)

    total_steps = 0
    enable_amp = score_args.amp_dtype is not None
    enable_grad_scaler = enable_amp and factor_args.amp_dtype == torch.float16
    scaler = GradScaler(init_scale=factor_args.amp_scale, enabled=enable_grad_scaler)
    if enable_grad_scaler:
        gradient_scale = 1.0 / scaler.get_scale()
        set_gradient_scale(model=model, gradient_scale=gradient_scale)

    with tqdm(
        total=len(train_loader),
        desc="Computing self-influence scores",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for index, batch in enumerate(train_loader):
            batch = send_to_device(
                tensor=batch,
                device=state.device,
            )

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                with autocast(device_type=state.device.type, enabled=enable_amp, dtype=score_args.amp_dtype):
                    loss = task.compute_train_loss(
                        batch=batch,
                        model=model,
                        sample=False,
                    )
                scaler.scale(loss).backward()

            if factor_args.has_shared_parameters:
                finalize_iteration(model=model, tracked_module_names=tracked_module_names)

            with torch.no_grad():
                if score_args.compute_per_module_scores:
                    for module in cached_module_lst:
                        score_chunks[module.name].append(
                            module.get_factor(factor_name=SELF_SCORE_VECTOR_NAME).to(device="cpu", copy=True)
                        )
                else:
                    self_scores = None
                    for module in cached_module_lst:
                        if self_scores is None:
                            self_scores = torch.zeros_like(
                                module.get_factor(factor_name=SELF_SCORE_VECTOR_NAME), requires_grad=False
                            )
                        self_scores.add_(module.get_factor(factor_name=SELF_SCORE_VECTOR_NAME))
                    self_scores = self_scores.cpu()
                    score_chunks[ALL_MODULE_NAME].append(self_scores)
                accumulate_iterations(model=model, tracked_module_names=tracked_module_names)

            if (
                state.use_distributed
                and total_steps % DISTRIBUTED_SYNC_INTERVAL == 0
                and index not in [len(train_loader) - 1, len(train_loader) - 2]
            ):
                state.wait_for_everyone()

            del loss
            total_steps += 1
            pbar.update(1)

    model.zero_grad(set_to_none=True)
    if enable_grad_scaler:
        set_gradient_scale(model=model, gradient_scale=1.0)
    set_mode(
        model=model,
        mode=ModuleMode.DEFAULT,
        tracked_module_names=tracked_module_names,
        release_memory=True,
    )
    release_memory()

    total_scores: SCORE_TYPE = {}
    for module_name, chunks in score_chunks.items():
        total_scores[module_name] = torch.cat(chunks, dim=0)
        if state.use_distributed:
            total_scores[module_name] = total_scores[module_name].to(device=state.device)
            gather_list = None
            if state.is_main_process:
                gather_list = [torch.zeros_like(total_scores[module_name]) for _ in range(state.num_processes)]
            dist.gather(total_scores[module_name], gather_list)
            if state.is_main_process:
                total_scores[module_name] = torch.cat(gather_list, dim=0)[:dataset_size].cpu()
            else:
                total_scores[module_name] = total_scores[module_name].cpu()
    state.wait_for_everyone()

    return total_scores


def compute_self_measurement_scores_with_loaders(
    loaded_factors: FACTOR_TYPE,
    model: nn.Module,
    state: State,
    task: Task,
    train_loader: data.DataLoader,
    score_args: ScoreArguments,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]],
    disable_tqdm: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes self-influence scores with measurement (instead of the loss) for a given model and task. See
    `compute_self_scores_with_loaders` for the detailed docstring."""
    update_factor_args(model=model, factor_args=factor_args)
    update_score_args(model=model, score_args=score_args)
    if tracked_module_names is None:
        tracked_module_names = get_tracked_module_names(model=model)
    if len(loaded_factors) > 0:
        for name in loaded_factors:
            set_factors(
                model=model,
                factor_name=name,
                factors=loaded_factors[name],
                clone=True,
            )
    prepare_modules(model=model, tracked_module_names=tracked_module_names, device=state.device)

    cached_module_lst = []
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            cached_module_lst.append(module)

    dataset_size = len(train_loader.dataset)
    score_chunks: Dict[str, List[torch.Tensor]] = {}
    if score_args.compute_per_module_scores:
        for module in cached_module_lst:
            score_chunks[module.name] = []
    else:
        score_chunks[ALL_MODULE_NAME] = []

    total_steps = 0
    enable_amp = score_args.amp_dtype is not None
    enable_grad_scaler = enable_amp and factor_args.amp_dtype == torch.float16
    scaler = GradScaler(init_scale=factor_args.amp_scale, enabled=enable_grad_scaler)
    if enable_grad_scaler:
        gradient_scale = 1.0 / scaler.get_scale()
        set_gradient_scale(model=model, gradient_scale=gradient_scale)

    with tqdm(
        total=len(train_loader),
        desc="Computing self-influence scores",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for index, batch in enumerate(train_loader):
            batch = send_to_device(
                tensor=batch,
                device=state.device,
            )

            set_mode(
                model=model,
                mode=ModuleMode.PRECONDITION_GRADIENT,
                tracked_module_names=tracked_module_names,
                release_memory=False,
            )
            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                with autocast(device_type=state.device.type, enabled=enable_amp, dtype=score_args.amp_dtype):
                    measurement = task.compute_measurement(batch=batch, model=model)
                scaler.scale(measurement).backward()

            if factor_args.has_shared_parameters:
                finalize_iteration(model=model, tracked_module_names=tracked_module_names)
            del measurement

            set_mode(
                model=model,
                mode=ModuleMode.SELF_MEASUREMENT_SCORE,
                tracked_module_names=tracked_module_names,
                release_memory=False,
            )
            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                with autocast(device_type=state.device.type, enabled=enable_amp, dtype=score_args.amp_dtype):
                    loss = task.compute_train_loss(
                        batch=batch,
                        model=model,
                        sample=False,
                    )
                scaler.scale(loss).backward()

            if factor_args.has_shared_parameters:
                finalize_iteration(model=model, tracked_module_names=tracked_module_names)
            del loss

            with torch.no_grad():
                if score_args.compute_per_module_scores:
                    for module in cached_module_lst:
                        score_chunks[module.name].append(
                            module.get_factor(factor_name=SELF_SCORE_VECTOR_NAME).to(device="cpu", copy=True)
                        )
                else:
                    self_scores = None
                    for module in cached_module_lst:
                        if self_scores is None:
                            self_scores = torch.zeros_like(
                                module.get_factor(factor_name=SELF_SCORE_VECTOR_NAME), requires_grad=False
                            )
                        self_scores.add_(module.get_factor(factor_name=SELF_SCORE_VECTOR_NAME))
                    self_scores = self_scores.cpu()
                    score_chunks[ALL_MODULE_NAME].append(self_scores)
                accumulate_iterations(model=model, tracked_module_names=tracked_module_names)

            if (
                state.use_distributed
                and total_steps % DISTRIBUTED_SYNC_INTERVAL == 0
                and index not in [len(train_loader) - 1, len(train_loader) - 2]
            ):
                state.wait_for_everyone()

            total_steps += 1
            pbar.update(1)

    model.zero_grad(set_to_none=True)
    if enable_grad_scaler:
        set_gradient_scale(model=model, gradient_scale=1.0)
    set_mode(
        model=model,
        mode=ModuleMode.DEFAULT,
        tracked_module_names=tracked_module_names,
        release_memory=True,
    )
    release_memory()

    total_scores: SCORE_TYPE = {}
    for module_name, chunks in score_chunks.items():
        total_scores[module_name] = torch.cat(chunks, dim=0)
        if state.use_distributed:
            total_scores[module_name] = total_scores[module_name].to(device=state.device)
            gather_list = None
            if state.is_main_process:
                gather_list = [torch.zeros_like(total_scores[module_name]) for _ in range(state.num_processes)]
            dist.gather(total_scores[module_name], gather_list)
            if state.is_main_process:
                total_scores[module_name] = torch.cat(gather_list, dim=0)[:dataset_size].cpu()
            else:
                total_scores[module_name] = total_scores[module_name].cpu()
    state.wait_for_everyone()

    return total_scores
