from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from accelerate.utils import send_to_device
from safetensors.torch import load_file, save_file
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    accumulate_iterations,
    finalize_all_iterations,
    finalize_iteration,
    get_tracked_module_names,
    prepare_modules,
    set_factors,
    set_gradient_scale,
    set_mode,
    synchronize_modules,
    truncate,
    update_factor_args,
    update_score_args,
)
from kronfluence.score.dot_product import (
    compute_aggregated_dot_products_with_loader,
    compute_dot_products_with_loader,
)
from kronfluence.task import Task
from kronfluence.utils.constants import FACTOR_TYPE, PARTITION_TYPE, SCORE_TYPE
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync


def pairwise_scores_save_path(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> Path:
    """Generates the path for saving or loading pairwise influence scores.

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
            f"pairwise_scores_data_partition{data_partition}_module_partition{module_partition}.safetensors"
        )
    return output_dir / "pairwise_scores.safetensors"


def save_pairwise_scores(
    output_dir: Path,
    scores: SCORE_TYPE,
    partition: Optional[PARTITION_TYPE] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves pairwise scores to disk.

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
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    save_file(tensors=scores, filename=save_path, metadata=metadata)


def load_pairwise_scores(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> SCORE_TYPE:
    """Loads pairwise scores from disk.

    Args:
        output_dir (Path):
            Directory to load the scores from.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        SCORE_TYPE:
            Dictionary of loaded scores.
    """
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return load_file(filename=save_path)


def pairwise_scores_exist(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> bool:
    """Checks if pairwise influence scores exist at the specified directory.

    Args:
        output_dir (Path):
            Directory to check for scores.
        partition (PARTITION_TYPE, optional):
            Partition information, if any.

    Returns:
        bool:
            `True` if scores exist, `False` otherwise.
    """
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return save_path.exists()


def compute_pairwise_scores_with_loaders(
    loaded_factors: FACTOR_TYPE,
    model: nn.Module,
    state: State,
    task: Task,
    query_loader: data.DataLoader,
    per_device_query_batch_size: int,
    train_loader: data.DataLoader,
    score_args: ScoreArguments,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]],
    disable_tqdm: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes pairwise influence scores for a given model and task.

    Args:
        loaded_factors (FACTOR_TYPE):
            Computed factors.
        model (nn.Module):
            The model for which pairwise influence scores will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        query_loader (data.DataLoader):
            The data loader that will be used to compute query gradients.
        per_device_query_batch_size (int):
            Per-device batch size for the query data loader.
        train_loader (data.DataLoader):
            The data loader that will be used to compute training gradients.
        score_args (ScoreArguments):
            Arguments for computing pairwise scores.
        factor_args (FactorArguments):
            Arguments used to compute factors.
        tracked_module_names (List[str], optional):
            A list of module names that pairwise scores will be computed. If not specified, scores
            will be computed for all available tracked modules.
        disable_tqdm (bool, optional):
            Whether to disable the progress bar. Defaults to `False`.

    Returns:
        SCORE_TYPE:
            A dictionary containing the module name and its pairwise influence scores.
    """
    update_factor_args(model=model, factor_args=factor_args)
    update_score_args(model=model, score_args=score_args)
    if tracked_module_names is None:
        tracked_module_names = get_tracked_module_names(model=model)
    set_mode(
        model=model,
        mode=ModuleMode.PRECONDITION_GRADIENT,
        tracked_module_names=tracked_module_names,
        release_memory=True,
    )
    if len(loaded_factors) > 0:
        for name in loaded_factors:
            set_factors(
                model=model,
                factor_name=name,
                factors=loaded_factors[name],
                clone=True,
            )
    prepare_modules(model=model, tracked_module_names=tracked_module_names, device=state.device)

    total_scores_chunks: Dict[str, Union[List[torch.Tensor], torch.Tensor]] = {}
    total_query_batch_size = per_device_query_batch_size * state.num_processes
    query_remainder = len(query_loader.dataset) % total_query_batch_size

    num_batches = len(query_loader)
    query_iter = iter(query_loader)
    num_accumulations = 0
    enable_amp = score_args.amp_dtype is not None
    enable_grad_scaler = enable_amp and factor_args.amp_dtype == torch.float16
    scaler = GradScaler(init_scale=factor_args.amp_scale, enabled=enable_grad_scaler)
    if enable_grad_scaler:
        gradient_scale = 1.0 / scaler.get_scale()
        set_gradient_scale(model=model, gradient_scale=gradient_scale)

    dot_product_func = (
        compute_aggregated_dot_products_with_loader
        if score_args.aggregate_train_gradients
        else compute_dot_products_with_loader
    )

    with tqdm(
        total=num_batches,
        desc="Computing pairwise scores (query gradient)",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for query_index in range(num_batches):
            query_batch = next(query_iter)
            query_batch = send_to_device(
                tensor=query_batch,
                device=state.device,
            )

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                with autocast(device_type=state.device.type, enabled=enable_amp, dtype=score_args.amp_dtype):
                    measurement = task.compute_measurement(batch=query_batch, model=model)
                scaler.scale(measurement).backward()

            if factor_args.has_shared_parameters:
                finalize_iteration(model=model, tracked_module_names=tracked_module_names)

            if state.use_distributed:
                # Stack preconditioned query gradient across multiple devices or nodes.
                synchronize_modules(
                    model=model, tracked_module_names=tracked_module_names, num_processes=state.num_processes
                )
                if query_index == len(query_loader) - 1 and query_remainder > 0:
                    # Removes duplicate data points if the dataset is not evenly divisible by the current batch size.
                    truncate(model=model, tracked_module_names=tracked_module_names, keep_size=query_remainder)
            accumulate_iterations(model=model, tracked_module_names=tracked_module_names)
            del query_batch, measurement

            num_accumulations += 1
            if (
                num_accumulations < score_args.query_gradient_accumulation_steps
                and query_index != len(query_loader) - 1
            ):
                pbar.update(1)
                continue

            # Computes the dot product between preconditioning query gradient and all training gradients.
            scores = dot_product_func(
                model=model,
                state=state,
                task=task,
                train_loader=train_loader,
                factor_args=factor_args,
                score_args=score_args,
                tracked_module_names=tracked_module_names,
                scaler=scaler,
                disable_tqdm=disable_tqdm,
            )

            if state.is_main_process:
                for module_name, current_scores in scores.items():
                    if module_name not in total_scores_chunks:
                        total_scores_chunks[module_name] = []
                    total_scores_chunks[module_name].append(current_scores)
            del scores
            state.wait_for_everyone()

            num_accumulations = 0
            pbar.update(1)

    if state.is_main_process:
        for module_name in total_scores_chunks:
            total_scores_chunks[module_name] = torch.cat(total_scores_chunks[module_name], dim=0)

    model.zero_grad(set_to_none=True)
    if enable_grad_scaler:
        set_gradient_scale(model=model, gradient_scale=1.0)
    finalize_all_iterations(model=model, tracked_module_names=tracked_module_names)
    set_mode(model=model, mode=ModuleMode.DEFAULT, release_memory=True)
    state.wait_for_everyone()

    return total_scores_chunks


def compute_pairwise_query_aggregated_scores_with_loaders(
    loaded_factors: FACTOR_TYPE,
    model: nn.Module,
    state: State,
    task: Task,
    query_loader: data.DataLoader,
    per_device_query_batch_size: int,
    train_loader: data.DataLoader,
    score_args: ScoreArguments,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]],
    disable_tqdm: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes pairwise influence scores (with query gradients aggregated) for a given model and task. See
    `compute_pairwise_scores_with_loaders` for detailed information."""
    del per_device_query_batch_size
    update_factor_args(model=model, factor_args=factor_args)
    update_score_args(model=model, score_args=score_args)
    if tracked_module_names is None:
        tracked_module_names = get_tracked_module_names(model=model)
    set_mode(
        model=model,
        mode=ModuleMode.GRADIENT_AGGREGATION,
        tracked_module_names=tracked_module_names,
        release_memory=True,
    )
    if len(loaded_factors) > 0:
        for name in loaded_factors:
            set_factors(model=model, factor_name=name, factors=loaded_factors[name], clone=True)
    prepare_modules(model=model, tracked_module_names=tracked_module_names, device=state.device)

    enable_amp = score_args.amp_dtype is not None
    enable_grad_scaler = enable_amp and factor_args.amp_dtype == torch.float16
    scaler = GradScaler(init_scale=factor_args.amp_scale, enabled=enable_grad_scaler)
    if enable_grad_scaler:
        gradient_scale = 1.0 / scaler.get_scale()
        set_gradient_scale(model=model, gradient_scale=gradient_scale)

    dot_product_func = (
        compute_aggregated_dot_products_with_loader
        if score_args.aggregate_train_gradients
        else compute_dot_products_with_loader
    )

    with tqdm(
        total=len(query_loader),
        desc="Computing pairwise scores (query gradient)",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for query_batch in query_loader:
            query_batch = send_to_device(
                tensor=query_batch,
                device=state.device,
            )

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                with autocast(device_type=state.device.type, enabled=enable_amp, dtype=score_args.amp_dtype):
                    measurement = task.compute_measurement(batch=query_batch, model=model)
                scaler.scale(measurement).backward()

            if factor_args.has_shared_parameters:
                finalize_iteration(model=model, tracked_module_names=tracked_module_names)

            del measurement
            pbar.update(1)

    if state.use_distributed:
        synchronize_modules(model=model, tracked_module_names=tracked_module_names)

    set_mode(
        model=model,
        mode=ModuleMode.PRECONDITION_GRADIENT,
        tracked_module_names=tracked_module_names,
        release_memory=False,
    )
    finalize_all_iterations(model=model, tracked_module_names=tracked_module_names)

    scores = dot_product_func(
        model=model,
        state=state,
        task=task,
        train_loader=train_loader,
        factor_args=factor_args,
        score_args=score_args,
        tracked_module_names=tracked_module_names,
        scaler=scaler,
        disable_tqdm=disable_tqdm,
    )

    model.zero_grad(set_to_none=True)
    if enable_grad_scaler:
        set_gradient_scale(model=model, gradient_scale=1.0)
    set_mode(model=model, mode=ModuleMode.DEFAULT, release_memory=True)
    state.wait_for_everyone()

    return scores
