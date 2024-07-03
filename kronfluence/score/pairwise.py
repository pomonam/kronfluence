from pathlib import Path
from typing import Dict, List, Optional, Union

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
    accumulate_preconditioned_gradient,
    aggregated_gradient_exist,
    compute_pairwise_scores_from_aggregation,
    compute_preconditioned_gradient_from_aggregation,
    finalize_gradient_aggregation,
    finalize_pairwise_scores,
    finalize_preconditioned_gradient,
    get_tracked_module_names,
    release_aggregated_gradient,
    release_preconditioned_gradient,
    release_scores,
    set_factors,
    set_gradient_scale,
    set_mode,
    synchronize_aggregated_gradient,
    synchronize_preconditioned_gradient,
    truncate_preconditioned_gradient,
    update_factor_args,
    update_score_args,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
    ALL_MODULE_NAME,
    FACTOR_TYPE,
    PAIRWISE_SCORE_MATRIX_NAME,
    PARTITION_TYPE,
    SCORE_TYPE,
)
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync, release_memory


def pairwise_scores_save_path(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> Path:
    """Generates the path for saving/loading pairwise scores."""
    if partition is not None:
        data_partition, module_partition = partition
        return output_dir / (
            f"pairwise_scores_data_partition{data_partition}_module_partition{module_partition}.safetensors"
        )
    return output_dir / "pairwise_scores.safetensors"


def pairwise_scores_exist(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> bool:
    """Checks if the pairwise scores exist at the specified path."""
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return save_path.exists()


def save_pairwise_scores(
    output_dir: Path,
    scores: SCORE_TYPE,
    partition: Optional[PARTITION_TYPE] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves pairwise influence scores to disk."""
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    save_file(tensors=scores, filename=save_path, metadata=metadata)


def load_pairwise_scores(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> Dict[str, torch.Tensor]:
    """Loads pairwise scores from disk."""
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return load_file(filename=save_path)


def _compute_dot_products_with_loader(
    model: nn.Module,
    task: Task,
    state: State,
    train_loader: data.DataLoader,
    factor_args: FactorArguments,
    score_args: ScoreArguments,
    tracked_module_names: List[str],
    scaler: GradScaler,
    disable_tqdm: bool = False,
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """After computing the preconditioned query gradient, compute dot products with individual training gradients."""
    with torch.no_grad():
        model.zero_grad(set_to_none=True)
        set_mode(
            model=model,
            mode=ModuleMode.PAIRWISE_SCORE,
            tracked_module_names=tracked_module_names,
            keep_factors=True,
        )
        release_memory()

    dataset_size = len(train_loader.dataset)
    score_chunks: Dict[str, List[torch.Tensor]] = {}
    if score_args.compute_per_module_scores:
        for module in model.modules():
            if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                score_chunks[module.name] = []
    else:
        score_chunks[ALL_MODULE_NAME] = []

    total_steps = 0
    enable_amp = score_args.amp_dtype is not None

    with tqdm(
        total=len(train_loader),
        desc="Computing pairwise scores (training gradient)",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for batch in train_loader:
            batch = send_to_device(tensor=batch, device=state.device)

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
                finalize_pairwise_scores(model=model, tracked_module_names=tracked_module_names)

            with torch.no_grad():
                if score_args.compute_per_module_scores:
                    for module in model.modules():
                        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                            score_chunks[module.name].append(
                                module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME).clone().cpu()
                            )
                else:
                    pairwise_scores = None
                    for module in model.modules():
                        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                            if pairwise_scores is None:
                                pairwise_scores = torch.zeros_like(
                                    module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME), requires_grad=False
                                )
                            try:
                                pairwise_scores.add_(module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME))
                            except RuntimeError:
                                if score_args.compute_per_token_scores:
                                    raise RuntimeError(
                                        "The model does not support token-wise score computation. "
                                        "Set `compute_per_module_scores=True` or `compute_per_token_scores=False` "
                                        "to avoid this error."
                                    )
                                raise
                    score_chunks[ALL_MODULE_NAME].append(pairwise_scores.cpu())
                release_scores(model=model)

            if state.use_distributed and total_steps % score_args.distributed_sync_interval == 0:
                # Periodically synchronizes all processes to avoid timeout at the final synchronization.
                state.wait_for_everyone()

            total_steps += 1
            pbar.update(1)

    with torch.no_grad():
        model.zero_grad(set_to_none=True)
        set_mode(
            model=model,
            mode=ModuleMode.PRECONDITION_GRADIENT,
            tracked_module_names=tracked_module_names,
            keep_factors=True,
        )
        release_preconditioned_gradient(model=model)
        release_memory()

        total_scores: SCORE_TYPE = {}
        for module_name, chunks in score_chunks.items():
            total_scores[module_name] = torch.cat(chunks, dim=1)
            if state.use_distributed:
                total_scores[module_name] = total_scores[module_name].to(device=state.device)
                gather_list = None
                if state.is_main_process:
                    gather_list = [torch.zeros_like(total_scores[module_name]) for _ in range(state.num_processes)]
                torch.distributed.gather(total_scores[module_name], gather_list)
                if state.is_main_process:
                    total_scores[module_name] = torch.cat(gather_list, dim=1)[:, :dataset_size].cpu()
        state.wait_for_everyone()

    return total_scores


def _compute_aggregated_dot_products_with_loader(
    model: nn.Module,
    task: Task,
    state: State,
    train_loader: data.DataLoader,
    factor_args: FactorArguments,
    score_args: ScoreArguments,
    tracked_module_names: List[str],
    scaler: GradScaler,
    disable_tqdm: bool = False,
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """After computing the preconditioned query gradient, compute dot products with aggregated training gradients."""
    with torch.no_grad():
        model.zero_grad(set_to_none=True)
        set_mode(
            model=model,
            mode=ModuleMode.GRADIENT_AGGREGATION,
            tracked_module_names=tracked_module_names,
            keep_factors=True,
        )
        release_memory()

    scores: Dict[str, Optional[torch.Tensor]] = {}
    if score_args.compute_per_module_scores:
        for module in model.modules():
            if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                scores[module.name] = None
    else:
        scores[ALL_MODULE_NAME] = None

    enable_amp = score_args.amp_dtype is not None

    if not aggregated_gradient_exist(model=model, tracked_module_names=tracked_module_names):
        release_aggregated_gradient(model=model)
        with tqdm(
            total=len(train_loader),
            desc="Computing pairwise scores (training gradient)",
            bar_format=TQDM_BAR_FORMAT,
            disable=not state.is_main_process or disable_tqdm,
        ) as pbar:
            for batch in train_loader:
                batch = send_to_device(tensor=batch, device=state.device)

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
                    finalize_gradient_aggregation(model=model, tracked_module_names=tracked_module_names)

                pbar.update(1)

        with torch.no_grad():
            if state.use_distributed:
                synchronize_aggregated_gradient(model=model, tracked_module_names=tracked_module_names)

    compute_pairwise_scores_from_aggregation(model=model, tracked_module_names=tracked_module_names)

    with torch.no_grad():
        if score_args.compute_per_module_scores:
            for module in model.modules():
                if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                    scores[module.name] = module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME).clone().cpu()
        else:
            pairwise_scores = None
            for module in model.modules():
                if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                    if pairwise_scores is None:
                        pairwise_scores = torch.zeros_like(
                            module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME), requires_grad=False
                        )
                    pairwise_scores.add_(module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME))
            scores[ALL_MODULE_NAME] = pairwise_scores.cpu()
        release_scores(model=model)

        model.zero_grad(set_to_none=True)
        set_mode(
            model=model,
            mode=ModuleMode.PRECONDITION_GRADIENT,
            tracked_module_names=tracked_module_names,
            keep_factors=True,
        )
        release_preconditioned_gradient(model=model)
        release_memory()

    return scores


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
            The factor results to load from, before computing the pairwise scores.
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
            Arguments related to computing pairwise scores.
        factor_args (FactorArguments):
            Arguments related to computing preconditioning factors.
        tracked_module_names (List[str], optional):
            A list of module names that pairwise scores will be computed. If not specified, scores
            will be computed for all available tracked modules.
        disable_tqdm (bool, optional):
            Disables TQDM progress bars. Defaults to False.

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary containing the module name and its pairwise influence scores.
    """
    with torch.no_grad():
        update_factor_args(model=model, factor_args=factor_args)
        update_score_args(model=model, score_args=score_args)
        if tracked_module_names is None:
            tracked_module_names = get_tracked_module_names(model=model)
        set_mode(
            model=model,
            mode=ModuleMode.PRECONDITION_GRADIENT,
            tracked_module_names=tracked_module_names,
            keep_factors=False,
        )
        # Loads necessary factors before computing pairwise influence scores.
        if len(loaded_factors) > 0:
            for name in loaded_factors:
                set_factors(
                    model=model,
                    factor_name=name,
                    factors=loaded_factors[name],
                )

    total_scores_chunks: Dict[str, Union[List[torch.Tensor], torch.Tensor]] = {}
    total_query_batch_size = per_device_query_batch_size * state.num_processes
    query_remainder = len(query_loader.dataset) % total_query_batch_size

    num_batches = len(query_loader)
    query_iter = iter(query_loader)
    num_accumulations = 0
    enable_amp = score_args.amp_dtype is not None
    scaler = GradScaler(enabled=enable_amp)
    if enable_amp:
        gradient_scale = 1.0 / scaler.get_scale()
        set_gradient_scale(model=model, gradient_scale=gradient_scale)

    dot_product_func = (
        _compute_aggregated_dot_products_with_loader
        if score_args.aggregate_train_gradients
        else _compute_dot_products_with_loader
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
                finalize_preconditioned_gradient(model=model, tracked_module_names=tracked_module_names)

            if state.use_distributed:
                # Stacks preconditioned query gradient across multiple devices or nodes.
                synchronize_preconditioned_gradient(
                    model=model, tracked_module_names=tracked_module_names, num_processes=state.num_processes
                )
                if query_index == len(query_loader) - 1 and query_remainder > 0:
                    # Removes duplicate data points if the dataset is not exactly divisible
                    # by the current batch size.
                    truncate_preconditioned_gradient(
                        model=model, tracked_module_names=tracked_module_names, keep_size=query_remainder
                    )

            accumulate_preconditioned_gradient(model=model, tracked_module_names=tracked_module_names)
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

            with torch.no_grad():
                if state.is_main_process:
                    for module_name, current_scores in scores.items():
                        if module_name not in total_scores_chunks:
                            total_scores_chunks[module_name] = []
                        total_scores_chunks[module_name].append(current_scores)
                state.wait_for_everyone()

            num_accumulations = 0
            pbar.update(1)

    with torch.no_grad():
        if state.is_main_process:
            for module_name in total_scores_chunks:
                total_scores_chunks[module_name] = torch.cat(total_scores_chunks[module_name], dim=0)

        # Clean up the memory.
        model.zero_grad(set_to_none=True)
        if enable_amp:
            set_gradient_scale(model=model, gradient_scale=1.0)
        release_aggregated_gradient(model=model)
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)
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
    """Computes pairwise influence scores (with query gradients aggregated) for a given model and task."""
    del per_device_query_batch_size
    with torch.no_grad():
        update_factor_args(model=model, factor_args=factor_args)
        update_score_args(model=model, score_args=score_args)
        if tracked_module_names is None:
            tracked_module_names = get_tracked_module_names(model=model)
        set_mode(
            model=model,
            mode=ModuleMode.GRADIENT_AGGREGATION,
            tracked_module_names=tracked_module_names,
            keep_factors=False,
        )
        # Loads necessary factors before computing pairwise influence scores.
        if len(loaded_factors) > 0:
            for name in loaded_factors:
                set_factors(
                    model=model,
                    factor_name=name,
                    factors=loaded_factors[name],
                )

    enable_amp = score_args.amp_dtype is not None
    scaler = GradScaler(enabled=enable_amp)
    if enable_amp:
        gradient_scale = 1.0 / scaler.get_scale()
        set_gradient_scale(model=model, gradient_scale=gradient_scale)

    dot_product_func = (
        _compute_aggregated_dot_products_with_loader
        if score_args.aggregate_train_gradients
        else _compute_dot_products_with_loader
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
                finalize_gradient_aggregation(model=model, tracked_module_names=tracked_module_names)

            pbar.update(1)

    with torch.no_grad():
        if state.use_distributed:
            synchronize_aggregated_gradient(model=model, tracked_module_names=tracked_module_names)

    compute_preconditioned_gradient_from_aggregation(model=model, tracked_module_names=tracked_module_names)
    accumulate_preconditioned_gradient(model=model, tracked_module_names=tracked_module_names)
    release_aggregated_gradient(model=model)

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

    with torch.no_grad():
        # Clean up the memory.
        model.zero_grad(set_to_none=True)
        if enable_amp:
            set_gradient_scale(model=model, gradient_scale=1.0)
        release_aggregated_gradient(model=model)
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)
        state.wait_for_everyone()

    return scores
