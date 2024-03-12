from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate.utils import find_batch_size, send_to_device
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.module import TrackedModule
from kronfluence.module.constants import (
    ALL_MODULE_NAME,
    FACTOR_TYPE,
    PAIRWISE_SCORE_MATRIX_NAME,
    SCORE_TYPE,
)
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    get_preconditioned_gradient_batch_size,
    get_tracked_module_names,
    release_scores,
    set_factors,
    set_mode,
    synchronize_preconditioned_gradient,
    truncate_preconditioned_gradient,
    update_factor_args,
    update_score_args,
)
from kronfluence.task import Task
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync, release_memory


def pairwise_scores_save_path(
    output_dir: Path,
    partition: Optional[Tuple[int, int]] = None,
) -> Path:
    """Generates the path for saving/loading pairwise scores."""
    if partition is not None:
        data_partition, module_partition = partition
        return output_dir / (
            f"pairwise_scores_data_partition{data_partition}"
            f"_module_partition{module_partition}.safetensors"
        )
    return output_dir / "pairwise_scores.safetensors"


def pairwise_scores_exist(
    output_dir: Path,
    partition: Optional[Tuple[int, int]] = None,
) -> bool:
    """Check if the pairwise scores exist at specified path."""
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return save_path.exists()


def save_pairwise_scores(
    output_dir: Path,
    scores: SCORE_TYPE,
    partition: Optional[Tuple[int, int]] = None,
) -> None:
    """Saves pairwise influence scores to disk."""
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    save_file(tensors=scores, filename=save_path)


def load_pairwise_scores(
    output_dir: Path,
    partition: Optional[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """Loads pairwise scores from disk."""
    save_path = pairwise_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return load_file(filename=save_path)


def _compute_pairwise_dot_products_with_loader(
    model: nn.Module,
    task: Task,
    state: State,
    train_loader: data.DataLoader,
    score_args: ScoreArguments,
    tracked_module_names: List[str],
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """After computing the preconditioned query gradient, the pairwise dot products with
    training gradients are computed."""

    dataset_size = len(train_loader.dataset)
    score_chunks: Dict[str, List[torch.Tensor]] = {}
    if score_args.per_module_score:
        for module in model.modules():
            if (
                isinstance(module, TrackedModule)
                and module.name in tracked_module_names
            ):
                score_chunks[module.name] = []
    else:
        score_chunks[ALL_MODULE_NAME] = []

    with torch.no_grad():
        total_query_batch_size = get_preconditioned_gradient_batch_size(model=model)
        set_mode(
            model=model,
            mode=ModuleMode.PAIRWISE_SCORE,
            tracked_module_names=tracked_module_names,
            keep_factors=True,
        )

    with tqdm(
        total=len(train_loader),
        desc="Computing dot products on training dataset",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process,
    ) as pbar:
        for batch in train_loader:
            batch = send_to_device(batch, device=state.device)

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                loss = task.compute_train_loss(
                    batch=batch,
                    model=model,
                    sample=False,
                )
                loss.backward()

            with torch.no_grad():
                if score_args.per_module_score:
                    for module in model.modules():
                        if (
                            isinstance(module, TrackedModule)
                            and module.name in tracked_module_names
                        ):
                            score_chunks[module.name].append(
                                module.get_factor(
                                    factor_name=PAIRWISE_SCORE_MATRIX_NAME
                                ).cpu()
                            )
                else:
                    # Aggregate the pairwise scores across all modules.
                    batch_size = find_batch_size(batch)
                    pairwise_scores = torch.zeros(
                        size=(total_query_batch_size, batch_size),
                        dtype=score_args.score_dtype,
                        device=state.device,
                        requires_grad=False,
                    )
                    for module in model.modules():
                        if (
                            isinstance(module, TrackedModule)
                            and module.name in tracked_module_names
                        ):
                            pairwise_scores.add_(
                                module.get_factor(
                                    factor_name=PAIRWISE_SCORE_MATRIX_NAME
                                )
                            )
                    # `.cpu()` synchronizes the CUDA stream.
                    score_chunks[ALL_MODULE_NAME].append(pairwise_scores.cpu())
                release_scores(model=model)

            pbar.update(1)

    with torch.no_grad():
        set_mode(
            model=model,
            mode=ModuleMode.PRECONDITION_GRADIENT,
            tracked_module_names=tracked_module_names,
            keep_factors=False,
        )
        total_scores: SCORE_TYPE = {}
        for module_name, chunks in score_chunks.items():
            total_scores[module_name] = torch.cat(chunks, dim=1)
            if state.use_distributed:
                size = total_scores[module_name].size()
                total_scores[module_name] = total_scores[module_name].to(
                    device=state.device,
                    non_blocking=False,
                )
                torch.cuda.synchronize(state.device)
                total_scores[module_name] = total_scores[module_name].t().contiguous()
                release_memory()
                stacked_scores = torch.empty(
                    size=(size[1] * state.num_processes, size[0]),
                    dtype=total_scores[module_name].dtype,
                    device=state.device,
                    requires_grad=False,
                )
                torch.distributed.all_gather_into_tensor(
                    output_tensor=stacked_scores,
                    input_tensor=total_scores[module_name],
                )
                stacked_scores = stacked_scores.t().contiguous()
                stacked_scores = stacked_scores[:, :dataset_size]
                total_scores[module_name] = stacked_scores.cpu()
                del stacked_scores
    return total_scores


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
) -> Dict[str, torch.Tensor]:
    """Computes pairwise influence scores for a given model and task.

    Args:
        loaded_factors (FACTOR_TYPE, optional):
            The factor results to load from, before computing the pairwise scores.
        model (nn.Module):
            The model that pairwise influence scores will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        query_loader (data.DataLoader):
            The data loader that will be used to compute query gradient.
        per_device_query_batch_size (int):
            Per-device batch size for the query data loader.
        train_loader (data.DataLoader):
            The data loader that will be used to compute training gradient.
        score_args (ScoreArguments):
            Arguments related to computing pairwise scores.
        factor_args (FactorArguments):
            Arguments related to computing preconditioning factors.
        tracked_module_names (List[str], optional):
            A list of module names that pairwise scores will be computed. If not specified, scores
            will be computed for all available tracked modules.

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary containing the module name and its pairwise influence scores. If
            `score_args.per_module_score` is False, the pairwise scores will be aggregated across all modules.
    """
    with torch.no_grad():
        update_factor_args(model=model, factor_args=factor_args)
        update_score_args(model=model, score_args=score_args)
        set_mode(
            model=model,
            mode=ModuleMode.DEFAULT,
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
        if tracked_module_names is None:
            tracked_module_names = get_tracked_module_names(model)

        set_mode(
            model=model,
            mode=ModuleMode.PRECONDITION_GRADIENT,
            tracked_module_names=tracked_module_names,
            keep_factors=True,
        )

    total_scores_chunks: Dict[str, Union[List[torch.Tensor], torch.Tensor]] = {}
    total_query_batch_size = per_device_query_batch_size * state.num_processes
    query_remainder = len(query_loader.dataset) % total_query_batch_size

    with tqdm(
        total=len(query_loader),
        desc="Computing pairwise influence scores",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process,
    ) as pbar:
        for query_index, query_batch in enumerate(query_loader):
            query_batch = send_to_device(
                query_batch,
                device=state.device,
            )

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                measurement = task.compute_measurement(batch=query_batch, model=model)
                measurement.backward()

                if state.use_distributed:
                    # Stack preconditioned query gradient across multiple devices or nodes.
                    synchronize_preconditioned_gradient(
                        model=model, num_processes=state.num_processes
                    )
                    if query_index == len(query_loader) - 1 and query_remainder > 0:
                        # Remove duplicate data points if the dataset is not exactly divisible
                        # by the current batch size.
                        truncate_preconditioned_gradient(
                            model=model, keep_size=query_remainder
                        )

            # Compute the dot product between preconditioning query gradient and all training gradients.
            release_memory()
            scores = _compute_pairwise_dot_products_with_loader(
                model=model,
                state=state,
                task=task,
                train_loader=train_loader,
                score_args=score_args,
                tracked_module_names=tracked_module_names,
            )

            with torch.no_grad():
                for module_name, current_scores in scores.items():
                    if module_name not in total_scores_chunks:
                        total_scores_chunks[module_name] = []
                    total_scores_chunks[module_name].append(current_scores)
            pbar.update(1)

    with torch.no_grad():
        set_mode(model=model, mode=ModuleMode.DEFAULT, keep_factors=False)

        for module_name in total_scores_chunks:
            total_scores_chunks[module_name] = torch.cat(
                total_scores_chunks[module_name], dim=0
            )

    return total_scores_chunks
