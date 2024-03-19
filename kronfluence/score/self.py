from pathlib import Path
from typing import Dict, List, Optional

import torch
from accelerate.utils import find_batch_size, send_to_device
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.module import TrackedModule
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    get_tracked_module_names,
    release_scores,
    set_factors,
    set_mode,
    update_factor_args,
    update_score_args,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
    ALL_MODULE_NAME,
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
    """Generates the path for saving/loading self-influence scores."""
    if partition is not None:
        data_partition, module_partition = partition
        return output_dir / (
            f"self_scores_data_partition{data_partition}_module_partition{module_partition}.safetensors"
        )
    return output_dir / "self_scores.safetensors"


def self_scores_exist(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> bool:
    """Check if the self-influence scores exist at specified path."""
    save_path = self_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return save_path.exists()


def save_self_scores(
    output_dir: Path,
    scores: SCORE_TYPE,
    partition: Optional[PARTITION_TYPE] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Saves self-influence scores to disk."""
    save_path = self_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    save_file(tensors=scores, filename=save_path, metadata=metadata)


def load_self_scores(
    output_dir: Path,
    partition: Optional[PARTITION_TYPE] = None,
) -> Dict[str, torch.Tensor]:
    """Loads self-influence scores from disk."""
    save_path = self_scores_save_path(
        output_dir=output_dir,
        partition=partition,
    )
    return load_file(filename=save_path)


def compute_self_scores_with_loaders(
    loaded_factors: FACTOR_TYPE,
    model: nn.Module,
    state: State,
    task: Task,
    train_loader: data.DataLoader,
    score_args: ScoreArguments,
    factor_args: FactorArguments,
    tracked_module_names: Optional[List[str]],
) -> Dict[str, torch.Tensor]:
    """Computes self-influence scores for a given model and task.

    Args:
        loaded_factors (FACTOR_TYPE):
            The factor results to load from, before computing the self-influence scores.
        model (nn.Module):
            The model that self-influence scores will be computed.
        state (State):
            The current process's information (e.g., device being used).
        task (Task):
            The specific task associated with the model.
        train_loader (data.DataLoader):
            The data loader that will be used to compute training gradient.
        score_args (ScoreArguments):
            Arguments related to computing self-influence scores.
        factor_args (FactorArguments):
            Arguments related to computing preconditioning factors.
        tracked_module_names (List[str], optional):
            A list of module names that self-influence scores will be computed. If not specified, scores
            will be computed for all available tracked modules.

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary containing the module name and its self-influence scores. If
            `score_args.per_module_score` is False, the self-influence scores will be aggregated across all modules.
    """
    with torch.no_grad():
        update_factor_args(model=model, factor_args=factor_args)
        update_score_args(model=model, score_args=score_args)
        set_mode(
            model=model,
            mode=ModuleMode.DEFAULT,
            keep_factors=False,
        )

        # Loads necessary factors before computing self-influence scores.
        if len(loaded_factors) > 0:
            for name in loaded_factors:
                set_factors(
                    model=model,
                    factor_name=name,
                    factors=loaded_factors[name],
                )
        if tracked_module_names is None:
            tracked_module_names = get_tracked_module_names(model=model)
        set_mode(
            model=model,
            mode=ModuleMode.SELF_SCORE,
            tracked_module_names=tracked_module_names,
            keep_factors=True,
        )
    dataset_size = len(train_loader.dataset)
    score_chunks: Dict[str, List[torch.Tensor]] = {}
    if score_args.per_module_score:
        for module in model.modules():
            if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                score_chunks[module.name] = []
    else:
        score_chunks[ALL_MODULE_NAME] = []
    total_steps = 0

    with tqdm(
        total=len(train_loader),
        desc="Computing self-influence scores",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process,
    ) as pbar:
        for batch in train_loader:
            batch = send_to_device(
                tensor=batch,
                device=state.device,
            )

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                loss = task.compute_train_loss(
                    batch=batch,
                    model=model,
                    sample=False,
                )
                loss.backward()
            total_steps += 1

            with torch.no_grad():
                if score_args.per_module_score:
                    for module in model.modules():
                        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                            score_chunks[module.name].append(
                                module.get_factor(factor_name=SELF_SCORE_VECTOR_NAME).cpu()
                            )
                else:
                    # Aggregate the self-influence scores across all modules.
                    batch_size = find_batch_size(batch)
                    self_scores = torch.zeros(
                        size=(batch_size,),
                        dtype=score_args.score_dtype,
                        device=state.device,
                        requires_grad=False,
                    )
                    for module in model.modules():
                        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                            self_scores.add_(module.get_factor(factor_name=SELF_SCORE_VECTOR_NAME))
                    score_chunks[ALL_MODULE_NAME].append(self_scores.cpu())
                release_scores(model=model)

            if state.use_distributed and total_steps % score_args.distributed_sync_steps == 0:
                state.wait_for_everyone()

            pbar.update(1)

    with torch.no_grad():
        set_mode(
            model=model,
            mode=ModuleMode.DEFAULT,
            tracked_module_names=tracked_module_names,
            keep_factors=False,
        )
        total_scores: SCORE_TYPE = {}
        for module_name, chunks in score_chunks.items():
            total_scores[module_name] = torch.cat(chunks, dim=0)
            if state.use_distributed:
                release_memory()
                total_scores[module_name] = total_scores[module_name].to(device=state.device)
                gather_list = None
                if state.is_main_process:
                    gather_list = [torch.zeros_like(total_scores[module_name]) for _ in range(state.num_processes)]
                torch.distributed.gather(total_scores[module_name], gather_list)
                if state.is_main_process:
                    total_scores[module_name] = torch.cat(gather_list, dim=0)[:dataset_size].cpu()
    state.wait_for_everyone()
    return total_scores
