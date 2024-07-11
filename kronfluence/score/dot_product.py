from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.module import TrackedModule
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import (
    accumulate_iterations,
    exist_for_all_modules,
    finalize_all_iterations,
    finalize_iteration,
    set_mode,
    synchronize_modules,
)
from kronfluence.task import Task
from kronfluence.utils.constants import (
    ALL_MODULE_NAME,
    DISTRIBUTED_SYNC_INTERVAL,
    PAIRWISE_SCORE_MATRIX_NAME,
    SCORE_TYPE,
)
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync, release_memory

DIMENSION_NOT_MATCH_ERROR_MSG = (
    "The model does not support token-wise score computation. "
    "Set `compute_per_module_scores=True` or `compute_per_token_scores=False` "
    "to avoid this error."
)


def compute_dot_products_with_loader(
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
    model.zero_grad(set_to_none=True)
    set_mode(
        model=model,
        mode=ModuleMode.PAIRWISE_SCORE,
        tracked_module_names=tracked_module_names,
        release_memory=False,
    )
    release_memory()

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
                finalize_iteration(model=model, tracked_module_names=tracked_module_names)

            with torch.no_grad():
                if score_args.compute_per_module_scores:
                    for module in cached_module_lst:
                        score_chunks[module.name].append(
                            module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME).to(device="cpu", copy=True)
                        )
                else:
                    pairwise_scores = None
                    for module in cached_module_lst:
                        if pairwise_scores is None:
                            pairwise_scores = torch.zeros_like(
                                module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME), requires_grad=False
                            )
                        try:
                            pairwise_scores.add_(module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME))
                        except RuntimeError as exc:
                            if score_args.compute_per_token_scores:
                                raise RuntimeError(DIMENSION_NOT_MATCH_ERROR_MSG) from exc
                            raise
                    pairwise_scores = pairwise_scores.cpu()
                    score_chunks[ALL_MODULE_NAME].append(pairwise_scores)
                    accumulate_iterations(model=model, tracked_module_names=tracked_module_names)

            if state.use_distributed and total_steps % DISTRIBUTED_SYNC_INTERVAL == 0:
                state.wait_for_everyone()

            del loss
            total_steps += 1
            pbar.update(1)

    model.zero_grad(set_to_none=True)
    finalize_all_iterations(model=model, tracked_module_names=tracked_module_names)
    set_mode(
        model=model,
        mode=ModuleMode.PRECONDITION_GRADIENT,
        tracked_module_names=tracked_module_names,
        release_memory=False,
    )
    release_memory()

    total_scores: SCORE_TYPE = {}
    for module_name, chunks in score_chunks.items():
        total_scores[module_name] = torch.cat(chunks, dim=1)
        if state.use_distributed:
            total_scores[module_name] = total_scores[module_name].to(device=state.device)
            gather_list = None
            if state.is_main_process:
                gather_list = [torch.zeros_like(total_scores[module_name]) for _ in range(state.num_processes)]
            dist.gather(total_scores[module_name], gather_list)
            if state.is_main_process:
                total_scores[module_name] = torch.cat(gather_list, dim=1)[:, :dataset_size].cpu()
            else:
                total_scores[module_name] = total_scores[module_name].cpu()
    state.wait_for_everyone()

    return total_scores


def compute_aggregated_dot_products_with_loader(
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
    model.zero_grad(set_to_none=True)
    set_mode(
        model=model,
        mode=ModuleMode.GRADIENT_AGGREGATION,
        tracked_module_names=tracked_module_names,
        release_memory=False,
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

    if not exist_for_all_modules(model=model, tracked_module_names=tracked_module_names):
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
                    finalize_iteration(model=model, tracked_module_names=tracked_module_names)

                del loss
                pbar.update(1)

        if state.use_distributed:
            synchronize_modules(model=model, tracked_module_names=tracked_module_names)

    set_mode(
        model=model,
        mode=ModuleMode.PAIRWISE_SCORE,
        tracked_module_names=tracked_module_names,
        release_memory=False,
    )
    finalize_all_iterations(model=model, tracked_module_names=tracked_module_names)

    with torch.no_grad():
        if score_args.compute_per_module_scores:
            for module in model.modules():
                if isinstance(module, TrackedModule) and module.name in tracked_module_names:
                    scores[module.name] = module.get_factor(factor_name=PAIRWISE_SCORE_MATRIX_NAME).to(
                        device="cpu", copy=True
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
                    except RuntimeError as exc:
                        if score_args.compute_per_token_scores:
                            raise RuntimeError(DIMENSION_NOT_MATCH_ERROR_MSG) from exc
                        raise
            scores[ALL_MODULE_NAME] = pairwise_scores.cpu()
        accumulate_iterations(model=model, tracked_module_names=tracked_module_names)

    model.zero_grad(set_to_none=True)
    set_mode(
        model=model,
        mode=ModuleMode.PRECONDITION_GRADIENT,
        tracked_module_names=tracked_module_names,
        release_memory=False,
    )
    release_memory()

    return scores
