from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from accelerate.utils import send_to_device
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from torch.utils import data
from tqdm import tqdm
from kronfluence.module import update_score_args
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
from kronfluence.score.dot_product import DIMENSION_NOT_MATCH_ERROR_MSG
from kronfluence.task import Task
from kronfluence.utils.constants import (
    ALL_MODULE_NAME,
    DISTRIBUTED_SYNC_INTERVAL,
    PAIRWISE_SCORE_MATRIX_NAME,
    SCORE_TYPE,
    SQUARED_GRADIENT_NORM_NAME,
)
from kronfluence.utils.logger import TQDM_BAR_FORMAT
from kronfluence.utils.state import State, no_sync, release_memory


def compute_gradient_norms_with_loaders(
    model: nn.Module,
    task: Task,
    state: State,
    score_args: ScoreArguments,
    train_loader: data.DataLoader,
    tracked_module_names: List[str],
    disable_tqdm: bool = False,
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """After computing the preconditioned query gradient, compute dot products with individual training gradients."""
    if score_args.has_shared_parameters:
        raise NotImplementedError("Shared parameters are not supported for gradient norm computation.")

    model.zero_grad(set_to_none=True)
    set_mode(
        model=model,
        mode=ModuleMode.GRADIENT_NORM,
        tracked_module_names=tracked_module_names,
        release_memory=False,
    )
    update_score_args(model=model, score_args=score_args)
    release_memory()

    cached_module_lst: list[TrackedModule] = []
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            cached_module_lst.append(module)

    dataset_size = len(train_loader.dataset)
    module_to_gradient_norm: Dict[str, list[torch.Tensor]] = {}
    if score_args.compute_per_module_scores:
        for module in cached_module_lst:
            module_to_gradient_norm[module.name] = []
    else:
        module_to_gradient_norm[ALL_MODULE_NAME] = []

    total_steps = 0

    with tqdm(
        total=len(train_loader),
        desc="Computing gradient norms (training gradient)",
        bar_format=TQDM_BAR_FORMAT,
        disable=not state.is_main_process or disable_tqdm,
    ) as pbar:
        for batch in train_loader:
            batch = send_to_device(tensor=batch, device=state.device)

            with no_sync(model=model, state=state):
                model.zero_grad(set_to_none=True)
                loss = task.compute_train_loss(
                    batch=batch,
                    model=model,
                    sample=False,
                )

                if score_args.compute_per_module_scores:
                    for module in cached_module_lst:
                        module_to_gradient_norm[module.name].append(
                            torch.sqrt(module.get_factor(factor_name=SQUARED_GRADIENT_NORM_NAME)).to(
                                device="cpu", copy=True
                            )
                        )
                else:
                    squared_gradient_norms = None
                    for module in cached_module_lst:
                        if squared_gradient_norms is None:
                            squared_gradient_norms = torch.zeros_like(
                                module.get_factor(factor_name=SQUARED_GRADIENT_NORM_NAME), requires_grad=False
                            )
                        try:
                            squared_gradient_norms.add_(module.get_factor(factor_name=SQUARED_GRADIENT_NORM_NAME))
                        except RuntimeError as exc:
                            raise RuntimeError(DIMENSION_NOT_MATCH_ERROR_MSG) from exc
                    assert squared_gradient_norms is not None
                    gradient_norm = torch.sqrt(squared_gradient_norms).cpu()
                    module_to_gradient_norm[ALL_MODULE_NAME].append(gradient_norm)
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

    total_gradient_norms: SCORE_TYPE = {}
    for module_name, chunks in module_to_gradient_norm.items():
        total_gradient_norms[module_name] = torch.cat(chunks, dim=0)
        if state.use_distributed:
            total_gradient_norms[module_name] = total_gradient_norms[module_name].to(device=state.device)
            gather_list = None
            if state.is_main_process:
                gather_list = [torch.zeros_like(total_gradient_norms[module_name]) for _ in range(state.num_processes)]
            dist.gather(total_gradient_norms[module_name], gather_list)
            if state.is_main_process:
                total_gradient_norms[module_name] = torch.cat(gather_list, dim=1)[:, :dataset_size].cpu()
            else:
                total_gradient_norms[module_name] = total_gradient_norms[module_name].cpu()
    state.wait_for_everyone()

    return total_gradient_norms
