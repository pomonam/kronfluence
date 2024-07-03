from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel.data_parallel import DataParallel as DP
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.module.tracked_module import ModuleMode, TrackedModule
from kronfluence.task import Task
from kronfluence.utils.exceptions import IllegalTaskConfigurationError


def _get_submodules(model: nn.Module, key: str) -> Tuple[nn.Module, str]:
    """Returns the parent module and its name given the name of the current module."""
    # The code is modified from: https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py.
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    return parent, target_name


def wrap_tracked_modules(
    model: nn.Module,
    task: Optional[Task] = None,
    factor_args: Optional[FactorArguments] = None,
    score_args: Optional[ScoreArguments] = None,
) -> nn.Module:
    """Inspects all modules within the model and, if supported modules are found, wraps them with `TrackedModule`.

    Args:
        model (nn.Module):
            The PyTorch model to install `TrackedModule`.
        task (Task):
            The specific task associated with the model.
        factor_args (FactorArguments, optional):
            Arguments related to computing the influence factors.
        score_args (ScoreArguments, optional):
            Arguments related to computing the influence scores.

    Returns:
        nn.Module:
            The wrapped Pytorch model with `TrackedModule` installed.
    """
    if isinstance(model, (DP, DDP, FSDP)):
        raise ValueError(
            "The model is wrapped with DataParallel, DistributedDataParallel "
            "or FullyShardedDataParallel. Call `prepare_model` before wrapping the model."
        )

    tracked_module_count = 0
    tracked_module_names = task.tracked_modules() if task is not None else None
    tracked_module_exists_dict = None
    if tracked_module_names is not None:
        tracked_module_exists_dict = {name: False for name in tracked_module_names}
    per_sample_gradient_process_fnc = None
    if task is not None and task.do_post_process_per_sample_gradient:
        per_sample_gradient_process_fnc = task.post_process_per_sample_gradient

    named_modules = model.named_modules()
    for module_name, module in named_modules:
        if len(list(module.children())) > 0:
            continue

        # Filters modules based on the task's `tracked_modules` if specified.
        if tracked_module_names is not None and module_name not in tracked_module_names:
            continue

        # Wraps the module if it is currently supported (e.g., nn.Linear & nn.Conv2d).
        if isinstance(module, tuple(TrackedModule.SUPPORTED_MODULES)):
            tracked_module = TrackedModule.SUPPORTED_MODULES[type(module)](
                name=module_name,
                original_module=module,
                per_sample_gradient_process_fnc=per_sample_gradient_process_fnc,
                factor_args=factor_args,
                score_args=score_args,
            )
            parent, target_name = _get_submodules(model=model, key=module_name)
            setattr(parent, target_name, tracked_module)
            tracked_module_count += 1

            if tracked_module_exists_dict is not None:
                tracked_module_exists_dict[module_name] = True

    if tracked_module_exists_dict is not None and not all(list(tracked_module_exists_dict.values())):
        error_msg = (
            f"Some provided tracked modules were not found. The current mapping: `{tracked_module_exists_dict}`."
        )
        raise IllegalTaskConfigurationError(error_msg)

    if tracked_module_count == 0:
        supported_modules_names = [module.__name__ for module in TrackedModule.SUPPORTED_MODULES]
        error_msg = (
            f"Kronfluence currently supports following PyTorch modules: `{supported_modules_names}`. "
            f"However, these modules were not found in the provided model. If you want to analyze "
            "custom layers, consider rewriting your model to use the supported modules, "
            "or define your own custom module by subclassing `TrackedModule`."
        )
        error_msg += f"\nCurrent Model:\n{model}"
        raise IllegalTaskConfigurationError(error_msg)

    return model


def make_modules_partition(total_module_names: List[str], partition_size: int) -> List[List[str]]:
    """Divides a list of module names into smaller partitions of a specified size."""
    # See https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length.
    div, mod = divmod(len(total_module_names), partition_size)
    return list(
        total_module_names[i * div + min(i, mod) : (i + 1) * div + min(i + 1, mod)] for i in range(partition_size)
    )


def set_mode(
    model: nn.Module,
    mode: ModuleMode,
    tracked_module_names: List[str] = None,
    keep_factors: bool = False,
) -> None:
    """Sets the module mode of all `TrackedModule` instances within a model. For example, to compute
    and update covariance matrices, the module mode needs to be set to `ModuleMode.COVARIANCE`. If
    `tracked_module_names` are provided, the module mode is only set for modules listed in `tracked_module_names`.

    Args:
        model (nn.Module):
            The PyTorch model which contains `TrackedModule`.
        mode (ModuleMode):
            The new mode to set for `TrackedModule`.
        tracked_module_names (List[str], optional):
            The list of names for `TrackedModule` to set the new mode. If not provided, the new mode is
            set for all available `TrackedModule` within the model.
        keep_factors (bool, optional):
            If True, existing factors are kept in memory. Defaults to False.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule):
            if tracked_module_names is not None and module.name not in tracked_module_names:
                continue
            module.set_mode(mode=mode, keep_factors=keep_factors)


def update_factor_args(model: nn.Module, factor_args: FactorArguments) -> None:
    """Updates the factor arguments for all `TrackedModule` instances within a model."""
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.update_factor_args(factor_args=factor_args)


def update_score_args(model: nn.Module, score_args: ScoreArguments) -> None:
    """Updates the score arguments for all `TrackedModule` instances within a model."""
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.update_score_args(score_args=score_args)


def get_tracked_module_names(model: nn.Module) -> List[str]:
    """Returns the names of `TrackedModule` instances within a model."""
    tracked_modules = []
    for module in model.modules():
        if isinstance(module, TrackedModule):
            tracked_modules.append(module.name)
    return tracked_modules


def load_factors(
    model: nn.Module,
    factor_name: str,
    tracked_module_names: List[str] = None,
    clone: bool = False,
) -> Dict[str, torch.Tensor]:
    """Loads factors with the given name from all `TrackedModule` instances within a model (or all modules listed
    in `tracked_module_names` if not `None`)."""
    loaded_factors = {}
    for module in model.modules():
        if isinstance(module, TrackedModule):
            if tracked_module_names is not None and module.name not in tracked_module_names:
                continue
            factor = module.get_factor(factor_name=factor_name)
            if factor is not None:
                loaded_factors[module.name] = factor.contiguous().clone() if clone else factor
    return loaded_factors


def set_factors(model: nn.Module, factor_name: str, factors: Dict[str, torch.Tensor]) -> None:
    """Sets new factor for all `TrackedModule` instances within a model."""
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.set_factor(factor_name=factor_name, factor=factors[module.name])


def set_attention_mask(
    model: nn.Module,
    attention_mask: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
) -> None:
    """Sets the attention mask for all `TrackedModule` instances within a model."""
    for module in model.modules():
        if isinstance(module, TrackedModule):
            if isinstance(attention_mask, dict):
                if module.name in attention_mask:
                    module.set_attention_mask(attention_mask=attention_mask[module.name])
                else:
                    module.set_attention_mask(attention_mask=None)
            elif isinstance(attention_mask, torch.Tensor):
                module.set_attention_mask(attention_mask=attention_mask)
            else:
                raise RuntimeError(f"Invalid attention mask `{attention_mask}` provided.")


def set_gradient_scale(
    model: nn.Module,
    gradient_scale: float = 1.0,
) -> None:
    """Sets the gradient scale for all `TrackedModule` instances within a model."""
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.set_gradient_scale(scale=gradient_scale)


def synchronize_covariance_matrices(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Synchronizes covariance matrices for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.synchronize_covariance_matrices()


def finalize_lambda_matrices(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Updates Lambda matrices for all modules listed in `tracked_module_names`."""
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.finalize_lambda_matrix()


def synchronize_lambda_matrices(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Synchronizes Lambda matrices for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.synchronize_lambda_matrices()


def finalize_preconditioned_gradient(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Computes preconditioned gradient for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.finalize_preconditioned_gradient()


def accumulate_preconditioned_gradient(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Accumulates preconditioned gradient for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.accumulate_preconditioned_gradient()


def release_preconditioned_gradient(model: nn.Module) -> None:
    """Releases preconditioned gradient of all `TrackedModule` instances within a model."""
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.release_preconditioned_gradient()


def truncate_preconditioned_gradient(model: nn.Module, tracked_module_names: List[str], keep_size: int) -> None:
    """Truncates preconditioned gradient for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.truncate_preconditioned_gradient(keep_size=keep_size)


def synchronize_preconditioned_gradient(model: nn.Module, tracked_module_names: List[str], num_processes: int) -> None:
    """Synchronizes preconditioned gradient for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.synchronize_preconditioned_gradient(num_processes=num_processes)


def release_scores(model: nn.Module) -> None:
    """Releases scores of all `TrackedModule` instances within a model."""
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.release_scores()


def finalize_pairwise_scores(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Computes pairwise influence scores for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.finalize_pairwise_score()


def finalize_self_scores(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Computes self-influence scores for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.finalize_self_score()


def finalize_self_measurement_scores(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Computes self-influence scores with measurement for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.finalize_self_measurement_score()


def finalize_gradient_aggregation(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Computes aggregated gradient for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.finalize_gradient_aggregation()


def synchronize_aggregated_gradient(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Synchronizes aggregated gradient for all modules listed in `tracked_module_names`."""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.synchronize_aggregated_gradient()


def release_aggregated_gradient(model: nn.Module) -> None:
    """Releases aggregated gradient of all `TrackedModule` instances within a model."""
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.release_aggregated_gradient()


def aggregated_gradient_exist(model: nn.Module, tracked_module_names: List[str]) -> bool:
    """Checks if the aggregated gradient is computed for all modules listed in `tracked_module_names`."""
    exists = True
    for name, module in model.named_modules():
        if (
            isinstance(module, TrackedModule)
            and module.name in tracked_module_names
            and module.aggregated_gradient is None
        ):
            exists = False
    return exists


def compute_preconditioned_gradient_from_aggregation(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Computes preconditioned aggregated gradient for all modules listed in `tracked_module_names`"""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.compute_preconditioned_gradient_from_aggregation()


def compute_pairwise_scores_from_aggregation(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Computes preconditioned aggregated gradient for all modules listed in `tracked_module_names`"""
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.compute_pairwise_scores_from_aggregation()
