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
    """Retrieves the parent module and its name given the name of the current module.

    Args:
        model (nn.Module):
            The PyTorch model to inspect.
        key (str):
            The full name of the current module.

    Returns:
        Tuple[nn.Module, str]:
            The parent module and the name of the target module.
    """
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
            Arguments related to computing influence factors.
        score_args (ScoreArguments, optional):
            Arguments related to computing influence scores.

    Returns:
        nn.Module:
            The processed model with `TrackedModule` installed.
    """
    if isinstance(model, (DP, DDP, FSDP)):
        raise ValueError(
            "The model is wrapped with DataParallel, DistributedDataParallel "
            "or FullyShardedDataParallel. Call `prepare_model` before wrapping the model."
        )

    tracked_module_names = task.get_influence_tracked_modules() if task is not None else None
    tracked_module_exists_dict = None
    if tracked_module_names is not None:
        tracked_module_exists_dict = {name: False for name in tracked_module_names}
    per_sample_gradient_process_fnc = None
    if task is not None and task.enable_post_process_per_sample_gradient:
        per_sample_gradient_process_fnc = task.post_process_per_sample_gradient

    named_modules = model.named_modules()
    for module_name, module in named_modules:
        if len(list(module.children())) > 0:
            continue

        # Filters modules based on the task's `get_influence_tracked_modules` if specified.
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

            if tracked_module_exists_dict is not None:
                tracked_module_exists_dict[module_name] = True

    if tracked_module_exists_dict is not None and not all(list(tracked_module_exists_dict.values())):
        error_msg = (
            f"Some provided tracked modules were not found. The current mapping: `{tracked_module_exists_dict}`."
        )
        raise IllegalTaskConfigurationError(error_msg)

    if not any(isinstance(module, TrackedModule) for module in model.modules()):
        supported_modules = ", ".join(module.__name__ for module in TrackedModule.SUPPORTED_MODULES)
        raise IllegalTaskConfigurationError(
            f"No supported modules found. Kronfluence supports: {supported_modules}. "
            "Consider rewriting your model or subclassing `TrackedModule` for custom layers.\n"
            f"Current Model:\n{model}"
        )
    return model


def make_modules_partition(total_module_names: List[str], partition_size: int) -> List[List[str]]:
    """Divides a list of module names into smaller partitions of a specified size.

    Args:
        total_module_names (List[str]):
            The list of all module names.
        partition_size (int):
            The number of partitions to create.

    Returns:
        List[List[str]]:
            A list of partitioned module names.

    Raises:
        ValueError: If `len(total_module_names)` is less than `partition_size`.
    """
    if len(total_module_names) < partition_size:
        raise ValueError("The total modules must be equal to or greater than the partition size.")
    # See https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length.
    div, mod = divmod(len(total_module_names), partition_size)
    return list(
        total_module_names[i * div + min(i, mod) : (i + 1) * div + min(i + 1, mod)] for i in range(partition_size)
    )


def set_mode(
    model: nn.Module,
    mode: ModuleMode,
    tracked_module_names: List[str] = None,
    release_memory: bool = False,
) -> None:
    """Sets the module mode of specified `TrackedModule` instances within a model.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        mode (ModuleMode):
            The new mode to set for `TrackedModule`.
        tracked_module_names (List[str], optional):
             Names of modules to update. If `None`, updates all.
        release_memory (bool, optional):
            If `True`, releases memory of existing factors.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule):
            if tracked_module_names is not None and module.name not in tracked_module_names:
                continue
            module.set_mode(mode=mode, release_memory=release_memory)


def update_factor_args(model: nn.Module, factor_args: FactorArguments) -> None:
    """Updates the factor arguments for all `TrackedModule` instances within a model.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        factor_args (FactorArguments):
            The new factor arguments to set.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.update_factor_args(factor_args=factor_args)


def update_score_args(model: nn.Module, score_args: ScoreArguments) -> None:
    """Updates the score arguments for all `TrackedModule` instances within a model.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        score_args (ScoreArguments):
            The new score arguments to set.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.update_score_args(score_args=score_args)


def get_tracked_module_names(model: nn.Module) -> List[str]:
    """Returns the names of `TrackedModule` instances within a model.

    Args:
        model (nn.Module):
            The PyTorch model to inspect.

    Returns:
        List[str]:
            A list of names of `TrackedModule` instances.
    """
    return [module.name for module in model.modules() if isinstance(module, TrackedModule)]


def load_factors(
    model: nn.Module,
    factor_name: str,
    tracked_module_names: List[str] = None,
    cpu: bool = True,
) -> Dict[str, torch.Tensor]:
    """Loads factors with the given name from specified `TrackedModule` instances.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        factor_name (str):
            The name of the factor to load.
        tracked_module_names (Optional[List[str]]):
            Names of modules to load from. If `None`, loads from all.
        cpu (bool):
            If `True`, moves factors to CPU and releases GPU memory.

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary of loaded factors, keyed by module name.
    """
    loaded_factors = {}
    for module in model.modules():
        if isinstance(module, TrackedModule):
            if tracked_module_names is not None and module.name not in tracked_module_names:
                continue
            factor = module.get_factor(factor_name=factor_name)
            if factor is not None:
                if cpu:
                    loaded_factors[module.name] = factor.to(device="cpu", copy=True)
                    module.release_factor(factor_name=factor_name)
                else:
                    loaded_factors[module.name] = factor
    return loaded_factors


def set_factors(model: nn.Module, factor_name: str, factors: Dict[str, torch.Tensor], clone: bool = False) -> None:
    """Sets new factors for all `TrackedModule` instances within a model.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        factor_name (str):
            The name of the factor to set.
        factors (Dict[str, torch.Tensor]):
            A dictionary of factors to set, keyed by module name.
        clone (bool):
            If `True`, clones the factors before setting.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.set_factor(
                factor_name=factor_name, factor=factors[module.name].clone() if clone else factors[module.name]
            )


def set_attention_mask(
    model: nn.Module,
    attention_mask: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
) -> None:
    """Sets the attention mask for all `TrackedModule` instances within a model.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        attention_mask (Optional[Union[Dict[str, torch.Tensor], torch.Tensor]]):
            The attention mask to set. Can be a dictionary (keyed by module name) or a single tensor.

    Raises:
        RuntimeError:
            If an invalid attention mask is provided.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule):
            if isinstance(attention_mask, dict):
                if module.name in attention_mask:
                    module.set_attention_mask(attention_mask=attention_mask[module.name])
                else:
                    module.set_attention_mask(attention_mask=None)
            elif isinstance(attention_mask, torch.Tensor):
                module.set_attention_mask(attention_mask=attention_mask)
            elif attention_mask is None:
                module.set_attention_mask(attention_mask=None)
            else:
                raise RuntimeError(f"Invalid attention mask `{attention_mask}` provided.")


def set_gradient_scale(
    model: nn.Module,
    gradient_scale: float = 1.0,
) -> None:
    """Sets the gradient scale for all `TrackedModule` instances within a model.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        gradient_scale (float):
            The gradient scale to set.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule):
            module.set_gradient_scale(scale=gradient_scale)


def prepare_modules(model: nn.Module, tracked_module_names: List[str], device: torch.device) -> None:
    """Prepares specified `TrackedModule` instances for score computation.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        tracked_module_names (List[str]):
            Names of modules to prepare.
        device (torch.device):
            The device to prepare the modules for.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.prepare_storage(device=device)


def synchronize_modules(model: nn.Module, tracked_module_names: List[str], num_processes: int = 1) -> None:
    """Synchronizes specified `TrackedModule` instances across processes.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        tracked_module_names (List[str]):
            Names of modules to synchronize.
        num_processes (int):
            The number of processes to synchronize across.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.synchronize(num_processes=num_processes)


def truncate(model: nn.Module, tracked_module_names: List[str], keep_size: int) -> None:
    """Truncates the data in specified `TrackedModule` instances.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        tracked_module_names (List[str]):
            Names of modules to truncate.
        keep_size (int):
            The number of elements to keep after truncation.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.truncate(keep_size=keep_size)


def exist_for_all_modules(model: nn.Module, tracked_module_names: List[str]) -> bool:
    """Checks if all specified `TrackedModule` instances have existing factor.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        tracked_module_names (List[str]):
            Names of modules to check.

    Returns:
        bool:
            `True` if all specified modules have existing factor, `False` otherwise.
    """
    return all(
        module.exist()
        for module in model.modules()
        if isinstance(module, TrackedModule) and module.name in tracked_module_names
    )


def accumulate_iterations(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Accumulates iterations for specified `TrackedModule` instances.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        tracked_module_names (List[str]):
            Names of modules to accumulate iterations for.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.accumulate_iterations()


def finalize_iteration(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Finalizes the current iteration for specified `TrackedModule` instances.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        tracked_module_names (List[str]):
            Names of modules to finalize iteration for.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.finalize_iteration()


def finalize_all_iterations(model: nn.Module, tracked_module_names: List[str]) -> None:
    """Finalizes all iterations for specified `TrackedModule` instances.

    Args:
        model (nn.Module):
            The PyTorch model containing `TrackedModule` instances.
        tracked_module_names (List[str]):
            Names of modules to finalize all iterations for.
    """
    for module in model.modules():
        if isinstance(module, TrackedModule) and module.name in tracked_module_names:
            module.finalize_all_iterations()
