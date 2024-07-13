import functools
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.parallel.distributed import DistributedDataParallel


def apply_ddp(
    model: nn.Module,
    local_rank: int,
    rank: int,
    world_size: int,
) -> DistributedDataParallel:
    """Applies DistributedDataParallel (DDP) to the given PyTorch model.

    Args:
        model (nn.Module):
            The PyTorch model to be parallelized.
        local_rank (int):
            The local rank of the current process within its node.
        rank (int):
            The global rank of the current process across all nodes.
        world_size (int):
            The total number of processes in the distributed setup.

    Returns:
        DistributedDataParallel:
            The input model wrapped with DDP.

    Raises:
        RuntimeError:
            If the distributed initialization fails.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    ddp_cfg = {
        "device_ids": [local_rank],
        "output_device": local_rank,
    }

    model = model.to(device=device)
    model = DistributedDataParallel(model, **ddp_cfg)

    return model


def apply_fsdp(
    model: nn.Module,
    local_rank: int,
    rank: int,
    world_size: int,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = True,
    is_transformer: bool = False,
    layer_to_wrap: Optional[nn.Module] = None,
) -> FSDP:
    """Applies FullyShardedDataParallel (FSDP) to the given PyTorch model.

    Args:
        model (nn.Module):
            The PyTorch model to be parallelized.
        local_rank (int):
            The local rank of the current process within its node.
        rank (int):
            The global rank of the current process across all nodes.
        world_size (int):
            The total number of processes in the distributed setup.
        sharding_strategy (str):
            The FSDP sharding strategy to use. Defaults to "FULL_SHARD".
        cpu_offload (bool):
            Whether to offload parameters to CPU. Defaults to `True`.
        is_transformer (bool):
            Whether the model is a transformer. Defaults to `False`.
        layer_to_wrap (nn.Module, optional):
            The specific layer to wrap for transformer models. Required if `is_transformer` is `True`.

    Returns:
        FullyShardedDataParallel:
            The input model wrapped with FSDP.

    Raises:
        ValueError:
            If an invalid sharding strategy is provided or if `layer_to_wrap` is not provided for transformer models.
        RuntimeError:
            If the distributed initialization fails.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if not hasattr(ShardingStrategy, sharding_strategy):
        msg = f"The provided sharding strategy {sharding_strategy} does not exist."
        raise ValueError(msg)
    sharding_strategy = getattr(ShardingStrategy, sharding_strategy)
    if is_transformer:
        if layer_to_wrap is None:
            raise ValueError("`layer_to_wrap` must be provided for transformer models.")
        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_to_wrap},
        )
    else:
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=100,
        )

    fsdp_cfg = {
        "use_orig_params": False,
        "sharding_strategy": sharding_strategy,
        "auto_wrap_policy": my_auto_wrap_policy,
        "cpu_offload": CPUOffload(offload_params=cpu_offload),
    }

    model = model.to(device=device)
    model = FSDP(model, **fsdp_cfg)

    return model
