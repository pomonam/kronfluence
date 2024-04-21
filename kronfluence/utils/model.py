import functools
import torch
import torch.distributed as dist

from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel.distributed import DistributedDataParallel


def apply_ddp(
    model: torch.nn.Module,
    local_rank: int,
    rank: int,
    world_size: int,
) -> DistributedDataParallel:
    """
    Applies DistributedDataParallel (DDP) to the given model.

    Args:
        model (torch.nn.Module): The model to apply DDP to.
        local_rank (int): The local rank of the current process.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        DistributedDataParallel: The model wrapped with DDP.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda:{}".format(local_rank))
    torch.cuda.set_device(local_rank)

    ddp_cfg = {
        "device_ids": [local_rank],
        "output_device": local_rank,
    }

    model = model.to(device=device)
    model = DistributedDataParallel(model, **ddp_cfg)

    return model


def apply_fsdp(
    model: torch.nn.Module,
    local_rank: int,
    rank: int,
    world_size: int,
) -> FSDP:
    """
    Applies FullyShardedDataParallel (FSDP) to the given model.

    Args:
        model (torch.nn.Module): The model to apply FSDP to.
        local_rank (int): The local rank of the current process.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Returns:
        FSDP: The model wrapped with FSDP.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda:{}".format(local_rank))
    torch.cuda.set_device(local_rank)

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    fsdp_cfg = {
        "use_orig_params": False,
        "auto_wrap_policy": my_auto_wrap_policy,
        "cpu_offload": CPUOffload(offload_params=True),
    }

    model = model.to(device=device)
    model = FSDP(model, **fsdp_cfg)

    return model