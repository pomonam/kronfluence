import torch
import torch.distributed as dist
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