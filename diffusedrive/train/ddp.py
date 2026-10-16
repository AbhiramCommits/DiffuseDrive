"""Distributed training helpers."""

import os

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler


def init_distributed():
    """Initialize distributed training (torchrun compatible)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

        return rank, world_size
    else:
        return 0, 1


def is_main_process():
    """Check if current process is main."""
    return int(os.environ.get("RANK", 0)) == 0


def get_rank():
    """Get current process rank."""
    return int(os.environ.get("RANK", 0))


def get_world_size():
    """Get number of processes."""
    return int(os.environ.get("WORLD_SIZE", 1))


def wrap_ddp(model, backend="nccl"):
    """Wrap model with DDP."""
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    return model


def create_distributed_sampler(dataset, shuffle=True):
    """Create sampler for distributed training."""
    return DistributedSampler(
        dataset,
        shuffle=shuffle,
        drop_last=True,
    )


def all_reduce_mean(value: torch.Tensor) -> float:
    """All-reduce and return mean across ranks."""
    dist.all_reduce(value)
    return (value / get_world_size()).item()
