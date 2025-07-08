"""Utility helpers for distributed training.

These functions wrap ``torch.distributed`` initialization so that the same
training scripts can run both locally on a single process and on a
multi-node cluster.  When ``WORLD_SIZE`` is greater than ``1`` we prefer the
``nccl`` backend if GPUs are available and fall back to ``gloo`` for CPU-only
clusters.  A single-process "gloo" setup keeps the code path consistent during
dry runs.
"""

import os
import torch
import torch.distributed as dist


def init_distributed(local_rank: int) -> None:
    """Set up ``torch.distributed`` environment.

    Parameters
    ----------
    local_rank : int
        GPU index assigned by ``torchrun``.  When running a CPU-only dry run this
        value defaults to ``0``.
    """

    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        # On a multi-process cluster we prefer NCCL when GPUs are available.
        # Otherwise fall back to a CPU-only ``gloo`` backend.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    else:
        # Single process execution. ``gloo`` works on any hardware.
        dist.init_process_group("gloo", rank=0, world_size=1)


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """Return ``True`` on rank 0.

    This is useful for logging or saving models only once when running under
    data parallelism.
    """

    return (not dist.is_initialized()) or dist.get_rank() == 0
