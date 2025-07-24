"""Utility helpers for distributed training.

The functions in this module hide the boilerplate required to set up
``torch.distributed`` so that the exact same training scripts can execute on
a single laptop or across many nodes in a cluster.  When ``WORLD_SIZE`` is
greater than ``1`` we attempt to use the high performance ``nccl`` backend if
GPUs are available. On CPU clusters we fall back to ``gloo``. Initialising a
"gloo" process group even when ``WORLD_SIZE`` equals ``1`` keeps the code path
consistent during dry runs and simplifies debugging.
"""

import os
import torch
import torch.distributed as dist


def init_distributed(local_rank: int) -> None:
    """Set up ``torch.distributed`` environment.

    Parameters
    ----------
    local_rank : int
        Device index assigned by ``torchrun``. During a CPU-only dry run this
        defaults to ``0`` so that the code path is identical to multi-GPU runs.

    This function reads ``WORLD_SIZE`` from the environment to decide whether
    to create a multi-process group. By wrapping this logic we avoid duplicating
    boilerplate across all training scripts.
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
    """Return ``True`` only on rank 0.

    Many training loops need to perform certain actions exactly once such as
    logging metrics or saving model checkpoints.  By querying this helper the
    scripts can stay agnostic to whether they are running with multiple
    processes or a single process.
    """

    return (not dist.is_initialized()) or dist.get_rank() == 0
