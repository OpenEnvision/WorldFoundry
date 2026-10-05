"""Distributed helpers shared by the LingBot World runtimes."""

from __future__ import annotations

import torch
import torch.distributed as dist


def distributed_barrier(device: torch.device) -> None:
    """Synchronize ranks while preserving the NCCL rank-to-device mapping."""

    if not dist.is_available() or not dist.is_initialized():
        return

    if str(dist.get_backend()).lower() == "nccl" and device.type == "cuda":
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        dist.barrier(device_ids=[device_index])
        return

    dist.barrier()


__all__ = ["distributed_barrier"]
