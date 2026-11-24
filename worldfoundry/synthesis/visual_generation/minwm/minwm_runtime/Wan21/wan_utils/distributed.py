"""Inference-only distributed helpers backed by WorldFoundry core."""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from worldfoundry.core.distributed.sequence_parallel import parallel_state
from worldfoundry.core.distributed.sequence_parallel.parallel_states import get_parallel_state


def get_sp_data_sampler(dataset, shuffle: bool = True, drop_last: bool = True):
    """Create a sampler whose peers in one SP group receive identical inputs."""

    if parallel_state.model_parallel_is_initialized() and get_parallel_state().sp_enabled:
        return torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=parallel_state.get_dp_world_size(),
            rank=parallel_state.get_dp_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)


def get_sp_seed_offset() -> int:
    """Use the DP rank so all ranks in an SP group share random inputs."""

    if parallel_state.model_parallel_is_initialized() and get_parallel_state().sp_enabled:
        return parallel_state.get_dp_rank()
    return dist.get_rank() if dist.is_initialized() else 0


def launch_distributed_job(backend: str = "nccl", sp_size: int = 1) -> None:
    """Initialize either WorldFoundry sequence parallelism or plain DDP."""

    if sp_size < 1:
        raise ValueError(f"sp_size must be positive, got {sp_size}")
    if sp_size > 1:
        parallel_state.maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=sp_size)
        return
    if dist.is_initialized():
        return
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    init_method = f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)


__all__ = ["get_sp_data_sampler", "get_sp_seed_offset", "launch_distributed_job"]
