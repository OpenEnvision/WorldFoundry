# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference-time distributed init and rank predicates.

``dist_init`` brings up the process group with a generous timeout so a
slow rank-0 download does not abort NCCL. :func:`is_last_rank` /
:func:`is_last_tp_cp_rank` decide who writes artifacts: only the last
TP×CP rank should dump video so shards do not overwrite each other.

This is not FSDP/PP setup — those live in ``fsdp_runtime`` and
``pipeline_parallel``.
"""

import os
from datetime import timedelta

import torch

import worldfoundry.core.distributed.model_parallel.groups as mpu
from worldfoundry.core.distributed.runtime.logging import print_rank_0
from worldfoundry.core.distributed.model_parallel.pipeline import init_pp_scheduler

from worldfoundry.core.distributed.collectives.generic import get_world_size  # noqa: F401 - public compatibility export

# ──────────────────────────────────────────────────────────────────────────
# Inference process-group + CP/PP mesh — not a wrapper of init_torch_distributed
# ──────────────────────────────────────────────────────────────────────────


def dist_init(config):
    """Initialize torch.distributed plus WorldFoundry CP/PP groups.

    Args:
        config: Runtime config whose ``engine_config`` provides
            ``distributed_backend``, ``distributed_timeout_minutes``,
            ``cp_size``, and ``pp_size``.

    Notes:
        Rank and world size come from ``RANK`` and ``WORLD_SIZE``. The function
        requires CUDA, binds each rank to a local device, verifies
        ``cp_size * pp_size == world_size``, and initializes the pipeline
        scheduler when pipeline parallelism is active.

    CC-17 WONTFIX: not a wrapper of ``init_torch_distributed``. Backend and
    timeout come from ``config``, device binding is ``rank % device_count``
    (not ``LOCAL_RANK``), and this function also creates CP/PP groups plus
    the PP scheduler.
    """
    # CC-17 WONTFIX: keep this body. Different default backend, required env,
    # device binding, and extra group creation — wrapping would change behavior.

    assert torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        print_rank_0("Torch distribution already initialized, skipping initialization ...")
    else:
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        # Manually set the device ids.
        if device_count > 0:
            device = rank % device_count
            torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=config.engine_config.distributed_backend,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=config.engine_config.distributed_timeout_minutes),
        )
    assert config.engine_config.cp_size * config.engine_config.pp_size == torch.distributed.get_world_size()
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print_rank_0("Model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                cp_size=config.engine_config.cp_size,
                pp_size=config.engine_config.pp_size,
                nccl_communicator_config_path=None,
                distributed_timeout_minutes=config.engine_config.distributed_timeout_minutes,
                order="tp-cp-pp-dp",
            )
    if mpu.get_pp_world_size() > 1:
        init_pp_scheduler()
    print_rank_0("Initialize torch distribution and model parallel successfully")


# ──────────────────────────────────────────────────────────────────────────
# Artifact-writer predicates and backend device — last TP×CP rank dumps video
# ──────────────────────────────────────────────────────────────────────────


def is_last_rank():
    """Return whether this worker is the final rank in the global process group."""
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def is_last_tp_cp_rank():
    """Return whether this worker is last in the combined tensor/context group."""
    return mpu.get_tp_rank(with_context_parallel=True) == mpu.get_tp_world_size(with_context_parallel=True) - 1


def get_device(local_rank=None):
    """Map the process-group backend to the tensor device collectives must use.

    NCCL tensors live on CUDA; Gloo is CPU-only. Any other backend is refused
    rather than guessed — a silent CPU fallback would hang the first NCCL op.
    """

    backend = torch.distributed.get_backend()
    if backend == "nccl":
        if local_rank is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{local_rank}")
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        raise RuntimeError
    return device
