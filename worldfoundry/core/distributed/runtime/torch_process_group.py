"""Torch process-group helpers shared by runtime and training code.

Create/destroy the default group and query backend. Model-parallel
subgroups are *not* created here — that is ``initialize_model_parallel``.

Public surface: :func:`init_torch_distributed`, :func:`get_rank`,
:func:`get_world_size`, :func:`rank0_first`, :class:`DistributedDataParallel`,
and the WORLD tensor collectives.
"""

from __future__ import annotations

import collections
import collections.abc
import ctypes
import functools
import logging
import math
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Callable, Container, Optional

import torch
import torch.distributed as dist
from torch.distributed import get_process_group_ranks

try:
    import pynvml
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency.
    pynvml = None

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group
    from torch.distributed.utils import (
        _sync_module_states,
        _verify_param_shape_across_processes,
    )

try:
    from worldfoundry.core.distributed.model_parallel.megatron_compat import parallel_state
except Exception:  # pragma: no cover - optional training dependency.

    class _NoParallelState:
        """Stub when ``megatron_compat.parallel_state`` cannot be imported."""
        @staticmethod
        def is_initialized() -> bool:
            """Always false — Megatron parallel_state is not importable here."""

            return False

        @staticmethod
        def get_data_parallel_group(with_context_parallel: bool = False):
            """``None`` so DDP falls back to the default process group."""

            return None

    parallel_state = _NoParallelState()


logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Device / affinity helpers — bind CUDA before any NCCL collective
# ──────────────────────────────────────────────────────────────────────────


def _get_local_cuda_device(local_rank: int) -> int:
    """Map ``LOCAL_RANK`` to a CUDA index, honoring an explicit override.

    ``WORLDFOUNDRY_GEN3C_LOCAL_CUDA_DEVICE`` is for Gen3C jobs that pin a
    process to a non-ordinal GPU. An unparsable value is ignored so a typo
    cannot crash launch after ranks have already rendezvoused.
    """

    override = os.getenv("WORLDFOUNDRY_GEN3C_LOCAL_CUDA_DEVICE")
    if override is None:
        return local_rank
    try:
        return int(override)
    except ValueError:
        logger.warning(
            "Ignoring invalid WORLDFOUNDRY_GEN3C_LOCAL_CUDA_DEVICE=%r.",
            override,
        )
        return local_rank


def _get_gpu_cpu_affinity(device_idx: int) -> list[int]:
    """Return CPU ids NVML associates with ``device_idx``, or ``[]`` without pynvml.

    Empty means "do not call ``sched_setaffinity``" so a missing NVML cannot
    pin the process to the empty set and starve it.
    """

    if pynvml is None:
        return []
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    affinity_elements = math.ceil((os.cpu_count() or 1) / 64)
    affinity_string = ""
    for element in pynvml.nvmlDeviceGetCpuAffinity(handle, affinity_elements):
        affinity_string = f"{element:064b}" + affinity_string
    affinity = [int(value) for value in affinity_string]
    affinity.reverse()
    return [index for index, enabled in enumerate(affinity) if enabled]


def _set_cuda_l2_fetch_granularity() -> None:
    """Best-effort ``cudaDeviceSetLimit`` for L2 fetch (enum ``0x05`` = 128 B).

    Missing ``libcudart.so`` is ignored: this is a throughput hint, not a
    correctness requirement.
    """

    if not torch.cuda.is_available():
        return
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        libcudart.cudaDeviceGetLimit(value, ctypes.c_int(0x05))
    except OSError:
        logger.debug("libcudart.so is unavailable; skipped CUDA device limit setup.")


# ──────────────────────────────────────────────────────────────────────────
# Canonical process-group init — other dist_init helpers must call this
# ──────────────────────────────────────────────────────────────────────────


def init_torch_distributed(
    backend: str = "nccl",
    init_method: str | None = "env://",
    world_size: int | None = None,
    rank: int | None = None,
    timeout: timedelta | None = None,
) -> int | None:
    """Canonical torch.distributed process-group initialization.

    Other entry points in this package (``generic_collectives.dist_init``,
    ``metric_sync.init_distributed``, ``evaluation_collectives.dist_init``,
    ``runtime_setup.init_distributed``, ``sequence_ops.init_distributed_group``,
    multiprocess launch) should call this function rather than
    ``torch.distributed.init_process_group`` directly.

    Reads torchrun-style ``LOCAL_RANK`` for device binding, applies NCCL
    defaults when unset, and initializes the default process group. Extra
    ``init_process_group`` arguments are forwarded so launchers that pass an
    explicit rendezvous URL / rank keep their call-site contract.
    """

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    local_cuda_device = _get_local_cuda_device(local_rank)
    if dist.is_available() and dist.is_initialized():
        return torch.cuda.current_device() if torch.cuda.is_available() else local_rank

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            affinity = _get_gpu_cpu_affinity(local_cuda_device)
            if affinity:
                os.sched_setaffinity(0, affinity)
        except Exception as exc:
            logger.warning("Failed to set GPU CPU affinity: %s", exc)

    # Respect explicit user overrides (e.g. TORCH_NCCL_BLOCKING_WAIT=1 for
    # debugging); only provide defaults when unset (CC-23).
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    if dist.is_available():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_cuda_device)
        if timeout is None:
            timeout = timedelta(seconds=int(os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)))
        init_kwargs: dict[str, Any] = {"backend": backend, "timeout": timeout}
        if init_method is not None:
            init_kwargs["init_method"] = init_method
        if world_size is not None:
            init_kwargs["world_size"] = world_size
        if rank is not None:
            init_kwargs["rank"] = rank
        dist.init_process_group(**init_kwargs)
        logger.info(
            "Initialized distributed process group with local rank %s, CUDA device %s, and timeout %s.",
            local_rank,
            local_cuda_device,
            timeout,
        )

    _set_cuda_l2_fetch_granularity()
    logger.info("Running with %s GPUs.", get_world_size())
    return None


# ──────────────────────────────────────────────────────────────────────────
# Canonical rank / world-size queries — other modules must delegate here
# ──────────────────────────────────────────────────────────────────────────


def init() -> int | None:
    """Initialize a NCCL process group from torchrun-style environment variables.

    Alias of :func:`init_torch_distributed`, the canonical initializer.
    """

    return init_torch_distributed()


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Return this worker's rank, or 0 outside distributed execution.

    Canonical rank query for ``worldfoundry.core.distributed``. Other
    ``get_rank`` helpers in this package should delegate here.
    """

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group)
    return 0


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Return the process-group size, or 1 outside distributed execution.

    Canonical world-size query for ``worldfoundry.core.distributed``. Other
    ``get_world_size`` helpers in this package should delegate here.
    """

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group)
    return 1


def get_local_rank() -> int:
    """Return this worker's local rank from ``LOCAL_RANK``, or 0 if not distributed.

    Canonical local-rank query for ``worldfoundry.core.distributed``.
    """

    if not dist.is_available() or not dist.is_initialized():
        return 0
    return int(os.getenv("LOCAL_RANK", 0))


def is_rank0() -> bool:
    """True on global rank 0, including the single-process fallback."""

    return get_rank() == 0


def is_local_rank0() -> bool:
    """Return whether this process is the first rank on its node.

    ``LOCAL_RANK`` (set by torchrun and most launchers) is authoritative:
    schedulers that isolate each process with ``CUDA_VISIBLE_DEVICES=<one
    GPU>`` make ``torch.cuda.current_device()`` report 0 in every process,
    which would wrongly mark all ranks as local rank 0 (CC-16).
    """

    local_rank = os.getenv("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank) == 0
        except ValueError:
            logger.warning("Ignoring invalid LOCAL_RANK=%r.", local_rank)
    if torch.cuda.is_available():
        return torch.cuda.current_device() == 0
    return True


def device_with_rank(device: str) -> str:
    """Qualify a bare ``cuda`` device string with this process's local device.

    Uses ``LOCAL_RANK`` (falling back to the current CUDA device) rather than
    the global rank: on multi-node jobs the global rank exceeds the per-node
    GPU count and would produce nonexistent devices such as ``cuda:8`` (CC-15).
    """

    if device == "cuda":
        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is not None:
            try:
                return f"cuda:{int(local_rank)}"
            except ValueError:
                logger.warning("Ignoring invalid LOCAL_RANK=%r.", local_rank)
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return f"cuda:{get_rank()}"
    return device


# ──────────────────────────────────────────────────────────────────────────
# Rank-0 helpers — barrier even on rank-0 failure so peers do not NCCL-timeout
# ──────────────────────────────────────────────────────────────────────────


def rank0_only(func: Callable) -> Callable:
    """Run ``func`` only on global rank 0."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        """Call ``func`` only when :func:`is_rank0`; otherwise return ``None``."""

        if is_rank0():
            return func(*args, **kwargs)
        return None

    return wrapper


def barrier() -> None:
    """Default-group barrier; no-op when torch.distributed is not initialized."""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank0_first(func: Callable) -> Callable:
    """Run ``func`` on rank 0 before all other ranks.

    Rank 0 failures are propagated to every rank: the barrier is reached even
    when ``func`` raises (otherwise the surviving ranks would block on the
    barrier until the NCCL timeout, CC-14), and a success flag is broadcast so
    non-zero ranks raise instead of running ``func`` against half-initialized
    state. Note ``func``'s rank-0 runtime is still bounded by the process
    group timeout of the barrier/broadcast collectives.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        """Run ``func`` on rank 0, barrier, then on other ranks if rank 0 succeeded."""

        distributed = dist.is_available() and dist.is_initialized() and get_world_size() > 1
        result = None
        rank0_error: BaseException | None = None
        if is_rank0():
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:  # re-raised below, after the barrier
                rank0_error = exc
        if distributed:
            barrier()
            success = [rank0_error is None]
            dist.broadcast_object_list(success, src=0)
            if rank0_error is not None:
                raise rank0_error
            if not success[0]:
                raise RuntimeError(
                    f"rank0_first({getattr(func, '__name__', func)!r}) failed on rank 0; "
                    "see the rank 0 log for the original exception."
                )
        elif rank0_error is not None:
            raise rank0_error
        if not is_rank0():
            result = func(*args, **kwargs)
        return result

    return wrapper


# ──────────────────────────────────────────────────────────────────────────
# DDP wrap — prefer Megatron DP+CP group when parallel_state is live
# ──────────────────────────────────────────────────────────────────────────


def parallel_model_wrapper(config_ddp: Any, model: torch.nn.Module) -> torch.nn.Module | DistributedDataParallel:
    """Wrap a model with DDP when a process group is initialized."""

    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        local_cuda_device = _get_local_cuda_device(local_rank)
        try:
            ddp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        except Exception as exc:
            logger.info("parallel_state not initialized; using the default DDP group: %s", exc)
            ddp_group = None

        model = DistributedDataParallel(
            model,
            device_ids=[local_cuda_device],
            output_device=local_cuda_device,
            find_unused_parameters=config_ddp.find_unused_parameters,
            static_graph=config_ddp.static_graph,
            broadcast_buffers=config_ddp.broadcast_buffers,
            process_group=ddp_group,
        )
    return model


# ──────────────────────────────────────────────────────────────────────────
# DDP wrapper — training_step must go through forward so hooks fire
# ──────────────────────────────────────────────────────────────────────────


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """DDP wrapper that redirects ``training_step`` through ``forward``."""

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        """Remember whether we already warned about static_graph + ddp_sync_grad."""

        super().__init__(model, *args, **kwargs)
        self.show_sync_grad_static_graph_warning = True

    def training_step(self, *args, **kwargs) -> Any:
        """Temporarily point ``module.forward`` at ``training_step`` and call DDP.

        Lightning-style modules implement ``training_step`` instead of
        ``forward``. DDP only hooks ``forward``, so we swap for one call.
        """

        original_forward = self.module.forward

        def wrapped_training_step(*_args, **_kwargs):  # noqa: ANN202
            """Restore ``forward`` before the real ``training_step`` to avoid recursion."""

            self.module.forward = original_forward
            return self.module.training_step(*_args, **_kwargs)

        self.module.forward = wrapped_training_step
        return self(*args, **kwargs)


@contextmanager
def ddp_sync_grad(model, enabled):
    """Temporarily enable or disable DDP gradient synchronization."""

    assert isinstance(model, torch.nn.Module)
    old_require_backward_grad_sync = None
    if isinstance(model, DistributedDataParallel):
        old_require_backward_grad_sync = model.require_backward_grad_sync
        if model.static_graph and model.require_backward_grad_sync != enabled:
            if model.show_sync_grad_static_graph_warning:
                logger.warning("DDP static_graph=True is incompatible with ddp_sync_grad().")
                model.show_sync_grad_static_graph_warning = False
        else:
            model.require_backward_grad_sync = enabled
    try:
        yield
    finally:
        if isinstance(model, DistributedDataParallel) and old_require_backward_grad_sync is not None:
            model.require_backward_grad_sync = old_require_backward_grad_sync


def collate_batches(data_batches: list[dict[str, torch.Tensor]]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Gather validation batches from all ranks in original rank order."""

    if isinstance(data_batches[0], torch.Tensor):
        data_concat = torch.cat(data_batches, dim=0)  # type: ignore[arg-type]
        if get_world_size() == 1:
            return data_concat
        max_num_local_samples = torch.tensor(len(data_concat), device=data_concat.device)
        dist.all_reduce(max_num_local_samples, op=dist.ReduceOp.MAX)
        max_num_local_samples_value = int(max_num_local_samples.item())
        if len(data_concat) < max_num_local_samples_value:
            assert len(data_concat) + 1 == max_num_local_samples_value
            dummy = torch.empty_like(data_concat[:1])
            data_concat = torch.cat([data_concat, dummy], dim=0)
            dummy_count = torch.tensor(1, device=data_concat.device)
        else:
            dummy_count = torch.tensor(0, device=data_concat.device)
        dist.all_reduce(dummy_count, op=dist.ReduceOp.SUM)
        dummy_count_value = int(dummy_count.item())
        gathered = all_gather_tensor(data_concat.contiguous())
        data_collate = torch.stack(gathered, dim=1).flatten(start_dim=0, end_dim=1)
        if dummy_count_value > 0:
            data_collate = data_collate[:-dummy_count_value]
    elif isinstance(data_batches[0], collections.abc.Mapping):
        # Recurse per key so a mixed tensor/dict batch stays aligned across ranks.
        data_collate = {}
        for key in data_batches[0].keys():
            data_collate[key] = collate_batches([data[key] for data in data_batches])  # type: ignore[index]
    else:
        raise TypeError(f"Unsupported batch type: {type(data_batches[0])!r}")
    return data_collate


# ──────────────────────────────────────────────────────────────────────────
# Tensor collectives — no-op when world_size < 2 so callers stay branch-free
# ──────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """All-gather equal tensors on WORLD; return ``[tensor]`` when alone."""

    if get_world_size() == 1:
        return [tensor]
    tensor_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast(tensor, src, group=None, async_op=False):
    """In-place broadcast; identity when ``world_size < 2``."""

    if get_world_size() < 2:
        return tensor
    dist.broadcast(tensor, src=src, group=group, async_op=async_op)
    return tensor


def dist_reduce_tensor(tensor, rank=0, reduce="mean"):
    """Reduce onto ``rank`` then optionally divide; only that rank sees the mean.

    Non-destination ranks still participate in ``dist.reduce`` but must not
    divide — they hold the unreduced local value.
    """

    if get_world_size() < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=rank)
        if get_rank() == rank:
            if reduce == "mean":
                tensor /= get_world_size()
            elif reduce != "sum":
                raise NotImplementedError(f"Unsupported reduce mode: {reduce}")
    return tensor


# ──────────────────────────────────────────────────────────────────────────
# Parameter broadcast — ignore list + 250 MiB buckets (c10d helper)
# ──────────────────────────────────────────────────────────────────────────


def sync_model_states(
    model: torch.nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
    src: int = 0,
    params_and_buffers_to_ignore: Optional[Container[str]] = None,
    broadcast_buffers: bool = True,
) -> None:
    """Broadcast model parameters and buffers from ``src`` to the process group."""

    if not dist.is_available() or not dist.is_initialized():
        return
    if process_group is None:
        process_group = _get_default_group()
    if not params_and_buffers_to_ignore:
        params_and_buffers_to_ignore = set()

    logger.info(
        "Synchronizing model states from rank %s to process group %s.",
        src,
        get_process_group_ranks(process_group),
    )

    modules_and_parameters = [
        (module, parameter)
        for module_name, module in model.named_modules()
        for parameter in [
            param
            for param_name, param in module.named_parameters(recurse=False)
            if f"{module_name}.{param_name}" not in params_and_buffers_to_ignore
        ]
    ]

    memo = set()
    modules_and_parameters = [
        (module, parameter)
        for module, parameter in modules_and_parameters
        if parameter not in memo and not memo.add(parameter)  # type: ignore[func-returns-value]
    ]
    parameters = [parameter for _, parameter in modules_and_parameters]
    if not parameters:
        return

    _verify_param_shape_across_processes(process_group, parameters)
    _sync_module_states(
        module=model,
        process_group=process_group,
        broadcast_bucket_size=int(250 * 1024 * 1024),
        src=src,
        params_and_buffers_to_ignore=params_and_buffers_to_ignore,
        broadcast_buffers=broadcast_buffers,
    )


__all__ = [
    "DistributedDataParallel",
    "all_gather_tensor",
    "barrier",
    "broadcast",
    "collate_batches",
    "ddp_sync_grad",
    "device_with_rank",
    "dist_reduce_tensor",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "init",
    "init_torch_distributed",
    "is_local_rank0",
    "is_rank0",
    "parallel_model_wrapper",
    "rank0_first",
    "rank0_only",
    "sync_model_states",
]
