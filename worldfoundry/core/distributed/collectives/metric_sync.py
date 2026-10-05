"""Distributed metric logging and synchronization helpers.

Metrics reduce on the DP group (not TP/CP) so a sequence-sharded rank
does not under-count tokens. Rank 0 logs; others only participate in
the collective.
"""

from __future__ import annotations

import builtins
import datetime
import os
import pickle
import time
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from logging import getLogger

import torch
import torch.distributed as dist

import worldfoundry.core.distributed.runtime.torch_process_group as _torch_process_group

from worldfoundry.core.distributed.collectives.generic import (
    get_collective_device,
    get_rank,
    get_world_size,
)
from worldfoundry.core.distributed.collectives.generic import (
    is_dist_initialized as is_dist_avail_and_initialized,
)
from worldfoundry.core.distributed.collectives.generic import (
    is_master as is_main_process,
)

logger = getLogger(__name__)

# Original builtins.print, captured the first time setup_for_distributed patches it.
_ORIGINAL_BUILTINS_PRINT = None

# ──────────────────────────────────────────────────────────────────────────
# builtins.print patch — process-wide; restore before third-party capture
# ──────────────────────────────────────────────────────────────────────────


def setup_for_distributed(is_master) -> None:
    """Disable plain print on non-master ranks unless forced.

    This monkey-patches ``builtins.print`` process-wide (including third-party
    libraries). Use :func:`restore_builtins_print` or
    :func:`builtins_print_unpatched` to undo it. Note the historical quirk
    inherited from the MAE codebase: with ``world_size > 8`` every rank prints
    (``force`` is implied); this behavior is preserved as-is.
    """

    global _ORIGINAL_BUILTINS_PRINT

    builtin_print = builtins.print
    if getattr(builtin_print, "_worldfoundry_rank_filtered_print", False):
        # Re-entrant setup: wrap the original once instead of chaining wrappers
        # (chaining would duplicate timestamps and capture stale master flags).
        builtin_print = _ORIGINAL_BUILTINS_PRINT
    if _ORIGINAL_BUILTINS_PRINT is None:
        _ORIGINAL_BUILTINS_PRINT = builtin_print

    def print(*args, **kwargs):  # noqa: A001
        """Rank-filtered print; ``force`` or ``world_size > 8`` prints on every rank."""

        force = kwargs.pop("force", False)
        force = force or get_world_size() > 8
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)

    setattr(print, "_worldfoundry_rank_filtered_print", True)
    builtins.print = print
    logger.info(
        "Patched builtins.print process-wide with a rank-filtered wrapper (is_master=%s); "
        "call worldfoundry.core.distributed.collectives.metric_sync.restore_builtins_print() to undo.",
        bool(is_master),
    )


def restore_builtins_print() -> bool:
    """Restore the original ``builtins.print``; returns True if a patch was removed."""

    if _ORIGINAL_BUILTINS_PRINT is None or not getattr(builtins.print, "_worldfoundry_rank_filtered_print", False):
        return False
    builtins.print = _ORIGINAL_BUILTINS_PRINT
    logger.info("Restored the original builtins.print.")
    return True


@contextmanager
def builtins_print_unpatched():
    """Temporarily restore the original ``builtins.print`` within a scope."""

    patched_print = builtins.print if getattr(builtins.print, "_worldfoundry_rank_filtered_print", False) else None
    restore_builtins_print()
    try:
        yield
    finally:
        if patched_print is not None:
            builtins.print = patched_print


# ──────────────────────────────────────────────────────────────────────────
# Deprecated NCCL init — torchrun / SLURM env, then init_torch_distributed
# ──────────────────────────────────────────────────────────────────────────


def init_distributed(port=37124, rank_and_world_size=(None, None)):
    """Initialize a process group from torchrun/SLURM env vars.

    Deprecated: call
    :func:`worldfoundry.core.distributed.runtime.torch_process_group.init_torch_distributed`
    after publishing the same env vars. Rank / world-size / ``MASTER_*``
    resolution is unchanged so existing call sites keep their contract.
    """

    warnings.warn(
        "worldfoundry.core.distributed.collectives.metric_sync.init_distributed is deprecated; "
        "use worldfoundry.core.distributed.runtime.torch_process_group.init_torch_distributed",
        DeprecationWarning,
        stacklevel=2,
    )
    rank, world_size = rank_and_world_size
    gpu = None
    dist_url = "env://"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(port))
    print("Using port", os.environ["MASTER_PORT"])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        except Exception as exc:
            raise RuntimeError(
                "RANK/WORLD_SIZE are set but the torchrun environment is incomplete or invalid "
                f"(RANK={os.environ.get('RANK')!r}, WORLD_SIZE={os.environ.get('WORLD_SIZE')!r}, "
                f"LOCAL_RANK={os.environ.get('LOCAL_RANK')!r}): {exc}"
            ) from exc
    elif "SLURM_PROCID" in os.environ:
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            gpu = rank % max(torch.cuda.device_count(), 1)
            os.environ["MASTER_ADDR"] = os.environ.get("HOSTNAME", "127.0.0.1")
        except Exception as exc:
            raise RuntimeError(
                "SLURM_PROCID is set but the SLURM environment is incomplete or invalid "
                f"(SLURM_NTASKS={os.environ.get('SLURM_NTASKS')!r}, "
                f"SLURM_PROCID={os.environ.get('SLURM_PROCID')!r}): {exc}"
            ) from exc
    else:
        rank = 0
        world_size = 1
        gpu = 0
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    if rank is None or world_size is None or gpu is None:
        raise RuntimeError(
            "init_distributed could not determine rank/world_size/local device; "
            "pass rank_and_world_size explicitly or launch via torchrun/SLURM."
        )
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("LOCAL_RANK", str(gpu))
    _torch_process_group.init_torch_distributed(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )
    return world_size, rank, gpu, True


# ──────────────────────────────────────────────────────────────────────────
# Windowed meters — deque is local; only count/total are all-reduced
# ──────────────────────────────────────────────────────────────────────────


class SmoothedValue:
    """Track a series of values and expose smoothed statistics."""

    def __init__(self, window_size=20, fmt=None):
        """Allocate a sliding window; the format string is used by ``__str__``."""

        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Append ``value`` with multiplicity ``n`` (e.g. token count)."""

        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """All-reduce ``count`` and ``total`` only — the deque stays process-local.

        Median / window average therefore remain per-rank after a sync; use
        :attr:`global_avg` for the reduced statistic.
        """

        if not is_dist_avail_and_initialized():
            return
        tensor = torch.tensor(
            [self.count, self.total],
            dtype=torch.float64,
            device=get_collective_device(),
        )
        dist.barrier()
        dist.all_reduce(tensor)
        values = tensor.tolist()
        self.count = int(values[0])
        self.total = values[1]

    @property
    def median(self):
        """Median of the local window (not synchronized)."""

        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        """Mean of the local window (not synchronized)."""

        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        """``total / count`` after an optional :meth:`synchronize_between_processes`."""

        return self.total / self.count

    @property
    def max(self):
        """Maximum of the local window."""

        return max(self.deque)

    @property
    def value(self):
        """Most recently appended local value."""

        return self.deque[-1]

    def __str__(self):
        """Render the configured format using median / avg / global_avg / max / value."""

        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """Named :class:`SmoothedValue` meters plus a progress-bar iterator wrapper."""

    def __init__(self, delimiter="\t"):
        """Create an empty meter map; ``delimiter`` joins :meth:`__str__` fields."""

        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Record scalar kwargs; tensors are ``.item()``'d, ``None`` is skipped."""

        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.meters[key].update(value)

    def __getattr__(self, attr):
        """Expose meters as attributes so ``logger.loss.global_avg`` works."""

        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    def __str__(self):
        """Join every meter with the configured delimiter."""

        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        """All-reduce ``count``/``total`` on every meter (windows stay local)."""

        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Install a pre-built meter (e.g. a custom window size)."""

        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Yield items while printing ETA / meters / peak CUDA memory.

        Uses the (possibly rank-filtered) ``print`` so only the master
        typically emits. Empty iterables would divide by zero in the
        summary line — callers must pass a non-empty sequence.
        """

        index = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        mb = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if index % print_freq == 0 or index == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - index)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            index,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / mb,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            index,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            index += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")
        self.update(total_time=total_time)


# ──────────────────────────────────────────────────────────────────────────
# FID merge — pickle per-rank state, then merge_state on rank-local metrics
# ──────────────────────────────────────────────────────────────────────────


def sync_fid_loss_fns(fid_loss_fn, device="cuda"):
    """Synchronize FID metric objects across all distributed ranks."""

    if not is_dist_avail_and_initialized():
        return fid_loss_fn

    serialized_fid_loss_fn = pickle.dumps(fid_loss_fn)
    gathered_fid_loss_fn = [None] * dist.get_world_size()

    dist.barrier()
    dist.all_gather_object(gathered_fid_loss_fn, serialized_fid_loss_fn)

    from torcheval.metrics import FrechetInceptionDistance

    final_fid_loss_fn = {sec: FrechetInceptionDistance(feature_dim=2048).to(device) for sec in [1, 2, 4, 8, 16]}
    for serialized_rank_metrics in gathered_fid_loss_fn:
        rank_metrics = pickle.loads(serialized_rank_metrics)
        for sec in [1, 2, 4, 8, 16]:
            final_fid_loss_fn[sec].merge_state([rank_metrics[sec]])

    return final_fid_loss_fn


__all__ = [
    "MetricLogger",
    "SmoothedValue",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_dist_avail_and_initialized",
    "is_main_process",
    "setup_for_distributed",
    "sync_fid_loss_fns",
]
