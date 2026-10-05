# SPDX-License-Identifier: Apache-2.0
# Extracted from trainer/utils.py — CUDA utility functions needed by SP infra.
"""CUDA stream/device helpers used by sequence-parallel communicators.

Keeps NCCL ops on the caller's stream and avoids import-time
``cuda.set_device``. Thread-local current device is the same CC-25 rule
as attention probing.

Not a CUDA runtime — this module only tracks the Python-visible current
stream and locates ``libnccl`` / ``librccl``. Graph capture still uses
:mod:`.device_communicators.pynccl`.

Public surface: :func:`find_nccl_library`, :func:`current_stream`,
:func:`install_torch_set_stream_patch`, :func:`restore_torch_set_stream`,
:func:`torch_set_stream_patch_removed`.
"""

import threading
from contextlib import contextmanager

import torch

import worldfoundry.core.distributed.sequence_parallel.envs as envs
from .logger import init_logger

logger = init_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# NCCL shared-object lookup — TRAINER_NCCL_SO_PATH overrides PyTorch's soname
# ──────────────────────────────────────────────────────────────────────────


def find_nccl_library() -> str:
    """
    We either use the library file specified by the `TRAINER_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """
    so_file = envs.TRAINER_NCCL_SO_PATH

    # manually load the nccl library
    if so_file:
        logger.info("Found nccl from environment variable TRAINER_NCCL_SO_PATH=%s", so_file)
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.info("Found nccl from library %s", so_file)
    return str(so_file)


class _StreamCache(threading.local):
    """Per-thread cache of the stream last passed to ``torch.cuda.set_stream``.

    PyTorch's current stream is per-thread (and per-device) state; a single
    process-wide cache would leak stream selections across threads. Each new
    thread starts at ``None`` and falls back to ``torch.cuda.current_stream()``
    on first read, matching unpatched behavior.
    """

    def __init__(self) -> None:
        """Start empty so the first :func:`current_stream` read hits PyTorch."""
        self.value: torch.cuda.Stream | None = None


_STREAM_CACHE = _StreamCache()
# Original torch.cuda.set_stream, captured when the patch installs.
prev_set_stream = None


# ──────────────────────────────────────────────────────────────────────────
# set_stream monkey-patch — cache the Python-visible stream for pynccl
# ──────────────────────────────────────────────────────────────────────────


def install_torch_set_stream_patch() -> None:
    """Monkey-patch ``torch.cuda.set_stream`` to track the current stream.

    Idempotent. The patch is installed at import of this module (historical
    default relied on by the sequence-parallel pynccl communicators); use
    :func:`restore_torch_set_stream` or :func:`torch_set_stream_patch_removed`
    to undo it.
    """

    global prev_set_stream
    current = torch.cuda.set_stream
    if getattr(current, "_worldfoundry_sp_set_stream_patch", False):
        return
    prev_set_stream = current

    def _patched_set_stream(stream: torch.cuda.Stream | None) -> None:
        """Record ``stream`` then forward; ``None`` only updates the cache."""
        _STREAM_CACHE.value = stream
        if stream is not None:
            prev_set_stream(stream)

    setattr(_patched_set_stream, "_worldfoundry_sp_set_stream_patch", True)
    torch.cuda.set_stream = _patched_set_stream
    logger.info(
        "Patched torch.cuda.set_stream process-wide to track the current stream for "
        "sequence-parallel pynccl collectives; call "
        "worldfoundry.core.distributed.sequence_parallel.cuda_utils.restore_torch_set_stream() to undo."
    )


def restore_torch_set_stream() -> bool:
    """Restore the original ``torch.cuda.set_stream``; returns True if unpatched."""

    if prev_set_stream is None or not getattr(torch.cuda.set_stream, "_worldfoundry_sp_set_stream_patch", False):
        return False
    torch.cuda.set_stream = prev_set_stream
    _STREAM_CACHE.value = None
    logger.info("Restored the original torch.cuda.set_stream.")
    return True


@contextmanager
def torch_set_stream_patch_removed():
    """Temporarily restore the original ``torch.cuda.set_stream`` in a scope.

    Needed when a callee (CUDA-graph capture, third-party NCCL) must see
    the unpatched API. Reinstalls only if this scope was the one that
    removed the patch, so nested uses do not leak a double-patch.
    """

    was_patched = restore_torch_set_stream()
    try:
        yield
    finally:
        if was_patched:
            install_torch_set_stream_patch()


install_torch_set_stream_patch()


def current_stream() -> torch.cuda.Stream | None:
    """
    replace `torch.cuda.current_stream()` with `current_stream()`.
    it turns out that `torch.cuda.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.

    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code. Stream switches performed by C++/CUDA-graph internals do
    not pass through the Python patch, so the cached value can be stale in
    those scenarios; the cache is per-thread to avoid cross-thread leakage.
    """
    if _STREAM_CACHE.value is None:
        _STREAM_CACHE.value = torch.cuda.current_stream()
    return _STREAM_CACHE.value
