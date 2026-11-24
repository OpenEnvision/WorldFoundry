from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from torch import nn

from ...ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ...ltx_core.devices import synchronize_device
from ...ltx_core.model.disposable import dispose_model
from .helpers import cleanup_memory

_M = TypeVar("_M", bound=nn.Module)


@contextmanager
def gpu_model(model: _M, alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM) -> Iterator[_M]:
    """Context manager that yields a model and releases its memory on exit.
    Releases parameter / persistent-buffer storage so it
    moves to meta (shell-safe; fused LoRA weights do not linger on a cached
    module). ``TRIM`` (default) also synchronizes and ``cleanup_memory()`` to
    return cached blocks to the OS. ``DEFER`` skips sync / ``empty_cache`` so
    the CUDA caching allocator stays warm for the next build.
    Usage::
        with gpu_model(build_encoder()) as encoder:
            ...  # use encoder -- typed as the concrete class
        # parameter storage released; TRIM also returns cached blocks to the OS
    """
    try:
        yield model
    finally:
        if alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            synchronize_device()
            dispose_model(model)
            cleanup_memory()
        else:
            dispose_model(model)
