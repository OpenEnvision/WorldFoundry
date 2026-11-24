"""Optional host ranges for Nsight Systems, without CUDA synchronization.

Set ``WORLDFOUNDRY_NVTX=1`` before running inference. Disabled ranges do not
import torch or probe CUDA. Ranges are synchronous/thread-local: do not keep
one open across an ``await`` or a generator ``yield``.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def nvtx_range(name: str) -> Generator[None, None, None]:
    """Annotate host submission of work; CPU-only installations are a no-op."""

    if os.environ.get("WORLDFOUNDRY_NVTX", "").strip().lower() not in {"1", "true", "yes", "on"}:
        yield
        return
    try:
        import torch
    except ImportError:
        yield
        return
    if not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
