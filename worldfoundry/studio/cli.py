from __future__ import annotations

import os
from typing import Sequence


def _prepare_cuda_allocator() -> None:
    """Default the CUDA allocator to expandable segments before any torch import.

    Large multi-GPU runtimes (e.g. the LingBot-World fast checkpoint sharded with
    FSDP) can transiently fragment GPU memory while every rank loads its shards in
    parallel, producing ``CUDA error: out of memory`` failures even though the
    settled footprint is small. ``expandable_segments`` lets the allocator reclaim
    fragmented blocks and must be set before CUDA initializes. ``setdefault`` keeps
    any explicit operator override intact.
    """

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main(argv: Sequence[str] | None = None) -> None:
    """Launch a standalone Studio browser frontend."""

    _prepare_cuda_allocator()
    from worldfoundry.studio.ui.launcher import main as standalone_main

    standalone_main(argv)


if __name__ == "__main__":
    main()
