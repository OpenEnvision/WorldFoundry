"""Memory residency. Each strategy lives in its own module.

- ``none``   — no-op handle; the pipeline still uses ``with offload.use(...)``.
- ``module`` — one resident module group at a time (CUDA, async H2D/D2H).
- ``block``  — two DiT blocks resident; the next block is prefetched while the current one runs.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Protocol

import torch

from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.base import ConfigModel
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.offload.block import BlockOffload
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.offload.common import OffloadStrategy, _module_to
from worldfoundry.base_models.diffusion_model.optimizations.kandinsky6.offload.module import ModuleOffload

__all__ = (
    "BlockOffload",
    "ModuleOffload",
    "NoOpOffload",
    "OffloadConfig",
    "OffloadHandle",
    "OffloadStrategy",
    "_module_to",
    "build_offload",
)


class OffloadConfig(ConfigModel):
    """Memory residency strategy.

    ``module`` — CUDA-only async stage offload (text, DiT, VAE).
    ``block`` — per-DiT-block offload. Two blocks stay resident, and the next
    one is prefetched on a transfer stream. Requires ``pin_memory``.
    """

    strategy: OffloadStrategy = "none"
    pin_memory: bool = True


class OffloadHandle(Protocol):
    """What the pipeline calls. Strategies implement these methods."""

    strategy: OffloadStrategy

    def register(self, name: str, module: Any) -> None: ...

    def release(self, *names: str) -> None: ...

    @contextmanager
    def use(
        self,
        *names: str,
        prefetch: str | Sequence[str] | None = None,
    ) -> Iterator[None]: ...


class NoOpOffload:
    """Zero-cost handle so pipeline stages never branch on offload mode."""

    strategy: OffloadStrategy = "none"

    def register(self, name: str, module: Any) -> None:
        return None

    def release(self, *names: str) -> None:
        return None

    @contextmanager
    def use(
        self,
        *names: str,
        prefetch: str | Sequence[str] | None = None,
    ) -> Iterator[None]:
        yield


def build_offload(
    strategy: OffloadStrategy,
    compute_device: torch.device,
    *,
    pin_memory: bool = True,
) -> OffloadHandle:
    if strategy == "none":
        return NoOpOffload()
    if strategy == "module":
        if compute_device.type != "cuda":
            raise ValueError(f"offload.strategy='module' currently supports CUDA only, got {compute_device}")
        return ModuleOffload(compute_device, pin_memory=pin_memory)
    if strategy == "block":
        if compute_device.type != "cuda":
            raise ValueError(f"offload.strategy='block' currently supports CUDA only, got {compute_device}")
        return BlockOffload(compute_device, pin_memory=pin_memory)
    raise ValueError(f"Unknown offload strategy: {strategy!r}")
