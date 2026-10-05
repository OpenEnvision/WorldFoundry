"""Variable temporal chunk geometry and dual-resolution KV storage.

The regular :mod:`block_pattern` primitive intentionally models a first block
followed by uniform blocks.  Flow-map autoregressive models use an explicit
partition instead (for example ``[1, 3, 3, 3, 3, 3, 3, 2]``) and retain recent
chunks at full spatial resolution while compressing older chunks.  This module
keeps that geometry in core so training and native model adapters do not each
reimplement frame boundaries or cache capacities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────
# Partition + token geometry — explicit chunks, not first-plus-uniform
# ──────────────────────────────────────────────────────────────────────────


def _positive_triplet(value: tuple[int, ...], *, field_name: str) -> tuple[int, int, int]:
    """Require a 3-tuple of positive integers (patch sizes)."""
    result = tuple(int(item) for item in value)
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError(f"{field_name} must contain three positive integers")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TemporalChunkPartition:
    """An explicit latent-frame partition with recent full-resolution chunks."""

    chunks: tuple[int, ...]
    full_chunk_limit: int
    patch_size: tuple[int, int, int] = (1, 2, 2)
    compressed_patch_size: tuple[int, int, int] = (1, 4, 4)

    def __post_init__(self) -> None:
        """Reject empty/non-positive chunks and finer compressed spatial patches."""
        chunks = tuple(int(value) for value in self.chunks)
        if not chunks or any(value <= 0 for value in chunks):
            raise ValueError("chunks must contain positive latent-frame counts")
        if isinstance(self.full_chunk_limit, bool):
            raise TypeError("full_chunk_limit must be an integer")
        limit = int(self.full_chunk_limit)
        if not 1 <= limit <= len(chunks):
            raise ValueError("full_chunk_limit must be in [1, len(chunks)]")
        patch = _positive_triplet(self.patch_size, field_name="patch_size")
        compressed = _positive_triplet(
            self.compressed_patch_size,
            field_name="compressed_patch_size",
        )
        if patch[0] != compressed[0]:
            raise ValueError("full and compressed temporal patch sizes must agree")
        if any(compressed[index] < patch[index] for index in (1, 2)):
            raise ValueError("compressed spatial patches cannot be finer than full patches")
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "full_chunk_limit", limit)
        object.__setattr__(self, "patch_size", patch)
        object.__setattr__(self, "compressed_patch_size", compressed)

    @property
    def frame_count(self) -> int:
        """Total latent frames across every chunk in the partition."""
        return sum(self.chunks)

    @property
    def chunk_count(self) -> int:
        """Number of temporal chunks (not frames) in the partition."""
        return len(self.chunks)

    def prefix(self, chunk_count: int) -> tuple[int, ...]:
        """Return the first ``chunk_count`` chunks after validating the range."""

        if isinstance(chunk_count, bool):
            raise TypeError("chunk_count must be an integer")
        count = int(chunk_count)
        if not self.full_chunk_limit <= count <= self.chunk_count:
            raise ValueError(
                "sampled chunk_count must include full_chunk_limit chunks and not exceed the partition"
            )
        return self.chunks[:count]

    def spans(self, chunk_count: int | None = None) -> tuple[tuple[int, int], ...]:
        """Return contiguous frame spans for the full partition or a prefix."""

        selected = self.chunks if chunk_count is None else self.prefix(chunk_count)
        result: list[tuple[int, int]] = []
        cursor = 0
        for size in selected:
            result.append((cursor, cursor + size))
            cursor += size
        return tuple(result)

    def context_target_frames(self, chunk_count: int) -> tuple[int, int]:
        """Return context and full-resolution target frame counts for training."""

        selected = self.prefix(chunk_count)
        target = sum(selected[-self.full_chunk_limit :])
        return sum(selected) - target, target

    def token_geometry(
        self,
        *,
        latent_height: int,
        latent_width: int,
        chunk_count: int | None = None,
    ) -> DualResolutionTokenGeometry:
        """Resolve full/compressed tokens and official cache capacities."""

        height = int(latent_height)
        width = int(latent_width)
        if height <= 0 or width <= 0:
            raise ValueError("latent height and width must be positive")
        for patch, name in (
            (self.patch_size, "patch_size"),
            (self.compressed_patch_size, "compressed_patch_size"),
        ):
            if height % patch[1] or width % patch[2]:
                raise ValueError(f"latent geometry must be divisible by {name}")
        selected = self.chunks if chunk_count is None else self.prefix(chunk_count)
        maximum = max(selected)
        full_per_frame = (height // self.patch_size[1]) * (width // self.patch_size[2])
        compressed_per_frame = (
            (height // self.compressed_patch_size[1])
            * (width // self.compressed_patch_size[2])
        )
        # AnyFlow deliberately over-allocates each slot to max(chunk_partition)
        # so every chunk can reuse the same cache views.
        full_capacity = self.full_chunk_limit * maximum * full_per_frame
        compressed_slots = len(selected) - self.full_chunk_limit + 1
        compressed_capacity = compressed_slots * maximum * compressed_per_frame
        return DualResolutionTokenGeometry(
            full_tokens_per_frame=full_per_frame,
            compressed_tokens_per_frame=compressed_per_frame,
            full_capacity=full_capacity,
            compressed_capacity=compressed_capacity,
        )


@dataclass(frozen=True, slots=True)
class DualResolutionTokenGeometry:
    """Token counts consumed by a full/recent plus compressed/history cache."""

    full_tokens_per_frame: int
    compressed_tokens_per_frame: int
    full_capacity: int
    compressed_capacity: int

    def __post_init__(self) -> None:
        """Require every token/capacity field to be positive."""
        if any(
            int(value) <= 0
            for value in (
                self.full_tokens_per_frame,
                self.compressed_tokens_per_frame,
                self.full_capacity,
                self.compressed_capacity,
            )
        ):
            raise ValueError("dual-resolution token geometry must be positive")


# ──────────────────────────────────────────────────────────────────────────
# Dual-resolution store — recent full-res chunks plus compressed history
# ──────────────────────────────────────────────────────────────────────────


class DualResolutionKVCache:
    """Layer-major key/value storage for recent and compressed FAR context."""

    def __init__(
        self,
        geometry: DualResolutionTokenGeometry,
        *,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        """Allocate layer-major full and compressed buffers.

        Layout is ``[layers, 2, batch, heads, capacity, head_dim]`` so a
        layer can index ``full[layer]`` / ``compressed[layer]`` as writable
        ``(k, v)`` pairs without slicing the layer axis at every step.
        """
        if not isinstance(geometry, DualResolutionTokenGeometry):
            raise TypeError("geometry must be DualResolutionTokenGeometry")
        dimensions = {
            "batch_size": batch_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        shape = (int(num_layers), 2, int(batch_size), int(num_heads))
        tail = (int(head_dim),)
        self.geometry = geometry
        self.full = torch.zeros(
            (*shape, geometry.full_capacity, *tail),
            device=device,
            dtype=dtype,
        )
        self.compressed = torch.zeros(
            (*shape, geometry.compressed_capacity, *tail),
            device=device,
            dtype=dtype,
        )
        self.num_cached_chunks = 0
        self.is_cache_step = False

    def layer(self, index: int) -> tuple[Tensor, Tensor]:
        """Return writable ``(full, compressed)`` key/value tensors for a layer."""

        if isinstance(index, bool) or not 0 <= int(index) < int(self.full.shape[0]):
            raise IndexError("cache layer index is out of range")
        return self.full[int(index)], self.compressed[int(index)]

    def reset(self) -> None:
        """Zero both caches and clear the committed-chunk counter."""
        self.full.zero_()
        self.compressed.zero_()
        self.num_cached_chunks = 0
        self.is_cache_step = False


__all__ = [
    "DualResolutionKVCache",
    "DualResolutionTokenGeometry",
    "TemporalChunkPartition",
]
