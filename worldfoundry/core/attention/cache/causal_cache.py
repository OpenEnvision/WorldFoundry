"""Shared raw-cache ABI for in-tree causal video transformer graphs.

Some causal Wan graphs predate :class:`BlockKVCache` and update four tensors
inside each layer directly.  Replacing that ABI at the training boundary would
change the model forward.  This module instead owns allocation, geometry,
overwrite validation, and logical block advancement while exposing the exact
raw dictionaries those graphs consume.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from operator import index as integer_index

import torch
from torch import Tensor

from worldfoundry.core.attention.sparse.block_pattern import BlockPattern
from worldfoundry.core.attention.cache.kv_cache_policy import CachedBlock, CacheState

# ──────────────────────────────────────────────────────────────────────────
# Geometry — validate the raw ABI so graphs cannot drift from BlockPattern
# ──────────────────────────────────────────────────────────────────────────


def _integer(value: object, *, name: str) -> int:
    """Coerce ``value`` to an integer index, rejecting bools.

    ``bool`` is a subclass of ``int``; accepting it would silently treat
    True/False as 1/0 and corrupt block geometry.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(integer_index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


@dataclass(frozen=True, slots=True)
class CausalVideoCacheGeometry:
    """Tensor and block geometry for one causal transformer cache."""

    batch_size: int
    total_frames: int
    frame_tokens: int
    frames_per_block: int
    num_layers: int
    num_heads: int
    head_dim: int
    local_attention_frames: int = -1
    sink_frames: int = 0

    def __post_init__(self) -> None:
        """Validate positive geometry and local-window vs sink invariants."""
        for name in (
            "batch_size",
            "total_frames",
            "frame_tokens",
            "frames_per_block",
            "num_layers",
            "num_heads",
            "head_dim",
        ):
            value = _integer(getattr(self, name), name=name)
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if self.total_frames % self.frames_per_block:
            raise ValueError("total_frames must be divisible by frames_per_block")
        local = _integer(
            self.local_attention_frames,
            name="local_attention_frames",
        )
        if local == 0 or local < -1:
            raise ValueError("local_attention_frames must be -1 or a positive integer")
        if local > 0 and local < self.frames_per_block:
            raise ValueError("local_attention_frames cannot be smaller than one block")
        sink = _integer(self.sink_frames, name="sink_frames")
        if sink < 0:
            raise ValueError("sink_frames must be a non-negative integer")
        if local > 0 and sink >= local:
            raise ValueError("sink_frames must be smaller than local_attention_frames")
        object.__setattr__(self, "local_attention_frames", local)
        object.__setattr__(self, "sink_frames", sink)

    @property
    def pattern(self) -> BlockPattern:
        """Uniform block pattern implied by ``frames_per_block``."""
        return BlockPattern(
            frame_tokens=self.frame_tokens,
            first_chunk_frames=self.frames_per_block,
            chunk_frames=self.frames_per_block,
        )

    @property
    def sequence_tokens(self) -> int:
        """Logical sequence length in tokens (``total_frames * frame_tokens``)."""
        return self.total_frames * self.frame_tokens

    @property
    def capacity_frames(self) -> int:
        """Frames retained in the rolling buffer; ``total_frames`` when unbounded."""
        return self.total_frames if self.local_attention_frames == -1 else self.local_attention_frames

    @property
    def capacity_tokens(self) -> int:
        """Token width of the allocated ``k`` / ``v`` buffers."""
        return self.capacity_frames * self.frame_tokens

    @property
    def num_blocks(self) -> int:
        """Number of equal-sized blocks in the configured sequence."""
        return self.total_frames // self.frames_per_block


# ──────────────────────────────────────────────────────────────────────────
# Raw ABI — allocate / recover the dictionaries causal Wan graphs consume
# ──────────────────────────────────────────────────────────────────────────


def allocate_causal_video_cache(
    geometry: CausalVideoCacheGeometry,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    """Allocate the raw per-layer ABI and its block lifecycle state."""

    if not isinstance(geometry, CausalVideoCacheGeometry):
        raise TypeError("geometry must be CausalVideoCacheGeometry")
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("causal video cache dtype must be floating point")
    resolved_device = torch.device(device)

    def layer_cache() -> dict[str, Tensor]:
        """Allocate one layer's ``k``/``v`` buffers and end-index scalars."""
        shape = (
            geometry.batch_size,
            geometry.capacity_tokens,
            geometry.num_heads,
            geometry.head_dim,
        )
        return {
            "k": torch.zeros(shape, device=resolved_device, dtype=dtype),
            "v": torch.zeros(shape, device=resolved_device, dtype=dtype),
            "global_end_index": torch.zeros(
                1,
                device=resolved_device,
                dtype=torch.long,
            ),
            "local_end_index": torch.zeros(
                1,
                device=resolved_device,
                dtype=torch.long,
            ),
        }

    return {
        "kv_cache": [layer_cache() for _ in range(geometry.num_layers)],
        "crossattn_cache": [{"is_init": False} for _ in range(geometry.num_layers)],
        "batch_size": geometry.batch_size,
        "total_frames": geometry.total_frames,
        "frame_tokens": geometry.frame_tokens,
        "frames_per_block": geometry.frames_per_block,
        "num_layers": geometry.num_layers,
        "num_heads": geometry.num_heads,
        "head_dim": geometry.head_dim,
        "local_attention_frames": geometry.local_attention_frames,
        "sink_frames": geometry.sink_frames,
        "sequence_length": geometry.sequence_tokens,
        "active_block": -1,
        "committed_blocks": 0,
        "committed_frames": 0,
    }


def causal_video_cache_geometry(cache: Mapping[str, object]) -> CausalVideoCacheGeometry:
    """Recover and validate geometry stored in a raw cache payload."""

    if not isinstance(cache, Mapping):
        raise TypeError("causal video cache must be a mapping")
    geometry = CausalVideoCacheGeometry(
        batch_size=cache.get("batch_size", 0),
        total_frames=cache.get("total_frames", 0),
        frame_tokens=cache.get("frame_tokens", 0),
        frames_per_block=cache.get("frames_per_block", 0),
        num_layers=cache.get("num_layers", 0),
        num_heads=cache.get("num_heads", 0),
        head_dim=cache.get("head_dim", 0),
        local_attention_frames=cache.get("local_attention_frames", 0),
        sink_frames=cache.get("sink_frames", -1),
    )
    if int(cache.get("sequence_length", 0)) != geometry.sequence_tokens:
        raise ValueError("causal video cache sequence length differs from its geometry")
    kv_cache = cache.get("kv_cache")
    crossattn_cache = cache.get("crossattn_cache")
    if not isinstance(kv_cache, list) or len(kv_cache) != geometry.num_layers:
        raise ValueError("causal video cache has the wrong self-attention layer count")
    if not isinstance(crossattn_cache, list) or len(crossattn_cache) != geometry.num_layers:
        raise ValueError("causal video cache has the wrong cross-attention layer count")
    expected_shape = (
        geometry.batch_size,
        geometry.capacity_tokens,
        geometry.num_heads,
        geometry.head_dim,
    )
    cache_device: torch.device | None = None
    cache_dtype: torch.dtype | None = None
    for index, layer in enumerate(kv_cache):
        if not isinstance(layer, Mapping) or set(layer) != {
            "k",
            "v",
            "global_end_index",
            "local_end_index",
        }:
            raise ValueError(f"causal video cache layer {index} fields differ")
        k = layer["k"]
        v = layer["v"]
        global_end = layer["global_end_index"]
        local_end = layer["local_end_index"]
        if not isinstance(k, Tensor) or not isinstance(v, Tensor):
            raise TypeError(f"causal video cache layer {index} k/v must be tensors")
        if tuple(k.shape) != expected_shape or tuple(v.shape) != expected_shape:
            raise ValueError(f"causal video cache layer {index} tensor shape differs")
        if not k.is_floating_point() or k.dtype != v.dtype or k.device != v.device:
            raise ValueError(f"causal video cache layer {index} k/v storage differs")
        if (
            not isinstance(global_end, Tensor)
            or not isinstance(local_end, Tensor)
            or global_end.shape != (1,)
            or local_end.shape != (1,)
            or global_end.dtype != torch.long
            or local_end.dtype != torch.long
        ):
            raise ValueError(f"causal video cache layer {index} index tensors differ")
        if global_end.device != k.device or local_end.device != k.device:
            raise ValueError(f"causal video cache layer {index} indices are on another device")
        if cache_device is None:
            cache_device = k.device
            cache_dtype = k.dtype
        elif k.device != cache_device or k.dtype != cache_dtype:
            raise ValueError("causal video cache layers disagree on device or dtype")
    for index, layer in enumerate(crossattn_cache):
        if not isinstance(layer, Mapping) or not isinstance(layer.get("is_init"), bool):
            raise ValueError(f"causal video cross-attention cache layer {index} differs")
        if bool(layer["is_init"]):
            cross_k = layer.get("k")
            cross_v = layer.get("v")
            if not isinstance(cross_k, Tensor) or not isinstance(cross_v, Tensor):
                raise ValueError(f"initialized causal video cross-attention cache layer {index} has no k/v")
            if (
                cross_k.ndim != 4
                or cross_k.shape != cross_v.shape
                or int(cross_k.shape[0]) != geometry.batch_size
                or int(cross_k.shape[1]) <= 0
                or tuple(cross_k.shape[2:]) != (geometry.num_heads, geometry.head_dim)
                or cross_k.dtype != cache_dtype
                or cross_k.device != cache_device
                or cross_v.dtype != cache_dtype
                or cross_v.device != cache_device
            ):
                raise ValueError(f"initialized causal video cross-attention cache layer {index} tensor layout differs")
    return geometry


# ──────────────────────────────────────────────────────────────────────────
# Block lifecycle — append-or-overwrite must stay ordered and audited
# ──────────────────────────────────────────────────────────────────────────


def _validate_block_span(
    geometry: CausalVideoCacheGeometry,
    *,
    block_index: object,
    start_frame: object,
    frame_count: object,
) -> tuple[int, int, int]:
    """Require ``block_index``/start/count to match :attr:`geometry.pattern`.

    Raises:
        ValueError: The block is out of range or the span disagrees with
            the configured first/later chunk sizes.
    """
    resolved_block = _integer(block_index, name="block_index")
    resolved_start = _integer(start_frame, name="start_frame")
    resolved_count = _integer(frame_count, name="frame_count")
    if not 0 <= resolved_block < geometry.num_blocks:
        raise ValueError("block_index falls outside the configured sequence")
    expected_start = geometry.pattern.blocks_to_frames(resolved_block)
    expected_count = geometry.pattern.block_size(resolved_block)
    if resolved_start != expected_start:
        raise ValueError("start_frame differs from the configured block pattern")
    if resolved_count != expected_count:
        raise ValueError("frame_count differs from the configured block pattern")
    return resolved_block, resolved_start, resolved_count


def begin_causal_video_cache_block(
    cache: MutableMapping[str, object],
    *,
    block_index: int,
    start_frame: int,
    frame_count: int,
) -> None:
    """Validate append-or-overwrite ordering before a graph call."""

    geometry = causal_video_cache_geometry(cache)
    block_index, start_frame, frame_count = _validate_block_span(
        geometry,
        block_index=block_index,
        start_frame=start_frame,
        frame_count=frame_count,
    )
    committed = int(cache.get("committed_blocks", -1))
    committed_frames = int(cache.get("committed_frames", -1))
    active = int(cache.get("active_block", -2))
    if committed != block_index or committed_frames != start_frame:
        raise RuntimeError("causal video cache call skipped or revisited a committed block")
    if active not in {block_index - 1, block_index}:
        raise RuntimeError("causal video cache is not positioned at the requested block")
    cache["active_block"] = block_index


def finish_causal_video_cache_call(
    cache: Mapping[str, object],
    *,
    start_frame: int,
    frame_count: int,
) -> None:
    """Audit that every graph layer overwrote/appended the requested span."""

    geometry = causal_video_cache_geometry(cache)
    resolved_start = _integer(start_frame, name="start_frame")
    resolved_count = _integer(frame_count, name="frame_count")
    if resolved_start < 0 or resolved_count <= 0:
        raise ValueError("cache span must contain non-negative start and positive frame count")
    if resolved_start + resolved_count > geometry.total_frames:
        raise ValueError("cache span falls outside the configured sequence")
    expected_global_end = (resolved_start + resolved_count) * geometry.frame_tokens
    expected_local_end = min(expected_global_end, geometry.capacity_tokens)
    kv_cache = cache["kv_cache"]
    assert isinstance(kv_cache, list)
    for index, layer in enumerate(kv_cache):
        assert isinstance(layer, Mapping)
        global_index = layer["global_end_index"]
        local_index = layer["local_end_index"]
        assert isinstance(global_index, Tensor) and isinstance(local_index, Tensor)
        global_end = int(global_index.item())
        local_end = int(local_index.item())
        if global_end != expected_global_end:
            raise RuntimeError(
                f"causal video cache layer {index} ended at token {global_end}, expected {expected_global_end}"
            )
        if local_end != expected_local_end:
            raise RuntimeError(
                f"causal video cache layer {index} local end is {local_end}, expected {expected_local_end}"
            )


def commit_causal_video_cache_block(
    cache: MutableMapping[str, object],
    *,
    block_index: int,
    start_frame: int,
    frame_count: int,
) -> CacheState:
    """Advance logical block state after the detached clean overwrite."""

    geometry = causal_video_cache_geometry(cache)
    block_index, start_frame, frame_count = _validate_block_span(
        geometry,
        block_index=block_index,
        start_frame=start_frame,
        frame_count=frame_count,
    )
    if int(cache.get("active_block", -1)) != block_index:
        raise RuntimeError("causal video cache commit has no active block")
    if int(cache.get("committed_blocks", -1)) != block_index:
        raise RuntimeError("causal video cache commit is out of order")
    if int(cache.get("committed_frames", -1)) != start_frame:
        raise RuntimeError("causal video cache commit frame offset differs")
    finish_causal_video_cache_call(
        cache,
        start_frame=start_frame,
        frame_count=frame_count,
    )
    cache["committed_blocks"] = block_index + 1
    cache["committed_frames"] = start_frame + frame_count
    return causal_video_cache_state(cache, geometry=geometry)


def causal_video_cache_state(
    cache: Mapping[str, object],
    *,
    geometry: CausalVideoCacheGeometry | None = None,
) -> CacheState:
    """Expose committed raw-cache progress through the core policy contract."""

    resolved = geometry or causal_video_cache_geometry(cache)
    committed = int(cache.get("committed_blocks", -1))
    committed_frames = int(cache.get("committed_frames", -1))
    if not 0 <= committed <= resolved.num_blocks:
        raise RuntimeError("causal video cache committed block count is invalid")
    if committed_frames != resolved.pattern.blocks_to_frames(committed):
        raise RuntimeError("causal video cache committed frame count differs from its block pattern")
    blocks: list[CachedBlock] = []
    cursor = 0
    for block_index in range(committed):
        frame_count = resolved.pattern.block_size(block_index)
        blocks.append(
            CachedBlock(
                block_idx=block_index,
                frame_start=cursor,
                frame_count=frame_count,
            )
        )
        cursor += frame_count
    return CacheState(
        blocks=tuple(blocks),
        frame_tokens=resolved.frame_tokens,
        current_block_idx=committed - 1,
    )


__all__ = [
    "CausalVideoCacheGeometry",
    "allocate_causal_video_cache",
    "begin_causal_video_cache_block",
    "causal_video_cache_geometry",
    "causal_video_cache_state",
    "commit_causal_video_cache_block",
    "finish_causal_video_cache_call",
]
