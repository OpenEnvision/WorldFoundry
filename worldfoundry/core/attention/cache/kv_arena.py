"""Inference-only, contiguous storage for [static | current | memory] K/V.

The owner supplies already-positioned keys: this module never applies RoPE,
evicts tokens, or decides when a persistent segment is stale. Rebuild the
arena when a persistent segment changes. Returned tensors alias writable
storage and are valid until the next stage; consume them on the same CUDA
stream. Each session/layer needs its own arena.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

KVPair = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class KVSegmentLayout:
    static_tokens: int
    current_tokens: int
    memory_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("static_tokens", "current_tokens", "memory_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "current_tokens" else 0):
                raise ValueError(f"{name} must be an integer >= {1 if name == 'current_tokens' else 0}")

    @property
    def total_tokens(self) -> int:
        return self.static_tokens + self.current_tokens + self.memory_tokens


def _validate_pair(pair: KVPair, name: str) -> None:
    key, value = pair
    if key.ndim != 4 or key.shape != value.shape:
        raise ValueError(f"{name} K/V must have matching four-dimensional shapes")
    if key.device != value.device or key.dtype != value.dtype:
        raise ValueError(f"{name} K/V must have matching dtype and device")
    if key.layout != torch.strided or value.layout != torch.strided:
        raise ValueError(f"{name} K/V must be strided tensors")


class KVSegmentArena:
    """Fixed-size K/V storage. ``seq_dim`` is 1 for BSHD or 2 for BHSD.

    No unused capacity is exposed to attention. Persistent views must retain
    their exact storage, offset, stride and dtype; a matching shape alone is
    insufficient. Allocation and writes require gradients to be disabled.
    """

    def __init__(
        self,
        static: KVPair,
        *,
        current_tokens: int,
        memory: KVPair | None = None,
        seq_dim: int = 2,
    ) -> None:
        if torch.is_grad_enabled():
            raise RuntimeError("KV arena requires no_grad() or inference_mode()")
        if type(seq_dim) is not int or seq_dim not in (1, 2):
            raise ValueError("seq_dim must be 1 (BSHD) or 2 (BHSD)")
        _validate_pair(static, "static")
        if static[0].is_meta:
            raise ValueError("KV arena requires materialized tensors")
        self.seq_dim = seq_dim
        if memory is None:
            memory = tuple(t.narrow(seq_dim, 0, 0) for t in static)
        _validate_pair(memory, "memory")
        shape = list(static[0].shape)
        expected_memory = list(shape)
        expected_memory[seq_dim] = memory[0].shape[seq_dim]
        if list(memory[0].shape) != expected_memory:
            raise ValueError("static and memory K/V must agree outside the sequence dimension")
        if memory[0].device != static[0].device or memory[0].dtype != static[0].dtype:
            raise ValueError("static and memory K/V must agree on dtype and device")
        self.layout = KVSegmentLayout(shape[seq_dim], current_tokens, memory[0].shape[seq_dim])
        shape[seq_dim] = self.layout.total_tokens
        self._kv = tuple(t.new_empty(shape) for t in static)
        self._device = static[0].device
        self._stream = self._current_stream()
        self.static = self._segment(0, self.layout.static_tokens)
        self.memory = self._segment(self.layout.static_tokens + current_tokens, self.layout.memory_tokens)
        self._current = self._segment(self.layout.static_tokens, current_tokens)
        for destination, source in zip(self.static + self.memory, static + memory):
            destination.copy_(source)
        self.stage_count = 0

    def _current_stream(self) -> int | None:
        return torch.cuda.current_stream(self._device).cuda_stream if self._device.type == "cuda" else None

    def _segment(self, start: int, length: int) -> KVPair:
        return tuple(t.narrow(self.seq_dim, start, length) for t in self._kv)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._kv)

    def validate_persistent(self, static: KVPair, memory: KVPair | None = None) -> None:
        """Reject detached copies, wrong offsets, or reinterpreted persistent views."""
        for name, supplied, expected in (
            ("static", static, self._segment(0, self.layout.static_tokens)),
            (
                "memory",
                memory,
                self._segment(self.layout.static_tokens + self.layout.current_tokens, self.layout.memory_tokens),
            ),
        ):
            if supplied is None:
                if name == "memory" and self.layout.memory_tokens == 0:
                    continue
                raise ValueError(f"{name} views are required")
            _validate_pair(supplied, name)
            for actual, wanted in zip(supplied, expected):
                if (
                    actual.device != wanted.device
                    or actual.dtype != wanted.dtype
                    or actual.shape != wanted.shape
                    or actual.stride() != wanted.stride()
                    or actual.storage_offset() != wanted.storage_offset()
                    or actual.untyped_storage().data_ptr() != wanted.untyped_storage().data_ptr()
                ):
                    raise ValueError(f"{name} K/V must alias the declared arena segment")

    def stage_current(self, current: KVPair) -> KVPair:
        """Replace the current segment, leaving static/memory tokens untouched."""
        if torch.is_grad_enabled():
            raise RuntimeError("KV arena requires no_grad() or inference_mode()")
        if self._current_stream() != self._stream:
            raise RuntimeError("KV arena must be staged and consumed on its owning CUDA stream")
        _validate_pair(current, "current")
        expected = self._current[0]
        if (
            current[0].shape != expected.shape
            or current[0].dtype != expected.dtype
            or current[0].device != expected.device
        ):
            raise ValueError("current K/V shape, dtype and device must match the arena segment")
        storage = {t.untyped_storage().data_ptr() for t in self._kv}
        if any(t.untyped_storage().data_ptr() in storage for t in current):
            raise ValueError("current K/V must not alias arena storage")
        for destination, source in zip(self._current, current):
            destination.copy_(source)
        self.stage_count += 1
        return self._kv


__all__ = ["KVPair", "KVSegmentArena", "KVSegmentLayout"]
