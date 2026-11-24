# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic structural protocols for inference acceleration seams.

Acceleration passes inspect these protocols rather than matching family class
names, so any transformer family (Wan, HunyuanVideo, LTX, Cosmos, Sana,
Kandinsky) inherits attention routing, sequence parallelism, projection fusion,
and piecewise graph capture through one contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AttentionBackendConfigurable(Protocol):
    """Module exposing an explicit, request-scoped attention backend seam."""

    attention_backend: str | None

    def set_attention_backend(self, backend: str | None) -> None:
        """Bind an attention provider for subsequent forward calls."""
        ...


@runtime_checkable
class AttentionProcessorHost(Protocol):
    """Self-attention module supporting pluggable processor dispatch (SP/sparse)."""

    num_heads: int

    def get_processor(self) -> Any:
        """Return the currently installed attention processor."""
        ...

    def set_processor(self, processor: Any) -> None:
        """Replace the active attention processor."""
        ...


@runtime_checkable
class TransformerBlockStack(Protocol):
    """Transformer backbone exposing an ordered sequence of residual blocks."""

    blocks: Sequence[Any]


@runtime_checkable
class QKVProjectionModule(Protocol):
    """Attention module exposing separate ``q``, ``k``, ``v`` linear projections."""

    q: Any
    k: Any
    v: Any


__all__ = [
    "AttentionBackendConfigurable",
    "AttentionProcessorHost",
    "QKVProjectionModule",
    "TransformerBlockStack",
]
