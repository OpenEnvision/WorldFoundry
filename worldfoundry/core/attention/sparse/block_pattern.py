"""Chunk schedules and mask predicates for block-causal video attention.

Autoregressive video models emit latent frames in chunks::

    [ first_chunk_frames | chunk_frames | chunk_frames | ... ]

That one schedule has to agree across packed training masks, self-forcing
rollout replay, streaming inference, image-to-video conditioning, and
long-horizon extrapolation. When each network re-derives it, the definitions
drift. :class:`BlockPattern` is the single source of truth, and
:func:`build_mask_fn` turns a :class:`AttnMaskSpec` into the boolean predicate
that ``torch.nn.attention.flex_attention.create_block_mask`` (or any equivalent
mask builder) consumes.

Nothing here allocates tensors or binds an attention backend — the predicates
are pure index arithmetic, so callers stay free to compile them, materialize
them into range lists, or evaluate them on CPU for visualization and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch

MaskMode = Literal["none", "block_causal", "teacher_forcing"]

# ──────────────────────────────────────────────────────────────────────────
# Chunk schedule — one geometry for masks, rollout, and streaming offsets
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockPattern:
    """Chunk schedule for a block-causal latent video sequence.

    Blocks are counted in latent frames; every frame contributes
    ``frame_tokens`` tokens (typically ``latent_height * latent_width``). Block 0
    holds ``first_chunk_frames`` frames and every later block holds
    ``chunk_frames``, which is what lets image-to-video seed a single conditioning
    frame and then stream wider chunks.

    Attributes:
        frame_tokens: Tokens per latent frame.
        first_chunk_frames: Frames in block 0.
        chunk_frames: Frames in every block after block 0.
    """

    frame_tokens: int = 0
    first_chunk_frames: int = 1
    chunk_frames: int = 1

    def get_block_tokens(self, block_idx: int) -> int:
        """Return the token count of one block."""
        return self.first_chunk_frames * self.frame_tokens if block_idx == 0 else self.chunk_frames * self.frame_tokens

    def blocks_to_frames(self, num_blocks: int) -> int:
        """Return the frames covered by the first ``num_blocks`` blocks."""
        return 0 if num_blocks <= 0 else self.first_chunk_frames + (num_blocks - 1) * self.chunk_frames

    def blocks_to_tokens(self, num_blocks: int) -> int:
        """Return the tokens covered by the first ``num_blocks`` blocks."""
        return self.blocks_to_frames(num_blocks) * self.frame_tokens

    def block_size(self, block_idx: int) -> int:
        """Return the frame count of one block."""
        return self.first_chunk_frames if block_idx == 0 else self.chunk_frames

    def token_to_rel_block(self, token_idx: Any, use_first_block: bool) -> Any:
        """Map token indices to block indices, elementwise over a tensor.

        Args:
            token_idx: Token index tensor.
            use_first_block: Whether index 0 addresses the wider/narrower first
                block. Streaming queries at a nonzero block offset never do, so
                they fall back to uniform ``chunk_frames`` blocks.
        """
        frame_idx = token_idx // self.frame_tokens
        if not use_first_block:
            return frame_idx // self.chunk_frames
        return torch.where(
            frame_idx < self.first_chunk_frames,
            torch.zeros_like(frame_idx),
            1 + (frame_idx - self.first_chunk_frames) // self.chunk_frames,
        )

    def spans(
        self,
        num_blocks: int,
        block_offset: int,
        count_tokens: bool = False,
    ) -> tuple[list[tuple[int, int]], int]:
        """Return per-block ``[start, end)`` spans and the total length.

        Args:
            num_blocks: Number of blocks to lay out.
            block_offset: Global index of the first block. Only offset 0 uses
                ``first_chunk_frames``.
            count_tokens: Measure spans in tokens instead of frames.
        """
        spans: list[tuple[int, int]] = []
        cursor = 0
        multiplier = self.frame_tokens if count_tokens else 1
        for index in range(num_blocks):
            first = block_offset == 0 and index == 0
            size = (self.first_chunk_frames if first else self.chunk_frames) * multiplier
            spans.append((cursor, cursor + size))
            cursor += size
        return spans, cursor

    def block_bounds(self, q_block_offset: int, num_tokens: int) -> list[int]:
        """Return block boundaries in token coordinates, from 0 through ``num_tokens``."""
        first_tokens = self.first_chunk_frames * self.frame_tokens
        block_tokens = self.chunk_frames * self.frame_tokens

        bounds = [0]
        cursor = 0
        if q_block_offset == 0:
            cursor = min(num_tokens, cursor + first_tokens)
            bounds.append(cursor)
        while cursor < num_tokens:
            cursor = min(num_tokens, cursor + block_tokens)
            bounds.append(cursor)
        return bounds


# ──────────────────────────────────────────────────────────────────────────
# Mask spec + predicates — pure index arithmetic, no tensors or backends
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttnMaskSpec:
    """Declarative description of a block-structured attention mask.

    Modes:
        ``none``: full bidirectional attention.
        ``block_causal``: KV blocks up to and including the query block, with
            bidirectional attention inside a block.
        ``teacher_forcing``: packed ``[clean | noisy]`` sequence where clean
            queries attend causally over clean blocks, and noisy queries attend
            to their own noisy block plus strictly earlier clean blocks.

    Attributes:
        mode: Which mask to build.
        pattern: Chunk schedule the mask is expressed over.
        local_attn_blocks: Sliding window in blocks, including the current one.
            ``0`` disables the window.
        sink_blocks: Leading blocks that stay attendable regardless of the
            window — attention sinks.
        q_block_offset: Global index of the first query block. Nonzero during
            streaming, when earlier blocks live in the KV cache.
        clean_blocks: Number of clean blocks, required by ``teacher_forcing``.
    """

    mode: MaskMode = "none"
    pattern: BlockPattern | None = None
    local_attn_blocks: int = 0
    sink_blocks: int = 0
    q_block_offset: int = 0
    clean_blocks: int = 0


def _block_causal_mask_fn(spec: AttnMaskSpec, q_real: int, kv_real: int) -> tuple[Callable[..., Any], tuple]:
    """Build the block-causal predicate and its cache signature."""
    pattern = spec.pattern
    q_block_offset = spec.q_block_offset
    # `local_attn_blocks` counts the current block, so the lookback is one less.
    # The upstream sentinel keeps -1 meaningful: 0 disables windowing entirely.
    local_lookback = spec.local_attn_blocks - 1
    sink_blocks = spec.sink_blocks
    use_first_q = q_block_offset == 0

    def mask(batch, head, q_idx, kv_idx):
        """Allow KV blocks up to the query block, plus optional sinks/window."""
        del batch, head
        valid = (q_idx < q_real) & (kv_idx < kv_real)

        q_block = q_block_offset + pattern.token_to_rel_block(q_idx, use_first_block=use_first_q)
        kv_block = pattern.token_to_rel_block(kv_idx, use_first_block=True)

        allow = kv_block <= q_block
        if local_lookback >= 0:
            allow = allow & (kv_block >= q_block - local_lookback)
        if sink_blocks > 0:
            allow = allow | ((kv_block < sink_blocks) & (kv_block <= q_block))
        return valid & allow

    signature = ("block_causal", pattern, q_block_offset, local_lookback, sink_blocks, q_real, kv_real)
    return mask, signature


def _teacher_forcing_mask_fn(spec: AttnMaskSpec, q_real: int, kv_real: int) -> tuple[Callable[..., Any], tuple]:
    """Build the packed teacher-forcing predicate and its cache signature."""
    pattern = spec.pattern
    clean_blocks = spec.clean_blocks
    local_lookback = spec.local_attn_blocks - 1
    sink_blocks = spec.sink_blocks

    if clean_blocks <= 0:
        raise ValueError("AttnMaskSpec.clean_blocks must be positive for teacher_forcing")

    clean_len = pattern.blocks_to_tokens(clean_blocks)
    total_len = 2 * clean_len

    def mask(batch, head, q_idx, kv_idx):
        """Allow packed [clean | noisy] teacher-forcing; keep the diagonal open."""
        del batch, head
        valid = (q_idx < q_real) & (kv_idx < kv_real) & (q_idx < total_len) & (kv_idx < total_len)

        # The diagonal always stays open so no query is left with an empty row.
        eye = q_idx == kv_idx
        q_in_clean = q_idx < clean_len
        kv_in_clean = kv_idx < clean_len

        q_block_clean = pattern.token_to_rel_block(q_idx, use_first_block=True)
        q_block_noisy = pattern.token_to_rel_block(q_idx - clean_len, use_first_block=True)
        kv_block_clean = pattern.token_to_rel_block(kv_idx, use_first_block=True)
        kv_block_noisy = pattern.token_to_rel_block(kv_idx - clean_len, use_first_block=True)

        clean_allow = q_in_clean & kv_in_clean & (kv_block_clean <= q_block_clean)
        if local_lookback >= 0:
            clean_allow = clean_allow & (kv_block_clean >= q_block_clean - local_lookback)
        clean_allow = clean_allow | (q_in_clean & eye)

        # Noisy queries see strictly earlier clean blocks plus their own noisy block.
        noisy_to_clean = (~q_in_clean) & kv_in_clean & (kv_block_clean < q_block_noisy)
        if local_lookback >= 0:
            noisy_to_clean = noisy_to_clean & (kv_block_clean >= q_block_noisy - local_lookback)
        noisy_to_noisy = (~q_in_clean) & (~kv_in_clean) & (kv_block_noisy == q_block_noisy)
        noisy_allow = noisy_to_clean | noisy_to_noisy | ((~q_in_clean) & eye)

        if sink_blocks > 0:
            noisy_allow = noisy_allow | (
                (~q_in_clean) & kv_in_clean & (kv_block_clean < sink_blocks) & (kv_block_clean < q_block_noisy)
            )
            clean_allow = clean_allow | (
                q_in_clean & kv_in_clean & (kv_block_clean < sink_blocks) & (kv_block_clean < q_block_clean)
            )

        return valid & (eye | clean_allow | noisy_allow)

    signature = ("teacher_forcing", pattern, clean_blocks, local_lookback, sink_blocks, q_real, kv_real)
    return mask, signature


def build_mask_fn(spec: AttnMaskSpec, *, q_real: int, kv_real: int) -> tuple[Callable[..., Any], tuple]:
    """Return the mask predicate for ``spec`` plus a hashable cache signature.

    Args:
        spec: Mask description.
        q_real: Unpadded query length; indices at or beyond it are masked out.
        kv_real: Unpadded key/value length.

    Returns:
        The ``(batch, head, q_idx, kv_idx) -> bool`` predicate and a signature
        suitable for keying a compiled block-mask cache.

    Raises:
        ValueError: If ``spec.mode`` has no predicate (including ``"none"``,
            which callers should short-circuit to dense attention).
    """
    if spec.mode == "block_causal":
        return _block_causal_mask_fn(spec, q_real, kv_real)
    if spec.mode == "teacher_forcing":
        return _teacher_forcing_mask_fn(spec, q_real, kv_real)
    raise ValueError(f"Unknown attention mask mode: {spec.mode!r}")


__all__ = [
    "AttnMaskSpec",
    "BlockPattern",
    "MaskMode",
    "build_mask_fn",
]
