"""Policy-driven KV cache for the vendored Causal-rCM runtime.

The upstream cache grows without bound: every committed chunk stays resident, so
a long rollout is limited by cache memory rather than by the model. This subclass
routes each committed chunk through a
:class:`~worldfoundry.core.attention.cache.kv_cache_policy.KVCachePolicy`, which decides
what survives, without touching the sampler or the attention kernels.

**The invariant that makes this safe.** The upstream runtime addresses the cache
by *global block cursor*: ``block_range`` is the index of the chunk being
generated, and the plain cache satisfies ``block_range == len(_cum_ends)`` because
it never drops anything. Eviction breaks that identity — the cursor keeps
advancing while the retained chunk count does not. This class therefore keeps the
two quantities separate:

* :attr:`committed_blocks` counts chunks ever appended, so the caller's cursor
  still matches and the runtime's ``APPEND`` assertion holds.
* :meth:`get` and :meth:`get_prefix_end` clamp a cursor-valued ``block_range`` to
  what is actually retained.

The clamp is exact rather than defensive: every policy here evicts only whole
chunks that strictly precede the current one, so "the first ``block_range``
chunks" and "everything retained" denote the same tokens once eviction has
happened. A policy that dropped part of the *current* chunk would violate that,
which is why :meth:`append` verifies it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from worldfoundry.core.attention.cache.kv_cache_policy import (
    CachedBlock,
    CacheDecision,
    CacheState,
    KeepAllPolicy,
    KVCachePolicy,
)

from .rcm_runtime.utils.kv_cache import KVCache


class PolicyKVCache(KVCache):
    """Upstream KV cache with a pluggable eviction policy.

    Args:
        max_len: Buffer capacity in tokens, as for :class:`KVCache`.
        frame_tokens: Tokens per latent frame; the policy reasons in frames.
        policy: Eviction policy. Defaults to keeping everything, which makes this
            class behave exactly like the upstream cache.
    """

    def __init__(self, max_len: int, *, frame_tokens: int, policy: KVCachePolicy | None = None) -> None:
        super().__init__(max_len)
        if frame_tokens < 1:
            raise ValueError(f"frame_tokens must be >= 1, got {frame_tokens}")
        self.frame_tokens = frame_tokens
        self.policy = policy if policy is not None else KeepAllPolicy()
        self._committed_blocks = 0
        self._blocks: list[CachedBlock] = []
        self._next_frame = 0
        self.last_decision: CacheDecision | None = None

    # ── Cursor bookkeeping ───────────────────────────────────

    @property
    def committed_blocks(self) -> int:
        """Chunks ever appended, which is what the caller's block cursor counts."""
        return self._committed_blocks

    @property
    def retained_blocks(self) -> tuple[CachedBlock, ...]:
        """Chunks still resident, oldest first."""
        return tuple(self._blocks)

    def _clamp(self, block_range: Optional[int]) -> Optional[int]:
        """Reinterpret a cursor-valued ``block_range`` against retained chunks."""
        if block_range is None:
            return None
        if block_range < 0:
            raise ValueError(f"block_range must be >= 0, got {block_range}")
        return min(block_range, len(self._cum_ends))

    # ── Overridden cache surface ─────────────────────────────

    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[int, int]:
        """Commit a chunk, then apply the policy.

        Returns the span the chunk occupied *before* eviction, matching the
        upstream contract; callers use it only for bookkeeping.
        """
        tokens = k.shape[1]
        if tokens % self.frame_tokens:
            raise ValueError(f"chunk of {tokens} tokens is not a multiple of frame_tokens ({self.frame_tokens})")

        span = super().append(k, v)
        frames = tokens // self.frame_tokens
        self._blocks.append(
            CachedBlock(block_idx=self._committed_blocks, frame_start=self._next_frame, frame_count=frames)
        )
        self._committed_blocks += 1
        self._next_frame += frames

        decision = self.policy.decide(self.state())
        self.last_decision = decision
        if decision.evicts:
            self._apply(decision)
        return span

    def state(self) -> CacheState:
        """Snapshot handed to the policy."""
        return CacheState(
            blocks=tuple(self._blocks),
            frame_tokens=self.frame_tokens,
            current_block_idx=self._committed_blocks - 1,
        )

    def _apply(self, decision: CacheDecision) -> None:
        """Compact the buffer and rebuild chunk bookkeeping."""
        kept = decision.kept_blocks
        if kept is None:
            raise ValueError(
                "PolicyKVCache requires a policy that reports kept_blocks; token-level "
                "eviction would desynchronize the runtime's block cursor"
            )
        if kept and kept[-1] != len(self._blocks) - 1:
            raise ValueError("a policy must retain the chunk it was just given; the current block is still being read")

        self.compact_(decision.keep_indices)

        retained = [self._blocks[position] for position in kept]
        self._blocks = retained
        # compact_ collapses the chunk list into one entry, so restore per-chunk
        # boundaries: the runtime slices the cache by chunk, not by token.
        self._chunk_lens = [block.frame_count * self.frame_tokens for block in retained]
        cumulative, ends = 0, []
        for length in self._chunk_lens:
            cumulative += length
            ends.append(cumulative)
        self._cum_ends = ends
        self.cur = cumulative

    def get(self, block_range: Optional[int] = None):
        """Return a retained prefix, reinterpreting a cursor-valued range."""
        return super().get(self._clamp(block_range))

    def get_prefix_end(self, block_range: Optional[int] = None) -> int:
        """Return the retained prefix end for a cursor-valued range."""
        return super().get_prefix_end(self._clamp(block_range))

    def reset(self, free_buffers: bool = True) -> None:
        """Reset both the buffer and the block bookkeeping."""
        super().reset(free_buffers=free_buffers)
        self._committed_blocks = 0
        self._blocks = []
        self._next_frame = 0
        self.last_decision = None


def policy_cache_factory(*, frame_tokens: int, policy: KVCachePolicy | None = None):
    """Return a factory for :meth:`WanModel.allocate_kv_caches`.

    Every layer gets its own cache but they must evict identically, so they share
    one policy instance. All structural policies are frozen dataclasses with no
    per-call state, which makes sharing safe.

    Example::

        factory = policy_cache_factory(frame_tokens=h * w, policy=SlidingWindowPolicy(window_blocks=6))
        kv_caches = net.allocate_kv_caches(max_len=total_tokens, cache_factory=factory)
    """
    shared = policy if policy is not None else KeepAllPolicy()

    def build(max_len: int) -> PolicyKVCache:
        return PolicyKVCache(max_len, frame_tokens=frame_tokens, policy=shared)

    return build


def allocate_policy_kv_caches(
    num_layers: int,
    max_len: int,
    *,
    frame_tokens: int,
    policy: KVCachePolicy | None = None,
) -> list[PolicyKVCache]:
    """Build one policy-driven cache per transformer layer."""
    factory = policy_cache_factory(frame_tokens=frame_tokens, policy=policy)
    return [factory(max_len) for _ in range(num_layers)]


__all__ = ["PolicyKVCache", "allocate_policy_kv_caches", "policy_cache_factory"]
