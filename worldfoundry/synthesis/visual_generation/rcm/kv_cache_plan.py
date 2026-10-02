"""Pure cache-allocation planning for Causal-rCM rollouts.

This sits outside the vendored runtime so cache-budget arithmetic is testable in
the base WorldFoundry environment, without importing the model, tokenizer, or
optional CUDA attention packages.
"""

from __future__ import annotations

from worldfoundry.core.attention.sparse.block_pattern import BlockPattern
from worldfoundry.core.attention.cache.kv_cache_policy import CachedBlock, CacheState, KVCachePolicy, SlidingWindowPolicy


def make_kv_cache_plan(
    pattern: BlockPattern,
    num_blocks: int,
    *,
    policy_name: str,
    window_blocks: int,
    sink_blocks: int,
) -> tuple[KVCachePolicy | None, int]:
    """Return ``(policy, capacity_frames)`` for one causal rollout.

    The upstream cache preallocates the entire rollout, even if a caller later
    evicts tokens. For a bounded-memory window we reserve only the largest
    retained prefix plus one in-flight chunk: ``READONLY`` writes that current
    chunk transiently after the prefix. ``keep_all`` returns ``None`` and the
    full-horizon capacity, preserving upstream allocation and cache semantics.

    RoPE-remapping policies are intentionally not exposed here. The causal
    entrypoint caches post-RoPE keys, so changing their temporal indices needs
    the dedicated pre-RoPE extrapolation runtime rather than a silent mismatch.
    """
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

    normalized = policy_name.strip().lower().replace("-", "_")
    total_frames = pattern.blocks_to_frames(num_blocks)
    if normalized == "keep_all":
        return None, total_frames
    if normalized != "sliding_window":
        raise ValueError(
            f"Unsupported --kv_cache_policy {policy_name!r}. "
            "Causal-rCM supports keep_all and sliding_window."
        )

    policy = SlidingWindowPolicy(window_blocks=window_blocks, sink_blocks=sink_blocks)
    retained: list[CachedBlock] = []
    max_frames = 0
    for block_idx in range(num_blocks):
        current = CachedBlock(
            block_idx=block_idx,
            frame_start=pattern.blocks_to_frames(block_idx),
            frame_count=pattern.block_size(block_idx),
        )
        # ``write_transient`` needs one additional current chunk beyond the
        # committed cache retained by the previous iteration.
        max_frames = max(max_frames, sum(block.frame_count for block in retained) + current.frame_count)
        state = CacheState(
            blocks=tuple([*retained, current]),
            frame_tokens=pattern.frame_tokens,
            current_block_idx=block_idx,
        )
        decision = policy.decide(state)
        if decision.kept_blocks is None:
            retained.append(current)
        else:
            committed = [*retained, current]
            retained = [committed[index] for index in decision.kept_blocks]

    return policy, max_frames


__all__ = ["make_kv_cache_plan"]
