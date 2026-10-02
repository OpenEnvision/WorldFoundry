import pytest
import torch

from worldfoundry.core.attention.kv_cache_policy import (
    POLICY_REGISTRY,
    BankedSinkPolicy,
    BlockRelativeRoPEPolicy,
    CachedBlock,
    CacheState,
    KeepAllPolicy,
    KVCachePolicy,
    SlidingWindowPolicy,
    build_policy,
)
from worldfoundry.core.attention.kv_quantization import KVQuantConfig
from worldfoundry.core.attention.kvcache import BlockKVCache, CompactingKVCache

FRAME_TOKENS = 4
FRAMES_PER_CHUNK = 3
CHUNK_TOKENS = FRAME_TOKENS * FRAMES_PER_CHUNK


def _state(num_blocks: int, *, frames: int = FRAMES_PER_CHUNK, frame_tokens: int = FRAME_TOKENS) -> CacheState:
    """Build a cache state holding ``num_blocks`` equal-sized chunks."""
    blocks = tuple(
        CachedBlock(block_idx=index, frame_start=index * frames, frame_count=frames) for index in range(num_blocks)
    )
    return CacheState(blocks=blocks, frame_tokens=frame_tokens, current_block_idx=max(num_blocks - 1, 0))


def _chunk(dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, CHUNK_TOKENS, 2, 64, dtype=dtype),
        torch.randn(1, CHUNK_TOKENS, 2, 64, dtype=dtype),
    )


# ── CacheState geometry ──────────────────────────────────────


def test_cache_state_reports_frames_and_token_spans() -> None:
    state = _state(3)

    assert state.cached_frames == 9
    assert state.cached_tokens == 36
    assert state.token_span(0) == (0, 12)
    assert state.token_span(2) == (24, 36)


def test_cached_block_frame_end() -> None:
    assert CachedBlock(block_idx=2, frame_start=6, frame_count=3).frame_end == 9


# ── Policies ─────────────────────────────────────────────────


def test_all_registered_policies_satisfy_the_protocol() -> None:
    for name, policy_cls in POLICY_REGISTRY.items():
        assert isinstance(policy_cls(), KVCachePolicy), f"{name} does not implement the policy protocol"


def test_keep_all_never_evicts() -> None:
    decision = KeepAllPolicy().decide(_state(50))

    assert not decision.evicts
    assert not decision.remaps_positions


def test_sliding_window_is_a_noop_below_budget() -> None:
    assert not SlidingWindowPolicy(window_blocks=4).decide(_state(4)).evicts


def test_sliding_window_keeps_sinks_and_recent_blocks() -> None:
    decision = SlidingWindowPolicy(window_blocks=4, sink_blocks=1).decide(_state(10))

    assert decision.kept_blocks == (0, 7, 8, 9)
    # Survivors keep their original positions, so a gap is visible to RoPE.
    assert not decision.remaps_positions
    assert decision.keep_indices.tolist() == list(range(0, 12)) + list(range(84, 120))


def test_sliding_window_without_sinks_is_pure_fifo() -> None:
    decision = SlidingWindowPolicy(window_blocks=3, sink_blocks=0).decide(_state(9))

    assert decision.kept_blocks == (6, 7, 8)


def test_banked_sink_pins_the_bank_and_closes_the_gap() -> None:
    decision = BankedSinkPolicy(bank_blocks=2, recent_blocks=2).decide(_state(10))

    assert decision.kept_blocks == (0, 1, 8, 9)
    assert decision.remaps_positions
    # Four surviving blocks of 3 frames renumber onto a contiguous 0..11 range.
    assert sorted(set(decision.frame_positions.tolist())) == list(range(12))


def test_banked_sink_reports_its_residency_bound() -> None:
    assert BankedSinkPolicy(bank_blocks=3, recent_blocks=2).resident_blocks == 5


def test_block_relative_rope_pins_the_newest_block() -> None:
    policy = BlockRelativeRoPEPolicy(cache_blocks=4, sink_blocks=1, frame_limit=21)

    decision = policy.decide(_state(10))

    assert decision.kept_blocks == (0, 7, 8, 9)
    positions = decision.frame_positions
    newest = positions[-CHUNK_TOKENS:]
    assert newest.min().item() == 21
    assert positions.max().item() == 21 + FRAMES_PER_CHUNK - 1


def test_block_relative_rope_clamps_distant_blocks_to_zero() -> None:
    # A short frame limit cannot fit every surviving block, so the oldest ones
    # collapse to frame 0 instead of extrapolating RoPE below the trained range.
    policy = BlockRelativeRoPEPolicy(cache_blocks=6, sink_blocks=1, frame_limit=4)

    positions = policy.decide(_state(20)).frame_positions

    assert positions.min().item() == 0
    assert positions.max().item() == 4 + FRAMES_PER_CHUNK - 1


def test_block_relative_rope_remaps_even_below_budget() -> None:
    """Positions must be pinned from the first chunk, not only after eviction."""
    decision = BlockRelativeRoPEPolicy(cache_blocks=6, frame_limit=21).decide(_state(2))

    assert not decision.evicts
    assert decision.remaps_positions


def test_policies_reject_impossible_budgets() -> None:
    with pytest.raises(ValueError, match="window_blocks must be >= 1"):
        SlidingWindowPolicy(window_blocks=0)
    with pytest.raises(ValueError, match="must be <"):
        SlidingWindowPolicy(window_blocks=2, sink_blocks=2)
    with pytest.raises(ValueError, match="bank_blocks must be >= 1"):
        BankedSinkPolicy(bank_blocks=0)
    with pytest.raises(ValueError, match="frame_limit must be >= 2"):
        BlockRelativeRoPEPolicy(frame_limit=1)


def test_keep_indices_are_ascending_and_in_range() -> None:
    state = _state(12)
    for policy in (
        SlidingWindowPolicy(window_blocks=4, sink_blocks=1),
        BankedSinkPolicy(bank_blocks=2, recent_blocks=2),
        BlockRelativeRoPEPolicy(cache_blocks=5, sink_blocks=1),
    ):
        indices = policy.decide(state).keep_indices
        assert indices is not None
        assert torch.equal(indices, indices.sort().values)
        assert indices.unique().numel() == indices.numel()
        assert indices.min() >= 0 and indices.max() < state.cached_tokens


def test_frame_positions_cover_every_surviving_token() -> None:
    state = _state(12)
    for policy in (BankedSinkPolicy(bank_blocks=2, recent_blocks=2), BlockRelativeRoPEPolicy(cache_blocks=5)):
        decision = policy.decide(state)
        assert decision.frame_positions.numel() == decision.keep_indices.numel()


def test_build_policy_resolves_registered_names() -> None:
    assert build_policy("sliding_window", window_blocks=3) == SlidingWindowPolicy(window_blocks=3)
    with pytest.raises(KeyError, match="Unknown KV cache policy"):
        build_policy("does-not-exist")


# ── CompactingKVCache ────────────────────────────────────────


def test_compacting_cache_grows_without_a_policy() -> None:
    cache = CompactingKVCache(frame_tokens=FRAME_TOKENS)

    for _ in range(5):
        cache.append(*_chunk(), frame_count=FRAMES_PER_CHUNK)

    assert cache.length == 5 * CHUNK_TOKENS
    assert [block.block_idx for block in cache.blocks] == [0, 1, 2, 3, 4]


def test_compacting_cache_bounds_memory_under_a_policy() -> None:
    cache = CompactingKVCache(
        frame_tokens=FRAME_TOKENS, policy=SlidingWindowPolicy(window_blocks=4, sink_blocks=1)
    )

    sizes = []
    for _ in range(30):
        cache.append(*_chunk(), frame_count=FRAMES_PER_CHUNK)
        sizes.append(cache.length)

    assert max(sizes) == 4 * CHUNK_TOKENS
    # Flat once the budget is reached: the horizon no longer costs memory.
    assert sizes[10:] == [4 * CHUNK_TOKENS] * 20
    assert [block.block_idx for block in cache.blocks] == [0, 27, 28, 29]


def test_compacting_cache_keeps_the_bank_pinned_forever() -> None:
    cache = CompactingKVCache(frame_tokens=FRAME_TOKENS, policy=BankedSinkPolicy(bank_blocks=2, recent_blocks=2))

    for _ in range(40):
        cache.append(*_chunk(), frame_count=FRAMES_PER_CHUNK)

    assert [block.block_idx for block in cache.blocks] == [0, 1, 38, 39]
    # RoPE positions stay inside a fixed range no matter how long the rollout runs.
    assert cache.frame_positions().max().item() == 11


def test_compacting_cache_returns_dense_tensors_of_the_source_dtype() -> None:
    cache = CompactingKVCache(frame_tokens=FRAME_TOKENS, quantization=KVQuantConfig("int4"))

    keys_in, values_in = _chunk()
    cache.append(keys_in, values_in, frame_count=FRAMES_PER_CHUNK)
    keys, values = cache.cached()

    assert keys.shape == keys_in.shape
    assert keys.dtype == keys_in.dtype
    assert values.dtype == values_in.dtype


def test_quantized_storage_shrinks_the_cache() -> None:
    def bytes_for(config: KVQuantConfig | None) -> int:
        cache = CompactingKVCache(frame_tokens=FRAME_TOKENS, quantization=config)
        for _ in range(6):
            cache.append(*_chunk(), frame_count=FRAMES_PER_CHUNK)
        return cache.nbytes()

    dense = bytes_for(None)
    assert bytes_for(KVQuantConfig("int8")) < dense
    assert bytes_for(KVQuantConfig("int4")) < bytes_for(KVQuantConfig("int8"))
    assert bytes_for(KVQuantConfig("int2")) < bytes_for(KVQuantConfig("int4"))


def test_keys_and_values_take_independent_precisions() -> None:
    cache = CompactingKVCache(
        frame_tokens=FRAME_TOKENS,
        quantization=KVQuantConfig("int8"),
        value_quantization=KVQuantConfig("int4"),
    )

    cache.append(*_chunk(), frame_count=FRAMES_PER_CHUNK)
    keys, values = cache.cached()

    assert keys.shape == values.shape


def test_compacting_cache_validates_chunk_length() -> None:
    cache = CompactingKVCache(frame_tokens=FRAME_TOKENS)
    keys, values = _chunk()

    with pytest.raises(ValueError, match="expected 8 tokens"):
        cache.append(keys, values, frame_count=2)
    with pytest.raises(ValueError, match="frame_count must be >= 1"):
        cache.append(keys, values, frame_count=0)


def test_compacting_cache_reset_restarts_numbering() -> None:
    cache = CompactingKVCache(frame_tokens=FRAME_TOKENS)
    for _ in range(3):
        cache.append(*_chunk(), frame_count=FRAMES_PER_CHUNK)

    cache.reset()
    cache.append(*_chunk(), frame_count=FRAMES_PER_CHUNK)

    assert cache.length == CHUNK_TOKENS
    assert [block.block_idx for block in cache.blocks] == [0]
    assert cache.frame_positions() is None


def test_block_kv_cache_still_works_unchanged() -> None:
    """The CUDA-graph fixed-window path must not regress."""
    keys = torch.randn(1, 4, 2, 8)
    values = torch.randn(1, 4, 2, 8)

    cache = BlockKVCache.from_tensor(keys, values, seq_dim=1)

    assert cache.size == 4
    assert torch.equal(cache.cached_k(), keys)
