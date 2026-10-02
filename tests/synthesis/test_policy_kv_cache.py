"""Correctness proofs for the policy-driven Causal-rCM KV cache.

Three levels, strongest last:

A. ``KeepAllPolicy`` reproduces the upstream cache's metadata and buffers
   bit-for-bit across a randomized operation sequence.
B. Under eviction, the retained buffer equals exactly the blocks the policy
   selected, verified against an independent oracle built from the append history.
C. Attention *outputs* match: ``KeepAllPolicy`` reproduces the upstream path
   exactly, and each evicting policy equals dense attention over precisely its
   surviving tokens.

The simulated loop mirrors the real runtime contract in
``MinimalA2AAttnOp._materialize_kv``: several ``READONLY`` denoising steps that
see the retained prefix plus the in-flight chunk, then one ``APPEND`` that
commits it. The current block is *not* resident during ``READONLY``.
"""

import random

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.attention.kv_cache_policy import (
    BankedSinkPolicy,
    BlockRelativeRoPEPolicy,
    CacheDecision,
    KeepAllPolicy,
    SlidingWindowPolicy,
)

kv_cache_module = pytest.importorskip(
    "worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.utils.kv_cache",
    reason="vendored rCM runtime needs its optional dependencies",
)
policy_kv_cache = pytest.importorskip(
    "worldfoundry.synthesis.visual_generation.rcm.policy_kv_cache",
    reason="vendored rCM runtime needs its optional dependencies",
)

KVCache = kv_cache_module.KVCache
PolicyKVCache = policy_kv_cache.PolicyKVCache
policy_cache_factory = policy_kv_cache.policy_cache_factory

FRAME_TOKENS = 4
FRAMES = 3
CHUNK = FRAMES * FRAME_TOKENS
BATCH, HEADS, HEAD_DIM = 2, 4, 16
MAX_LEN = 16384
DENOISE_STEPS = 4


def _chunk(tokens: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(BATCH, tokens, HEADS, HEAD_DIM),
        torch.randn(BATCH, tokens, HEADS, HEAD_DIM),
    )


def _cursor(cache) -> int:
    """What the runtime's APPEND assertion compares the block cursor against."""
    return cache.committed_blocks


def _materialize_readonly(cache, keys, values, cursor):
    """Mirror ``_materialize_kv`` in READONLY + fast_infer mode."""
    prefix_end = cache.get_prefix_end(cursor)
    if prefix_end == 0:
        return keys, values
    return cache.write_transient(keys, values, prefix_end)


def _sdpa(query, keys, values):
    out = F.scaled_dot_product_attention(query.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2))
    return out.transpose(1, 2).contiguous()


def _oracle(history: dict[int, torch.Tensor], block_ids: list[int]) -> torch.Tensor:
    return torch.cat([history[block] for block in block_ids], dim=1)


# ── A. Equivalence with the upstream cache ───────────────────


def test_keep_all_matches_the_upstream_cache_bit_for_bit() -> None:
    random.seed(1234)
    torch.manual_seed(1234)

    for _ in range(8):
        plain = KVCache(MAX_LEN)
        policy = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=KeepAllPolicy())
        committed = 0

        for _ in range(random.randint(3, 10)):
            keys, values = _chunk(random.randint(1, 4) * FRAME_TOKENS)
            assert plain.append(keys, values) == policy.append(keys, values)
            committed += 1

            assert policy.cur == plain.cur
            assert policy._cum_ends == plain._cum_ends
            assert policy._chunk_lens == plain._chunk_lens
            assert policy.committed_blocks == committed == len(plain._cum_ends)

            for block_range in [None, *range(committed + 1)]:
                plain_keys, plain_values = plain.get(block_range)
                policy_keys, policy_values = policy.get(block_range)
                if plain_keys is None:
                    assert policy_keys is None
                else:
                    assert torch.equal(plain_keys, policy_keys)
                    assert torch.equal(plain_values, policy_values)
                assert plain.get_prefix_end(block_range) == policy.get_prefix_end(block_range)

            at = plain.get_prefix_end(committed)
            transient_keys, transient_values = _chunk(FRAME_TOKENS)
            expected = plain.write_transient(transient_keys, transient_values, at)
            actual = policy.write_transient(transient_keys, transient_values, at)
            assert torch.equal(expected[0], actual[0])
            assert torch.equal(expected[1], actual[1])


def test_reset_clears_policy_bookkeeping() -> None:
    torch.manual_seed(0)
    cache = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=SlidingWindowPolicy(window_blocks=2))
    for _ in range(5):
        cache.append(*_chunk())

    cache.reset()

    assert cache.committed_blocks == 0
    assert cache.retained_blocks == ()
    assert cache.cur == 0
    assert cache.get_prefix_end(0) == 0


# ── B. Eviction selects exactly what the policy asked for ────


@pytest.mark.parametrize(
    ("policy", "budget"),
    [
        (SlidingWindowPolicy(window_blocks=4, sink_blocks=1), 4),
        (BankedSinkPolicy(bank_blocks=2, recent_blocks=2), 4),
        (BlockRelativeRoPEPolicy(cache_blocks=5, sink_blocks=1, frame_limit=21), 5),
    ],
)
def test_retained_buffer_equals_the_policy_selection(policy, budget: int) -> None:
    torch.manual_seed(7)
    cache = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=policy)
    history: dict[int, torch.Tensor] = {}

    for block in range(25):
        keys, values = _chunk()
        history[block] = keys
        retained_before = [item.block_idx for item in cache.retained_blocks]

        for _ in range(DENOISE_STEPS):
            prefix_end = cache.get_prefix_end(block)
            # The cursor keeps advancing; the retained prefix does not.
            assert prefix_end == cache.cur
            view_keys, _ = _materialize_readonly(cache, keys, values, block)
            expected = _oracle(history, [*retained_before, block]) if retained_before else keys
            assert torch.equal(view_keys, expected)

        assert block == _cursor(cache)
        cache.append(keys, values)

        kept = [item.block_idx for item in cache.retained_blocks]
        assert kept == sorted(kept)
        assert kept[-1] == block, "the in-flight block must survive its own commit"
        assert len(kept) <= budget

        cached_keys, _ = cache.get()
        assert torch.equal(cached_keys, _oracle(history, kept))
        assert cache.cur == cached_keys.shape[1] == sum(cache._chunk_lens) == cache._cum_ends[-1]
        assert cache._chunk_lens == [item.frame_count * FRAME_TOKENS for item in cache.retained_blocks]

        # Chunk-wise slicing must still land on block boundaries after compaction.
        for count in range(1, len(kept) + 1):
            prefix_keys, _ = cache.get(count)
            assert torch.equal(prefix_keys, _oracle(history, kept[:count]))


def test_memory_stops_growing_once_the_budget_is_reached() -> None:
    torch.manual_seed(3)
    cache = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=SlidingWindowPolicy(window_blocks=4, sink_blocks=1))

    lengths = []
    for _ in range(40):
        cache.append(*_chunk())
        lengths.append(cache.cur)

    assert max(lengths) == 4 * CHUNK
    assert lengths[10:] == [4 * CHUNK] * 30
    assert cache.committed_blocks == 40


def test_bounded_plan_fits_the_real_readonly_and_append_lifecycle() -> None:
    """The allocation plan must fit the transient current block at every step."""
    from worldfoundry.core.attention.block_pattern import BlockPattern
    from worldfoundry.synthesis.visual_generation.rcm.kv_cache_plan import make_kv_cache_plan

    pattern = BlockPattern(frame_tokens=2, first_chunk_frames=1, chunk_frames=3)
    policy, capacity_frames = make_kv_cache_plan(
        pattern,
        5,
        policy_name="sliding_window",
        window_blocks=3,
        sink_blocks=1,
    )
    assert policy is not None
    cache = PolicyKVCache(capacity_frames * pattern.frame_tokens, frame_tokens=pattern.frame_tokens, policy=policy)

    for block in range(5):
        tokens = pattern.block_size(block) * pattern.frame_tokens
        keys = torch.full((1, tokens, 1, 1), float(block))
        values = keys + 100
        prefix_end = cache.get_prefix_end(block)
        if prefix_end:
            transient_keys, _ = cache.write_transient(keys, values, prefix_end)
            assert transient_keys.shape[1] <= cache.max_len
        assert cache.committed_blocks == block
        cache.append(keys, values)

    assert cache.committed_blocks == 5
    assert [item.block_idx for item in cache.retained_blocks] == [0, 3, 4]
    assert cache.current_len == 7 * pattern.frame_tokens


def test_cursor_clamping_never_reads_past_the_retained_prefix() -> None:
    torch.manual_seed(5)
    cache = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=SlidingWindowPolicy(window_blocks=3))
    for _ in range(20):
        cache.append(*_chunk())

    # A far-future cursor resolves to everything retained, not an IndexError.
    assert cache.get_prefix_end(999) == cache.cur
    keys, _ = cache.get(999)
    assert keys.shape[1] == cache.cur


def test_policy_must_report_kept_blocks() -> None:
    """Token-level eviction would desynchronize the runtime's block cursor."""

    class _TokenLevelPolicy:
        def decide(self, state):
            return CacheDecision(keep_indices=torch.arange(state.cached_tokens // 2))

    cache = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=_TokenLevelPolicy())

    with pytest.raises(ValueError, match="kept_blocks"):
        cache.append(*_chunk())


def test_policy_must_retain_the_current_block() -> None:
    class _DropsCurrentPolicy:
        def decide(self, state):
            keep = torch.arange(state.token_span(0)[1]) if len(state.blocks) > 1 else None
            return CacheDecision(keep_indices=keep, kept_blocks=(0,) if keep is not None else None)

    torch.manual_seed(0)
    cache = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=_DropsCurrentPolicy())
    cache.append(*_chunk())

    with pytest.raises(ValueError, match="retain the chunk it was just given"):
        cache.append(*_chunk())


def test_chunk_length_must_align_to_frames() -> None:
    cache = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=KeepAllPolicy())

    with pytest.raises(ValueError, match="not a multiple of frame_tokens"):
        cache.append(*_chunk(tokens=FRAME_TOKENS + 1))


# ── C. Attention outputs ─────────────────────────────────────


def _run_attention(cache, seed: int) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    outputs = []
    for block in range(12):
        query = torch.randn(BATCH, CHUNK, HEADS, HEAD_DIM)
        keys, values = _chunk()
        for _ in range(DENOISE_STEPS):
            view_keys, view_values = _materialize_readonly(cache, keys, values, block)
            outputs.append(_sdpa(query, view_keys, view_values))
        assert block == _cursor(cache)
        cache.append(keys, values)
    return outputs


def test_keep_all_reproduces_upstream_attention_exactly() -> None:
    upstream = _run_attention(KVCache(MAX_LEN), seed=99)
    policy = _run_attention(PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=KeepAllPolicy()), seed=99)

    assert len(upstream) == len(policy) == 12 * DENOISE_STEPS
    for expected, actual in zip(upstream, policy):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "policy",
    [SlidingWindowPolicy(window_blocks=4, sink_blocks=1), BankedSinkPolicy(bank_blocks=2, recent_blocks=2)],
)
def test_eviction_equals_attention_over_surviving_tokens(policy) -> None:
    torch.manual_seed(99)
    cache = PolicyKVCache(MAX_LEN, frame_tokens=FRAME_TOKENS, policy=policy)
    history: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for block in range(12):
        query = torch.randn(BATCH, CHUNK, HEADS, HEAD_DIM)
        keys, values = _chunk()
        history[block] = (keys, values)

        kept_before = [item.block_idx for item in cache.retained_blocks]
        if kept_before:
            reference_keys = torch.cat([history[j][0] for j in kept_before] + [keys], dim=1)
            reference_values = torch.cat([history[j][1] for j in kept_before] + [values], dim=1)
        else:
            reference_keys, reference_values = keys, values
        expected = _sdpa(query, reference_keys, reference_values)

        for _ in range(DENOISE_STEPS):
            view_keys, view_values = _materialize_readonly(cache, keys, values, block)
            assert torch.equal(_sdpa(query, view_keys, view_values), expected)

        cache.append(keys, values)


# ── Runtime wiring ───────────────────────────────────────────


def test_cache_factory_builds_one_cache_per_layer() -> None:
    factory = policy_cache_factory(frame_tokens=FRAME_TOKENS, policy=SlidingWindowPolicy(window_blocks=3))

    caches = [factory(MAX_LEN) for _ in range(4)]

    assert all(isinstance(cache, PolicyKVCache) for cache in caches)
    # One shared policy instance: layers must evict identically.
    assert len({id(cache.policy) for cache in caches}) == 1


def test_allocate_kv_caches_defaults_to_the_upstream_cache() -> None:
    """The vendored hook must not change behaviour when no factory is passed."""
    import inspect

    from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.networks import wan2pt1

    signature = inspect.signature(wan2pt1.WanModel.allocate_kv_caches)
    assert signature.parameters["cache_factory"].default is None
