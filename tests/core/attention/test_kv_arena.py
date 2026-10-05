from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.attention.cache.kv_arena import KVSegmentArena, KVSegmentLayout


def pair(tokens, *, seq_dim=2, dtype=torch.float32, device="cpu"):
    shape = [2, 3, tokens, 8] if seq_dim == 2 else [2, tokens, 3, 8]
    return tuple(torch.randn(shape, dtype=dtype, device=device) for _ in range(2))


@pytest.mark.parametrize("seq_dim", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_tokens,memory_tokens", [(0, 0), (0, 5), (7, 0), (7, 5)])
@torch.no_grad()
def test_arena_matches_cat_with_stable_storage(seq_dim, dtype, static_tokens, memory_tokens):
    static = pair(static_tokens, seq_dim=seq_dim, dtype=dtype)
    memory = pair(memory_tokens, seq_dim=seq_dim, dtype=dtype)
    arena = KVSegmentArena(static, current_tokens=4, memory=memory, seq_dim=seq_dim)
    pointers = None
    for _ in range(5):
        # Non-contiguous current input is legal; arena output is contiguous.
        current = tuple(
            t[..., ::2]
            for t in (torch.randn(*pair(4, seq_dim=seq_dim)[0].shape[:-1], 16, dtype=dtype) for _ in range(2))
        )
        actual = arena.stage_current(current)
        expected = tuple(torch.cat(parts, dim=seq_dim) for parts in zip(static, current, memory))
        arena.validate_persistent(arena.static, arena.memory)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
            assert value.is_contiguous()
        current_pointers = tuple(t.data_ptr() for t in actual)
        if pointers is not None:
            assert current_pointers == pointers
        pointers = current_pointers
    assert pointers[0] != pointers[1]
    assert arena.stage_count == 5
    assert arena.nbytes == 2 * actual[0].numel() * actual[0].element_size()


@pytest.mark.parametrize("bad", [-1, 0, 1.5, True])
def test_invalid_current_capacity(bad):
    with pytest.raises(ValueError, match="current_tokens"):
        KVSegmentLayout(1, bad)


@torch.no_grad()
def test_alias_and_geometry_checks_do_not_mutate_persistent_segments():
    arena = KVSegmentArena(pair(7), current_tokens=4, memory=pair(5))
    static_before = tuple(t.clone() for t in arena.static)
    with pytest.raises(ValueError, match="alias"):
        arena.validate_persistent(tuple(t.clone() for t in arena.static), arena.memory)
    with pytest.raises(ValueError, match="alias"):
        wrong_offset = tuple(t.narrow(2, 1, 7) for t in arena._kv)
        arena.validate_persistent(wrong_offset, arena.memory)
    with pytest.raises(ValueError, match="alias"):
        arena.validate_persistent(tuple(t.view(torch.int32) for t in arena.static), arena.memory)
    with pytest.raises(ValueError, match="alias"):
        arena.stage_current(tuple(t.narrow(2, 0, 4) for t in arena._kv))
    with pytest.raises(ValueError, match="match"):
        arena.stage_current(pair(3))
    with pytest.raises(ValueError, match="match"):
        arena.stage_current(pair(4, dtype=torch.float16))
    for actual, expected in zip(arena.static, static_before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert arena.stage_count == 0


def test_autograd_is_rejected():
    with pytest.raises(RuntimeError, match="no_grad"):
        KVSegmentArena(pair(3), current_tokens=4)
    with torch.no_grad():
        arena = KVSegmentArena(pair(3), current_tokens=4)
    with pytest.raises(RuntimeError, match="no_grad"):
        arena.stage_current(pair(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.no_grad()
def test_cuda_long_sequence_attention_and_stream_ownership(dtype):
    static = pair(2048, dtype=dtype, device="cuda")
    memory = pair(1024, dtype=dtype, device="cuda")
    arena = KVSegmentArena(static, current_tokens=64, memory=memory)
    q = pair(64, dtype=dtype, device="cuda")[0]
    outputs = []
    for _ in range(64):
        current = pair(64, dtype=dtype, device="cuda")
        keys, values = arena.stage_current(current)
        reference = tuple(torch.cat(parts, dim=2) for parts in zip(static, current, memory))
        outputs.append((F.scaled_dot_product_attention(q, keys, values), F.scaled_dot_product_attention(q, *reference)))
    torch.cuda.synchronize()
    for actual, expected in outputs:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream), pytest.raises(RuntimeError, match="owning CUDA stream"):
        arena.stage_current(current)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_cuda_graph_replay_reads_updated_current_segment():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        static, memory, current = pair(7, device="cuda"), pair(5, device="cuda"), pair(4, device="cuda")
        arena = KVSegmentArena(static, current_tokens=4, memory=memory)
        # Warm up the copy path before capture, on the same owning stream.
        arena.stage_current(current)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = tuple(t.clone() for t in arena.stage_current(current))
    with torch.cuda.stream(stream):
        for _ in range(4):
            for t in current:
                t.add_(1)
            graph.replay()
            expected = tuple(torch.cat(parts, dim=2) for parts in zip(static, current, memory))
            for actual, reference in zip(output, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
