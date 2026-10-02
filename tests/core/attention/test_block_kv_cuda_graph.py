"""Real replay of a rolling cache with host-owned preparation and commit.

Filling runs eagerly. ``before_update`` (including rolling), Python chunk and
length bookkeeping, and ``after_update`` remain outside the graph. Each graph
captures only ``update`` plus full-window K/V snapshots and attention after the
cache is full: tensor lengths and physical write bounds then stay constant for
both new chunks and same-chunk rewrites. No changing-length replay is claimed.

These are fixed-cache seam tests, not evidence that a model's dynamic RoPE or
conditioning inputs already support full-model CUDA Graph replay.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.attention.cache.kvcache import BlockKVCache

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph replay requires CUDA"),
]


def _cache(sink: int, window: int, chunk: int, seq_dim: int, dtype: torch.dtype) -> BlockKVCache:
    shape = (1, sink + window, 2, 8) if seq_dim == 1 else (1, 2, sink + window, 8)
    return BlockKVCache(
        k_shape=shape,
        v_shape=shape,
        seq_dim=seq_dim,
        chunk_size=chunk,
        window_size=window,
        sink_size=sink,
        device="cuda",
        dtype=dtype,
    )


def _inputs(cache: BlockKVCache, value: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = list(cache.k_shape)
    shape[cache.seq_dim] = cache.chunk_size
    base = torch.arange(math.prod(shape), device="cuda", dtype=torch.float32).reshape(shape)
    # Every chunk/rewrite changes K, V and Q, so replay cannot pass by retaining
    # captured inputs or attending to a previously populated cache.
    return tuple(
        ((base + value * 11).sin() * scale).to(cache.dtype)
        for scale in (0.25, 0.75, 0.5)
    )


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_dim: int) -> torch.Tensor:
    if seq_dim == 1:
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v)


def _oracle(
    history: list[tuple[torch.Tensor, torch.Tensor]], cache: BlockKVCache
) -> tuple[torch.Tensor, torch.Tensor]:
    # Independent sink + tail oracle; it never uses physical cache positions.
    result = []
    for column in (0, 1):
        full = torch.cat([pair[column] for pair in history], dim=cache.seq_dim)
        if full.shape[cache.seq_dim] > cache.sink_size + cache.window_size:
            full = torch.cat(
                (
                    full.narrow(cache.seq_dim, 0, cache.sink_size),
                    full.narrow(cache.seq_dim, full.shape[cache.seq_dim] - cache.window_size, cache.window_size),
                ),
                dim=cache.seq_dim,
            )
        result.append(full)
    return tuple(result)


class _CapturedUpdate:
    """Test-owned raw graph: no capture exception handler or eager fallback."""

    def __init__(self, cache: BlockKVCache, inputs: tuple[torch.Tensor, ...]) -> None:
        self.cache = cache
        self.inputs = tuple(t.clone() for t in inputs)
        self.geometry = (cache.size, cache._current_write_bounds(), tuple(t.shape for t in self.inputs))
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            # All host lifecycle work has already run. Warmup and capture only
            # overwrite the current fixed physical range and evaluate attention.
            for _ in range(2):
                self._forward()
        torch.cuda.current_stream().wait_stream(self.stream)
        self.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.outputs = self._forward()
        self.replays = 0
        self.storage = tuple(t.data_ptr() for t in (*self.inputs, *self.outputs, cache._k, cache._v))

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k, v, q = self.inputs
        self.cache.update(k, v)
        keys, values = self.cache.cached_k(), self.cache.cached_v()
        return keys.clone(), values.clone(), _attention(q, keys, values, self.cache.seq_dim)

    def replay(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        assert self.cache.is_steady_state()
        assert (self.cache.size, self.cache._current_write_bounds(), tuple(t.shape for t in inputs)) == self.geometry
        for static, fresh in zip(self.inputs, inputs):
            static.copy_(fresh)
        self.graph.replay()
        self.replays += 1
        assert tuple(
            t.data_ptr() for t in (*self.inputs, *self.outputs, self.cache._k, self.cache._v)
        ) == self.storage
        # Raw CUDA Graph outputs are reused buffers. Retain snapshots at the
        # caller boundary before the next replay, just as a model owner must.
        return tuple(output.clone() for output in self.outputs)


@pytest.mark.parametrize("sink,window,chunk", [(0, 8, 4), (3, 14, 4), (3, 5, 8)])
@pytest.mark.parametrize("seq_dim", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.no_grad()
def test_replay_matches_eager_and_history_across_rollover_and_rewrites(sink, window, chunk, seq_dim, dtype):
    captured_cache = _cache(sink, window, chunk, seq_dim, dtype)
    eager_cache = _cache(sink, window, chunk, seq_dim, dtype)
    history = []
    filling_chunks = math.ceil((sink + window) / chunk)
    for index in range(filling_chunks):
        k, v, _ = _inputs(captured_cache, index)
        history.append((k, v))
        for cache in (captured_cache, eager_cache):
            cache.before_update(index)
            cache.update(k, v)
            for actual, expected in zip((cache.cached_k(), cache.cached_v()), _oracle(history, cache)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            cache.after_update(index)

    assert captured_cache._n_cached == sink + window
    graph = None
    retained = []
    for index in range(filling_chunks, filling_chunks + 6):
        for rewrite in range(3):
            inputs = _inputs(captured_cache, index * 7 + rewrite)
            if rewrite == 0:
                history.append(inputs[:2])
            else:
                history[-1] = inputs[:2]
            captured_cache.before_update(index)
            eager_cache.before_update(index)
            assert captured_cache.is_steady_state()
            if graph is None:
                graph = _CapturedUpdate(captured_cache, inputs)
            actual = graph.replay(inputs)
            eager_cache.update(*inputs[:2])
            reference = _oracle(history, captured_cache)
            for got, eager, expected in zip(actual[:2], (eager_cache.cached_k(), eager_cache.cached_v()), reference):
                torch.testing.assert_close(got, eager, rtol=0, atol=0)
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
            expected_attention = _attention(inputs[2], *reference, seq_dim)
            eager_attention = _attention(inputs[2], eager_cache.cached_k(), eager_cache.cached_v(), seq_dim)
            torch.testing.assert_close(actual[2], eager_attention, rtol=0, atol=0)
            torch.testing.assert_close(actual[2], expected_attention, rtol=1e-3, atol=1e-3)
            retained.append((actual, tuple(t.clone() for t in actual)))
            captured_cache.after_update(index)
            eager_cache.after_update(index)

    assert graph.replays == 18
    assert graph.geometry[0] == sink + window
    assert captured_cache._prev_chunk_idx == filling_chunks + 5
    for outputs, snapshots in retained:
        for output, snapshot in zip(outputs, snapshots):
            torch.testing.assert_close(output, snapshot, rtol=0, atol=0)


@torch.no_grad()
def test_conditional_branches_and_new_sessions_keep_graph_and_cache_storage_independent():
    owners = []
    for owner_index in range(4):  # conditional/unconditional branches in two sessions
        cache = _cache(sink=1, window=7, chunk=4, seq_dim=2, dtype=torch.float16)
        for index in range(2):
            cache.before_update(index)
            cache.update(*_inputs(cache, owner_index * 100 + index)[:2])
            cache.after_update(index)
        cache.before_update(2)
        inputs = _inputs(cache, owner_index * 100 + 2)
        graph = _CapturedUpdate(cache, inputs)
        owners.append((cache, graph, graph.replay(inputs)))
        cache.after_update(2)

    all_addresses = [address for _, graph, _ in owners for address in graph.storage]
    assert len(set(all_addresses)) == len(all_addresses)
    preserved = [tuple(t.clone() for t in (cache._k, cache._v, *output)) for cache, _, output in owners[1:]]
    first_cache, first_graph, first_output = owners[0]
    first_snapshot = tuple(t.clone() for t in first_output)
    for index in range(3, 9):
        first_cache.before_update(index)
        first_graph.replay(_inputs(first_cache, index))
        first_cache.after_update(index)

    assert first_graph.replays == 7
    assert all(graph.replays == 1 for _, graph, _ in owners[1:])
    for (cache, _, output), snapshot in zip(owners[1:], preserved):
        for actual, expected in zip((cache._k, cache._v, *output), snapshot):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(first_output, first_snapshot):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
