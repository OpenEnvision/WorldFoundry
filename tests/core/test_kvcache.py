from __future__ import annotations

import pytest
import torch

from worldfoundry.core.attention.kvcache import BlockKVCache


class _NaiveKVCache:
    """Allocation-heavy [sink | rolling window] reference implementation."""

    def __init__(
        self,
        *,
        window_size: int,
        chunk_size: int,
        sink_size: int,
    ) -> None:
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.sink_size = sink_size
        self.total_size = sink_size + window_size
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.previous_chunk = -1

    def update(
        self,
        chunk_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        assert chunk_index in (self.previous_chunk, self.previous_chunk + 1)
        if self.k is None or self.v is None:
            assert chunk_index == 0
            self.k = k.clone()
            self.v = v.clone()
            self.previous_chunk = 0
            return

        if chunk_index == self.previous_chunk:
            overlaps_sink = (
                self.sink_size > 0
                and chunk_index * self.chunk_size < self.sink_size
            )
            if self.k.shape[1] == self.total_size and not overlaps_sink:
                if self.chunk_size <= self.window_size:
                    self.k[:, -self.chunk_size :] = k
                    self.v[:, -self.chunk_size :] = v
                else:
                    self.k = torch.cat(
                        [self.k[:, : self.sink_size], k[:, -self.window_size :]],
                        dim=1,
                    )
                    self.v = torch.cat(
                        [self.v[:, : self.sink_size], v[:, -self.window_size :]],
                        dim=1,
                    )
            else:
                self.k[:, -self.chunk_size :] = k
                self.v[:, -self.chunk_size :] = v
            return

        sink_k = self.k[:, : self.sink_size]
        sink_v = self.v[:, : self.sink_size]
        window_k = torch.cat([self.k[:, self.sink_size :], k], dim=1)[
            :, -self.window_size :
        ]
        window_v = torch.cat([self.v[:, self.sink_size :], v], dim=1)[
            :, -self.window_size :
        ]
        self.k = torch.cat([sink_k, window_k], dim=1)
        self.v = torch.cat([sink_v, window_v], dim=1)
        self.previous_chunk += 1


@pytest.mark.parametrize(
    ("sink_size", "window_size"),
    [
        (0, 8),
        (3, 5),
        (3, 21),
        (1, 16),
    ],
)
def test_block_kvcache_matches_reference_for_divisible_and_partial_overflow(
    sink_size: int,
    window_size: int,
) -> None:
    torch.manual_seed(0)
    chunk_size = 8
    total_size = sink_size + window_size
    shape = (2, total_size, 2, 4)
    cache = BlockKVCache(
        k_shape=shape,
        v_shape=shape,
        seq_dim=1,
        chunk_size=chunk_size,
        window_size=window_size,
        sink_size=sink_size,
        device="cpu",
        dtype=torch.float32,
    )
    reference = _NaiveKVCache(
        window_size=window_size,
        chunk_size=chunk_size,
        sink_size=sink_size,
    )

    for chunk_index in range(6):
        k = torch.randn(2, chunk_size, 2, 4)
        v = torch.randn(2, chunk_size, 2, 4)
        reference.update(chunk_index, k, v)

        cache.before_update(chunk_index)
        cache.update(k, v)
        torch.testing.assert_close(cache.cached_k(), reference.k)
        torch.testing.assert_close(cache.cached_v(), reference.v)
        cache.after_update(chunk_index)

        replacement_k = torch.randn(2, chunk_size, 2, 4)
        replacement_v = torch.randn(2, chunk_size, 2, 4)
        reference.update(chunk_index, replacement_k, replacement_v)

        cache.before_update(chunk_index)
        cache.update(replacement_k, replacement_v)
        torch.testing.assert_close(cache.cached_k(), reference.k)
        torch.testing.assert_close(cache.cached_v(), reference.v)
        cache.after_update(chunk_index)


def test_nondivisible_cache_reset_preserves_storage() -> None:
    cache = BlockKVCache(
        k_shape=(1, 17, 1, 2),
        v_shape=(1, 17, 1, 2),
        seq_dim=1,
        chunk_size=8,
        window_size=16,
        sink_size=1,
        device="cpu",
        dtype=torch.float32,
    )
    pointers = (cache._k.data_ptr(), cache._v.data_ptr())
    cache.before_update(0)
    cache.update(
        torch.ones((1, 8, 1, 2)),
        torch.ones((1, 8, 1, 2)),
    )
    cache.after_update(0)

    cache.reset()

    assert (cache._k.data_ptr(), cache._v.data_ptr()) == pointers
    assert cache._n_cached == 0
    assert cache._prev_chunk_idx == -1
    assert cache._curr_chunk_idx is None


def test_reset_discards_inflight_update_state() -> None:
    cache = BlockKVCache(
        k_shape=(1, 17, 1, 2),
        v_shape=(1, 17, 1, 2),
        seq_dim=1,
        chunk_size=8,
        window_size=16,
        sink_size=1,
        device="cpu",
        dtype=torch.float32,
    )
    pointers = (cache._k.data_ptr(), cache._v.data_ptr())

    cache.before_update(0)
    cache.update(
        torch.ones((1, 8, 1, 2)),
        torch.ones((1, 8, 1, 2)),
    )
    cache.reset()

    assert (cache._k.data_ptr(), cache._v.data_ptr()) == pointers
    assert cache._n_cached == 0
    assert cache._prev_chunk_idx == -1
    assert cache._curr_chunk_idx is None

    cache.before_update(0)
    cache.update(
        torch.full((1, 8, 1, 2), 2.0),
        torch.full((1, 8, 1, 2), 3.0),
    )
    torch.testing.assert_close(
        cache.cached_k(),
        torch.full((1, 8, 1, 2), 2.0),
    )
    torch.testing.assert_close(
        cache.cached_v(),
        torch.full((1, 8, 1, 2), 3.0),
    )
    cache.after_update(0)
