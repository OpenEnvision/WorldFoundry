"""Tests for WanModel.rotary_frequencies per-request memoization.

The assembled 3D RoPE frequencies depend only on grid size + device and are
constant across a request's denoise steps, so they are memoized. These tests
lock: cache hit reuse, bitwise parity with a fresh build, per-grid keying, and
the bounded cache size. Exercised via the bound method on a light stub (a full
WanModel is heavy to construct) plus the real frequency tables.
"""

from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import WanModel
from worldfoundry.core.attention.complex_rope import complex_rotary_frequencies_3d


class _Stub:
    pass


def _bound(head_dim: int = 128):
    stub = _Stub()
    stub.freqs = complex_rotary_frequencies_3d(head_dim)
    stub._rope_freq_cache = {}
    return stub, WanModel.rotary_frequencies.__get__(stub)


def _fresh(freqs, grid, device):
    f, h, w = grid
    return (
        torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(device)
    )


def test_hit_reuses_same_tensor() -> None:
    stub, rf = _bound()
    grid, device = (4, 16, 16), torch.device("cpu")
    first = rf(grid, device=device)
    second = rf(grid, device=device)
    assert first is second
    assert len(stub._rope_freq_cache) == 1


def test_bitwise_parity_with_fresh_build() -> None:
    stub, rf = _bound()
    grid, device = (4, 16, 16), torch.device("cpu")
    cached = rf(grid, device=device)
    assert torch.equal(cached, _fresh(stub.freqs, grid, device))


def test_distinct_grids_get_distinct_entries() -> None:
    stub, rf = _bound()
    rf((4, 16, 16), device=torch.device("cpu"))
    rf((2, 8, 8), device=torch.device("cpu"))
    assert len(stub._rope_freq_cache) == 2


def test_cache_is_bounded() -> None:
    stub, rf = _bound()
    for i in range(1, 11):
        rf((i, 4, 4), device=torch.device("cpu"))
    # Cache clears at 8 entries, so it never exceeds the bound.
    assert len(stub._rope_freq_cache) <= 8
