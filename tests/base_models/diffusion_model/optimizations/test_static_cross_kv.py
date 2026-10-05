"""Tests for the static cross-attention K/V cache.

Numerical parity + hit/miss + invalidation are exercised on CPU against the
real Wan CrossAttention; GPU speedup is a separate microbenchmark.
"""

from __future__ import annotations

import copy

import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import CrossAttention, WanModel
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import (
    StaticCrossKVProcessor,
    install_static_cross_kv_cache,
    reset_static_cross_kv,
)


class _Wrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ca = CrossAttention(128, 4)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.ca(x, ctx)


class _ImageWrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ca = CrossAttention(128, 4, has_image_input=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.ca(x, ctx)


def test_cached_matches_uncached() -> None:
    torch.manual_seed(0)
    m = _Wrap().eval()
    cached = copy.deepcopy(m)
    install_static_cross_kv_cache(cached)
    ctx = torch.randn(2, 16, 128)
    for _ in range(3):
        x = torch.randn(2, 32, 128)
        ref = m(x, ctx)
        out = cached(x, ctx)
        torch.testing.assert_close(out, ref)


def test_hits_across_repeated_context() -> None:
    torch.manual_seed(0)
    m = _Wrap().eval()
    cache = install_static_cross_kv_cache(m)
    ctx = torch.randn(1, 16, 128)
    with torch.no_grad():
        for _ in range(4):
            m(torch.randn(1, 8, 128), ctx)
    report = cache.report()
    assert report["misses"] == 1  # first step computes
    assert report["hits"] == 3  # next three reuse


def test_invalidation_recomputes() -> None:
    torch.manual_seed(0)
    m = _Wrap().eval()
    cache = install_static_cross_kv_cache(m)
    ctx = torch.randn(1, 16, 128)
    with torch.no_grad():
        m(torch.randn(1, 8, 128), ctx)
        reset_static_cross_kv(cache)
        new_ctx = torch.randn(1, 16, 128)
        m(torch.randn(1, 8, 128), new_ctx)
    assert cache.report()["version"] == 1
    # Runtime-effective counters describe only the current request window;
    # lifetime counters remain available for diagnostics.
    assert cache.report()["misses"] == 1
    assert cache.report()["lifetime_misses"] == 2


def test_request_window_does_not_reuse_previous_hits() -> None:
    torch.manual_seed(0)
    m = _Wrap().eval()
    cache = install_static_cross_kv_cache(m)
    ctx = torch.randn(1, 16, 128)
    with torch.no_grad():
        m(torch.randn(1, 8, 128), ctx)
        m(torch.randn(1, 8, 128), ctx)
    assert cache.report()["effective"] == "kv-reuse"
    assert cache.report()["hits"] == 1

    reset_static_cross_kv(cache)

    report = cache.report()
    assert report["effective"] == "installed (runtime-pending)"
    assert report["hits"] == 0
    assert report["misses"] == 0
    assert report["lifetime_hits"] == 1


def test_unknown_processor_kwargs_are_reported_as_dense_bypass() -> None:
    torch.manual_seed(0)
    m = _Wrap().eval()
    cache = install_static_cross_kv_cache(m)
    with torch.no_grad():
        output = m.ca(
            torch.randn(1, 8, 128),
            torch.randn(1, 16, 128),
            attention_mask=None,
        )

    assert output.shape == (1, 8, 128)
    report = cache.report()
    assert report["effective"] == "dense-bypass"
    assert report["bypasses"] == 1
    assert report["bypass_kwargs"] == ["attention_mask"]
    assert report["hits"] == 0
    assert report["misses"] == 0


def test_cfg_alternating_contexts_stay_correct() -> None:
    """Positive/negative CFG contexts alternate every step and must not collide.

    Regression for a cache-key bug: keying on ``data_ptr`` (with a single entry
    per module) collided across the two CFG branches. The cache now keys on
    tensor identity, retains the tensors strongly so identities cannot be
    recycled, and holds both branches at once.
    """

    torch.manual_seed(0)
    m = _Wrap().eval()
    ref = copy.deepcopy(m)
    cache = install_static_cross_kv_cache(m)
    ctx_pos = torch.randn(1, 16, 128)
    ctx_neg = torch.randn(1, 16, 128)  # distinct content = distinct branch
    with torch.no_grad():
        for _ in range(8):
            x = torch.randn(1, 8, 128)
            out_pos = m(x, ctx_pos)
            out_neg = m(x, ctx_neg)
            ref_p = ref(x, ctx_pos)
            ref_n = ref(x, ctx_neg)
            torch.testing.assert_close(out_pos, ref_p)
            torch.testing.assert_close(out_neg, ref_n)
    # Both branches cached side by side, reused after their first miss each.
    assert cache.report()["misses"] == 2
    assert cache.report()["hits"] >= 14


def test_image_and_text_kv_are_both_cached() -> None:
    torch.manual_seed(0)
    m = _ImageWrap().eval()
    ref = copy.deepcopy(m)
    cache = install_static_cross_kv_cache(m)
    ctx = torch.randn(1, 300, 128)  # first 257 tokens are the image branch
    with torch.no_grad():
        first = m(torch.randn(1, 8, 128), ctx)
        x = torch.randn(1, 8, 128)
        out = m(x, ctx)
        expected = ref(x, ctx)
    assert first.shape == out.shape
    torch.testing.assert_close(out, expected)
    assert cache.report()["misses"] == 1
    assert cache.report()["hits"] == 1


def test_wan_projected_context_reuses_identity_without_content_checksum() -> None:
    torch.manual_seed(0)
    model = WanModel(
        dim=128,
        in_dim=4,
        ffn_dim=256,
        out_dim=4,
        text_dim=64,
        freq_dim=64,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=1,
        has_image_input=False,
        require_vae_embedding=False,
        require_clip_embedding=False,
    ).eval()
    cache = install_static_cross_kv_cache(model)
    raw_context = torch.randn(1, 16, 64)
    with torch.no_grad():
        first = model.prepare_condition_context(raw_context)
        second = model.prepare_condition_context(raw_context)
        cloned = model.prepare_condition_context(raw_context.clone())

    assert second is first
    assert cloned is not first
    torch.testing.assert_close(cloned, first)
    report = cache.report()
    assert report["condition_hits"] == 1
    assert report["condition_misses"] == 2


def test_install_is_idempotent() -> None:
    m = _Wrap().eval()
    install_static_cross_kv_cache(m)
    proc1 = m.ca.get_processor()
    assert isinstance(proc1, StaticCrossKVProcessor)
    install_static_cross_kv_cache(m)
    proc2 = m.ca.get_processor()
    # Re-install unwraps then re-wraps; no nested StaticCrossKVProcessor.
    assert isinstance(proc2, StaticCrossKVProcessor)
    assert not isinstance(proc2._inner, StaticCrossKVProcessor)
