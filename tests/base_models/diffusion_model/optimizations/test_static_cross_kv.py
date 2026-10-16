"""Tests for the static cross-attention K/V cache.

Numerical parity + hit/miss + invalidation are exercised on CPU against the
real Wan CrossAttention; GPU speedup is a separate microbenchmark.
"""

from __future__ import annotations

import copy

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.wan.model import CrossAttention, WanModel
from worldfoundry.base_models.diffusion_model.optimizations.static_cross_kv import (
    StaticConditionCache,
    StaticCrossKVProcessor,
    install_static_cross_kv_cache,
    reset_static_cross_kv,
    reuse_cross_kv,
)

pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize("image", [False, True])
def test_request_condition_projection_and_live_query(image: bool) -> None:
    model = WanModel(
        dim=128,
        in_dim=4,
        ffn_dim=256,
        out_dim=4,
        text_dim=16,
        freq_dim=32,
        num_heads=4,
        num_layers=1,
        has_image_input=image,
        eps=1e-6,
        patch_size=(1, 2, 2),
        require_vae_embedding=False,
        require_clip_embedding=image,
    ).eval()
    source = torch.randn(1, 4, 16)
    clip = torch.randn(1, 257, 1280) if image else None
    condition = StaticConditionCache(source, clip)
    ca = model.blocks[0].cross_attn
    counts = {name: 0 for name in ("text", "q", "k", "v")}

    def count(name: str):
        def hook(*_args):
            counts[name] += 1

        return hook

    handles = [
        module.register_forward_hook(count(name))
        for name, module in (("text", model.text_embedding), ("q", ca.q), ("k", ca.k), ("v", ca.v))
    ]
    keys = tuple(model.state_dict())
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(1, 8, 128)
            raw_context = model.prepare_condition_context(source, clip_feature=clip)
            projected = model.prepare_condition_context(condition)
            assert torch.equal(projected, raw_context)
            raw = ca(x, raw_context)
            cached = ca(x, projected, _worldfoundry_condition_store=condition.store)
            assert torch.equal(cached, raw)
            assert cached.count_nonzero() > 0
    for handle in handles:
        handle.remove()
    assert counts == {"text": 4, "q": 6, "k": 4, "v": 4}
    assert condition.store.report()["hits"] == 2
    assert condition.nbytes > 0
    assert tuple(model.state_dict()) == keys


def test_request_condition_admission_and_raw_gradients() -> None:
    encoder = torch.nn.Linear(16, 32)
    source = torch.randn(2, 4, 16, requires_grad=True)
    condition = StaticConditionCache(source)

    def project(values: torch.Tensor, clip: torch.Tensor | None) -> torch.Tensor:
        return encoder(values)

    with torch.no_grad(), pytest.raises(RuntimeError, match="eval no-grad"):
        condition.project_context(encoder, project)
    encoder.eval()
    with pytest.raises(RuntimeError, match="eval no-grad"):
        condition.project_context(encoder, project)
    with pytest.raises(RuntimeError, match="eval no-grad"):
        reuse_cross_kv(encoder, source, condition.store, lambda: (encoder(source),))
    encoder(source).square().sum().backward()
    assert source.grad is not None and source.grad.count_nonzero() > 0
    assert encoder.weight.grad is not None and encoder.weight.grad.count_nonzero() > 0


def test_request_replacement_observes_condition_and_weight_changes() -> None:
    encoder = torch.nn.Linear(16, 32).eval()
    source = torch.randn(2, 4, 16)

    def project(values: torch.Tensor, clip: torch.Tensor | None) -> torch.Tensor:
        return encoder(values)

    first = StaticConditionCache(source)
    with torch.no_grad():
        previous = first.project_context(encoder, project).clone()
        encoder.bias.add_(1)
        for values in (source, torch.randn_like(source)):
            fresh = StaticConditionCache(values)
            assert torch.equal(fresh.project_context(encoder, project), encoder(values))
            assert not torch.equal(fresh.project_context(encoder, project), previous)
    assert first.store is not fresh.store


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("cross_type", ["t2v_cross_attn", "i2v_cross_attn", "cross_attn"])
@pytest.mark.parametrize("batch,skip", [(1, False), (2, True), (4, True)])
def test_wan_fun_request_condition_full_forward(device: str, cross_type: str, batch: int, skip: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.video_x_fun.transformer import (
        WanTransformer3DModel,
    )

    image = cross_type == "i2v_cross_attn"
    model = (
        WanTransformer3DModel(
            model_type="i2v" if image else "t2v",
            cross_attn_type=cross_type,
            in_dim=4,
            out_dim=4,
            dim=64,
            ffn_dim=128,
            freq_dim=16,
            text_dim=16,
            text_len=8,
            num_heads=4,
            num_layers=1,
        )
        .to(device)
        .eval()
    )
    if skip:
        model.enable_cfg_skip(0.5, 4)
    source = torch.randn(batch, 4, 16, device=device)
    clip = torch.randn(batch, 257, 1280, device=device) if image else None
    condition = StaticConditionCache(source, clip)
    keys = tuple(model.state_dict())
    calls = {"text": 0, "q": 0, "k": 0}

    def count(name: str):
        def hook(*_args):
            calls[name] += 1

        return hook

    ca = model.blocks[0].cross_attn
    handles = [
        module.register_forward_hook(count(name))
        for name, module in (("text", model.text_embedding), ("q", ca.q), ("k", ca.k))
    ]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        model.head.head.weight.normal_(std=0.01)
        for step in range(4):
            model.current_steps = step
            kwargs = dict(
                x=torch.randn(batch, 4, 3, 4, 4, device=device),
                seq_len=12,
                t=torch.full((batch,), 1000 - step * 200, device=device, dtype=torch.float32),
            )
            raw = model(context=source, clip_fea=clip, **kwargs)
            cached = model(context=condition, **kwargs)
            assert torch.equal(raw, cached)
            assert cached.count_nonzero() > 0
    for handle in handles:
        handle.remove()
    expected = {1: (5, 8), 2: (8, 12), 4: (12, 16)}[batch]
    assert calls == {"text": expected[0], "k": expected[0], "q": expected[1]}
    assert condition.nbytes > 0
    assert tuple(model.state_dict()) == keys


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_dualcamctrl_request_cache_keeps_both_dynamic_streams(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("ftfy")
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.dualcamctrl import WanControlNet
    from worldfoundry.synthesis.visual_generation.dualcamctrl.native_pipeline import model_fn_wan_video

    model = (
        WanControlNet(
            dim=96,
            in_dim=4,
            ffn_dim=192,
            out_dim=4,
            text_dim=16,
            freq_dim=16,
            num_heads=4,
            num_layers=1,
            has_image_input=False,
            eps=1e-6,
            patch_size=(1, 2, 2),
            require_vae_embedding=False,
            require_clip_embedding=False,
            control_block_index=[0],
            rgb_inject_blocks=[0],
            depth_inject_blocks=[0],
        )
        .to(device)
        .eval()
    )
    source = torch.randn(1, 4, 16, device=device)
    condition = StaticConditionCache(source)
    kwargs = dict(dit=model, return_control_latents=True)
    counts = {"text": 0, "main": 0, "control": 0}

    def count(name: str):
        def hook(*_args):
            counts[name] += 1

        return hook

    handles = [
        module.register_forward_hook(count(name))
        for name, module in (
            ("text", model.text_embedding),
            ("main", model.blocks[0]),
            ("control", model.control_blocks[0]),
        )
    ]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        model.head.head.weight.normal_(std=0.01)
        model.control_zero_inits[0].weight.normal_(std=0.01)
        for step in range(3):
            inputs = dict(
                latents=torch.randn(1, 4, 2, 4, 4, device=device),
                control_latents=torch.randn(1, 4, 2, 4, 4, device=device),
                timestep=torch.tensor([1000 - step * 200], device=device, dtype=torch.float32),
            )
            raw = model_fn_wan_video(context=source, **inputs, **kwargs)
            cached = model_fn_wan_video(context=condition, **inputs, **kwargs)
            assert all(torch.equal(a, b) for a, b in zip(raw, cached))
            assert all(a.count_nonzero() > 0 for a in cached)
    for handle in handles:
        handle.remove()
    assert counts == {"text": 4, "main": 6, "control": 6}
    assert condition.store.report()["misses"] == 2
    assert condition.store.report()["hits"] == 4
    assert len(model._rope_freq_cache) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_request_condition_sequential_offload(dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from accelerate import cpu_offload

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text = torch.nn.Linear(16, 128)
            self.ca = CrossAttention(128, 4)

        def forward(self, x: torch.Tensor, condition: torch.Tensor | StaticConditionCache) -> torch.Tensor:
            if isinstance(condition, StaticConditionCache):
                context = condition.project_context(self, lambda source, clip: self.text(source))
                return self.ca(x, context, _worldfoundry_condition_store=condition.store)
            return self.ca(x, self.text(condition))

    model = Model().eval()
    cpu_offload(model, execution_device=torch.device("cuda"))
    source = torch.randn(2, 4, 16, device="cuda")
    condition = StaticConditionCache(source)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        for _ in range(3):
            values = torch.randn(2, 8, 128, device="cuda")
            assert torch.equal(model(values, source), model(values, condition))
            assert model.text.weight.device.type == "meta"
            assert model.ca.k.weight.device.type == "meta"
    assert condition.store.report()["hits"] == 2


@pytest.mark.parametrize("image", [False, True])
def test_wan_fun_raw_training_and_cache_admission(image: bool) -> None:
    from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.video_x_fun.transformer import (
        WanTransformer3DModel,
    )

    model = WanTransformer3DModel(
        model_type="i2v" if image else "t2v",
        dim=64,
        ffn_dim=128,
        text_dim=16,
        text_len=8,
        freq_dim=16,
        in_dim=4,
        out_dim=4,
        num_heads=4,
        num_layers=1,
    ).train()
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.01)
    source = torch.randn(1, 4, 16, requires_grad=True)
    clip = torch.randn(1, 257, 1280, requires_grad=True) if image else None
    inputs = dict(x=torch.randn(1, 4, 2, 4, 4), t=torch.tensor([500.0]), seq_len=8)
    model(context=source, clip_fea=clip, **inputs).square().mean().backward()
    assert source.grad is not None and source.grad.count_nonzero() > 0
    assert model.blocks[0].cross_attn.k.weight.grad is not None
    if image:
        assert clip.grad is not None and clip.grad.count_nonzero() > 0
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval no-grad"):
        model(context=StaticConditionCache(source, clip), **inputs)
    model.eval()
    with pytest.raises(RuntimeError, match="eval no-grad"):
        model(context=StaticConditionCache(source, clip), **inputs)


@pytest.mark.parametrize("precision,dtype", [("fp32", torch.complex64), ("fp64", torch.complex128)])
def test_dualcamctrl_rotary_cache_preserves_original_assembly(precision: str, dtype: torch.dtype) -> None:
    model = WanModel(
        dim=96,
        in_dim=4,
        ffn_dim=192,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        num_heads=4,
        num_layers=1,
        has_image_input=False,
        eps=1e-6,
        patch_size=(1, 2, 2),
    )
    model.freqs = tuple(axis.to(dtype=dtype) for axis in model.freqs)
    for f, h, w in ((3, 2, 4), (2, 4, 3)):
        raw = torch.cat(
            [
                model.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                model.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                model.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        cached = model.rotary_frequencies((f, h, w), device=torch.device("cpu"), precision=precision)
        assert torch.equal(raw, cached)
        assert cached is model.rotary_frequencies((f, h, w), device=torch.device("cpu"), precision=precision)
    assert len(model._rope_freq_cache) == 2
