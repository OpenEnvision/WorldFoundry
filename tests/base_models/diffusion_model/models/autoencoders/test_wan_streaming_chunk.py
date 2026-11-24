"""Causal Wan chunks match dense decoding and own compact, stable caches."""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.streaming import (
    StreamingWanDecoder,
    decode_wan_chunk,
    new_decode_cache,
)
from worldfoundry.core.execution.realtime.feature_cache import update_feature_cache

if TYPE_CHECKING:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import WanVideoVAE


@pytest.mark.parametrize("batch", [1, 2])
def test_feature_tail_is_compact_and_reuses_storage(batch):
    volume = torch.randn(batch, 4, 16, 3, 5).contiguous(memory_format=torch.channels_last_3d)
    tail = volume[:, :, -2:]
    with torch.no_grad():
        cache = update_feature_cache(None, tail)
        assert cache.untyped_storage().nbytes() == cache.numel() * cache.element_size()
        assert cache.is_contiguous(memory_format=torch.channels_last_3d)
        pointer = cache.data_ptr()
        updated = update_feature_cache(cache, tail + 1)
        assert updated.data_ptr() == pointer
        torch.testing.assert_close(updated, tail + 1, rtol=0, atol=0)
        independent = update_feature_cache(None, tail)
        assert independent.data_ptr() != pointer


def _tiny_vae(kind: str, device: str = "cpu") -> SimpleNamespace:
    if kind == "reference":
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21_streaming import WanVAE_
    elif kind == "forcing":
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21 import WanVAE_
    else:
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import VideoVAE_ as WanVAE_
    model = (
        WanVAE_(
            dim=8,
            z_dim=4,
            dim_mult=[1, 2, 2, 2],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True],
        )
        .eval()
        .to(device)
    )
    return SimpleNamespace(
        model=model,
        scale=[torch.zeros(4, device=device), torch.ones(4, device=device)],
        parameters=model.parameters,
    )


@pytest.mark.parametrize("kind", ["reference", "native", "forcing"])
def test_dense_and_streaming_decode_match_across_chunks_and_reset(kind: str) -> None:
    torch.manual_seed(13)
    vae = _tiny_vae(kind)
    latents = torch.randn(1, 4, 6, 2, 2)
    with torch.no_grad():
        expected = vae.model.decode(latents, vae.scale).float().clamp(-1, 1)
        decoder = StreamingWanDecoder(vae)
        first = decoder.decode(latents[:, :, :2])
        steady = decoder.decode(latents[:, :, 2:4])
        pointers = [value.data_ptr() for value in decoder.cache if isinstance(value, torch.Tensor)]
        last = decoder.decode(latents[:, :, 4:])
        assert pointers == [value.data_ptr() for value in decoder.cache if isinstance(value, torch.Tensor)]
        output = torch.cat((first, steady, last), dim=2)
        assert first.shape[2] == 5 and steady.shape[2] == last.shape[2] == 8
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        for value in decoder.cache:
            if isinstance(value, torch.Tensor):
                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        other = StreamingWanDecoder(vae)
        torch.testing.assert_close(other.decode(latents[:, :, :2]), first, rtol=0, atol=0)
        decoder.reset()
        assert all(value is None for value in decoder.cache)
        assert decoder.report()["latent_decode_calls"] == decoder.report()["cache_bytes"] == 0
        changed = latents.neg()
        changed_expected = vae.model.decode(changed, vae.scale).float().clamp(-1, 1)
        torch.testing.assert_close(decoder.decode(changed), changed_expected, rtol=0, atol=0)
        decoder.reset()
        repeated = torch.cat([decoder.decode(latents[:, :, :1]), decoder.decode(latents[:, :, 1:])], dim=2)
        torch.testing.assert_close(repeated, output, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("flavor", ["native", "reference"])
@pytest.mark.parametrize("tf32,atol", [(False, 2e-4), (True, 2e-3)])
def test_small_chunk_compile_and_channels_last_match_eager(monkeypatch, flavor, tf32, atol):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", tf32)
    torch.manual_seed(13)
    vae = _tiny_vae(flavor, "cuda")
    from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

    convert_convolution_weight_layouts(vae.model, conv2d=True, conv3d=True)
    latents = torch.randn(1, 4, 3, 2, 2, device="cuda")
    decoder = StreamingWanDecoder(vae, channels_last_3d=True, compile_decoder=True)
    cache = new_decode_cache(vae.model)
    with torch.no_grad():
        expected = torch.cat(
            [
                decode_wan_chunk(vae.model, vae.scale, part, cache, channels_last_3d=True)
                for part in (latents[:, :, :1], latents[:, :, 1:])
            ],
            dim=2,
        )
        actual = torch.cat([decoder.decode(latents[:, :, :1]), decoder.decode(latents[:, :, 1:])], dim=2)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=atol)
    assert (actual - expected).norm() / expected.norm() < (1e-3 if tf32 else 1e-5)
    assert decoder.report()["compiled_decode_calls"] == 3


def _snapshot_cache(cache: list) -> list:
    return [value.clone() if isinstance(value, torch.Tensor) else value for value in cache]


def _assert_cache_matches(cache: list, snapshot: list) -> None:
    for actual, expected in zip(cache, snapshot, strict=True):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected


@pytest.mark.parametrize("kind", ["reference", "native", "forcing"])
def test_streaming_decode_failure_clears_partial_history_and_restarts(kind: str) -> None:
    torch.manual_seed(14)
    vae = _tiny_vae(kind)
    latents = torch.randn(1, 4, 3, 2, 2)
    decoder, other = StreamingWanDecoder(vae), StreamingWanDecoder(vae)
    decoder.decode(latents[:, :, :1])
    other.decode(-latents[:, :, :1])
    other_cache = _snapshot_cache(other.cache)
    other_report = other.report()
    previous_tail = decoder.cache[0]
    assert isinstance(previous_tail, torch.Tensor)
    previous_tail = previous_tail.clone()
    error = RuntimeError("later latent failure")
    completed = 0

    def fail_second_step(module: torch.nn.Module, args: tuple, output: object) -> None:
        nonlocal completed
        completed += 1
        if completed == 2:
            current_tail = decoder.cache[0]
            assert isinstance(current_tail, torch.Tensor)
            assert not torch.equal(current_tail, previous_tail)
            raise error

    hook = vae.model.decoder.register_forward_hook(fail_second_step)
    try:
        with pytest.raises(RuntimeError, match="later latent failure") as caught:
            decoder.decode(latents[:, :, 1:])
        assert caught.value is error
    finally:
        hook.remove()
    assert completed == 2
    assert all(value is None for value in decoder.cache)
    assert decoder.report()["latent_decode_calls"] == decoder.report()["compiled_decode_calls"] == 0
    assert decoder.report()["cache_bytes"] == 0
    _assert_cache_matches(other.cache, other_cache)
    assert other.report() == other_report
    fresh = StreamingWanDecoder(vae)
    torch.testing.assert_close(decoder.decode(latents), fresh.decode(latents), rtol=0, atol=0)
    torch.testing.assert_close(
        other.decode(-latents[:, :, 1:]),
        StreamingWanDecoder(vae).decode(-latents)[:, :, 1:],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("kind", ["reference", "native", "forcing"])
def test_streaming_decode_preserves_channels_last_inputs_and_resident_dtype(kind: str) -> None:
    from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

    torch.manual_seed(15)
    vae = _tiny_vae(kind)
    vae.model.double()
    vae.scale = [torch.arange(4).float() / 10, torch.arange(1, 5).float()]
    convert_convolution_weight_layouts(vae.model, conv2d=True, conv3d=True)
    latents = torch.randn(2, 4, 3, 2, 2)
    seen = []

    def check_layout(module: torch.nn.Module, args: tuple) -> None:
        value = args[0]
        assert value.dtype == torch.float64
        assert value.device == vae.model.conv2.weight.device
        assert value.is_contiguous(memory_format=torch.channels_last_3d)
        if module is vae.model.decoder:
            assert value.shape[2] == 1
        seen.append(module)

    hooks = [module.register_forward_pre_hook(check_layout) for module in (vae.model.conv2, vae.model.decoder)]
    try:
        whole = StreamingWanDecoder(vae, channels_last_3d=True).decode(latents)
        split = StreamingWanDecoder(vae, channels_last_3d=True)
        actual = torch.cat([split.decode(latents[:, :, :1]), split.decode(latents[:, :, 1:])], dim=2)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(seen) == 9
    assert actual.dtype == torch.float32
    assert actual.shape == (2, 3, 9, 16, 16)
    torch.testing.assert_close(actual, whole, rtol=0, atol=0)
    for value in split.cache:
        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.float64
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


@pytest.mark.parametrize("kind", ["reference", "native", "forcing"])
def test_streaming_compile_reuses_fullgraph_wrapper_across_requests_and_native_component(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import WanVideoDecoder
    from worldfoundry.core.acceleration.convolution_layout import convert_convolution_weight_layouts

    torch.manual_seed(16)
    vae = _tiny_vae(kind)
    convert_convolution_weight_layouts(vae.model, conv2d=True, conv3d=True)
    latents = torch.randn(1, 4, 2, 2, 2)
    expected = StreamingWanDecoder(vae, channels_last_3d=True).decode(latents)
    changed_expected = StreamingWanDecoder(vae, channels_last_3d=True).decode(-latents)
    compile_calls = []
    compile_fn = torch.compile
    monkeypatch.setenv("WORLDFOUNDRY_COMPILE_CACHE_DIR", str(tmp_path))

    def compile_eager(module: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        compile_calls.append((module, kwargs))
        return cast(torch.nn.Module, compile_fn(module, backend="eager", dynamic=False, fullgraph=True))

    monkeypatch.setattr(torch, "compile", compile_eager)
    first = StreamingWanDecoder(vae, channels_last_3d=True, compile_decoder=True)
    second = StreamingWanDecoder(vae, channels_last_3d=True, compile_decoder=True)
    actual = torch.cat([first.decode(latents[:, :, :1]), first.decode(latents[:, :, 1:])], dim=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    first.reset()
    torch.testing.assert_close(first.decode(-latents), changed_expected, rtol=0, atol=0)
    torch.testing.assert_close(second.decode(-latents), changed_expected, rtol=0, atol=0)
    first.reset()
    torch.testing.assert_close(first.decode(latents), expected, rtol=0, atol=0)
    assert first.decoder is second.decoder
    if kind == "native":
        component = WanVideoDecoder(
            cast("WanVideoVAE", vae),
            device=torch.device("cpu"),
            channels_last_3d_requested=True,
            streaming_compile=True,
        )
        torch.testing.assert_close(
            component.decode_chunk(latents, state=component.new_stream_state()),
            expected,
            rtol=0,
            atol=0,
        )
        assert first.decoder is component._streaming_decoder
        runtime = component.runtime_optimization_report()["runtime"]
        assert isinstance(runtime, dict)
        assert runtime["streaming_compile_effective"] is True
        assert runtime["streaming_compiled_decode_calls"] == 2
    assert compile_calls == [
        (
            vae.model.decoder,
            {
                "backend": "inductor",
                "dynamic": False,
                "fullgraph": True,
                "options": {"triton.cudagraphs": False},
            },
        )
    ]
    assert first.report()["compile_effective"] is True
    assert first.report()["compiled_decode_calls"] == 2


def test_streaming_compile_receipts_follow_returned_callable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import WanVideoDecoder

    vae = _tiny_vae("native")
    latents = torch.randn(1, 4, 1, 2, 2)
    monkeypatch.setenv("WORLDFOUNDRY_COMPILE_CACHE_DIR", str(tmp_path))

    def return_eager(module: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        return module

    monkeypatch.setattr(torch, "compile", return_eager)
    decoder = StreamingWanDecoder(vae, compile_decoder=True)
    component = WanVideoDecoder(cast("WanVideoVAE", vae), device=torch.device("cpu"), streaming_compile=True)
    expected = decoder.decode(latents)
    actual = component.decode_chunk(latents, state=component.new_stream_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert decoder.decoder is component._streaming_decoder is vae.model.decoder
    assert decoder.report()["compile_requested"] is True
    assert decoder.report()["compile_effective"] is False
    assert decoder.report()["latent_decode_calls"] == 1
    assert decoder.report()["compiled_decode_calls"] == 0
    runtime = component.runtime_optimization_report()["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["streaming_compile_effective"] is False
    assert runtime["streaming_latent_decode_calls"] == 1
    assert runtime["streaming_compiled_decode_calls"] == 0


def test_streaming_compile_errors_propagate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import WanVideoDecoder

    vae = _tiny_vae("native")
    error = RuntimeError("compile failed")
    monkeypatch.setenv("WORLDFOUNDRY_COMPILE_CACHE_DIR", str(tmp_path))

    def fail_compile(*args: object, **kwargs: object) -> torch.nn.Module:
        raise error

    monkeypatch.setattr(torch, "compile", fail_compile)
    with pytest.raises(RuntimeError, match="compile failed") as caught:
        StreamingWanDecoder(vae, compile_decoder=True)
    assert caught.value is error
    component = WanVideoDecoder(cast("WanVideoVAE", vae), device=torch.device("cpu"), streaming_compile=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="compile failed") as caught:
            component.decode_chunk(torch.randn(1, 4, 1, 2, 2), state=component.new_stream_state())
        assert caught.value is error
        assert component._streaming_decoder is None
        runtime = component.runtime_optimization_report()["runtime"]
        assert isinstance(runtime, dict)
        assert runtime["streaming_latent_decode_calls"] == 0


def test_component_streaming_failure_preserves_other_request_and_lifetime_counts() -> None:
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import WanVideoDecoder

    vae = _tiny_vae("native")
    component = WanVideoDecoder(cast("WanVideoVAE", vae), device=torch.device("cpu"))
    state = cast(dict[str, Any], component.new_stream_state())
    other = cast(dict[str, Any], component.new_stream_state())
    latents = torch.randn(1, 4, 3, 2, 2)
    component.decode_chunk(latents[:, :, :1], state=state)
    component.decode_chunk(-latents[:, :, :1], state=other)
    other_cache = _snapshot_cache(other["decode"])
    encoder_cache = _snapshot_cache(state["encode"])
    runtime = component.runtime_optimization_report()["runtime"]
    completed = 0

    def fail_second_step(module: torch.nn.Module, args: tuple, output: object) -> None:
        nonlocal completed
        completed += 1
        if completed == 2:
            raise RuntimeError("later latent failure")

    hook = vae.model.decoder.register_forward_hook(fail_second_step)
    try:
        with pytest.raises(RuntimeError, match="later latent failure"):
            component.decode_chunk(latents[:, :, 1:], state=state)
    finally:
        hook.remove()
    assert completed == 2
    assert all(value is None for value in state["decode"])
    _assert_cache_matches(other["decode"], other_cache)
    _assert_cache_matches(state["encode"], encoder_cache)
    assert component.runtime_optimization_report()["runtime"] == runtime
    actual = component.decode_chunk(latents, state=state)
    torch.testing.assert_close(actual, StreamingWanDecoder(vae).decode(latents), rtol=0, atol=0)
    final_runtime = component.runtime_optimization_report()["runtime"]
    assert isinstance(final_runtime, dict)
    assert final_runtime["streaming_latent_decode_calls"] == 5
