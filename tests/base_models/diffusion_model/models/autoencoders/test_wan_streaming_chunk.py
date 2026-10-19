"""Causal Wan chunks match dense decoding and own compact, stable caches."""

from types import SimpleNamespace

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.streaming import (
    StreamingWanDecoder,
    decode_wan_chunk,
    new_decode_cache,
)
from worldfoundry.core.execution.realtime.feature_cache import update_feature_cache


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


def _tiny_vae(kind, device="cpu"):
    if kind == "reference":
        from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.reference_21_streaming import WanVAE_
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
    return SimpleNamespace(model=model, scale=[torch.zeros(4, device=device), torch.ones(4, device=device)])


@pytest.mark.parametrize("kind", ["reference", "native"])
def test_dense_and_streaming_decode_match_across_chunks_and_reset(kind):
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
        torch.testing.assert_close(decoder.decode(latents[:, :, :2]), first, rtol=0, atol=0)


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
