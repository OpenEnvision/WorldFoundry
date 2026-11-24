"""CPU contracts for the Wan codec's ``std_reciprocal_in_codec_dtype`` latent scale option."""

from __future__ import annotations

import pytest
import torch

from worldfoundry.base_models.diffusion_model.components import (
    ComponentBuildContext,
    ComponentKey,
    ComponentKind,
)
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import (
    component as wan_component,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import (
    WanVideoDecoder,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import (
    WanVideoVAE,
)
from worldfoundry.base_models.diffusion_model.optimizations import RuntimePolicy


def _build_tiny_wan21_codec(monkeypatch, **component_options: object) -> WanVideoDecoder:
    """Build the 16-channel codec through its recipe factory around a small random Wan2.1 VAE."""

    def fake_load(self, spec, checkpoint, policy):
        del self, spec, checkpoint
        torch.manual_seed(0)
        return WanVideoVAE(z_dim=16, base_dim=8).to(dtype=policy.dtype)

    monkeypatch.setattr(wan_component.NativeModuleLoader, "load", fake_load)
    decoder = wan_component.build_wan_video_decoder(
        ComponentBuildContext(
            model_id="wan-codec",
            key=ComponentKey(ComponentKind.DECODER),
            policy=RuntimePolicy(device="cpu", dtype=torch.bfloat16),
            checkpoints={"weights": CheckpointSpec(source="unused.pth")},
            component_options=component_options,
        )
    )
    assert isinstance(decoder, WanVideoDecoder)
    return decoder


def test_wan_codec_keeps_float32_std_reciprocal_by_default(monkeypatch) -> None:
    decoder = _build_tiny_wan21_codec(monkeypatch, dtype=torch.bfloat16)

    mean, inverse_std = decoder.vae.scale
    assert decoder.dtype is torch.bfloat16
    assert mean is decoder.vae.mean
    assert inverse_std.dtype == torch.float32
    assert torch.equal(inverse_std, 1.0 / decoder.vae.std)


def test_wan_codec_std_reciprocal_in_codec_dtype_matches_self_forcing_wrapper(monkeypatch) -> None:
    decoder = _build_tiny_wan21_codec(
        monkeypatch,
        dtype=torch.bfloat16,
        std_reciprocal_in_codec_dtype=True,
    )

    mean, inverse_std = decoder.vae.scale
    # Self-Forcing's WanVAEWrapper: scale = [mean.to(dtype), 1.0 / std.to(dtype)].
    official = [decoder.vae.mean.to(torch.bfloat16), 1.0 / decoder.vae.std.to(torch.bfloat16)]
    assert mean is decoder.vae.mean
    assert inverse_std.dtype == torch.bfloat16
    assert torch.equal(inverse_std, official[1])
    assert not torch.equal((1.0 / decoder.vae.std).to(torch.bfloat16), official[1])

    latents = torch.randn(1, 16, 3, 2, 2, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = decoder.vae.model.decode(latents.to(torch.bfloat16), official).float().clamp_(-1, 1)
        whole = decoder.decode(latents)
        state = decoder.new_stream_state()
        streamed = torch.cat(
            [
                decoder.decode_chunk(latents[:, :, :1], state=state, is_first_chunk=True),
                decoder.decode_chunk(latents[:, :, 1:], state=state),
            ],
            dim=2,
        )
    assert torch.equal(whole.float(), expected)
    assert torch.equal(streamed, expected)


def test_wan_codec_std_reciprocal_option_requires_a_bool(monkeypatch) -> None:
    with pytest.raises(TypeError, match="std_reciprocal_in_codec_dtype must be a bool"):
        _build_tiny_wan21_codec(monkeypatch, std_reciprocal_in_codec_dtype="yes")
