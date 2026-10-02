from __future__ import annotations

import torch

from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.models.autoencoders.cosmos1.component import Cosmos1VideoCodec


class _VideoEncoder(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :, :16]


class _IdentityDecoder(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _codec(*, latent_scale: float = 0.5) -> Cosmos1VideoCodec:
    return Cosmos1VideoCodec(
        _VideoEncoder(),
        _IdentityDecoder(),
        image_mean=torch.zeros(16),
        image_std=torch.ones(16),
        video_mean=torch.zeros(16 * 16),
        video_std=torch.ones(16 * 16),
        dtype=torch.float32,
        latent_scale=latent_scale,
    )


def test_cosmos1_codec_applies_predict1_sigma_data_scale_symmetrically() -> None:
    codec = _codec()
    pixels = torch.randn(1, 16, 121, 2, 2)
    encoded_pixels = pixels[:, :, :16]

    latents = codec.encode(pixels)
    request = DiffusionRequest(
        prompt="",
        height=16,
        width=16,
        num_frames=121,
        sampling=SamplingConfig(num_inference_steps=1),
    )

    assert torch.equal(latents, encoded_pixels * 0.5)
    assert torch.equal(codec.decode(latents, request), encoded_pixels)


def test_cosmos1_codec_rejects_non_positive_latent_scale() -> None:
    try:
        _codec(latent_scale=0.0)
    except ValueError as error:
        assert "latent_scale must be positive" in str(error)
    else:
        raise AssertionError("expected non-positive Cosmos latent scale to be rejected")
