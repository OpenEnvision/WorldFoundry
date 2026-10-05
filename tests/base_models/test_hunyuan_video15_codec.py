from __future__ import annotations

from types import SimpleNamespace

import torch

from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
from worldfoundry.base_models.diffusion_model.models.autoencoders.hunyuan_video.component import (
    HunyuanVideo15Codec,
)


class _FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.scaling_factor = 1.0
        self.tiling_calls: list[bool] = []

    def enable_tiling(self, enabled: bool = True) -> None:
        self.tiling_calls.append(enabled)

    def encode(self, images: torch.Tensor):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: images))

    def decode(self, latents: torch.Tensor):
        return SimpleNamespace(sample=latents)


def test_hunyuan_video15_codec_enables_spatial_tiling_for_full_resolution() -> None:
    vae = _FakeVAE()
    codec = HunyuanVideo15Codec(vae, tiled=True)
    tensor = torch.zeros(1, 3, 2, 4, 4)

    codec.encode(tensor)
    codec.decode(tensor, DiffusionRequest(prompt="test"))

    assert vae.tiling_calls == [True, True]
