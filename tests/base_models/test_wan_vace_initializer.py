from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.contracts import (
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import (
    WanVideoDecoder,
)
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import (
    WanVaceLatentInitializer,
)
from worldfoundry.base_models.diffusion_model.recipes.wan import wan21_vace_14b_recipe
from worldfoundry.pipelines.wan.pipeline_wan_vace import Wan2p1VACEPipeline
from worldfoundry.pipelines.native_diffusion_video import NativeTextToVideoPipeline


class _FakeEncoder:
    dtype = torch.float32

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = pixels.shape
        latent_frames = 1 + (frames - 1) // 4
        value = pixels.mean(dim=(1, 2, 3, 4), keepdim=True)
        return value.expand(batch, 16, latent_frames, height // 8, width // 8).clone()


class _FakeVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.decoded_latents: torch.Tensor | None = None

    def decode(self, latents: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        self.decoded_latents = latents
        return latents


def _request(*, images: list[str], reference_count: int) -> DiffusionRequest:
    return DiffusionRequest(
        prompt="reference composition",
        height=16,
        width=16,
        num_frames=5,
        sampling=SamplingConfig(num_inference_steps=2, guidance_scale=5.0, seed=7),
        inputs={
            "images": images,
            "vace_reference_count": reference_count,
            "vace_context_scale": 1.0,
        },
    )


def test_vace_reference_latents_are_prefixed_and_trimmed_before_decode(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(np.full((16, 16, 3), 32, dtype=np.uint8)).save(first)
    Image.fromarray(np.full((16, 16, 3), 224, dtype=np.uint8)).save(second)
    request = _request(images=[str(first), str(second)], reference_count=2)

    initialized = WanVaceLatentInitializer().initialize_with_encoder(
        request,
        latent_encoder=_FakeEncoder(),
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert initialized.latents.shape == (1, 16, 4, 2, 2)
    context = initialized.conditioning["vace_context"]
    assert isinstance(context, torch.Tensor)
    assert context.shape == (1, 96, 4, 2, 2)
    assert torch.count_nonzero(context[:, 32:, :2]) == 0
    assert context[:, :16, 0].mean() < context[:, :16, 1].mean()

    vae = _FakeVAE()
    decoder = WanVideoDecoder(vae, device=torch.device("cpu"))
    decoded = decoder.decode(initialized.latents, request)
    assert decoded.shape == (1, 16, 2, 2, 2)
    assert vae.decoded_latents is not None
    assert torch.equal(vae.decoded_latents, initialized.latents[:, :, 2:])


def test_vace_rejects_reference_count_drift(tmp_path) -> None:
    image = tmp_path / "reference.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image)

    with pytest.raises(ValueError, match="must match"):
        WanVaceLatentInitializer().initialize_with_encoder(
            _request(images=[str(image)], reference_count=2),
            latent_encoder=_FakeEncoder(),
            generator=torch.Generator().manual_seed(7),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_vace_official_reference_defaults_use_shift_16() -> None:
    assert Wan2p1VACEPipeline.DEFAULT_SCHEDULER_OPTIONS == {"shift": 16.0}
    recipe = wan21_vace_14b_recipe()
    scheduler = next(component for component in recipe.components if component.key.kind == "scheduler")
    assert scheduler.options["shift"] == 16.0


def test_vace_accepts_studio_reference_images(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_call(self, **kwargs):
        del self
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(NativeTextToVideoPipeline, "__call__", fake_call)
    pipeline = object.__new__(Wan2p1VACEPipeline)
    references = [object(), object()]

    pipeline(prompt="compose", reference_images=references)

    assert captured["images"] == references
    assert captured["vace_reference_count"] == 2
