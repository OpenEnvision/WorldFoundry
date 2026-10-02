"""TI2V mask semantics shared by initialization, denoising, and step hooks."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DiffusionRequest,
)
from worldfoundry.base_models.diffusion_model.extensions.base import (
    DiffusionRunContext,
)
from worldfoundry.base_models.diffusion_model.extensions.frozen_mask import (
    FrozenLatentMaskExtension,
)
from worldfoundry.base_models.diffusion_model.models.initializers.wan.component import (
    WAN_DENOISE_MASK_IS_ALL_ONES,
    WanTextImageToVideoLatentInitializer,
)


class _LatentEncoder:
    dtype = torch.float32

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            pixels.shape[0],
            4,
            1,
            pixels.shape[-2] // 8,
            pixels.shape[-1] // 8,
            device=pixels.device,
            dtype=pixels.dtype,
        )


def _initializer() -> WanTextImageToVideoLatentInitializer:
    return WanTextImageToVideoLatentInitializer(
        channels=4,
        spatial_compression=8,
        temporal_compression=4,
    )


def _initialize(request: DiffusionRequest):
    return _initializer().initialize_with_encoder(
        request,
        latent_encoder=_LatentEncoder(),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_text_only_initializer_emits_explicit_all_ones_proof() -> None:
    initialized = _initialize(
        DiffusionRequest(prompt="test", height=16, width=16, num_frames=5)
    )

    assert initialized.conditioning[WAN_DENOISE_MASK_IS_ALL_ONES] is True
    assert torch.all(initialized.conditioning["denoise_mask"] == 1)


def test_image_initializer_marks_first_frame_frozen(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(image_path)

    initialized = _initialize(
        DiffusionRequest(
            prompt="test",
            height=16,
            width=16,
            num_frames=5,
            inputs={"image": str(image_path)},
        )
    )

    mask = initialized.conditioning["denoise_mask"]
    assert initialized.conditioning[WAN_DENOISE_MASK_IS_ALL_ONES] is False
    assert torch.count_nonzero(mask[:, :, :1]) == 0
    assert torch.all(mask[:, :, 1:] == 1)


def _extension_context(shared: dict[str, object]) -> DiffusionRunContext:
    return DiffusionRunContext(
        request=DiffusionRequest(prompt="test"),
        components=SimpleNamespace(),
        conditioning=Conditioning(positive={}, shared=shared),
        generator=torch.Generator(),
    )


def test_frozen_mask_step_is_identity_only_with_explicit_proof() -> None:
    latents = torch.randn(1, 4, 2, 2, 2)
    extension = FrozenLatentMaskExtension()
    context = _extension_context({WAN_DENOISE_MASK_IS_ALL_ONES: True})

    output = extension.after_step(context, latents)

    assert output is latents


def test_frozen_mask_does_not_infer_all_ones_from_tensor() -> None:
    latents = torch.randn(1, 4, 2, 2, 2)
    clean = torch.zeros_like(latents)
    mask = torch.ones_like(latents)
    extension = FrozenLatentMaskExtension()
    context = _extension_context(
        {"clean_latents": clean, "denoise_mask": mask}
    )

    output = extension.after_step(context, latents)

    assert output is not latents
    torch.testing.assert_close(output, latents)


def test_frozen_mask_rejects_non_boolean_semantic_flag() -> None:
    context = _extension_context({WAN_DENOISE_MASK_IS_ALL_ONES: torch.tensor(True)})
    with pytest.raises(TypeError, match="must be a bool"):
        FrozenLatentMaskExtension().after_step(context, torch.zeros(1))
