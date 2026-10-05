from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan.clip import (
    CLIPModel,
    VisionTransformer,
)


class _IdentityTransform:
    transforms = (lambda value: value,)


class _RecordingVisual:
    image_size = 4

    def __init__(self) -> None:
        self.seen_shape: tuple[int, ...] | None = None

    def visual(self, images: torch.Tensor, *, use_31_block: bool) -> torch.Tensor:
        assert use_31_block is True
        self.seen_shape = tuple(images.shape)
        return images


def test_clip_visual_accepts_upstream_sequence_of_cthw_samples() -> None:
    wrapper = CLIPModel.__new__(CLIPModel)
    wrapper.dtype = torch.float16
    wrapper.device = torch.device("cpu")
    wrapper.model = _RecordingVisual()
    wrapper.transforms = _IdentityTransform()

    output = wrapper.visual([torch.zeros(3, 1, 4, 4)])

    assert tuple(output.shape) == (1, 3, 4, 4)
    assert output.dtype == torch.float16
    assert wrapper.model.seen_shape == (1, 3, 4, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the offload boundary")
def test_vision_transformer_materializes_direct_embeddings_on_activation_device() -> None:
    model = VisionTransformer(
        image_size=4,
        patch_size=2,
        dim=4,
        mlp_ratio=2,
        out_dim=4,
        num_heads=1,
        num_layers=0,
        pool_type="token",
    )
    model.patch_embedding.to("cuda")
    model.dropout.to("cuda")
    model.pre_norm.to("cuda")

    output = model(torch.zeros(1, 3, 4, 4, device="cuda"), use_31_block=True)

    assert output.device.type == "cuda"
