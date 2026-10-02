from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.encoders.wan.variants.video_x_fun import (
    CLIPModel,
)


class _IdentityTransform:
    transforms = (lambda value: value,)


class _MetaVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.empty((), device="meta"))
        self.image_size = 4
        self.seen_device: torch.device | None = None

    def visual(self, frames: torch.Tensor, *, use_31_block: bool) -> torch.Tensor:
        assert use_31_block is True
        self.seen_device = frames.device
        return frames


def test_video_x_fun_clip_wrapper_does_not_copy_offloaded_inputs_to_meta() -> None:
    wrapper = CLIPModel.__new__(CLIPModel)
    nn.Module.__init__(wrapper)
    wrapper.model = _MetaVisual()
    wrapper.transforms = _IdentityTransform()

    output = wrapper([torch.zeros(1, 3, 4, 4)])

    assert output.device.type == "cpu"
    assert wrapper.model.seen_device == torch.device("cpu")
