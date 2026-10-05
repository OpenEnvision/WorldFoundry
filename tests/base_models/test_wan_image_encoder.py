from __future__ import annotations

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan.image import (
    WanImageEncoderStateDictConverter,
)


def test_civitai_converter_keeps_visual_and_textual_clip_weights() -> None:
    state_dict = {
        "visual.patch_embedding.weight": torch.ones(1),
        "textual.token_embedding.weight": torch.ones(1),
    }

    converted = WanImageEncoderStateDictConverter().from_civitai(state_dict)

    assert set(converted) == {
        "model.visual.patch_embedding.weight",
        "model.textual.token_embedding.weight",
    }
