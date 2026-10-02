"""Regression coverage for WorldCam checkpoint-class bindings."""

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

from worldfoundry.base_models.diffusion_model.models.encoders.wan import (
    WanTextEncoder,
)
from worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime import (
    model_registry,
)


def test_worldcam_registry_text_encoder_has_native_checkpoint_converter() -> None:
    classes = model_registry._MODEL_CLASSES

    assert classes["WanTextEncoder"] is WanTextEncoder
    converter = WanTextEncoder.state_dict_converter()
    state_dict = {"token_embedding.weight": object()}
    assert converter.from_civitai(state_dict) is state_dict
    assert converter.from_diffusers(state_dict) is state_dict
