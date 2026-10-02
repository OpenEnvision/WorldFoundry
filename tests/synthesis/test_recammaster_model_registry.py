"""Regression coverage for ReCamMaster checkpoint-class bindings."""

import pytest

# This test module imports worldfoundry code that requires the optional
# "ftfy" dependency at import time; skip when it is unavailable.
pytest.importorskip("ftfy")

from worldfoundry.synthesis.visual_generation.kling.recammaster_runtime import (
    recammaster_model_registry,
)
from worldfoundry.synthesis.visual_generation.kling.recammaster_runtime.models.wan_video_image_encoder import (
    WanImageEncoder,
)
from worldfoundry.synthesis.visual_generation.kling.recammaster_runtime.models.wan_video_text_encoder import (
    WanTextEncoder,
)


def test_registry_uses_checkpoint_compatible_encoder_classes() -> None:
    classes = recammaster_model_registry._MODEL_CLASSES
    assert classes["WanTextEncoder"] is WanTextEncoder
    assert classes["WanImageEncoder"] is WanImageEncoder
    assert callable(WanTextEncoder.state_dict_converter)
    assert callable(WanImageEncoder.state_dict_converter)
