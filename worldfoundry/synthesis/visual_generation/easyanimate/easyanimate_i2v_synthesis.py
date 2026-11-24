"""Provides a specialized wrapper for the EasyAnimate Image-to-Video (I2V) synthesis model.

This module integrates the EasyAnimate I2V model into a broader video synthesis runtime
framework, allowing it to be used consistently with other synthesis models.
"""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.easyanimate.worldfoundry_runtime import EasyAnimate

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class EasyAnimateI2VSynthesis(RuntimeVideoSynthesis):
    """A specialized wrapper class for performing Image-to-Video (I2V) synthesis using the EasyAnimate model.

    This class extends `RuntimeVideoSynthesis` and configures it specifically for the
    EasyAnimate I2V model, defining its name, generation type, runtime class, and
    configuration paths.
    """

    MODEL_NAME = "easyanimate_i2v"  # Identifier for the EasyAnimate I2V model.
    GENERATION_TYPE = "i2v"  # Specifies the type of generation, in this case, Image-to-Video.
    RUNTIME_CLS = EasyAnimate  # The underlying EasyAnimate runtime class used for synthesis.
    PRIMARY_PATH_KEY = (
        "model_path"  # The key in the runtime configuration pointing to the primary model checkpoint path.
    )
    RUNTIME_CONFIG_PATH = "models/runtime/configs/easyanimate/runtime_defaults.yaml"  # Default path to the runtime configuration YAML file.
    RUNTIME_CONFIG_KEY = MODEL_NAME  # The specific key within the runtime configuration to load for this model.
