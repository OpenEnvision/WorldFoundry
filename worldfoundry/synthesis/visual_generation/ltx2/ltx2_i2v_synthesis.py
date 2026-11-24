"""Provides an independent synthesis wrapper for the LTX2 Image-to-Video (I2V) model.

This module defines the `LTX2I2VSynthesis` class, which configures the runtime
for LTX2 I2V generation by setting model-specific attributes like name,
generation type, runtime class, and configuration paths.
"""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .ltx2_runtime import LTX2Video


class LTX2I2VSynthesis(RuntimeVideoSynthesis):
    """Configures an independent synthesis wrapper for the LTX2 Image-to-Video (I2V) model.

    This class inherits from `RuntimeVideoSynthesis` and specifies the necessary
    parameters for running the LTX2 I2V model. It defines the model's name,
    generation type, the concrete runtime class to be used (`LTX2Video`),
    the key for its primary checkpoint path, and its default runtime
    configuration file path and key within that file.
    """

    MODEL_NAME = "ltx2_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = LTX2Video
    PRIMARY_PATH_KEY = "checkpoint_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/ltx2/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
