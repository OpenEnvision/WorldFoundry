"""Provides an independent synthesis wrapper for the 'ltx2_3_i2v' model.

This module defines the LTX23I2VSynthesis class, which configures the
runtime environment for performing video synthesis using the LTX2_3
model specifically for image-to-video (i2v) generation. It inherits
from RuntimeVideoSynthesis to leverage a standardized interface for
model integration and execution.
"""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .ltx2_runtime import LTX2Video


class LTX23I2VSynthesis(RuntimeVideoSynthesis):
    """An independent synthesis wrapper for the 'ltx2_3_i2v' model.

    This class extends `RuntimeVideoSynthesis` to provide a concrete
    implementation for handling the LTX2_3 image-to-video (i2v) model.
    It defines all necessary configuration parameters, such as the model name,
    generation type, and paths to runtime configuration files, to enable
    its integration and execution within a larger synthesis framework.
    """

    MODEL_NAME = "ltx2_3_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = LTX2Video
    PRIMARY_PATH_KEY = "checkpoint_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/ltx2/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
