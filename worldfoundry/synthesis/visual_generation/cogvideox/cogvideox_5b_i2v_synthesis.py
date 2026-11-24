"""
This module defines a concrete implementation of `RuntimeVideoSynthesis` for
integrating the CogVideoX 5 billion parameter "image-to-video" (i2v) model.

It provides a standardized wrapper for independent video generation using the
specified CogVideoX variant, allowing it to be managed within a broader
evaluation or synthesis framework.
"""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .worldfoundry_runtime import CogVideoX


class CogVideoX5bI2VSynthesis(RuntimeVideoSynthesis):
    """
    A concrete implementation of `RuntimeVideoSynthesis` specifically for the
    CogVideoX 5 billion parameter "image-to-video" (i2v) model.

    This class configures the necessary parameters to integrate and utilize the
    `CogVideoX` runtime for generating videos from images, acting as an
    independent synthesis wrapper within a larger system.
    """

    MODEL_NAME = "cogvideox_5b_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = CogVideoX
    PRIMARY_PATH_KEY = "model_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/cogvideox/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
