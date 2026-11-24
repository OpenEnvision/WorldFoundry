"""Module defining the LTXVideoI2VSynthesis class, an independent synthesis wrapper for the ltx_video_i2v model."""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .ltx_video_runtime import LTXVideo


class LTXVideoI2VSynthesis(RuntimeVideoSynthesis):
    """
    Represents an independent synthesis wrapper for the 'ltx_video_i2v' model.

    This class configures the `RuntimeVideoSynthesis` base class for the specific
    'Image-to-Video' (i2v) synthesis task using the LTXVideo runtime.
    It defines model-specific attributes such as its name, generation type,
    associated runtime class, primary configuration path key, and default
    runtime configuration paths.
    """

    MODEL_NAME = "ltx_video_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = LTXVideo
    PRIMARY_PATH_KEY = "model_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/ltx_video/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
