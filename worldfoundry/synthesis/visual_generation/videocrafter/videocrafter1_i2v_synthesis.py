from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .worldfoundry_runtime import VideoCrafter


class VideoCrafter1I2VSynthesis(RuntimeVideoSynthesis):
    """Independent synthesis wrapper for videocrafter1_i2v."""

    MODEL_NAME = "videocrafter1_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = VideoCrafter
    PRIMARY_PATH_KEY = "ckpt_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/videocrafter/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
