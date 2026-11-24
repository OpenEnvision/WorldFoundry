"""Module for the CogVideoX5bT2VSynthesis class, providing a specialized wrapper for text-to-video synthesis using the CogVideoX 5 billion parameter model."""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .worldfoundry_runtime import CogVideoX


class CogVideoX5bT2VSynthesis(RuntimeVideoSynthesis):
    """
    Independent synthesis wrapper for the `cogvideox_5b_t2v` model.

    This class specializes `RuntimeVideoSynthesis` for the CogVideoX 5 billion parameter
    text-to-video model. It configures the necessary metadata and runtime class
    for model loading and video generation, making it ready for use within the
    evaluation framework.

    Attributes:
        MODEL_NAME (str): The unique identifier for this specific model variant, "cogvideox_5b_t2v".
        GENERATION_TYPE (str): The type of generation task this model performs, "t2v" for text-to-video.
        RUNTIME_CLS (type): The base runtime class responsible for actual model inference, which is CogVideoX.
        PRIMARY_PATH_KEY (str): The key used in configuration to specify the primary model weight path.
        RUNTIME_CONFIG_PATH (str): The default file path to the runtime configuration YAML for CogVideoX.
        RUNTIME_CONFIG_KEY (str): The specific key within the runtime configuration to load
                                  defaults for this particular model.
    """

    MODEL_NAME = "cogvideox_5b_t2v"
    GENERATION_TYPE = "t2v"
    RUNTIME_CLS = CogVideoX
    PRIMARY_PATH_KEY = "model_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/cogvideox/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
