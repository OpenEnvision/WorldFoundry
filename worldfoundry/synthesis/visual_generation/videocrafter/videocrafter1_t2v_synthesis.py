"""
This module defines a specialized wrapper for the VideoCrafter1 T2V (text-to-video) synthesis model.

It integrates the VideoCrafter model into a broader video synthesis runtime system,
providing a standardized interface for configuration, loading, and execution
of the model within the `worldfoundry` framework.
"""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .worldfoundry_runtime import VideoCrafter


class VideoCrafter1T2VSynthesis(RuntimeVideoSynthesis):
    """
    Concrete implementation of `RuntimeVideoSynthesis` for the VideoCrafter1 T2V model.

    This class serves as a configuration entry point for loading and running
    the VideoCrafter1 model. It specifies all model-specific metadata and
    paths required by the `RuntimeVideoSynthesis` base class to correctly
    initialize and manage the model's lifecycle.

    Attributes:
        MODEL_NAME (str): A unique identifier for this specific model instance.
        GENERATION_TYPE (str): Specifies the type of content generation this model performs (text-to-video).
        RUNTIME_CLS (type): The actual runtime class (`VideoCrafter`) that handles the model's loading and inference.
        PRIMARY_PATH_KEY (str): The key used in the runtime configuration to locate the model's primary checkpoint file.
        RUNTIME_CONFIG_PATH (str): The relative path to the default YAML configuration file for VideoCrafter models.
        RUNTIME_CONFIG_KEY (str): The key within the `RUNTIME_CONFIG_PATH` YAML file that corresponds to this model's specific configuration.
    """

    MODEL_NAME = "videocrafter1_t2v"
    GENERATION_TYPE = "t2v"
    RUNTIME_CLS = VideoCrafter
    PRIMARY_PATH_KEY = "ckpt_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/videocrafter/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
