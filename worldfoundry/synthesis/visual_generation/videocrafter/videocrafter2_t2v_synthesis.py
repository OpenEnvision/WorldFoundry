"""
Defines the VideoCrafter2T2VSynthesis class, a specialized runtime wrapper
for the VideoCrafter2 Text-to-Video (T2V) diffusion model.

This module configures the specific parameters required to integrate the
VideoCrafter2 T2V model into the worldfoundry synthesis framework, including
model identification, generation type, runtime class, and configuration paths.
"""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .worldfoundry_runtime import VideoCrafter


class VideoCrafter2T2VSynthesis(RuntimeVideoSynthesis):
    """
    A concrete implementation of RuntimeVideoSynthesis for the VideoCrafter2 Text-to-Video (T2V) model.

    This class defines the specific configuration and integration points required
    to use the VideoCrafter2 T2V model within the worldfoundry runtime framework.
    It specifies the model name, generation type, underlying runtime class,
    primary path key for checkpoints, and paths to runtime configuration files.
    """

    MODEL_NAME = "videocrafter2_t2v"
    GENERATION_TYPE = "t2v"
    RUNTIME_CLS = VideoCrafter
    PRIMARY_PATH_KEY = "ckpt_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/videocrafter/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
