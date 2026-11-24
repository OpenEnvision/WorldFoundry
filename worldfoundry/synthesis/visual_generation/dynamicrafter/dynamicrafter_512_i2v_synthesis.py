"""
Defines a specific runtime synthesis wrapper for the 'dynamicrafter_512_i2v' model.

This module sets up the necessary configuration and class references for integrating
DynamiCrafter with a standardized video synthesis runtime, making it easily
pluggable into evaluation or deployment systems.
"""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.dynamicrafter import DynamiCrafter

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class DynamiCrafter512I2VSynthesis(RuntimeVideoSynthesis):
    """
    Independent synthesis wrapper for the `dynamicrafter_512_i2v` model.

    This class extends `RuntimeVideoSynthesis` to provide specific configuration
    for the DynamiCrafter model when used for image-to-video (i2v) generation
    with 512x512 resolution. It defines the model name, generation type,
    the concrete runtime class, and paths for configuration and checkpoints.
    """

    MODEL_NAME = "dynamicrafter_512_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = DynamiCrafter
    PRIMARY_PATH_KEY = "ckpt_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/dynamicrafter/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
