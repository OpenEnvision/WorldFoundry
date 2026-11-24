"""
Module for configuring and wrapping the DynamiCrafter 1024 i2v model for video synthesis.

This module defines a specialized class, `DynamiCrafter1024I2VSynthesis`,
which extends `RuntimeVideoSynthesis` to provide a specific configuration
for the `dynamicrafter_1024_i2v` model. It sets up various class attributes
to specify model details, generation type, and configuration paths,
allowing for standardized integration within a larger synthesis framework.
"""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.dynamicrafter import DynamiCrafter

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class DynamiCrafter1024I2VSynthesis(RuntimeVideoSynthesis):
    """
    Independent synthesis wrapper for the `dynamicrafter_1024_i2v` model.

    This class configures the base `RuntimeVideoSynthesis` with specific
    parameters required to run the DynamiCrafter model at 1024 resolution
    for image-to-video generation. It specifies the model name, generation type,
    the underlying runtime class (`DynamiCrafter`), the key for the primary
    model checkpoint path, and the paths to its runtime configuration files.
    """

    MODEL_NAME = "dynamicrafter_1024_i2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = DynamiCrafter
    PRIMARY_PATH_KEY = "ckpt_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/dynamicrafter/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
