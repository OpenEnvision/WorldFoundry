"""Provides the `AllegroTi2VSynthesis` class, a specialized wrapper for integrating the 'allegro_ti2v'
model into the video synthesis runtime system.

This module defines the necessary configurations and linkages for the Allegro Text-to-Video (Ti2V)
synthesis process, inheriting from `RuntimeVideoSynthesis`.
"""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.allegro.worldfoundry_runtime import Allegro

from ..runtime_video_synthesis import RuntimeVideoSynthesis


class AllegroTi2VSynthesis(RuntimeVideoSynthesis):
    """This class serves as an independent synthesis wrapper for the 'allegro_ti2v' model.

    It extends `RuntimeVideoSynthesis` to provide specific configurations and runtime
    integration for Text-to-Video (Ti2V) generation using the Allegro framework.
    It defines critical metadata such as model name, generation type, runtime class,
    and paths to configuration files required for the synthesis process.
    """

    MODEL_NAME = "allegro_ti2v"
    GENERATION_TYPE = "i2v"
    RUNTIME_CLS = Allegro
    PRIMARY_PATH_KEY = "model_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/allegro/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
