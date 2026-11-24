"""
Defines a specialized wrapper for the 'cogvideox_2b_t2v' model,
integrating it into the `RuntimeVideoSynthesis` framework.

This module sets up the necessary configuration and runtime class
for performing text-to-video synthesis using the specified CogVideoX variant.
"""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .worldfoundry_runtime import CogVideoX


class CogVideoX2bT2VSynthesis(RuntimeVideoSynthesis):
    """
    A concrete implementation of `RuntimeVideoSynthesis` specifically configured
    for the 'cogvideox_2b_t2v' model.

    This class serves as a declarative wrapper, defining the model's
    identity, generation type (text-to-video), associated runtime class,
    and configuration paths for its deployment within the evaluation framework.
    """

    MODEL_NAME = "cogvideox_2b_t2v"
    GENERATION_TYPE = "t2v"
    RUNTIME_CLS = CogVideoX
    PRIMARY_PATH_KEY = "model_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/cogvideox/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME
