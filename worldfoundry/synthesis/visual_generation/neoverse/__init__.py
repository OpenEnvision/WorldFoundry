"""
This package provides a unified interface for interacting with the Neoverse visual generation system.

It re-exports key components, utilities, and default configurations
from submodules, making them directly accessible from the `neoverse` package
root for convenience.

Components include runtime management functions, path resolvers for Neoverse
assets, the official Neoverse runtime class, and the Neoverse synthesis
workflow class.
"""

from worldfoundry.synthesis.visual_generation.neoverse.runtime_env import (
    DEFAULT_NEOVERSE_LORA_NAME,
    DEFAULT_NEOVERSE_REPO,
    resolve_neoverse_lora_path,
    resolve_neoverse_model_dir,
    resolve_neoverse_reconstructor_path,
)
from worldfoundry.synthesis.visual_generation.neoverse.worldfoundry_runtime import NeoVerseOfficialRuntime

from .neoverse_synthesis import DEFAULT_PROMPT, NeoVerseSynthesis

__all__ = [
    "DEFAULT_NEOVERSE_LORA_NAME",
    "DEFAULT_NEOVERSE_REPO",
    "DEFAULT_PROMPT",
    "NeoVerseOfficialRuntime",
    "NeoVerseSynthesis",
    "resolve_neoverse_lora_path",
    "resolve_neoverse_model_dir",
    "resolve_neoverse_reconstructor_path",
]
