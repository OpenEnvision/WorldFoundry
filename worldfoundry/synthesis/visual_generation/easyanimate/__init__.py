"""This module provides the main entry point and utilities for interacting with the EasyAnimate runtime environment.

It re-exports key components from the `worldfoundry_runtime` subpackage, including the `EasyAnimate` class
for animating models, functions to resolve default model paths, and configuration paths.
It also defines a utility function `runtime_root` to locate the base directory for EasyAnimate runtime assets.
"""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """
    Determines and returns the root directory for EasyAnimate runtime assets.

    This function calculates the path to a directory named 'easyanimate_runtime' located
    alongside the current Python file. This directory is expected to contain necessary
    runtime files for EasyAnimate operations.

    Returns:
        Path: An absolute path to the `easyanimate_runtime` directory.
    """
    # Resolve the absolute path of the current file and then get its parent directory
    # Finally, append 'easyanimate_runtime' to locate the specific runtime asset directory.
    return Path(__file__).resolve().parent / "easyanimate_runtime"


from .worldfoundry_runtime import (  # noqa: E402 - Follows module-level definitions, as upstream.
    EasyAnimate,
    default_model_path,
    resolve_config_path,
)

# Defines the public API of this module when a user imports using `from easyanimate import *`.
# It ensures that only the specified names are exported, making the module's interface clear.
__all__ = ["EasyAnimate", "default_model_path", "resolve_config_path", "runtime_root"]
