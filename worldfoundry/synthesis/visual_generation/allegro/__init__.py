"""
Provides a high-level interface for the Allegro runtime environment.

This module re-exports the main `Allegro` class and `preprocess_images` function
from the `worldfoundry_runtime` submodule, and offers a utility function to determine
the root directory of the Allegro runtime assets. It serves as an entry point
for interacting with the Allegro evaluation system.
"""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """
    Determines and returns the absolute path to the 'allegro_runtime' directory.

    This directory is expected to contain various assets or configurations
    required by the Allegro runtime environment. The path is resolved relative
    to the current file's location.

    Returns:
        Path: An absolute `pathlib.Path` object pointing to the 'allegro_runtime' directory.
    """
    return Path(__file__).resolve().parent / "allegro_runtime"


from .worldfoundry_runtime import (  # noqa: E402 - Follows module-level definitions, as upstream.
    Allegro,
    preprocess_images,
)

__all__ = ["Allegro", "preprocess_images", "runtime_root"]
