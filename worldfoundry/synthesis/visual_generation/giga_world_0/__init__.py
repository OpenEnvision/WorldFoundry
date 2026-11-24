"""
This module provides utility functions for resolving application-specific file paths,
such as the designated runtime root directory.
"""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """
    Calculates and returns the absolute path to the 'giga_world_0_runtime' directory.

    This directory is assumed to be a sibling directory to the parent directory
    of the current module. It ensures a consistent location for runtime assets
    regardless of how the application is executed.

    Returns:
        Path: The absolute path to the 'giga_world_0_runtime' directory.
    """
    return Path(__file__).resolve().parent / "giga_world_0_runtime"


__all__ = ["runtime_root"]
