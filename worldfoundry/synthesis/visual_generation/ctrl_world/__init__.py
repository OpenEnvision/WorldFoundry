"""
This module provides utility functions for locating the runtime root directory
of the 'ctrl_world' application.

It defines `runtime_root()` to determine the path to the 'ctrl_world_runtime'
directory relative to the current script's location.
"""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """
    Determines and returns the absolute path to the 'ctrl_world_runtime' directory.

    This directory is expected to be a sibling directory named 'ctrl_world_runtime'
    to the directory containing the current script.

    Returns:
        Path: The absolute path to the 'ctrl_world_runtime' directory.
    """
    # Get the absolute path of the directory containing this script,
    # then append 'ctrl_world_runtime' to locate the runtime root.
    return Path(__file__).resolve().parent / "ctrl_world_runtime"


__all__ = ["runtime_root"]
