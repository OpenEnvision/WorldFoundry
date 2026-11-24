"""
This module provides utility functions for locating runtime-specific directories
within the 'genie_envisioner' project structure.
"""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """
    Determines and returns the absolute path to the 'genie_envisioner_runtime' directory.

    This directory is expected to be a sibling of the directory containing this
    Python file. It's intended to serve as a base for runtime-specific assets
    or configurations.

    Returns:
        Path: An absolute Path object representing the
              'genie_envisioner_runtime' directory.
    """
    # Resolve the absolute path of the current file, find its parent directory,
    # and then append the 'genie_envisioner_runtime' subdirectory name.
    return Path(__file__).resolve().parent / "genie_envisioner_runtime"


__all__ = ["runtime_root"]
