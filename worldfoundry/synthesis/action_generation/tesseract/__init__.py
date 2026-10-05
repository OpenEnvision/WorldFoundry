"""Utility functions for locating the Tesseract runtime directory.

This module provides a function to determine the absolute path to the
'tesseract_runtime' directory, which is typically expected to be
a sibling directory to where the current script resides.
"""
from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """
    Determines and returns the absolute path to the Tesseract runtime directory.

    This function calculates the path to the 'tesseract_runtime' directory.
    It assumes 'tesseract_runtime' is located within the same parent directory
    as the script defining this function.

    Example:
        If this file is at `/project/src/utils.py`, and 'tesseract_runtime'
        is at `/project/tesseract_runtime`, this function will return
        `/project/tesseract_runtime`.

    Returns:
        Path: An absolute Path object pointing to the Tesseract runtime directory.
    """
    # Resolve the absolute path of the current file, then get its parent directory,
    # and append the 'tesseract_runtime' subdirectory name.
    return Path(__file__).resolve().parent / 'tesseract_runtime'


__all__ = ["runtime_root"]