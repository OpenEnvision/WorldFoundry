"""
Module for re-exporting the `RT1Synthesis` class.

This file serves as a convenient entry point to access the `RT1Synthesis` class
defined in the `rt1_synthesis` submodule, making it directly available
when importing from the parent package.
"""

from .rt1_synthesis import RT1Synthesis

__all__ = ["RT1Synthesis"]
