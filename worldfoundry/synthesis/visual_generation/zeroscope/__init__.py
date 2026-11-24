"""
Module for re-exporting the `ZeroScopeSynthesis` class.

This module serves as an entry point to make the `ZeroScopeSynthesis` class
readily accessible from the package's top level. The `ZeroScopeSynthesis` class
is responsible for handling the synthesis process using the ZeroScope model.
"""

from .zeroscope_synthesis import ZeroScopeSynthesis

__all__ = ["ZeroScopeSynthesis"]
