"""
Initializes the package and exposes the main LAPA Synthesis component.

This module serves as an entry point for the LAPA Synthesis functionality,
exporting the `LAPASynthesis` class for direct access when importing from this package.
"""

from .lapa_synthesis import LAPASynthesis

__all__ = ["LAPASynthesis"]
