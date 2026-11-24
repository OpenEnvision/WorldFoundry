"""
This package provides the PixelSplatSynthesis module.

It re-exports the PixelSplatSynthesis class from the .pixelsplat_synthesis submodule
to allow for direct imports like `from your_package import PixelSplatSynthesis`.
"""

from .pixelsplat_synthesis import PixelSplatSynthesis

# Define the public API of this package to control what is imported by `from package import *`
__all__ = ["PixelSplatSynthesis"]
