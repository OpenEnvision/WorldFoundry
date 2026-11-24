"""
Package initialization file for the open_magvit2 library.

This module serves to expose the main components (runtime and synthesis classes)
of the open_magvit2 library directly under the package namespace,
allowing for easier access when the package is imported.
"""

from .open_magvit2_synthesis import OpenMAGVIT2Synthesis
from .worldfoundry_runtime import OpenMAGVIT2Runtime

# Defines the public API for this package, specifying which names are
# imported when 'from open_magvit2 import *' is used.
__all__ = ["OpenMAGVIT2Runtime", "OpenMAGVIT2Synthesis"]
