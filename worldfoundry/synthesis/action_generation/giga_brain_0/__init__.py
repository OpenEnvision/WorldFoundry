"""
Provides the GigaBrain0Synthesis class, re-exporting it from its internal definition.

This module acts as an aggregation point, making the GigaBrain0Synthesis class
directly accessible for external imports as part of the public API, without
requiring knowledge of its specific file path within the package structure.
"""

from .giga_brain_0_synthesis import GigaBrain0Synthesis

# Define the public API of this module for 'from module import *' statements.
__all__ = ["GigaBrain0Synthesis"]
