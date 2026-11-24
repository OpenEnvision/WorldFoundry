"""
Top-level package for SCOPESynthesis, exposing core components.

This module serves as the primary entry point for the SCOPESynthesis
library, making key classes and functions directly accessible
when importing from the package.
"""

from .scope_synthesis import SCOPESynthesis, runtime_root

# Defines the public API of this module, specifying which names are
# imported when 'from package import *' is used.
# This explicitly exposes the SCOPESynthesis class and the runtime_root variable.
__all__ = ["SCOPESynthesis", "runtime_root"]
