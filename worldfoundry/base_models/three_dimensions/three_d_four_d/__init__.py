"""Exposes core runtime synthesis components and specifications from the `runtime` submodule.

This `__init__.py` file makes key classes and functions related to 3D/4D runtime
synthesis directly accessible when the package is imported, simplifying
package-level imports for users.
"""
from .runtime import (
    ThreeDFourDRuntimeSynthesis,
    three_d_four_d_runtime_spec,
)

# Defines the public API for the package when `from package import *` is used.
__all__ = [
    "ThreeDFourDRuntimeSynthesis",
    "three_d_four_d_runtime_spec",
]