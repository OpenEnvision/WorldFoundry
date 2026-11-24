"""
A collection of useful functions for 3D computer vision and graphics researchers in Python.
- Use `utils3d.{function}` to call the function automatically selecting the backend based on the input type (Numpy ndarray or Pytorch tensor).
- Use `utils3d.{np/pt}.{function}` to specifically call the Numpy or Pytorch version.
"""

import contextlib
from typing import TYPE_CHECKING

from .helpers import lazy_import
from .helpers import lazy_import_from as lazy_import_from

with contextlib.suppress(ImportError):
    from .interface import *  # noqa: F403 - Upstream star import kept as written.

__all__ = ["np", "numpy", "pt", "torch"]


lazy_import(globals(), ".numpy", "numpy")
lazy_import(globals(), ".numpy", "np")
lazy_import(globals(), ".torch", "torch")
lazy_import(globals(), ".torch", "pt")


if TYPE_CHECKING:
    from . import numpy

    # isort: split
    # isort: split
    from . import numpy as np  # short alias

    # isort: split
    # isort: split
    from . import torch

    # isort: split
    # isort: split
    from . import torch as pt  # short alias

    # isort: split
    # isort: split
    from . import io as io
