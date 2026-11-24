# Copyright (c) Tencent Inc. All rights reserved.
import importlib.metadata as importlib_metadata

try:
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"


from .models import *  # noqa: F403, I001 - Re-exports the subpackage's public names; keeps the upstream submodule import order.
from .datasets import *  # noqa: F403 - Re-exports the subpackage's public names.
from .engine import *  # noqa: F403 - Re-exports the subpackage's public names.
