"""Provides core functionalities for Yume video runtime integrations.

This module sets up paths to the package and its associated runtime directory,
and implements lazy loading for YumeRuntime and Yume1p5Runtime classes
to manage dependencies and improve startup performance.
"""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    """Returns the absolute path to the root directory of the current Python package.

    This is useful for locating resources relative to the package installation.
    """
    return Path(__file__).resolve().parent


def runtime_root() -> Path:
    """Returns the absolute path to the 'yume_runtime' directory.

    This directory is expected to contain runtime-specific assets or executables.
    """
    return package_root() / "yume_runtime"


RUNTIME_ROOT = runtime_root()


def __getattr__(name: str):
    """Lazily loads YumeRuntime and Yume1p5Runtime classes upon first access.

    This mechanism is used to avoid circular imports or to defer the loading
    of potentially heavy modules until they are actually needed, improving
    module import times.
    """
    if name == "YumeRuntime":
        # Import YumeRuntime only when it's explicitly accessed via this module.
        from .worldfoundry_runtime import YumeRuntime

        return YumeRuntime
    if name == "Yume1p5Runtime":
        # Import Yume1p5Runtime only when it's explicitly accessed via this module.
        from .worldfoundry_runtime import Yume1p5Runtime

        return Yume1p5Runtime
    raise AttributeError(name)


__all__ = [
    # Defines the public API of this module, specifying which names
    # are exported when 'from yume_video_runtime import *' is used.
    "RUNTIME_ROOT",
    "Yume1p5Runtime",
    "YumeRuntime",
    "package_root",
    "runtime_root",
]
