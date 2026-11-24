"""
This package provides the core components for the MotionCtrl system,
including synthesis capabilities and runtime evaluation utilities.

It uses a lazy loading mechanism via `__getattr__` to defer the import
of `MotionCtrlSynthesis` and `MotionCtrlRuntime` until they are explicitly
accessed, improving package import performance.

The public API of this package is defined by `__all__`.
"""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """
    Determines and returns the root directory for MotionCtrl runtime assets.

    This path typically points to the 'motionctrl_runtime' directory
    located alongside the current package's `__init__.py` file.

    Returns:
        Path: The absolute path to the MotionCtrl runtime root directory.
    """
    return Path(__file__).resolve().parent / "motionctrl_runtime"


def __getattr__(name: str):
    """
    Implements lazy loading for top-level attributes of the 'motionctrl' package.

    This function is called when an attribute (like a class) is accessed
    from the package (e.g., 'from motionctrl import MotionCtrlSynthesis')
    but has not been explicitly imported or defined in the package's
    __init__.py. It defers the actual import until the attribute is needed,
    improving initial package import speed.

    Args:
        name (str): The name of the attribute being accessed.

    Returns:
        Any: The requested attribute (e.g., a class from a submodule).

    Raises:
        AttributeError: If the requested attribute is not recognized
                        or cannot be lazily loaded.
    """
    if name == "MotionCtrlSynthesis":
        # Lazily import and return the MotionCtrlSynthesis class
        # from the .synthesis submodule when it's first accessed.
        from .synthesis import MotionCtrlSynthesis

        return MotionCtrlSynthesis
    if name == "MotionCtrlRuntime":
        # Lazily import and return the MotionCtrlRuntime class
        # from the .worldfoundry_runtime submodule when it's first accessed.
        from .worldfoundry_runtime import MotionCtrlRuntime

        return MotionCtrlRuntime
    # If the requested name is not one of the lazily loaded attributes,
    # raise an AttributeError to indicate it doesn't exist.
    raise AttributeError(name)


__all__ = ["MotionCtrlRuntime", "MotionCtrlSynthesis", "runtime_root"]
