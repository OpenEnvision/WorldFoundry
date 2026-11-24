"""Depth Anything v2 model and WorldFoundry adapters."""

from typing import Any

__all__ = ["DepthAnythingDepthModel", "DepthAnythingV2"]


def __getattr__(name: str) -> Any:
    if name == "DepthAnythingDepthModel":
        from .adapter import DepthAnythingDepthModel

        return DepthAnythingDepthModel
    if name == "DepthAnythingV2":
        from .dpt import DepthAnythingV2

        return DepthAnythingV2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
