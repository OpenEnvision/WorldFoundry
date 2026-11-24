"""Vendored Depth Anything 3 package for WorldFoundry."""

from typing import Any

__all__ = ["DepthAnything3", "DepthAnything3Model"]


def __getattr__(name: str) -> Any:
    if name == "DepthAnything3":
        from .api import DepthAnything3

        return DepthAnything3
    if name == "DepthAnything3Model":
        from .adapter import DepthAnything3Model

        return DepthAnything3Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
