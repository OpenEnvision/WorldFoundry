"""Depth Anything v1 family adapters and model code."""

from typing import Any

__all__ = ["DAPModel", "DepthAnything", "DepthAnythingAdapter", "make_dap_model"]


def __getattr__(name: str) -> Any:
    if name == "DepthAnything":
        from .dpt import DepthAnything

        return DepthAnything
    if name == "DepthAnythingAdapter":
        from .adapter import DepthAnythingAdapter

        return DepthAnythingAdapter
    if name == "DAPModel":
        from .dap_adapter import DAPModel

        return DAPModel
    if name == "make_dap_model":
        from .dap_model import make_dap_model

        return make_dap_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
