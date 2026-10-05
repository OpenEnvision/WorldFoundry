"""Shared ViGeo video geometry model, loaded only when requested."""

from typing import Any

__all__ = ["ViGeo", "ViGeoModel"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .vigeo import ViGeo, ViGeoModel

        return {"ViGeo": ViGeo, "ViGeoModel": ViGeoModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
