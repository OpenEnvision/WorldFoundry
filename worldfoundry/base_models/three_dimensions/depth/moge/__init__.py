"""Canonical MoGe depth base model integration."""

from importlib import import_module

__all__ = [
    "MoGeModelV1",
    "MoGeModelV2",
    "MogeModel",
    "model",
    "utils",
]


def __getattr__(name: str):
    if name == "MogeModel":
        from .adapter import MogeModel

        return MogeModel
    if name == "model":
        return import_module(f"{__name__}.model")
    if name == "utils":
        return import_module(f"{__name__}.utils")
    if name == "MoGeModelV1":
        from .model.v1 import MoGeModel

        return MoGeModel
    if name == "MoGeModelV2":
        from .model.v2 import MoGeModel

        return MoGeModel
    raise AttributeError(name)
