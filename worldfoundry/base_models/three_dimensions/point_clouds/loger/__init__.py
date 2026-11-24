"""Canonical LoGeR point-cloud base model integration."""

from importlib import import_module

__all__ = ["Pi3", "layers", "utils"]


def __getattr__(name: str):
    if name == "Pi3":
        from .pi3 import Pi3

        return Pi3
    if name in {"layers", "utils"}:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)
