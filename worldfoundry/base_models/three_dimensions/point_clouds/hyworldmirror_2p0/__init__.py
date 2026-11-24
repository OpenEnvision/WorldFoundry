"""Canonical Hunyuan World 2.0 WorldMirror base model integration."""

from importlib import import_module

__all__ = ["WorldMirror", "models", "utils"]


def __getattr__(name: str):
    if name == "WorldMirror":
        from .models.models.worldmirror import WorldMirror

        return WorldMirror
    if name in {"models", "utils"}:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(name)
