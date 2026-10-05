"""Canonical VFIMamba frame-interpolation runtime."""

from __future__ import annotations

from .paths import checkpoint_path

__all__ = ["InputPadder", "MODEL_CONFIG", "Model", "checkpoint_path", "init_model_config"]


def __getattr__(name: str):
    if name == "Model":
        from .inference import Model

        return Model
    if name == "InputPadder":
        from .benchmark.utils.padder import InputPadder

        return InputPadder
    if name in {"MODEL_CONFIG", "init_model_config"}:
        from . import config

        return getattr(config, name)
    raise AttributeError(name)
