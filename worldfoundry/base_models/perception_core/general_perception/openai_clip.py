"""Canonical access to WorldFoundry's in-tree OpenAI CLIP runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

RUNTIME_ROOT = Path(__file__).resolve().parent / "openai_clip_runtime"


def add_runtime_to_path() -> Path:
    """Return the compatibility runtime root without mutating ``sys.path``."""

    return RUNTIME_ROOT


def load(*args: Any, **kwargs: Any):
    from .openai_clip_runtime.clip import load as runtime_load

    return runtime_load(*args, **kwargs)


def tokenize(*args: Any, **kwargs: Any):
    from .openai_clip_runtime.clip import tokenize as runtime_tokenize

    return runtime_tokenize(*args, **kwargs)


def available_models() -> list[str]:
    from .openai_clip_runtime.clip import available_models as runtime_available_models

    return runtime_available_models()


def clear_checkpoint_cache() -> None:
    from .openai_clip_runtime.clip import clear_checkpoint_cache as runtime_clear_checkpoint_cache

    runtime_clear_checkpoint_cache()


__all__ = [
    "RUNTIME_ROOT",
    "add_runtime_to_path",
    "available_models",
    "clear_checkpoint_cache",
    "load",
    "tokenize",
]
