"""Module for base_models -> three_dimensions -> general_3d -> splatt3r -> __init__.py functionality."""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """Runtime root.

    Returns:
        The return value.
    """
    return Path(__file__).resolve().parent / 'splatt3r_runtime'


from .worldfoundry_runtime import (
    DEFAULT_SPLATT3R_FILENAME,
    DEFAULT_SPLATT3R_LOCAL_CKPT,
    DEFAULT_SPLATT3R_REPO,
    Splatt3RRuntime,
)

__all__ = [
    "DEFAULT_SPLATT3R_FILENAME",
    "DEFAULT_SPLATT3R_LOCAL_CKPT",
    "DEFAULT_SPLATT3R_REPO",
    "Splatt3RRuntime",
    "Splatt3RSynthesis",
    "runtime_root",
]


def __getattr__(name: str):
    if name == "Splatt3RSynthesis":
        from .splatt3r_synthesis import Splatt3RSynthesis

        return Splatt3RSynthesis
    raise AttributeError(name)
