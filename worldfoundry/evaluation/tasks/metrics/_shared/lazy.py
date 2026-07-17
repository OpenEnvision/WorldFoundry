"""Factories for lightweight metric package exports."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any


@lru_cache(maxsize=None)
def _load_module(module_path: str) -> ModuleType:
    return import_module(module_path)


def lazy_export(module_path: str, name: str, *, owner: str) -> Callable[..., Any]:
    """Create a function proxy that imports its implementation on first use."""

    def proxy(*args: Any, **kwargs: Any) -> Any:
        return getattr(_load_module(module_path), name)(*args, **kwargs)

    proxy.__name__ = name
    proxy.__qualname__ = name
    proxy.__module__ = owner
    return proxy


def package_root_export(module_file: str, *, owner: str) -> Callable[[], Path]:
    """Create the standard ``package_root()`` export for a metric package."""
    root = Path(module_file).resolve().parent

    def package_root() -> Path:
        return root

    package_root.__qualname__ = "package_root"
    package_root.__module__ = owner
    return package_root


__all__ = ["lazy_export", "package_root_export"]
