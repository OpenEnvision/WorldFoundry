"""Wrapper exposing the shared OpenFlamingo source as top-level ``open_flamingo``.

Modified by WorldFoundry: shadow detection and path registration now go
through ``worldfoundry.base_models._vendor_imports`` (idempotent insert, no
``remove + insert(0)`` preemption; conflicts raise ImportError).
"""

from __future__ import annotations

from pathlib import Path

from worldfoundry.base_models._vendor_imports import (
    assert_top_level_not_shadowed,
    prepend_import_path,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
IMPORT_ROOT = PACKAGE_ROOT.parent


def _assert_top_level_package_not_shadowed() -> None:
    """Fail fast when a foreign ``open_flamingo`` is already imported."""
    assert_top_level_not_shadowed("open_flamingo", PACKAGE_ROOT)


def ensure_import_paths() -> tuple[Path, ...]:
    """Expose the shared OpenFlamingo source as the top-level ``open_flamingo`` package."""

    _assert_top_level_package_not_shadowed()
    prepend_import_path(IMPORT_ROOT)
    return (IMPORT_ROOT,)


__all__ = ["IMPORT_ROOT", "PACKAGE_ROOT", "ensure_import_paths"]
