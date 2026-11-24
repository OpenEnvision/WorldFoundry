"""Import helpers shared by metric wrappers."""

from __future__ import annotations

import sys
from pathlib import Path


def prepend_import_path(path: str | Path) -> None:
    """Make a bundled implementation importable without duplicating path setup."""
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


__all__ = ["prepend_import_path"]
