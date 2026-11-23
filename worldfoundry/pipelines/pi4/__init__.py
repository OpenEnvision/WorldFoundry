# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Pi4 (pi^4) 4D geometry and motion pipeline package."""

from __future__ import annotations

from typing import Any

__all__ = ["Pi4Pipeline", "Pi4Result"]


def __getattr__(name: str) -> Any:
    """Lazily expose pipeline and result classes."""
    if name in ("Pi4Pipeline", "Pi4Result"):
        from .pipeline_pi4 import Pi4Pipeline, Pi4Result

        return Pi4Pipeline if name == "Pi4Pipeline" else Pi4Result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
