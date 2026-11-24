# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Pi4 (pi^4) 4D scene geometry and motion inference package.

Ported from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``
(CC BY-NC 4.0; upstream code and weights are non-commercial).
"""

from __future__ import annotations

from typing import Any

__all__ = ["Pi4", "Pi4Representation"]


def __getattr__(name: str) -> Any:
    """Lazily expose heavy PyTorch inference classes."""
    if name == "Pi4":
        from .models.pi4 import Pi4

        return Pi4
    if name == "Pi4Representation":
        from .runtime import Pi4Representation

        return Pi4Representation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
