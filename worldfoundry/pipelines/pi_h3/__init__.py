# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""pi-H3 2/4-step MiniMax-H3 pipeline with optional Sol-H3 DWD pixel decoding."""

from __future__ import annotations

from typing import Any

__all__ = ["PiH3Pipeline"]


def __getattr__(name: str) -> Any:
    if name == "PiH3Pipeline":
        from .pipeline_pi_h3 import PiH3Pipeline

        return PiH3Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
