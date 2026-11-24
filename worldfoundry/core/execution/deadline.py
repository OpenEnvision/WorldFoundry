# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Cooperative deadlines at inference stage and artifact boundaries."""

from __future__ import annotations

import time


def check_deadline(deadline_monotonic: float | None, *, stage: str) -> None:
    """Raise ``TimeoutError`` after a monotonic deadline; never interrupt a kernel."""
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError(f"Inference deadline exceeded before {stage}")
