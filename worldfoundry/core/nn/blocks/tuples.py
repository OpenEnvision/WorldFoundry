# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from timm's iterable-preserving tuple parser (Apache-2.0), Ross Wightman.
"""Scalar repetition for model parsers that retain iterable configuration values."""

from collections.abc import Callable, Iterable
from itertools import repeat
from typing import Any


def ntuple(n: int) -> Callable[[Any], Any]:
    """Repeat scalars n times; return existing non-string iterables unchanged."""

    def parse(value: Any) -> Any:
        if isinstance(value, Iterable) and not isinstance(value, str):
            return value
        return tuple(repeat(value, n))

    return parse
