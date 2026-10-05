"""High-sigma sampling-strategy identifiers for Cosmos inference.

:class:`HighSigmaStrategy` names the official Predict2 / Predict2.5
schedules that reshape the first few noise levels (uniform, log-uniform,
shift-24, balanced two-head, hardcoded 20-step).  ``NONE`` leaves the
solver's native sigma grid unchanged.

These are string enums stored on recipe configs; the actual schedule
math lives in the Cosmos runner, not in this module.
"""

from enum import Enum


class HighSigmaStrategy(str, Enum):
    """Named high-sigma grids used by Predict2 / Predict2.5 recipes."""

    NONE = "none"
    UNIFORM80_2000 = "uniform80_2000"
    LOGUNIFORM200_100000 = "LOGUNIFORM200_100000"
    SHIFT24 = "shift24"
    BALANCED_TWO_HEADS_V1 = "balanced_two_heads_v1"
    HARDCODED_20steps = "hardcoded_20steps"

    def __str__(self) -> str:
        return self.value


__all__ = ["HighSigmaStrategy"]
