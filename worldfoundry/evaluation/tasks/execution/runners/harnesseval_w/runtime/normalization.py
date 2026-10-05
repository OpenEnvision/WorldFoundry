from __future__ import annotations


HPSV3_NORMALIZATION_VERSION = "harnesseval.hpsv3_affine"
HPSV3_LOWER_BOUND = -8.0
HPSV3_UPPER_BOUND = 12.0

__all__ = [
    "HPSV3_NORMALIZATION_VERSION",
    "HPSV3_LOWER_BOUND",
    "HPSV3_UPPER_BOUND",
    "hpsv3_to_unit",
    "hpsv3_from_unit",
]


def hpsv3_to_unit(raw: float) -> float:
    span = HPSV3_UPPER_BOUND - HPSV3_LOWER_BOUND
    value = (float(raw) - HPSV3_LOWER_BOUND) / span
    return max(0.0, min(1.0, value))


def hpsv3_from_unit(value: float) -> float:
    span = HPSV3_UPPER_BOUND - HPSV3_LOWER_BOUND
    return float(value) * span + HPSV3_LOWER_BOUND
