"""MiniMax H3 pipeline constants (ported from SGLang Apache-2.0)."""

from __future__ import annotations

MINIMAX_H3_SUPPORTED_FPS = 24
MINIMAX_H3_MIN_DURATION_SECONDS = 4.0
MINIMAX_H3_MAX_DURATION_SECONDS = 15.0

# The distilled checkpoint has exactly one positive denoise branch.
MINIMAX_H3_DEFAULT_BRANCHES: tuple = ({"name": "cond_1"},)

# Audited 4xH200 T2VA cache-quality profiles:
# (warmup steps, residual-difference threshold, max consecutive cached steps).
MINIMAX_H3_QUALITY_PROFILES: dict[str, tuple[int, float, int] | None] = {
    "lossless": None,
    "high": (4, 0.04, 1),
    "medium": (4, 0.12, 3),
    "low": (4, 0.24, 3),
}

__all__ = [
    "MINIMAX_H3_DEFAULT_BRANCHES",
    "MINIMAX_H3_MAX_DURATION_SECONDS",
    "MINIMAX_H3_MIN_DURATION_SECONDS",
    "MINIMAX_H3_QUALITY_PROFILES",
    "MINIMAX_H3_SUPPORTED_FPS",
]
