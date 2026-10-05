"""Immutable SLAM inference constants.

Kept out of ``system.py`` so tests and callers can inspect reuse policy without
importing the native ``lietorch_ext`` module.
"""

from __future__ import annotations

# Graph / buffer objects that must be rebuilt for every video. Weight-bearing
# networks (DROID, UniDepth) stay resident on the reused SLAMSystem.
SLAM_PER_VIDEO_COMPONENTS = (
    "inner_filler",
    "backend",
    "frontend",
    "motion_filter",
    "buffer",
    "sparse_tracks",
)
