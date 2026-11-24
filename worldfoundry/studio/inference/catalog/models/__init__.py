"""Curated per-model catalog overrides, grouped by model family."""

from __future__ import annotations

from typing import Any

from .embodied import OVERRIDES as _EMBODIED
from .remote_api import OVERRIDES as _REMOTE_API
from .scene_3d import OVERRIDES as _SCENE_3D
from .video_generation import OVERRIDES as _VIDEO_GENERATION
from .world_models import OVERRIDES as _WORLD_MODELS

CURATED_OVERRIDES: dict[str, dict[str, Any]] = {
    **_VIDEO_GENERATION,
    **_WORLD_MODELS,
    **_SCENE_3D,
    **_EMBODIED,
    **_REMOTE_API,
}
