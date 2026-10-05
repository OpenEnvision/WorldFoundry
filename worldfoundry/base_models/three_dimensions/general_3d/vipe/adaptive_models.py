"""Resident depth networks reused across videos of one ViPE annotation process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AdaptiveDepthModels:
    """Weight holders for AdaptiveDepthProcessor; safe to keep across videos."""

    video_depth_model: Any
    depth_model: Any
    prompt_model: Any
