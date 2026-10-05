"""Adapter for the MotionCtrl synthesis runtime."""

from __future__ import annotations

from ..runtime_facade import RuntimeFacadeSynthesis
from .worldfoundry_runtime import (
    DEFAULT_MOTIONCTRL_CKPT,
    DEFAULT_MOTIONCTRL_COND_DIR,
    DEFAULT_MOTIONCTRL_CONFIG,
    MotionCtrlRuntime,
)


class MotionCtrlSynthesis(RuntimeFacadeSynthesis):
    """Thin synthesis facade over the MotionCtrl runtime."""

    RUNTIME_CLS = MotionCtrlRuntime
    MODEL_ID = "motionctrl"
    DISPLAY_NAME = "MotionCtrl"


__all__ = [
    "DEFAULT_MOTIONCTRL_CKPT",
    "DEFAULT_MOTIONCTRL_COND_DIR",
    "DEFAULT_MOTIONCTRL_CONFIG",
    "MotionCtrlSynthesis",
]
