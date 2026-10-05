"""WorldFoundry synthesis adapter for DualCamCtrl."""

from __future__ import annotations

from ..runtime_facade import RuntimeAdapterSynthesis
from .runtime import (
    DEFAULT_BASE_REPO,
    DEFAULT_CONFIG,
    DEFAULT_DUALCAMCTRL_CHECKPOINT,
    DEFAULT_DUALCAMCTRL_REPO,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_TEST_CASE_DIR,
    DualCamCtrlRuntime,
    OFFICIAL_SOURCE_REPO,
)


class DualCamCtrlSynthesis(RuntimeAdapterSynthesis):
    """Synthesis adapter delegating inference to :class:`DualCamCtrlRuntime`."""

    RUNTIME_CLS = DualCamCtrlRuntime
    MODEL_ID = DualCamCtrlRuntime.MODEL_ID
    DISPLAY_NAME = DualCamCtrlRuntime.DISPLAY_NAME


__all__ = [
    "DEFAULT_BASE_REPO",
    "DEFAULT_CONFIG",
    "DEFAULT_DUALCAMCTRL_CHECKPOINT",
    "DEFAULT_DUALCAMCTRL_REPO",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_TEST_CASE_DIR",
    "DualCamCtrlSynthesis",
    "OFFICIAL_SOURCE_REPO",
]
