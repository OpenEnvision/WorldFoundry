"""WorldFoundry synthesis adapter for CameraCtrl.

Re-exports the CameraCtrl default asset paths so callers can reach them without
importing the runtime module directly.
"""

from __future__ import annotations

from ..runtime_facade import RuntimeAdapterSynthesis
from .runtime import (
    DEFAULT_CAMERACTRL_CKPT,
    DEFAULT_CAMERACTRL_CONFIG,
    DEFAULT_CAMERACTRL_IMAGE_LORA,
    DEFAULT_CAMERACTRL_MOTION_ADAPTER,
    DEFAULT_SD15_ROOT,
    CameraCtrlRuntime,
)


class CameraCtrlSynthesis(RuntimeAdapterSynthesis):
    """Synthesis adapter delegating inference to :class:`CameraCtrlRuntime`."""

    RUNTIME_CLS = CameraCtrlRuntime
    MODEL_ID = CameraCtrlRuntime.MODEL_ID
    DISPLAY_NAME = CameraCtrlRuntime.DISPLAY_NAME

    #: CameraCtrl callers read the resolved asset paths straight off the adapter.
    MIRRORED_RUNTIME_ATTRS = RuntimeAdapterSynthesis.MIRRORED_RUNTIME_ATTRS + (
        "sd15_path",
        "pose_adaptor_ckpt",
        "model_config",
        "motion_module_ckpt",
        "motion_adapter_ckpt",
        "image_lora_ckpt",
        "image_lora_rank",
        "unet_subfolder",
        "personalized_base_model",
    )


__all__ = [
    "CameraCtrlSynthesis",
    "DEFAULT_CAMERACTRL_CKPT",
    "DEFAULT_CAMERACTRL_CONFIG",
    "DEFAULT_CAMERACTRL_IMAGE_LORA",
    "DEFAULT_CAMERACTRL_MOTION_ADAPTER",
    "DEFAULT_SD15_ROOT",
]
