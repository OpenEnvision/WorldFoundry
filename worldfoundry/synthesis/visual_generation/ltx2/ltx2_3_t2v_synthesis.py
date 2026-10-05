"""LTX-2.3 text-to-video synthesis wrapper."""

from __future__ import annotations

from ..runtime_video_synthesis import RuntimeVideoSynthesis
from .ltx2_runtime import LTX2Video


class LTX23T2VSynthesis(RuntimeVideoSynthesis):
    """Expose the native LTX-2.3 T2V recipe through the shared runtime surface."""

    MODEL_NAME = "ltx2_3_t2v"
    GENERATION_TYPE = "t2v"
    RUNTIME_CLS = LTX2Video
    PRIMARY_PATH_KEY = "checkpoint_path"
    RUNTIME_CONFIG_PATH = "models/runtime/configs/ltx2/runtime_defaults.yaml"
    RUNTIME_CONFIG_KEY = MODEL_NAME


__all__ = ["LTX23T2VSynthesis"]
