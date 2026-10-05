"""Public SkyReels-V2 pipeline backed by WorldFoundry native diffusion."""

from __future__ import annotations

from worldfoundry.pipelines.native_diffusion_video import NativeTextToVideoPipeline


class SkyReelsV2Pipeline(NativeTextToVideoPipeline):
    """Studio-facing adapter for the native SkyReels-V2 1.3B recipe."""

    MODEL_ID = "skyreels-v2"
    OWNER = "SkyReels-V2"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "vae")
    DEFAULT_HEIGHT = 544
    DEFAULT_WIDTH = 960
    DEFAULT_NUM_FRAMES = 97
    DEFAULT_NUM_INFERENCE_STEPS = 30
    DEFAULT_GUIDANCE_SCALE = 6.0
    DEFAULT_FPS = 24
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 8.0}


__all__ = ["SkyReelsV2Pipeline"]
