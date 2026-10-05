"""Public Step-Video-T2V pipeline backed by WorldFoundry native diffusion."""

from __future__ import annotations

from worldfoundry.pipelines.native_diffusion_video import NativeTextToVideoPipeline


# The released StepVideo pipeline defaults both prompt suffixes to empty.
# Keep those checkpoint-native defaults: forcing an HDR/surrealism suffix here
# materially changes the prompt and produces the yellow, over-saturated output
# that the Workspace quality check exposed.
DEFAULT_POSITIVE_MAGIC = ""
DEFAULT_NEGATIVE_MAGIC = ""


class StepVideoT2VPipeline(NativeTextToVideoPipeline):
    """Studio-facing adapter for the native Step-Video-T2V recipe."""

    MODEL_ID = "step-video-t2v"
    OWNER = "StepVideo"
    CHECKPOINT_ROLES = ("transformer", "vae", "resources")
    DEFAULT_HEIGHT = 544
    DEFAULT_WIDTH = 992
    DEFAULT_NUM_FRAMES = 204
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 9.0
    DEFAULT_FPS = 25
    DEFAULT_SCHEDULER_OPTIONS = {"time_shift": 13.0}
    REQUEST_INPUT_DEFAULTS = {
        "positive_magic": DEFAULT_POSITIVE_MAGIC,
        "negative_magic": DEFAULT_NEGATIVE_MAGIC,
    }
    REQUEST_INPUT_ALIASES = {
        "pos_magic": "positive_magic",
        "neg_magic": "negative_magic",
    }


__all__ = ["StepVideoT2VPipeline"]
