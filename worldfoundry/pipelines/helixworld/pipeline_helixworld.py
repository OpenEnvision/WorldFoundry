"""Public HelixWorld adapter over the shared native visual pipeline."""

from __future__ import annotations

from ..native_diffusion import NativeVisualDiffusionPipeline


class HelixWorldPipeline(NativeVisualDiffusionPipeline):
    MODEL_ID = "helixworld"
    OWNER = "HelixWorld"
    GENERATION_TYPE = "i2v"
    CHECKPOINT_ROLES = ("model", "gemma", "tokenizer")
    PRIMARY_CHECKPOINT_ROLE = "model"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 768
    DEFAULT_NUM_FRAMES = 121
    DEFAULT_NUM_INFERENCE_STEPS = 4
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_FPS = 24
    OUTPUT_LAYOUT = "FHWC"
    OUTPUT_VALUE_RANGE = "0,1"
    REQUEST_INPUT_DEFAULTS = {
        "actions": "W", "perspective": "first_person", "audio_prompt": None,
        "av_prompt": None, "control_path": None, "frame_rate": 24.0,
    }
    REQUEST_INPUT_ALIASES = {"action_plan": "actions"}

    @classmethod
    def _checkpoint_overrides(cls, model_path, options):
        overrides = super()._checkpoint_overrides(model_path, options) or {}
        text_encoder_path = options.get("text_encoder_path")
        if text_encoder_path is not None:
            overrides.update(gemma=str(text_encoder_path), tokenizer=str(text_encoder_path))
        return overrides or None
