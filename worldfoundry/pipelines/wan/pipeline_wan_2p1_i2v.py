"""Public Wan2.1 image-to-video adapters for the native diffusion infra."""

from __future__ import annotations

from ..native_diffusion_video import NativeTextToVideoPipeline
from .pipeline_wan_2p1_t2v import WAN21_NEGATIVE_PROMPT


class Wan2p1I2VPipeline(NativeTextToVideoPipeline):
    """Wan2.1 I2V 14B 480P on the framework-owned native runner."""

    MODEL_ID = "wan2.1-i2v-14b-480p"
    OWNER = "Wan2.1 I2V 14B 480P"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "image-encoder", "vae")
    GENERATION_TYPE = "i2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 40
    DEFAULT_GUIDANCE_SCALE = 5.0
    DEFAULT_NEGATIVE_PROMPT = "镜头晃动，" + WAN21_NEGATIVE_PROMPT
    DEFAULT_FPS = 16
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 3.0}


class Wan2p1I2V720PPipeline(Wan2p1I2VPipeline):
    """Wan2.1 I2V 14B 720P using the same component contracts."""

    MODEL_ID = "wan2.1-i2v-14b-720p"
    OWNER = "Wan2.1 I2V 14B 720P"
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}


__all__ = ["Wan2p1I2V720PPipeline", "Wan2p1I2VPipeline"]
