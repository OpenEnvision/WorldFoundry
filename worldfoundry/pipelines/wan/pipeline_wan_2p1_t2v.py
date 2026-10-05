"""Public Wan2.1 text-to-video adapters for the native diffusion infra."""

from __future__ import annotations

from ..native_diffusion_video import NativeTextToVideoPipeline

WAN21_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class Wan2p1T2VPipeline(NativeTextToVideoPipeline):
    """Wan2.1 T2V 1.3B on the framework-owned native runner."""

    MODEL_ID = "wan2.1-t2v-1.3b"
    OWNER = "Wan2.1 T2V 1.3B"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "vae")
    PEFT_ADAPTER_COMPONENT = "denoiser:main"
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 6.0
    DEFAULT_NEGATIVE_PROMPT = WAN21_NEGATIVE_PROMPT
    DEFAULT_FPS = 16
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 8.0}


class Wan2p1T2V14BPipeline(Wan2p1T2VPipeline):
    """Wan2.1 T2V 14B using the same component and execution contracts."""

    MODEL_ID = "wan2.1-t2v-14b"
    OWNER = "Wan2.1 T2V 14B"
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}


__all__ = ["Wan2p1T2V14BPipeline", "Wan2p1T2VPipeline"]
