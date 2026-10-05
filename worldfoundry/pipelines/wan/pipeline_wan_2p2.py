"""Public Wan2.2 adapter for the native diffusion infra."""

from __future__ import annotations

from typing import Any

from ..native_diffusion_video import NativeTextToVideoPipeline
from .pipeline_wan_2p1_t2v import WAN21_NEGATIVE_PROMPT


class Wan2p2Pipeline(NativeTextToVideoPipeline):
    """Wan2.2 TI2V 5B with optional first-frame conditioning."""

    MODEL_ID = "wan2.2-ti2v-5b"
    OWNER = "Wan2.2 TI2V 5B"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "vae")
    # Distilled/accelerated Wan releases are commonly distributed as PEFT
    # LoRAs. Reuse the audited Wan merge path so 4-step adapters remain an
    # opt-in checkpoint choice instead of requiring a separate pipeline.
    PEFT_ADAPTER_COMPONENT = "denoiser:main"
    GENERATION_TYPE = "ti2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = False
    DEFAULT_HEIGHT = 704
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 121
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 5.0
    DEFAULT_NEGATIVE_PROMPT = WAN21_NEGATIVE_PROMPT
    DEFAULT_FPS = 24
    OUTPUT_VALUE_RANGE = "-1,1"
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}

    def __call__(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        *,
        size: str | None = None,
        frame_num: int | None = None,
        sample_solver: str = "unipc",
        sample_steps: int | None = None,
        sample_shift: float | None = None,
        sample_guide_scale: float | None = None,
        base_seed: int | None = None,
        offload_model: bool | None = None,
        height: int | None = None,
        width: int | None = None,
        **kwargs: Any,
    ) -> Any:
        del offload_model
        if str(sample_solver).strip().lower() != "unipc":
            raise ValueError("native Wan2.2 currently supports the shared UniPC scheduler")
        standard_frames = kwargs.pop("num_frames", None)
        standard_steps = kwargs.pop("num_inference_steps", None)
        standard_guidance = kwargs.pop("guidance_scale", None)
        standard_seed = kwargs.pop("seed", None)
        standard_shift = kwargs.pop("shift", None)
        if frame_num is not None and standard_frames is not None and int(frame_num) != int(standard_frames):
            raise ValueError("Wan2.2 frame_num conflicts with num_frames")
        if sample_steps is not None and standard_steps is not None and int(sample_steps) != int(standard_steps):
            raise ValueError("Wan2.2 sample_steps conflicts with num_inference_steps")
        if (
            sample_guide_scale is not None
            and standard_guidance is not None
            and float(sample_guide_scale) != float(standard_guidance)
        ):
            raise ValueError("Wan2.2 sample_guide_scale conflicts with guidance_scale")
        if base_seed is not None and standard_seed is not None and int(base_seed) != int(standard_seed):
            raise ValueError("Wan2.2 base_seed conflicts with seed")
        if sample_shift is not None and standard_shift is not None and float(sample_shift) != float(standard_shift):
            raise ValueError("Wan2.2 sample_shift conflicts with shift")
        if size is not None:
            separator = "*" if "*" in size else "x"
            try:
                parsed_width, parsed_height = (int(value) for value in size.lower().split(separator, 1))
            except (TypeError, ValueError) as error:
                raise ValueError("Wan2.2 size must use WIDTH*HEIGHT, for example 1280*704") from error
            if (width is not None and width != parsed_width) or (
                height is not None and height != parsed_height
            ):
                raise ValueError("Wan2.2 size conflicts with explicit height/width")
            width, height = parsed_width, parsed_height
        return super().__call__(
            prompt=prompt,
            images=images,
            height=height,
            width=width,
            num_frames=frame_num if frame_num is not None else standard_frames,
            num_inference_steps=sample_steps if sample_steps is not None else standard_steps,
            guidance_scale=(
                sample_guide_scale if sample_guide_scale is not None else standard_guidance
            ),
            seed=(
                base_seed
                if base_seed is not None
                else (standard_seed if standard_seed is not None else 0)
            ),
            shift=sample_shift if sample_shift is not None else standard_shift,
            **kwargs,
        )


class Wan2p2T2VA14BPipeline(NativeTextToVideoPipeline):
    """Official Wan2.2 T2V A14B dual-expert generation."""

    MODEL_ID = "wan2.2-t2v-a14b"
    OWNER = "Wan2.2 T2V A14B"
    CHECKPOINT_ROLES = ("high-dit", "low-dit", "text-encoder", "tokenizer", "vae")
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 40
    DEFAULT_GUIDANCE_SCALE = 4.0
    DEFAULT_NEGATIVE_PROMPT = WAN21_NEGATIVE_PROMPT
    DEFAULT_FPS = 16
    OUTPUT_VALUE_RANGE = "-1,1"
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 12.0}
    REQUEST_INPUT_DEFAULTS = {
        "low_noise_guidance_scale": 3.0,
        "high_noise_guidance_scale": 4.0,
    }


class Wan2p2I2VA14BPipeline(NativeTextToVideoPipeline):
    """Official Wan2.2 I2V A14B dual-expert generation."""

    MODEL_ID = "wan2.2-i2v-a14b"
    OWNER = "Wan2.2 I2V A14B"
    CHECKPOINT_ROLES = ("high-dit", "low-dit", "text-encoder", "tokenizer", "vae")
    GENERATION_TYPE = "i2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 40
    DEFAULT_GUIDANCE_SCALE = 3.5
    DEFAULT_NEGATIVE_PROMPT = WAN21_NEGATIVE_PROMPT
    DEFAULT_FPS = 16
    OUTPUT_VALUE_RANGE = "-1,1"
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}
    REQUEST_INPUT_DEFAULTS = {
        "low_noise_guidance_scale": 3.5,
        "high_noise_guidance_scale": 3.5,
    }


__all__ = [
    "Wan2p2I2VA14BPipeline",
    "Wan2p2Pipeline",
    "Wan2p2T2VA14BPipeline",
]
