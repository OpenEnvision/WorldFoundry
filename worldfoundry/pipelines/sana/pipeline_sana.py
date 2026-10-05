"""Public Sana products assembled by the unified native diffusion infra."""

from __future__ import annotations

from typing import ClassVar

from worldfoundry.base_models.diffusion_model.recipes.sana_variants import get_sana_variant

from ..native_diffusion import NativeVisualDiffusionPipeline


class SanaPipeline(NativeVisualDiffusionPipeline):
    """Select Sana graph/checkpoints declaratively; reuse the shared runner."""

    MODEL_ID: ClassVar[str] = "sana"
    OWNER: ClassVar[str] = "Sana"
    CHECKPOINT_ROLES: ClassVar[tuple[str, ...]] = (
        "dit",
        "text-encoder",
        "tokenizer",
        "codec",
    )
    PRIMARY_CHECKPOINT_ROLE: ClassVar[str] = "dit"
    ALLOW_MODEL_ID_OVERRIDE: ClassVar[bool] = True
    NUM_INFERENCE_STEP_ALIASES: ClassVar[tuple[str, ...]] = ("step", "infer_steps")
    SCHEDULER_OPTION_ALIASES: ClassVar[dict[str, str]] = {"flow_shift": "shift"}

    def __init__(self, *, native_pipeline, device: str, model_id: str | None = None) -> None:
        variant = get_sana_variant(model_id or self.MODEL_ID)
        self.variant = variant
        self.GENERATION_TYPE = (
            "i2v" if variant.runner == "controlnet" else "v2v" if variant.runner == "streaming" else "t2v"
        )
        self.ACCEPTS_IMAGES = variant.runner == "controlnet"
        self.REQUIRES_IMAGES = variant.runner == "controlnet"
        self.ACCEPTS_VIDEO = variant.runner == "streaming"
        self.DEFAULT_HEIGHT, self.DEFAULT_WIDTH = self._default_size(variant)
        self.DEFAULT_NUM_FRAMES = variant.default_num_frames or (
            81 if variant.task in {"text-to-video", "video-to-video"} else 1
        )
        self.DEFAULT_NUM_INFERENCE_STEPS = variant.default_steps or 50
        self.DEFAULT_GUIDANCE_SCALE = variant.default_cfg_scale or 1.0
        self.DEFAULT_FPS = variant.default_fps or 16
        self.DEFAULT_NEGATIVE_PROMPT = ""
        self.REQUEST_INPUT_DEFAULTS = (
            {
                "motion_score": 10 if variant.mode == "bidirectional_short" else 0,
                "num_cached_blocks": 2,
                "sink_token": True,
            }
            if variant.runner == "streaming"
            else {}
        )
        if variant.runner == "sprint":
            self.DEFAULT_SCHEDULER_OPTIONS = {}
        else:
            self.DEFAULT_SCHEDULER_OPTIONS = {
                "shift": 8.0 if variant.resolution == "720p" else 7.0
                if variant.resolution == "480p"
                else 4.0 if "600m" in variant.model_id.split("-")
                else 3.0
            }
        super().__init__(
            native_pipeline=native_pipeline,
            device=device,
            model_id=variant.model_id,
        )
        self.model_name = variant.display_name
        self.generation_type = variant.task

    @staticmethod
    def _default_size(variant) -> tuple[int, int]:
        if variant.default_height is not None and variant.default_width is not None:
            return int(variant.default_height), int(variant.default_width)
        if variant.resolution == "720p":
            return 720, 1280
        if variant.resolution == "480p":
            return 480, 832
        side = int(variant.resolution.removesuffix("px"))
        return side, side


class Sana600M512pxPipeline(SanaPipeline):
    MODEL_ID = "sana-600m-512px"


class Sana600M1024pxPipeline(SanaPipeline):
    MODEL_ID = "sana-600m-1024px"


class Sana1600M512pxPipeline(SanaPipeline):
    MODEL_ID = "sana-1600m-512px"


class Sana1600M512pxMultilingPipeline(SanaPipeline):
    MODEL_ID = "sana-1600m-512px-multiling"


class Sana1600M1024pxPipeline(SanaPipeline):
    MODEL_ID = "sana-1600m-1024px"


class Sana1600M1024pxMultilingPipeline(SanaPipeline):
    MODEL_ID = "sana-1600m-1024px-multiling"


class Sana1600M1024pxBf16Pipeline(SanaPipeline):
    MODEL_ID = "sana-1600m-1024px-bf16"


class Sana1600M2kBf16Pipeline(SanaPipeline):
    MODEL_ID = "sana-1600m-2k-bf16"


class Sana1600M4kBf16Pipeline(SanaPipeline):
    MODEL_ID = "sana-1600m-4k-bf16"


class Sana1p51600M1024pxPipeline(SanaPipeline):
    MODEL_ID = "sana1p5-1600m-1024px"


class Sana1p54800M1024pxPipeline(SanaPipeline):
    MODEL_ID = "sana1p5-4800m-1024px"


class SanaSprint600M1024pxPipeline(SanaPipeline):
    MODEL_ID = "sana-sprint-600m-1024px"


class SanaSprint1600M1024pxPipeline(SanaPipeline):
    MODEL_ID = "sana-sprint-1600m-1024px"


class SanaControlnet600M1024pxPipeline(SanaPipeline):
    MODEL_ID = "sana-controlnet-600m-1024px"


class SanaControlnet1600M1024pxBf16Pipeline(SanaPipeline):
    MODEL_ID = "sana-controlnet-1600m-1024px-bf16"


class SanaVideo2b480pPipeline(SanaPipeline):
    MODEL_ID = "sana-video-2b-480p"


class SanaVideo2b720pPipeline(SanaPipeline):
    MODEL_ID = "sana-video-2b-720p"


class LongsanaVideo2b480pPipeline(SanaPipeline):
    MODEL_ID = "longsana-video-2b-480p"


class SanaStreaming2b720pPipeline(SanaPipeline):
    MODEL_ID = "sana-streaming-2b-720p"


class SanaStreamingBidirectional2b720pPipeline(SanaPipeline):
    MODEL_ID = "sana-streaming-bidirectional-2b-720p"


__all__ = [
    "LongsanaVideo2b480pPipeline",
    "Sana1600M1024pxBf16Pipeline",
    "Sana1600M1024pxMultilingPipeline",
    "Sana1600M1024pxPipeline",
    "Sana1600M2kBf16Pipeline",
    "Sana1600M4kBf16Pipeline",
    "Sana1600M512pxMultilingPipeline",
    "Sana1600M512pxPipeline",
    "Sana1p51600M1024pxPipeline",
    "Sana1p54800M1024pxPipeline",
    "Sana600M1024pxPipeline",
    "Sana600M512pxPipeline",
    "SanaControlnet1600M1024pxBf16Pipeline",
    "SanaControlnet600M1024pxPipeline",
    "SanaPipeline",
    "SanaSprint1600M1024pxPipeline",
    "SanaSprint600M1024pxPipeline",
    "SanaStreaming2b720pPipeline",
    "SanaStreamingBidirectional2b720pPipeline",
    "SanaVideo2b480pPipeline",
    "SanaVideo2b720pPipeline",
]
