"""Public Sana products assembled by the unified native diffusion infra."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from worldfoundry.base_models.diffusion_model.models.networks.sana.acceleration import (
    SanaAccelerationConfig,
    accelerate_sana_vae,
    accelerate_sana_video,
    sana_runtime_receipt,
)
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
    PRIMARY_CHECKPOINT_ROLE: ClassVar[str | None] = "dit"
    ALLOW_MODEL_ID_OVERRIDE: ClassVar[bool] = True
    NUM_INFERENCE_STEP_ALIASES: ClassVar[tuple[str, ...]] = ("step", "infer_steps")
    SCHEDULER_OPTION_ALIASES: ClassVar[Mapping[str, str]] = {"flow_shift": "shift"}

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
                "shift": 8.0
                if variant.resolution == "720p"
                else 7.0
                if variant.resolution == "480p"
                else 4.0
                if "600m" in variant.model_id.split("-")
                else 3.0
            }
        super().__init__(
            native_pipeline=native_pipeline,
            device=device,
            model_id=variant.model_id,
        )
        self.model_name = variant.display_name
        self.generation_type = variant.task
        self._sol_acceleration_receipt: dict[str, Any] | None = None

    def _resolve_native_components(self) -> tuple[Any, Any]:
        components = getattr(self.native_pipeline, "components", None)
        denoiser = getattr(components, "denoiser", None) if components is not None else None
        if denoiser is None:
            denoiser = getattr(self.native_pipeline, "denoiser", None)
        decoder = getattr(components, "decoder", None) if components is not None else None
        if decoder is None:
            decoder = getattr(self.native_pipeline, "decoder", None)
        return denoiser, decoder

    def apply_sol_engine(
        self,
        config: SanaAccelerationConfig | str = "sol_balanced",
        *,
        vae_channels_last_3d: bool | None = None,
        vae_decode_autocast_bf16: bool | None = None,
        vae_compile: bool | None = None,
    ) -> dict[str, Any]:
        """Apply the SANA SOL-Engine stack to this pipeline's denoiser and VideoVAE."""
        denoiser, decoder = self._resolve_native_components()
        model = getattr(denoiser, "model", None)
        if model is None:
            raise RuntimeError("SanaPipeline native_pipeline does not expose a denoiser.model")
        if denoiser is not None and hasattr(denoiser, "apply_sol_engine"):
            receipt = denoiser.apply_sol_engine(config)
        else:
            receipt = accelerate_sana_video(model, config)
            if denoiser is not None and hasattr(denoiser, "_feature_cache_config"):
                denoiser._feature_cache_config = getattr(model, "_worldfoundry_easycache_config", None)
        if decoder is not None:
            cfg_dict = receipt.get("config", {})
            eff_cl3d = (
                bool(cfg_dict.get("vae_channels_last_3d", True))
                if vae_channels_last_3d is None
                else bool(vae_channels_last_3d)
            )
            eff_bf16 = (
                bool(cfg_dict.get("vae_decode_autocast", True))
                if vae_decode_autocast_bf16 is None
                else bool(vae_decode_autocast_bf16)
            )
            eff_compile = bool(cfg_dict.get("vae_compile", False)) if vae_compile is None else bool(vae_compile)
            receipt["vae"] = accelerate_sana_vae(
                decoder,
                channels_last_3d=eff_cl3d,
                decode_autocast_bf16=eff_bf16,
                compile_decoder=eff_compile,
                compile_mode=receipt.get("compile", {}).get("mode") or "default",
            )
        self._sol_acceleration_receipt = receipt
        return receipt

    def sol_engine_report(self) -> dict[str, Any]:
        """Return the live SOL-Engine optimization receipt for this pipeline."""
        denoiser, _ = self._resolve_native_components()
        if denoiser is not None and hasattr(denoiser, "runtime_optimization_report"):
            return denoiser.runtime_optimization_report()
        model = getattr(denoiser, "model", None)
        if model is not None:
            return sana_runtime_receipt(model)
        return {"installed": [], "fallbacks": []}

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
    "Sana1p51600M1024pxPipeline",
    "Sana1p54800M1024pxPipeline",
    "Sana600M512pxPipeline",
    "Sana600M1024pxPipeline",
    "Sana1600M2kBf16Pipeline",
    "Sana1600M4kBf16Pipeline",
    "Sana1600M512pxMultilingPipeline",
    "Sana1600M512pxPipeline",
    "Sana1600M1024pxBf16Pipeline",
    "Sana1600M1024pxMultilingPipeline",
    "Sana1600M1024pxPipeline",
    "SanaControlnet600M1024pxPipeline",
    "SanaControlnet1600M1024pxBf16Pipeline",
    "SanaPipeline",
    "SanaSprint600M1024pxPipeline",
    "SanaSprint1600M1024pxPipeline",
    "SanaStreaming2b720pPipeline",
    "SanaStreamingBidirectional2b720pPipeline",
    "SanaVideo2b480pPipeline",
    "SanaVideo2b720pPipeline",
]
