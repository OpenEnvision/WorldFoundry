"""Public HunyuanVideo pipelines backed only by WorldFoundry native diffusion."""

from __future__ import annotations

import secrets
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.optimizations import (
    AttentionBackend,
    RuntimePolicy,
    parse_offload_policy,
    parse_torch_dtype,
)
from worldfoundry.core.media.codecs.video import save_image_or_video_tensor
from worldfoundry.operators.runtime_video_operator import RuntimeVideoOperator
from worldfoundry.synthesis.visual_generation.memory.video import VideoArtifactMemory

from ..pipeline_utils import PipelineABC


def _resolve_hunyuan_checkpoint_source(
    source: object,
    *,
    model_id: str,
) -> object:
    """Resolve stale integration paths to the unified staged checkpoint root."""

    if model_id.startswith("hunyuanvideo-1.5"):
        transformer_dir = "720p_i2v" if model_id.endswith("i2v") else "720p_t2v"
        sentinels = (Path("transformer") / transformer_dir / "diffusion_pytorch_model.safetensors",)
    elif model_id == "hunyuanvideo-i2v":
        prefix = Path("hunyuan-video-i2v-720p/transformers")
        sentinels = (
            prefix / "mp_rank_00_model_states.safetensors",
            prefix / "mp_rank_00_model_states.pt",
        )
    else:
        prefix = Path("hunyuan-video-t2v-720p/transformers")
        sentinels = (
            prefix / "mp_rank_00_model_states.safetensors",
            prefix / "mp_rank_00_model_states.pt",
        )
    if isinstance(source, (str, Path)):
        explicit = Path(source).expanduser()
        if explicit.is_dir() and any((explicit / sentinel).is_file() for sentinel in sentinels):
            return source
    checkpoint_root = os.getenv("WORLDFOUNDRY_CKPT_DIR") or os.getenv("WORLDEVALS_CKPT_DIR")
    if not checkpoint_root:
        return source
    if model_id.startswith("hunyuanvideo-1.5"):
        directory = "tencent--HunyuanVideo-1.5"
    elif model_id == "hunyuanvideo-i2v":
        directory = "tencent--HunyuanVideo-I2V"
    else:
        directory = "tencent--HunyuanVideo"
    candidate = Path(checkpoint_root).expanduser() / directory
    return candidate if any((candidate / sentinel).is_file() for sentinel in sentinels) else source


class NativeHunyuanVideoPipeline(PipelineABC):
    """Stable Studio-facing adapter around one declarative HunyuanVideo recipe."""

    MODEL_ID = "hunyuanvideo-t2v"
    GENERATION_TYPE = "t2v"
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 129
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_FPS = 24
    DEFAULT_GUIDANCE_SCALE = 6.0

    @staticmethod
    def _attention_policy(value: object, *, model_id: str) -> AttentionBackend:
        """Select the memory-bounded official attention path for HunyuanVideo 1.5.

        The 720p/121-frame 1.5 recipes have roughly 40k visual tokens.  Native
        torch SDPA falls back to a dense masked kernel for that shape and tries
        to materialize an attention matrix larger than 23 GiB.  FlashAttention
        is the upstream full-resolution path and changes neither sampling steps
        nor output geometry, so treat ``auto`` as Flash for 1.5 while preserving
        explicit Torch/SDPA requests for diagnostics.
        """

        normalized = str(value or "auto").strip().lower().replace("-", "_")
        if normalized == "auto" and model_id.startswith("hunyuanvideo-1.5"):
            return AttentionBackend.FLASH
        if normalized in {"flash_attention", "flash_attention_2", "flash2", "flash_attn"}:
            return AttentionBackend.FLASH
        return AttentionBackend(normalized)

    def __init__(self, *, native_pipeline: NativeDiffusionPipeline, device: str, model_id: str) -> None:
        super().__init__(
            model_id=model_id,
            operator=RuntimeVideoOperator(generation_type=self.GENERATION_TYPE),
            memory_module=VideoArtifactMemory(model_id=model_id),
            device=device,
        )
        self.native_pipeline = native_pipeline

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "NativeHunyuanVideoPipeline":
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        resolved_model_id = str(options.get("variant_id") or model_id or cls.MODEL_ID)
        source = options.get("checkpoint_path", options.get("model_path", model_path))
        source = _resolve_hunyuan_checkpoint_source(source, model_id=resolved_model_id)
        overrides: dict[str, Any] = {}
        source_root: Path | None = None
        if isinstance(source, (str, Path)) and Path(source).expanduser().is_dir():
            source_root = Path(source).expanduser().resolve()
            root = str(source_root)
            overrides.update({name: root for name in ("transformer", "vae")})
            if resolved_model_id.startswith("hunyuanvideo-1.5"):
                overrides["resources"] = root
                local_vision = source_root / "vision_encoder" / "siglip"
                vision_files = (
                    "image_encoder/config.json",
                    "image_encoder/model.safetensors",
                    "feature_extractor/preprocessor_config.json",
                )
                if resolved_model_id == "hunyuanvideo-1.5-i2v" and all(
                    (local_vision / name).is_file() for name in vision_files
                ):
                    overrides["vision"] = str(local_vision.resolve())

        if resolved_model_id in {"hunyuanvideo-t2v", "hunyuanvideo-i2v"}:
            # The original Tencent repositories do not ship either text
            # encoder. Prefer explicit paths, then legacy merged-directory
            # layouts, and otherwise let each recipe role resolve its actual
            # XTuner/OpenAI repository from WORLDFOUNDRY_CKPT_DIR.
            primary_source = options.get(
                "primary_text_encoder_path",
                options.get("text_encoder_path"),
            )
            clip_source = options.get("clip_text_encoder_path", options.get("clip_path"))
            if primary_source is not None:
                overrides["primary"] = str(Path(primary_source).expanduser())
            elif source_root is not None:
                primary_name = (
                    "text_encoder_i2v"
                    if resolved_model_id == "hunyuanvideo-i2v"
                    else "text_encoder"
                )
                local_primary = source_root / primary_name
                if (local_primary / "config.json").is_file():
                    overrides["primary"] = str(local_primary.resolve())
            if clip_source is not None:
                overrides["clip"] = str(Path(clip_source).expanduser())
            elif source_root is not None:
                local_clip = source_root / "text_encoder_2"
                if (local_clip / "config.json").is_file():
                    overrides["clip"] = str(local_clip.resolve())

        explicit_overrides = options.get("checkpoint_overrides")
        if explicit_overrides is not None:
            if not isinstance(explicit_overrides, Mapping):
                raise TypeError("checkpoint_overrides must be a mapping of recipe roles to local checkpoints")
            overrides.update(dict(explicit_overrides))

        native = NativeDiffusionPipeline.from_pretrained(
            resolved_model_id,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner="HunyuanVideo",
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
                    allow_disk=False,
                    owner="HunyuanVideo",
                ),
                attention=cls._attention_policy(
                    options.get("attention_backend", options.get("attention")),
                    model_id=resolved_model_id,
                ),
            ),
            checkpoint_overrides=overrides or None,
            component_options={
                "latent_encoder:codec": {
                    "tiled": bool(options.get("vae_tiling", True)),
                },
                "conditioner:main": {
                    "release_after_encode": bool(options.get("release_text_encoders_after_encode", False)),
                },
            },
        )
        return cls(native_pipeline=native, device=device, model_id=resolved_model_id)

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        source = options.get("checkpoint_path", options.get("model_path", model_path))
        return {
            "model_id": str(options.get("variant_id") or options.get("model_id") or cls.MODEL_ID),
            "checkpoint": str(source) if source is not None else None,
            "backend": "worldfoundry-native-diffusion",
            "native_inference": True,
            "blocked": False,
        }

    def _process_prompt(self, prompt: str) -> str:
        self.operator.get_interaction(prompt)
        try:
            interaction = self.operator.process_interaction()
        finally:
            self.operator.delete_last_interaction()
        return str(interaction["processed_prompt"])

    def process(self, prompt: str | list[str], images: Any = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if self.GENERATION_TYPE == "i2v" and images is None:
            raise ValueError(f"{self.model_id} requires an input image")
        if self.GENERATION_TYPE == "t2v" and images is not None:
            raise ValueError(f"{self.model_id} is text-to-video and does not accept images")
        processed = [self._process_prompt(value) for value in prompt] if isinstance(prompt, list) else self._process_prompt(prompt)
        return {"prompt": processed, "images": images}

    def __call__(
        self,
        prompt: str | list[str],
        images: Any = None,
        image: Any = None,
        image_path: str | None = None,
        output_path: str | Path | None = None,
        num_frames: int | None = None,
        frames: int | None = None,
        frame_num: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        embedded_guidance_scale: float | None = None,
        seed: int = 0,
        fps: int | None = None,
        output_type: str = "video",
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        image_value = image if image is not None else images
        image_value = image_value if image_value is not None else image_path
        processed = self.process(prompt=prompt, images=image_value)
        actual_seed = secrets.randbits(63) if int(seed) < 0 else int(seed)
        scale = float(guidance_scale if guidance_scale is not None else self.DEFAULT_GUIDANCE_SCALE)
        frame_values = [value for value in (num_frames, frames, frame_num) if value is not None]
        if len({int(value) for value in frame_values}) > 1:
            raise ValueError("num_frames, frames, and frame_num must agree when provided together")
        resolved_num_frames = int(frame_values[0]) if frame_values else self.DEFAULT_NUM_FRAMES
        inputs: dict[str, object] = {
            "fps": int(fps or self.DEFAULT_FPS),
            "return_latent": output_type == "latent",
            "embedded_guidance_scale": float(embedded_guidance_scale or scale),
        }
        if image_value is not None:
            inputs["image"] = image_value
        request = DiffusionRequest(
            prompt=processed["prompt"],
            height=int(height or self.DEFAULT_HEIGHT),
            width=int(width or self.DEFAULT_WIDTH),
            num_frames=resolved_num_frames,
            sampling=SamplingConfig(
                num_inference_steps=int(num_inference_steps or self.DEFAULT_NUM_INFERENCE_STEPS),
                guidance_scale=scale,
                seed=actual_seed,
            ),
            inputs=inputs,
        )
        output = self.native_pipeline(request)
        artifact_path = None
        if output_path is not None and output_type != "latent":
            artifact_path = save_image_or_video_tensor(output.sample, output_path, fps=int(fps or self.DEFAULT_FPS))
        result = {
            "video": output.sample,
            "latents": output.latents,
            "artifact_path": artifact_path,
            "generated_video_path": artifact_path,
            "model_name": self.model_id,
            "generation_type": self.GENERATION_TYPE,
            "metadata": dict(output.metadata),
        }
        return result if return_dict else (artifact_path or output.sample)

    def stream(self, prompt: str, images: Any = None, **kwargs: Any) -> Any:
        result = self(prompt=prompt, images=images, return_dict=True, **kwargs)
        value = result.get("artifact_path") or result["video"]
        self.memory_module.record(value, metadata={"prompt": prompt, "model_name": self.model_id})
        return value

    def get_synthesis_model(self) -> NativeDiffusionPipeline:
        return self.native_pipeline


class HunyuanVideoT2VPipeline(NativeHunyuanVideoPipeline):
    MODEL_ID = "hunyuanvideo-t2v"


class HunyuanVideoI2VPipeline(NativeHunyuanVideoPipeline):
    MODEL_ID = "hunyuanvideo-i2v"
    GENERATION_TYPE = "i2v"


class HunyuanVideo15T2VPipeline(NativeHunyuanVideoPipeline):
    MODEL_ID = "hunyuanvideo-1.5-t2v"
    DEFAULT_HEIGHT = 720
    DEFAULT_WIDTH = 1280
    DEFAULT_NUM_FRAMES = 121
    DEFAULT_GUIDANCE_SCALE = 6.0


class HunyuanVideo15I2VPipeline(HunyuanVideo15T2VPipeline):
    MODEL_ID = "hunyuanvideo-1.5-i2v"
    GENERATION_TYPE = "i2v"
    DEFAULT_WIDTH = 544


__all__ = [
    "HunyuanVideo15I2VPipeline",
    "HunyuanVideo15T2VPipeline",
    "HunyuanVideoI2VPipeline",
    "HunyuanVideoT2VPipeline",
    "NativeHunyuanVideoPipeline",
]
