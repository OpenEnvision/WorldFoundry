"""Public Cosmos Transfer 2.5 pipeline backed by native diffusion infra."""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.optimizations import (
    RuntimePolicy,
    parse_offload_policy,
    parse_torch_dtype,
)
from worldfoundry.core.media.codecs.video import save_image_or_video_tensor

from ..pipeline_utils import PipelineABC
from .reason1_options import reason1_component_options

COSMOS_TRANSFER2P5_MODEL_ID = "cosmos-transfer2.5-2b-controlled-video"
COSMOS_TRANSFER2P5_DEFAULT_FPS = 16
COSMOS_TRANSFER2P5_DEFAULT_NUM_FRAMES = 93
COSMOS_TRANSFER2P5_DEFAULT_NUM_INFERENCE_STEPS = 35
DEFAULT_NEGATIVE_PROMPT = (
    "The video captures ugly scenes, static motion, motion blur, over-saturation, shaky footage, low resolution, "
    "grainy texture, poor lighting, artifacts, unnatural transitions, visual noise, and flickering."
)


def _local_override(value: object) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else None


class CosmosTransfer2p5Pipeline(PipelineABC):
    """Studio-facing wrapper around the native 2B VACE recipe."""

    MODEL_ID = "cosmos-transfer-2.5"

    def __init__(self, *, native_pipeline: NativeDiffusionPipeline, device: str) -> None:
        super().__init__(model_id=COSMOS_TRANSFER2P5_MODEL_ID, synthesis_model=None, device=device)
        self.native_pipeline = native_pipeline

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "CosmosTransfer2p5Pipeline":
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        encoder_options = reason1_component_options(options)
        from worldfoundry.core.distributed.model_parallel.inference_parallel import validate_parallel_degrees

        tp, cp = validate_parallel_degrees(options.get("tensor_parallel", 1), options.get("context_parallel", 1))
        if tp * cp > 1:
            raise ValueError("multi-rank Cosmos Transfer inference is not supported")
        requested = str(options.get("variant_id") or model_id or "").lower()
        supported = {
            "",
            "edge",
            "cosmos-transfer-2.5",
            "cosmos-transfer2.5",
            "cosmos-transfer2p5",
            "cosmos-transfer-2.5-2b",
            COSMOS_TRANSFER2P5_MODEL_ID,
        }
        if requested not in supported:
            raise ValueError(f"unsupported Cosmos Transfer 2.5 selector: {requested!r}")
        control_variant = str(options.get("controlnet_variant", "edge")).lower()
        if control_variant != "edge":
            raise ValueError("native Cosmos Transfer 2.5 currently integrates the official general/edge checkpoint")

        transformer_source = options.get(
            "controlnet_model_path",
            options.get("checkpoint_path", options.get("transformer_model_path", model_path)),
        )
        overrides: dict[str, str] = {}
        if source := _local_override(transformer_source):
            overrides["transformer"] = source
        if source := _local_override(options.get("vae_model_path")):
            overrides["vae"] = source
        text_source = _local_override(options.get("text_encoder_model_path"))
        if text_source is not None:
            overrides.update({"text-encoder": text_source, "tokenizer": text_source})

        native = NativeDiffusionPipeline.from_pretrained(
            COSMOS_TRANSFER2P5_MODEL_ID,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner="Cosmos Transfer2.5",
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
                    owner="Cosmos Transfer2.5",
                ),
            ),
            checkpoint_overrides=overrides or None,
            extensions=options.get("extensions", ()),
            component_options={
                "conditioner:main": encoder_options,
                "latent_initializer:main": {
                    "tiled": bool(options.get("vae_tiling", False)),
                    "tile_size": tuple(options.get("vae_tile_size", (34, 34))),
                    "tile_stride": tuple(options.get("vae_tile_stride", (18, 16))),
                }
            },
        )
        return cls(native_pipeline=native, device=device)

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
        control_variant = str(options.get("controlnet_variant", "edge")).lower()
        if control_variant != "edge":
            raise ValueError("native Cosmos Transfer 2.5 currently supports controlnet_variant='edge'")
        source = options.get("checkpoint_path", options.get("transformer_model_path", model_path))
        local = _local_override(source)
        return {
            "model_id": COSMOS_TRANSFER2P5_MODEL_ID,
            "checkpoint": local or (str(source) if source is not None else None),
            "controlnet_variant": control_variant,
            "backend": "worldfoundry-native-diffusion",
            "native_inference": True,
            "blocked": source is not None and local is None and Path(str(source)).expanduser().is_absolute(),
        }

    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        images: Any = None,
        image: Any = None,
        image_path: str | None = None,
        video: Any = None,
        video_path: str | None = None,
        control_video: Any = None,
        control: Any = None,
        controlnet_variant: str = "edge",
        control_context_scale: float = 1.0,
        control_is_preprocessed: bool = False,
        canny_low_threshold: int = 100,
        canny_high_threshold: int = 200,
        output_path: str | None = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = COSMOS_TRANSFER2P5_DEFAULT_NUM_INFERENCE_STEPS,
        fps: int = COSMOS_TRANSFER2P5_DEFAULT_FPS,
        num_frames: int = COSMOS_TRANSFER2P5_DEFAULT_NUM_FRAMES,
        height: int = 704,
        width: int = 1280,
        seed: int = 0,
        conditional_frame_timestep: float = 0.0,
        output_type: str = "video",
        return_dict: bool = False,
        interactions: Any = None,
        **kwargs: Any,
    ) -> Any:
        del interactions, kwargs
        if controlnet_variant.lower() != "edge":
            raise ValueError("native Cosmos Transfer 2.5 currently supports controlnet_variant='edge'")
        control_value = control_video if control_video is not None else control
        control_value = control_value if control_value is not None else video
        control_value = control_value if control_value is not None else video_path
        if control_value is None:
            raise ValueError("Cosmos Transfer 2.5 requires video/control_video input")
        image_value = image if image is not None else images
        image_value = image_value if image_value is not None else image_path
        inputs: dict[str, object] = {
            "control_video": control_value,
            "controlnet_variant": "edge",
            "control_context_scale": float(control_context_scale),
            "control_is_preprocessed": bool(control_is_preprocessed),
            "canny_low_threshold": int(canny_low_threshold),
            "canny_high_threshold": int(canny_high_threshold),
            "fps": int(fps),
            "return_latent": output_type == "latent",
            "conditional_frame_timestep": float(conditional_frame_timestep),
        }
        if image_value is not None:
            inputs["image"] = image_value
        actual_seed = secrets.randbits(63) if int(seed) < 0 else int(seed)
        request = DiffusionRequest(
            prompt=prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            sampling=SamplingConfig(
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                seed=actual_seed,
            ),
            inputs=inputs,
        )
        output = self.native_pipeline(request)
        artifact_path = None
        if output_path is not None and output_type != "latent":
            artifact_path = save_image_or_video_tensor(output.sample, output_path, fps=int(fps))
        result = {
            "video": output.sample,
            "latents": output.latents,
            "artifact_path": artifact_path,
            "metadata": dict(output.metadata),
        }
        return result if return_dict else (artifact_path or output.sample)


__all__ = ["CosmosTransfer2p5Pipeline"]
