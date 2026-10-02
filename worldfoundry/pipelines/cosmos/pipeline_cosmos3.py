"""Public Cosmos3 pipeline backed by the canonical native diffusion infra."""

from __future__ import annotations

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

from .cosmos3_runtime import resolve_cosmos3_runtime_options
from ..pipeline_utils import PipelineABC


def _model_id(requested: str | None, source: object) -> str:
    value = str(requested or "").strip().lower()
    if value in {"cosmos3-super", "super"} or "super" in str(source).lower():
        return "cosmos3-super"
    if value not in {"", "cosmos3", "cosmos3-nano", "nano"}:
        raise ValueError(f"unsupported Cosmos3 model selector: {requested!r}")
    return "cosmos3-nano"


class Cosmos3Pipeline(PipelineABC):
    """Stable Studio-facing wrapper around Cosmos3 Nano or Super recipes."""

    MODEL_ID = "cosmos3"

    def __init__(self, *, native_pipeline: NativeDiffusionPipeline, device: str, model_id: str) -> None:
        super().__init__(model_id=model_id, synthesis_model=None, device=device)
        self.native_pipeline = native_pipeline

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "Cosmos3Pipeline":
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        source = options.get("checkpoint_path", options.get("model_path", model_path))
        resolved_model_id = _model_id(str(options.get("variant_id") or model_id or ""), source)
        checkpoint_overrides = None
        if isinstance(source, (str, Path)) and Path(source).expanduser().is_dir():
            root = str(Path(source).expanduser().resolve())
            checkpoint_overrides = {name: root for name in ("transformer", "vae", "sound", "tokenizer", "scheduler")}
        visible_gpus = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        offload_mode, device_map = resolve_cosmos3_runtime_options(
            options,
            model_id=resolved_model_id,
            visible_gpus=visible_gpus,
        )
        policy_options = {"device_map": device_map} if device_map else {}
        native = NativeDiffusionPipeline.from_pretrained(
            resolved_model_id,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(options.get("torch_dtype", options.get("dtype")), owner="Cosmos3"),
                offload=parse_offload_policy(
                    offload_mode,
                    allow_disk=False,
                    owner="Cosmos3",
                ),
                options=policy_options,
            ),
            checkpoint_overrides=checkpoint_overrides,
            component_options={
                "decoder:main": {
                    "tiled": bool(options.get("vae_tiling", False)),
                    "tile_size": tuple(options.get("vae_tile_size", (34, 34))),
                    "tile_stride": tuple(options.get("vae_tile_stride", (18, 16))),
                }
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
        model_id = _model_id(str(options.get("variant_id") or options.get("model_id") or ""), source)
        return {
            "model_id": model_id,
            "checkpoint": str(source) if source is not None else None,
            "backend": "worldfoundry-native-diffusion",
            "native_inference": True,
            "blocked": False,
        }

    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        image: Any = None,
        images: Any = None,
        image_path: str | None = None,
        input_path: str | None = None,
        video: Any = None,
        videos: Any = None,
        video_path: str | None = None,
        action: Mapping[str, Any] | None = None,
        action_mode: str | None = None,
        action_chunk_size: int | None = None,
        domain_name: str | None = None,
        raw_actions: Any = None,
        enable_sound: bool = False,
        output_path: str | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        output_type: str = "video",
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        inputs: dict[str, object] = {
            "fps": float(fps or 24.0),
            "enable_sound": bool(enable_sound),
            "return_latent": output_type == "latent",
        }
        image_value = image if image is not None else images
        image_value = image_value if image_value is not None else image_path
        video_value = video if video is not None else videos
        video_value = video_value if video_value is not None else video_path
        if image_value is None and video_value is None and input_path is not None:
            if Path(input_path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                video_value = input_path
            else:
                image_value = input_path
        if image_value is not None:
            inputs["image"] = image_value
        if video_value is not None:
            inputs["video"] = video_value

        action_values = dict(action or {})
        explicit_action = {
            "action_mode": action_mode,
            "action_chunk_size": action_chunk_size,
            "action_domain_name": domain_name,
            "raw_actions": raw_actions,
        }
        for key, value in explicit_action.items():
            if value is not None:
                action_values[key] = value
        if "mode" in action_values:
            action_values.setdefault("action_mode", action_values.pop("mode"))
        if "chunk_size" in action_values:
            action_values.setdefault("action_chunk_size", action_values.pop("chunk_size"))
        if "domain_name" in action_values:
            action_values.setdefault("action_domain_name", action_values.pop("domain_name"))
        inputs.update(action_values)

        request = DiffusionRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=int(height or 720),
            width=int(width or 1280),
            num_frames=int(num_frames or 189),
            sampling=SamplingConfig(
                num_inference_steps=int(num_inference_steps or 35),
                guidance_scale=float(guidance_scale or 1.0),
                seed=int(seed or 0),
            ),
            inputs=inputs,
        )
        output = self.native_pipeline(request)
        artifact_path = None
        if output_path is not None:
            artifact_path = save_image_or_video_tensor(output.sample, output_path, fps=int(fps or 24))
        result = {
            "video": output.sample,
            "artifact_path": artifact_path,
            "latents": output.latents,
            **dict(output.artifacts),
            "metadata": dict(output.metadata),
        }
        return result if return_dict else (artifact_path or output.sample)


__all__ = ["Cosmos3Pipeline", "resolve_cosmos3_runtime_options"]
