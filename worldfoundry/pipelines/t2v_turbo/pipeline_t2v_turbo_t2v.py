"""Public T2V-Turbo pipeline backed by WorldFoundry native diffusion."""

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
from worldfoundry.operators.runtime_video_operator import RuntimeVideoOperator
from worldfoundry.synthesis.visual_generation.memory.video import VideoArtifactMemory

from ..pipeline_utils import PipelineABC


class T2VTurboT2VPipeline(PipelineABC):
    """Studio-facing adapter for the native VideoCrafter2 T2V-Turbo recipe."""

    MODEL_ID = "t2v_turbo_t2v"
    DEFAULT_HEIGHT = 320
    DEFAULT_WIDTH = 512
    DEFAULT_NUM_FRAMES = 16
    DEFAULT_NUM_INFERENCE_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 7.5
    DEFAULT_FPS = 8

    def __init__(self, *, native_pipeline: NativeDiffusionPipeline, device: str) -> None:
        super().__init__(
            model_id=self.MODEL_ID,
            operator=RuntimeVideoOperator(generation_type="t2v"),
            memory_module=VideoArtifactMemory(model_id=self.MODEL_ID),
            device=device,
        )
        self.native_pipeline = native_pipeline
        self.synthesis_model = native_pipeline
        self.generation_type = "t2v"
        self.model_name = self.MODEL_ID

    @staticmethod
    def _checkpoint_overrides(
        model_path: str | Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> dict[str, str] | None:
        overrides: dict[str, str] = {}
        base = options.get("base_checkpoint", options.get("model_ckpt", model_path))
        lora = options.get("lora_checkpoint", options.get("lora_path"))
        if isinstance(base, (str, Path)):
            value = str(base)
            overrides["base"] = value if "://" in value else str(Path(value).expanduser())
        if isinstance(lora, (str, Path)):
            value = str(lora)
            overrides["lora"] = value if "://" in value else str(Path(value).expanduser())
        return overrides or None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "T2VTurboT2VPipeline":
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        native = NativeDiffusionPipeline.from_pretrained(
            cls.MODEL_ID,
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner="T2V-Turbo",
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"),
                    allow_disk=False,
                    owner="T2V-Turbo",
                ),
            ),
            checkpoint_overrides=cls._checkpoint_overrides(model_path, options),
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
        return {
            "model_id": cls.MODEL_ID,
            "backend": "worldfoundry-native-diffusion",
            "native_inference": True,
            "checkpoints": cls._checkpoint_overrides(model_path, options),
            "blocked": False,
        }

    def process(self, prompt: str | list[str] = "", images: Any = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if images is not None:
            raise ValueError("T2V-Turbo is text-to-video and does not accept input images")
        values = prompt if isinstance(prompt, list) else [prompt]
        processed = []
        for value in values:
            self.operator.get_interaction(value)
            try:
                interaction = self.operator.process_interaction()
            finally:
                self.operator.delete_last_interaction()
            processed.append(str(interaction["processed_prompt"]))
        return {"prompt": processed if isinstance(prompt, list) else processed[0]}

    def __call__(
        self,
        prompt: str | list[str] = "",
        images: Any = None,
        output_path: str | Path | None = None,
        fps: int | None = None,
        return_dict: bool = False,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        lcm_origin_steps: int = 200,
        seed: int = 0,
        output_type: str = "video",
        **kwargs: Any,
    ) -> Any:
        del kwargs
        processed = self.process(prompt=prompt, images=images)
        actual_seed = secrets.randbits(63) if int(seed) < 0 else int(seed)
        output = self.native_pipeline(
            DiffusionRequest(
                prompt=processed["prompt"],
                height=int(height or self.DEFAULT_HEIGHT),
                width=int(width or self.DEFAULT_WIDTH),
                num_frames=int(num_frames or self.DEFAULT_NUM_FRAMES),
                sampling=SamplingConfig(
                    num_inference_steps=int(num_inference_steps or self.DEFAULT_NUM_INFERENCE_STEPS),
                    guidance_scale=float(
                        guidance_scale if guidance_scale is not None else self.DEFAULT_GUIDANCE_SCALE
                    ),
                    seed=actual_seed,
                    scheduler_options={"lcm_origin_steps": int(lcm_origin_steps)},
                ),
                inputs={
                    "fps": int(fps or self.DEFAULT_FPS),
                    "return_latent": output_type == "latent",
                },
            )
        )
        artifact_path = None
        if output_path is not None and output_type != "latent":
            artifact_path = save_image_or_video_tensor(
                output.sample,
                output_path,
                fps=int(fps or self.DEFAULT_FPS),
            )
        result = {
            "video": output.sample,
            "latents": output.latents,
            "artifact_path": artifact_path,
            "generated_video_path": artifact_path,
            "model_name": self.MODEL_ID,
            "generation_type": "t2v",
            "metadata": dict(output.metadata),
        }
        return result if return_dict else (artifact_path or output.sample)

    def stream(self, prompt: str = "", images: Any = None, **kwargs: Any) -> Any:
        result = self(prompt=prompt, images=images, return_dict=True, **kwargs)
        value = result.get("artifact_path") or result["video"]
        self.memory_module.record(value, metadata={"prompt": prompt, "model_name": self.MODEL_ID})
        return value

    def get_synthesis_model(self) -> NativeDiffusionPipeline:
        return self.native_pipeline


__all__ = ["T2VTurboT2VPipeline"]
