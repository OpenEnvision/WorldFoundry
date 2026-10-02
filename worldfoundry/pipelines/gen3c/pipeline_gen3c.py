"""Public GEN3C pipeline backed by the native Cosmos Predict1 recipe."""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.optimizations import (
    RuntimePolicy,
    parse_offload_policy,
    parse_torch_dtype,
)
from worldfoundry.core.media.codecs.video import save_image_or_video_tensor
from worldfoundry.operators.gen3c_operator import Gen3COperator
from worldfoundry.synthesis.visual_generation.memory.stream import VisualFrameMemory

from ..pipeline_utils import PipelineABC
from .constants import DEFAULT_GEN3C_NEGATIVE_PROMPT, DEFAULT_GEN3C_PROMPT

GEN3C_DEFAULT_HEIGHT = 704
GEN3C_DEFAULT_WIDTH = 1280
GEN3C_DEFAULT_NUM_FRAMES = 121
GEN3C_DEFAULT_NUM_INFERENCE_STEPS = 35
GEN3C_DEFAULT_FPS = 24


def _existing_subdirectory(root: Path, names: tuple[str, ...], required_file: str) -> Path | None:
    for name in names:
        candidate = root / name
        if (candidate / required_file).is_file():
            return candidate
    return None


class Gen3CPipeline(PipelineABC):
    """Studio-facing wrapper around the native GEN3C diffusion components."""

    MODEL_ID = "gen3c"

    def __init__(
        self,
        *,
        native_pipeline: NativeDiffusionPipeline,
        operator: Gen3COperator | None = None,
        memory_module: Any = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            model_id=self.MODEL_ID,
            operator=operator or Gen3COperator(),
            synthesis_model=native_pipeline,
            memory_module=memory_module or VisualFrameMemory(model_id=self.MODEL_ID),
            device=device,
        )
        self.native_pipeline = native_pipeline

    @classmethod
    def _options(
        cls,
        model_path: str | Mapping[str, Any] | None,
        required_components: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        options = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        return options

    @classmethod
    def _checkpoint_overrides(
        cls,
        model_path: str | Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> dict[str, str] | None:
        overrides: dict[str, str] = {}
        source = options.get(
            "checkpoint_dir",
            options.get("checkpoint_path", options.get("pretrained_model_path", model_path)),
        )
        if isinstance(source, (str, Path)):
            root = Path(source).expanduser()
            if root.is_file():
                overrides["transformer"] = str(root.resolve())
            elif root.is_dir():
                transformer = root if (root / "model.pt").is_file() else _existing_subdirectory(
                    root, ("GEN3C-Cosmos-7B", "nvidia--GEN3C-Cosmos-7B"), "model.pt"
                )
                tokenizer = root if (root / "encoder.jit").is_file() else _existing_subdirectory(
                    root,
                    ("Cosmos-Tokenize1-CV8x8x8-720p", "nvidia--Cosmos-Tokenize1-CV8x8x8-720p"),
                    "encoder.jit",
                )
                text = root if (root / "pytorch_model.bin").is_file() else _existing_subdirectory(
                    root, ("t5-11b", "google-t5/t5-11b", "google-t5--t5-11b"), "pytorch_model.bin"
                )
                depth = root if (root / "model.pt").is_file() and root.name == "moge-vitl" else _existing_subdirectory(
                    root, ("moge-vitl", "Ruicheng--moge-vitl"), "model.pt"
                )
                if transformer is not None:
                    overrides["transformer"] = str(transformer.resolve())
                if tokenizer is not None:
                    overrides["tokenizer"] = str(tokenizer.resolve())
                if text is not None:
                    overrides.update(
                        {"text-encoder": str(text.resolve()), "text-tokenizer": str(text.resolve())}
                    )
                if depth is not None:
                    overrides["depth-model"] = str(depth.resolve())
        explicit = {
            "transformer": options.get("transformer_model_path", options.get("cosmos_checkpoint")),
            "tokenizer": options.get("tokenizer_model_path"),
            "text-encoder": options.get("text_encoder_model_path"),
            "text-tokenizer": options.get("text_tokenizer_path", options.get("text_encoder_model_path")),
            "depth-model": options.get("moge_pretrained", options.get("depth_model_path")),
        }
        for role, value in explicit.items():
            if isinstance(value, (str, Path)) and Path(value).expanduser().exists():
                overrides[role] = str(Path(value).expanduser().resolve())
        return overrides or None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "Gen3CPipeline":
        options = cls._options(model_path, required_components, kwargs)
        native = NativeDiffusionPipeline.from_pretrained(
            "gen3c",
            policy=RuntimePolicy(
                device=torch.device(device),
                dtype=parse_torch_dtype(
                    options.get("torch_dtype", options.get("weight_dtype", options.get("dtype"))),
                    owner="GEN3C",
                ),
                offload=parse_offload_policy(
                    options.get("offload_mode", "block"), allow_disk=False, owner="GEN3C"
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
        options = cls._options(model_path, required_components, kwargs)
        return {
            "model_id": "gen3c-cosmos1-7b",
            "backend": "worldfoundry-native-diffusion",
            "native_inference": True,
            "checkpoints": cls._checkpoint_overrides(model_path, options),
            "blocked": False,
        }

    def process(
        self,
        images: Any = None,
        interactions: Sequence[str | Mapping[str, Any]] | None = None,
        prompt: str = DEFAULT_GEN3C_PROMPT,
        trajectory: str | None = None,
    ) -> dict[str, Any]:
        selected = interactions or [trajectory or self.operator.DEFAULT_TRAJECTORY]
        image = self.operator.process_perception(images)
        self.operator.get_interaction(selected)
        try:
            condition = self.operator.process_interaction(prompt=prompt)
        finally:
            self.operator.delete_last_interaction()
        return {"image": image, "prompt": prompt or "", **condition}

    def __call__(
        self,
        images: Any = None,
        interactions: Sequence[str | Mapping[str, Any]] | None = None,
        prompt: str = "",
        negative_prompt: str | None = DEFAULT_GEN3C_NEGATIVE_PROMPT,
        trajectory: str | None = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        height: int = GEN3C_DEFAULT_HEIGHT,
        width: int = GEN3C_DEFAULT_WIDTH,
        num_frames: int = GEN3C_DEFAULT_NUM_FRAMES,
        num_inference_steps: int = GEN3C_DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = 1.0,
        fps: int = GEN3C_DEFAULT_FPS,
        seed: int = 1,
        output_type: str = "video",
        camera_path: Mapping[str, Any] | None = None,
        region_hint: str | None = None,
        rendered_warp_images: Any = None,
        rendered_warp_masks: Any = None,
        camera_to_world: Any = None,
        camera_intrinsics: Any = None,
        **kwargs: Any,
    ) -> Any:
        input_path = kwargs.pop("input_path", kwargs.pop("image_path", None))
        if images is None:
            images = input_path
        interaction_signal = kwargs.pop("interaction_signal", None)
        if interactions is None and interaction_signal is not None:
            interactions = interaction_signal
        # The curated Workspace contract mirrors the official GEN3C CLI while
        # this native pipeline uses WorldFoundry's common diffusion names.
        # Normalize the official aliases before rejecting genuinely unknown
        # options so the catalog's own default task remains executable.
        num_frames = int(kwargs.pop("num_video_frames", num_frames))
        num_inference_steps = int(kwargs.pop("num_steps", num_inference_steps))
        guidance_scale = float(kwargs.pop("guidance", guidance_scale))
        output_dir = kwargs.pop("output_dir", None)
        if output_path is None and output_dir:
            output_root = Path(output_dir).expanduser()
            output_root.mkdir(parents=True, exist_ok=True)
            output_path = output_root / "gen3c.mp4"
        for compatibility_option in (
            "disable_prompt_upsampler",
            "disable_guardrail",
            "disable_prompt_encoder",
            "offload_diffusion_transformer",
            "offload_tokenizer",
            "offload_text_encoder_model",
            "offload_prompt_upsampler",
            "offload_guardrail_models",
            "num_gpus",
            "save_buffer",
            "filter_points_threshold",
            "foreground_masking",
            "noise_aug_strength",
            "moge_pretrained",
        ):
            kwargs.pop(compatibility_option, None)
        condition_augment_sigma = float(kwargs.pop("condition_augment_sigma", 0.001))
        camera_rotation = str(kwargs.pop("camera_rotation", "center_facing"))
        movement_distance = float(kwargs.pop("movement_distance", 0.3))
        center_depth = float(kwargs.pop("center_depth", 1.0))
        center_depth_quantile = bool(kwargs.pop("center_depth_quantile", False))
        center_depth_quantile_value = float(kwargs.pop("center_depth_quantile_value", 0.5))
        if kwargs:
            raise TypeError(f"unsupported GEN3C inference options: {sorted(kwargs)}")
        if region_hint and not prompt:
            prompt = region_hint
        # Cosmos/GEN3C accepts an empty prompt but deterministically decodes an
        # almost-all-zero video for the packaged checkpoint. Treat blank text
        # as an omitted optional field so Studio's default demo remains valid.
        if not str(prompt or "").strip():
            prompt = DEFAULT_GEN3C_PROMPT
        if camera_path is not None and camera_to_world is None:
            from worldfoundry.core.geometry.path import sample_camera_path

            sampled_path = sample_camera_path(camera_path, frame_count=int(num_frames))
            camera_to_world = np.linalg.inv(sampled_path["camera_w2c"])[None]
            first_fov = float(sampled_path["camera_path"]["keyframes"][0]["fov"])
            base_focal = 0.5 * float(width) / math.tan(math.radians(first_fov) * 0.5)
            focal_lengths = base_focal * sampled_path["zoom_factors"]
            camera_intrinsics = np.repeat(
                np.asarray(
                    [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]],
                    dtype=np.float32,
                ),
                int(num_frames),
                axis=0,
            )
            camera_intrinsics[:, 0, 0] = focal_lengths / float(width)
            camera_intrinsics[:, 1, 1] = focal_lengths / float(height)
            camera_intrinsics = camera_intrinsics[None]
        processed = self.process(
            images=images,
            interactions=interactions,
            prompt=prompt,
            trajectory=trajectory,
        )
        inputs: dict[str, object] = {
            "image": processed["image"],
            "trajectory": processed["trajectory"],
            "camera_rotation": camera_rotation,
            "movement_distance": movement_distance,
            "center_depth": center_depth,
            "center_depth_quantile": center_depth_quantile,
            "center_depth_quantile_value": center_depth_quantile_value,
            "fps": int(fps),
            "condition_augment_sigma": condition_augment_sigma,
            "return_latent": output_type == "latent",
        }
        if rendered_warp_images is not None:
            inputs["rendered_warp_images"] = rendered_warp_images
            inputs["rendered_warp_masks"] = rendered_warp_masks
        if camera_to_world is not None:
            inputs["camera_to_world"] = camera_to_world
        if camera_intrinsics is not None:
            inputs["camera_intrinsics"] = camera_intrinsics
        actual_seed = secrets.randbits(63) if int(seed) < 0 else int(seed)
        output = self.native_pipeline(
            DiffusionRequest(
                prompt=processed["prompt"],
                negative_prompt=negative_prompt,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                sampling=SamplingConfig(
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale),
                    seed=actual_seed,
                ),
                inputs=inputs,
                metadata={"trajectory": processed["trajectory"]},
            )
        )
        artifact_path = None
        if output_path is not None and output_type != "latent":
            artifact_path = save_image_or_video_tensor(output.sample, output_path, fps=int(fps))
        result = {
            "video": output.sample,
            "generated_video": output.sample,
            "latents": output.latents,
            "artifact_path": artifact_path,
            "generated_video_path": artifact_path,
            "actions": processed["actions"],
            "mapped_trajectories": processed["mapped_trajectories"],
            "trajectory": processed["trajectory"],
            "metadata": dict(output.metadata),
            "artifacts": dict(output.artifacts),
        }
        return result if return_dict else (artifact_path or output.sample)

    def stream(
        self,
        interactions: Sequence[str | Mapping[str, Any]] | None = None,
        images: Image.Image | Any = None,
        prompt: str = "",
        reset_memory: bool = False,
        **kwargs: Any,
    ) -> Any:
        return_dict = bool(kwargs.pop("return_dict", False))
        if reset_memory:
            self.memory_module.manage(action="reset")
        if images is not None:
            self.memory_module.record(self.operator.process_perception(images), metadata={"mode": "init"})
        current_image = self.memory_module.select()
        if current_image is None:
            raise ValueError("No input image found in memory. Provide images on the first stream turn.")
        result = self(
            images=current_image,
            interactions=interactions,
            prompt=prompt,
            return_dict=True,
            **kwargs,
        )
        self.memory_module.record(result, metadata={"prompt": prompt, "interactions": interactions})
        return result if return_dict else result.get("artifact_path") or result["video"]

    def get_synthesis_model(self) -> NativeDiffusionPipeline:
        return self.native_pipeline


__all__ = ["Gen3CPipeline"]
