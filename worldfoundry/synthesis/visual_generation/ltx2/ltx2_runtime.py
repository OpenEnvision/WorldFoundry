"""Native WorldFoundry runtime for LTX-2 and LTX-2.3 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.optimizations import (
    OffloadMode,
    OffloadPolicy,
    RuntimePolicy,
)


def _existing_file(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"LTX native inference requires {label}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _gemma_checkpoint(root_value: str) -> tuple[CheckpointSpec, CheckpointSpec]:
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"gemma_root does not exist: {root}")
    shards = tuple(sorted(root.glob("model-*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"gemma_root contains no model-*.safetensors shards: {root}")
    tokenizer_root = root
    if not (tokenizer_root / "tokenizer.model").is_file():
        sibling = root.parent / "tokenizer"
        if sibling.is_dir():
            tokenizer_root = sibling
    if not (tokenizer_root / "tokenizer.model").is_file():
        raise FileNotFoundError(f"LTX tokenizer.model was not found beside gemma_root: {root}")
    return (
        CheckpointSpec(source=tuple(str(path) for path in shards)),
        CheckpointSpec(source=str(tokenizer_root)),
    )


def _offload_policy(mode: str | None, cpu_offload: bool) -> OffloadPolicy:
    normalized = str(mode or ("cpu" if cpu_offload else "none")).strip().lower()
    if normalized in {"none", "false", "0"}:
        return OffloadPolicy()
    if normalized in {"cpu", "block", "layer"}:
        return OffloadPolicy(mode=OffloadMode.BLOCK, target="cpu", pin_memory=True)
    if normalized == "disk":
        return OffloadPolicy(mode=OffloadMode.DISK, target="disk")
    raise ValueError(f"unsupported LTX offload_mode: {mode!r}")


class LTX2Video:
    """Expose native LTX recipes through the existing video runtime surface."""

    def __init__(
        self,
        model_name: str,
        generation_type: Literal["i2v", "t2v"],
        version_hint: str,
        checkpoint_path: str,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        num_inference_steps: int = 11,
        image_frame_index: int = 0,
        image_strength: float = 1.0,
        enhance_prompt: bool = False,
        negative_prompt: str = "",
        seed: int = 171198,
        guidance_scale: float = 1.0,
        cpu_offload: bool = True,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        pipeline_variant: str | None = None,
        spatial_upsampler_path: str | None = None,
        gemma_root: str | None = None,
        offload_mode: str | None = None,
        quantization: str | None = None,
        vae_tiling: dict[str, object] | None = None,
        **kwargs,
    ) -> None:
        del pipeline_variant, kwargs
        if generation_type not in {"i2v", "t2v"}:
            raise ValueError(f"unsupported LTX generation type: {generation_type!r}")
        if image_frame_index != 0:
            raise ValueError("LTX image conditioning currently targets the first frame")
        if enhance_prompt:
            raise ValueError("prompt enhancement is separate from native diffusion inference")
        if quantization and str(quantization).lower() not in {"none", "false", "0"}:
            raise NotImplementedError("LTX quantization requires a shared core quantization pass")
        if int(height) % 64 or int(width) % 64:
            raise ValueError("two-stage LTX height and width must be divisible by 64")

        self.model_name = str(model_name)
        self.generation_type = generation_type
        version = "2.3" if str(version_hint).startswith("2.3") else "2"
        self.model_id = f"ltx-{version}-{generation_type}"
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.frame_rate = int(frame_rate)
        self.num_inference_steps = int(num_inference_steps)
        self.image_strength = float(image_strength)
        self.negative_prompt = str(negative_prompt)
        self.seed = int(seed)
        self.guidance_scale = float(guidance_scale)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            dtype = getattr(torch, str(torch_dtype))
        except AttributeError as error:
            raise ValueError(f"unknown torch dtype: {torch_dtype!r}") from error

        main = _existing_file(checkpoint_path, "checkpoint_path")
        upsampler = _existing_file(spatial_upsampler_path, "spatial_upsampler_path")
        gemma, tokenizer = _gemma_checkpoint(str(gemma_root or ""))
        tiling = dict(vae_tiling or {})
        self.pipeline = NativeDiffusionPipeline.from_pretrained(
            self.model_id,
            policy=RuntimePolicy(
                device=self.device,
                dtype=dtype,
                offload=_offload_policy(offload_mode, cpu_offload),
            ),
            checkpoint_overrides={
                "model": CheckpointSpec(source=str(main)),
                "upsampler": CheckpointSpec(source=str(upsampler)),
                "gemma": gemma,
                "tokenizer": tokenizer,
            },
            component_options={
                "decoder:main": {
                    "tiled": bool(tiling.get("enabled", True)),
                    "spatial_tile_size": int(tiling.get("spatial_tile_size", 768)),
                    "spatial_overlap": int(tiling.get("spatial_overlap", 64)),
                    "temporal_tile_size": int(tiling.get("temporal_tile_size", 80)),
                    "temporal_overlap": int(tiling.get("temporal_overlap", 24)),
                }
            },
        )
        self.last_audio: torch.Tensor | None = None
        self.last_audio_sampling_rate: int | None = None

    def generate_video(self, prompt: str, image_path: str | None = None) -> torch.Tensor:
        if self.generation_type == "i2v" and image_path is None:
            raise ValueError("LTX image-to-video inference requires image_path")
        if self.generation_type == "t2v" and image_path is not None:
            raise ValueError("LTX text-to-video inference does not accept image_path")
        inputs: dict[str, object] = {
            "frame_rate": self.frame_rate,
        }
        if image_path is not None:
            inputs.update(
                {
                    "image": str(Path(image_path).expanduser().resolve()),
                    "image_strength": self.image_strength,
                }
            )
        request = DiffusionRequest(
            prompt=str(prompt or ""),
            negative_prompt=self.negative_prompt or None,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            sampling=SamplingConfig(
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                seed=self.seed,
            ),
            inputs=inputs,
        )
        output = self.pipeline(request)
        audio = output.artifacts.get("audio")
        self.last_audio = audio if isinstance(audio, torch.Tensor) else None
        sampling_rate = output.artifacts.get("audio_sampling_rate")
        self.last_audio_sampling_rate = int(sampling_rate) if sampling_rate is not None else None
        return output.sample


__all__ = ["LTX2Video"]
