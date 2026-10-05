"""Native WorldFoundry runtime for LTX-Video 0.9.8 image-to-video inference."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
from worldfoundry.base_models.diffusion_model.optimizations import OffloadMode, OffloadPolicy, RuntimePolicy


def _resolve_assets(
    model_path: str,
) -> tuple[CheckpointSpec, CheckpointSpec, CheckpointSpec, CheckpointSpec]:
    path = Path(model_path).expanduser().resolve()
    if path.is_dir():
        root = path
        checkpoint = root / "ltxv-13b-0.9.8-distilled.safetensors"
    elif path.is_file():
        checkpoint = path
        root = path.parent
    else:
        raise FileNotFoundError(f"LTX-Video model_path does not exist: {path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LTX-Video checkpoint does not exist: {checkpoint}")
    upsampler = root / "ltxv-spatial-upscaler-0.9.8.safetensors"
    if not upsampler.is_file():
        raise FileNotFoundError(f"LTX-Video spatial upsampler does not exist: {upsampler}")

    text_root = root / "text_encoder"
    text_shards = tuple(sorted(text_root.glob("model-*.safetensors")))
    if not text_shards or not (text_root / "config.json").is_file():
        raise FileNotFoundError(f"LTX-Video text encoder assets are incomplete: {text_root}")
    tokenizer_root = root / "tokenizer"
    if not (tokenizer_root / "spiece.model").is_file():
        raise FileNotFoundError(f"LTX-Video tokenizer assets are incomplete: {tokenizer_root}")
    return (
        CheckpointSpec(source=str(checkpoint)),
        CheckpointSpec(source=str(upsampler)),
        CheckpointSpec(source=tuple(str(value) for value in text_shards)),
        CheckpointSpec(source=str(tokenizer_root)),
    )


def _offload(value: str | None, cpu_offload: bool) -> OffloadPolicy:
    mode = str(value or ("block" if cpu_offload else "none")).strip().lower()
    if mode in {"none", "false", "0"}:
        return OffloadPolicy()
    if mode in {"cpu", "block", "layer"}:
        return OffloadPolicy(mode=OffloadMode.BLOCK, target="cpu", pin_memory=True)
    if mode == "disk":
        return OffloadPolicy(mode=OffloadMode.DISK, target="disk")
    raise ValueError(f"unsupported LTX-Video offload_mode: {value!r}")


class LTXVideo:
    """Expose the native LTX-Video recipe through the existing runtime surface."""

    def __init__(
        self,
        model_name: str,
        generation_type: Literal["t2v", "i2v"],
        num_images_per_prompt: int,
        image_cond_noise_scale: float,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        pipeline_config: str | None = None,
        negative_prompt: str = "",
        conditioning_start_frames: int = 0,
        model_path: str = "Lightricks/LTX-Video",
        seed: int = 171198,
        num_inference_steps: int = 10,
        guidance_scale: float = 1.0,
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        cpu_offload: bool = True,
        offload_mode: str | None = None,
        vae_tiling: dict[str, object] | None = None,
        **kwargs,
    ) -> None:
        del pipeline_config, kwargs
        if generation_type != "i2v":
            raise ValueError("native LTX-Video currently supports image-to-video inference")
        if int(num_images_per_prompt) != 1:
            raise ValueError("native LTX-Video runtime currently emits one video per prompt")
        if int(conditioning_start_frames) != 0:
            raise ValueError("LTX-Video image conditioning targets the first frame")
        if not 0.0 <= float(image_cond_noise_scale) <= 1.0:
            raise ValueError("image_cond_noise_scale must be between zero and one")
        if int(height) % 32 or int(width) % 32:
            raise ValueError("LTX-Video height and width must be divisible by 32")
        if int(num_inference_steps) != 10:
            raise ValueError("the distilled LTX-Video 0.9.8 two-pass recipe requires ten steps")
        if float(guidance_scale) != 1.0:
            raise ValueError("the distilled LTX-Video 0.9.8 recipe requires guidance_scale=1")

        self.model_name = str(model_name)
        self.generation_type = generation_type
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
        self.frame_rate = int(frame_rate)
        self.image_strength = 1.0 - float(image_cond_noise_scale)
        self.negative_prompt = str(negative_prompt)
        self.seed = int(seed)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            dtype = getattr(torch, str(torch_dtype))
        except AttributeError as error:
            raise ValueError(f"unknown torch dtype: {torch_dtype!r}") from error

        model, upsampler, text_encoder, tokenizer = _resolve_assets(model_path)
        tiling = dict(vae_tiling or {})
        self.pipeline = NativeDiffusionPipeline.from_pretrained(
            "ltx-video-i2v",
            policy=RuntimePolicy(
                device=self.device,
                dtype=dtype,
                offload=_offload(offload_mode, cpu_offload),
            ),
            checkpoint_overrides={
                "model": model,
                "upsampler": upsampler,
                "text_encoder": text_encoder,
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

    def generate_video(self, prompt: str, image_path: str | None = None) -> torch.Tensor:
        if image_path is None:
            raise ValueError("LTX-Video image-to-video inference requires image_path")
        output = self.pipeline(
            DiffusionRequest(
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
                inputs={
                    "image": str(Path(image_path).expanduser().resolve()),
                    "image_strength": self.image_strength,
                    "frame_rate": self.frame_rate,
                },
            )
        )
        return output.sample


__all__ = ["LTXVideo"]
