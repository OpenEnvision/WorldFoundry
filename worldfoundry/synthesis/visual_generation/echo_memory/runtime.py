"""Native infer-only runtime shared by independent Echo-Memory model IDs."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
from worldfoundry.base_models.diffusion_model.contracts import (
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.base_models.diffusion_model.models.denoisers.echo_memory_spec import (
    get_echo_memory_model_spec,
)
from worldfoundry.base_models.diffusion_model.models.initializers.echo_memory_actions import (
    echo_camera_trajectory_actions,
)
from worldfoundry.base_models.diffusion_model.optimizations import (
    OffloadMode,
    OffloadPolicy,
    RuntimePolicy,
)

from .memory import EchoRolloutMemory


def _select_echo_checkpoint_path(
    checkpoint_path: str | os.PathLike[str],
    checkpoint_file: str,
) -> Path:
    """Select a recipe checkpoint without dereferencing HF snapshot symlinks."""

    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    candidates = [checkpoint]
    for ancestor in (checkpoint, *checkpoint.parents):
        if ancestor.name != "Echo-Memory":
            continue
        replacement = ancestor.parent / "Echo-Team--Echo-Memory"
        candidates.append(replacement / checkpoint.relative_to(ancestor))
        break
    for candidate in candidates:
        resolved = candidate / checkpoint_file if candidate.is_dir() else candidate
        if resolved.is_file():
            return resolved
    return checkpoint / checkpoint_file if checkpoint.is_dir() else checkpoint


def _resolve_wan_backbone_path(wan_base_dir: str | os.PathLike[str]) -> Path:
    """Resolve both plain and owner-prefixed local Hugging Face layouts."""

    requested = Path(wan_base_dir).expanduser().resolve()
    candidates = [requested]
    if not requested.name.startswith("Wan-AI--"):
        candidates.append(requested.parent / f"Wan-AI--{requested.name}")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return requested


class EchoMemoryRuntime:
    """Execute one Echo recipe through the shared native diffusion pipeline."""

    def __init__(
        self,
        *,
        model_id: str,
        wan_base_dir: str,
        checkpoint_path: str,
        device: str = "cuda",
        task: str = "t2v-1.3B",
        width: int = 640,
        height: int = 352,
        frames: int = 81,
        fps: int = 15,
        num_chunks: int = 2,
        sample_steps: int = 50,
        sample_shift: float | None = None,
        sample_solver: str = "unipc",
        sample_guide_scale: float = 5.0,
        base_seed: int = 42,
        negative_prompt: str = "oversaturated colors, overexposed, static, blurry details",
        camera_trajectory: Any = "z*80",
        camera_translation_step: float = 0.00125,
        camera_rotation_step_degrees: float = 0.5625,
        offload_model: bool = True,
        t5_cpu: bool = False,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        ulysses_size: int = 1,
        ring_size: int = 1,
    ) -> None:
        self.spec = get_echo_memory_model_spec(model_id)
        if task != "t2v-1.3B":
            raise ValueError("released Echo-Memory checkpoints require Wan 2.1 T2V 1.3B")
        if int(frames) <= 0 or (int(frames) - 1) % 4:
            raise ValueError(f"frames must have form 4n+1, got {frames}")
        if int(num_chunks) <= 0:
            raise ValueError("num_chunks must be positive")
        if int(width) % 16 or int(height) % 16:
            raise ValueError("Echo width and height must be divisible by 16")
        if int(ulysses_size) != 1 or int(ring_size) != 1:
            raise ValueError("Echo-Memory USP checkpoint parity is not validated yet")
        if bool(t5_fsdp) or bool(dit_fsdp):
            raise ValueError("Echo-Memory native inference does not use training-time FSDP")
        if bool(t5_cpu):
            raise ValueError("use the shared offload policy instead of Echo-specific t5_cpu")
        if str(sample_solver).strip().lower() != "unipc":
            raise ValueError("Echo-Memory native inference currently supports sample_solver='unipc'")

        target_device = torch.device(device)
        if target_device.type != "cuda":
            raise ValueError("Echo-Memory inference requires a CUDA device")
        device_id = target_device.index
        if device_id is None:
            device_id = int(os.getenv("LOCAL_RANK", "0"))

        backbone = _resolve_wan_backbone_path(wan_base_dir)
        if not backbone.is_dir():
            raise FileNotFoundError(f"Wan 2.1 backbone directory not found: {backbone}")
        checkpoint = _select_echo_checkpoint_path(
            checkpoint_path,
            self.spec.checkpoint_file,
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Echo-Memory checkpoint not found: {checkpoint}. "
                f"Official upstream availability is "
                f"{self.spec.checkpoint_availability.value!r}; a local checkpoint may still "
                "be supplied for research models whose weights are unavailable upstream."
            )

        self.width = int(width)
        self.height = int(height)
        self.frames = int(frames)
        self.fps = int(fps)
        self.num_chunks = int(num_chunks)
        self.sample_steps = int(sample_steps)
        self.sample_shift = float(self.spec.recipe.default_sample_shift if sample_shift is None else sample_shift)
        self.sample_solver = str(sample_solver)
        self.sample_guide_scale = float(sample_guide_scale)
        self.base_seed = int(base_seed)
        self.negative_prompt = str(negative_prompt)
        self.camera_trajectory = camera_trajectory
        self.camera_translation_step = float(camera_translation_step)
        self.camera_rotation_step_degrees = float(camera_rotation_step_degrees)
        self.offload_model = bool(offload_model)
        self.device = torch.device("cuda", device_id)
        self.memory = EchoRolloutMemory(context_frames=self.spec.recipe.context_frames)
        policy = RuntimePolicy(
            device=self.device,
            dtype=torch.bfloat16,
            offload=OffloadPolicy(
                mode=(OffloadMode.BLOCK if self.offload_model else OffloadMode.NONE),
                target="cpu",
            ),
        )
        self.pipeline = NativeDiffusionPipeline.from_pretrained(
            self.spec.model_id,
            policy=policy,
            checkpoint_overrides={
                "dit": str(checkpoint),
                "text-encoder": str(backbone),
                "tokenizer": str(backbone),
                "vae": str(backbone),
            },
        )

    def _target_actions(self) -> torch.Tensor:
        actions = echo_camera_trajectory_actions(
            self.camera_trajectory,
            frame_count=self.frames,
            temporal_stride=self.spec.recipe.action_temporal_stride,
            translation_step=self.camera_translation_step,
            rotation_step_degrees=self.camera_rotation_step_degrees,
        )
        return torch.as_tensor(actions, dtype=torch.float32, device=self.device)

    def _prepare_frame(self, frame: Image.Image) -> Image.Image:
        return frame.convert("RGB").resize((self.width, self.height), Image.Resampling.LANCZOS)

    def _encode_context(self, frames: list[Image.Image]) -> torch.Tensor:
        decoder = self.pipeline.components.decoder
        vae = getattr(decoder, "vae", None)
        if vae is None or not callable(getattr(vae, "encode", None)):
            raise TypeError("Echo-Memory requires a Wan decoder with a latent encoder")
        encoded: list[torch.Tensor] = []
        for frame in frames:
            array = np.asarray(self._prepare_frame(frame), dtype=np.float32)
            tensor = torch.from_numpy(array).permute(2, 0, 1).to(self.device)
            tensor = tensor.div(127.5).sub(1.0).unsqueeze(1)
            encoded.append(
                vae.encode(
                    [tensor],
                    self.device,
                    tiled=bool(getattr(decoder, "tiled", False)),
                    tile_size=getattr(decoder, "tile_size", (34, 34)),
                    tile_stride=getattr(decoder, "tile_stride", (18, 16)),
                )[0]
            )
        return torch.cat(encoded, dim=1)

    @staticmethod
    def _identity_rt12(*, device, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _video_to_pil(video: torch.Tensor) -> list[Image.Image]:
        values = video.detach().float().cpu().clamp(-1, 1)
        values = ((values + 1.0) * 127.5).round().to(torch.uint8)
        values = values.permute(1, 2, 3, 0).numpy()
        return [Image.fromarray(frame, mode="RGB") for frame in values]

    def generate_video(self, prompt: str, image_path: str | None = None) -> torch.Tensor:
        """Generate one or more chunks while retaining only native memory state."""

        if image_path is None:
            raise ValueError(f"{self.spec.model_id} requires an initial image")
        with Image.open(image_path) as image:
            initial = self._prepare_frame(image)
        self.memory.reset_records()
        target_actions = self._target_actions()
        chunks: list[torch.Tensor] = []
        context_frames = [initial]

        for chunk_index in range(self.num_chunks):
            if chunk_index:
                context_frames = self.memory.select()
                if not context_frames:
                    raise RuntimeError("Echo rollout memory did not retain the previous chunk")
            context_latents = self._encode_context(context_frames)
            memory_actions = self._identity_rt12(device=self.device).view(1, 12).expand(context_latents.shape[1], 12)
            actions = torch.cat([target_actions, memory_actions], dim=0).unsqueeze(0)
            request = DiffusionRequest(
                prompt=str(prompt or ""),
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.frames,
                sampling=SamplingConfig(
                    num_inference_steps=self.sample_steps,
                    guidance_scale=self.sample_guide_scale,
                    seed=self.base_seed + chunk_index,
                    scheduler_options={"shift": self.sample_shift},
                ),
                inputs={
                    "frozen_context_latents": context_latents.unsqueeze(0),
                    "actions": actions,
                    "num_context_frames": int(context_latents.shape[1]),
                },
                metadata={
                    "chunk_index": chunk_index,
                    "recipe_id": self.spec.recipe_id,
                },
            )
            video = self.pipeline(request).sample[0]
            frames = self._video_to_pil(video)
            self.memory.record(
                frames,
                metadata={
                    "chunk_index": chunk_index,
                    "model_id": self.spec.model_id,
                    "recipe_id": self.spec.recipe_id,
                },
            )
            chunk = video.detach().float().cpu().permute(1, 0, 2, 3)
            if chunk_index and len(chunk) > 1:
                chunk = chunk[1:]
            chunks.append(((chunk + 1.0) / 2.0).clamp(0, 1))
        return torch.cat(chunks, dim=0)

    def close(self) -> None:
        self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["EchoMemoryRuntime"]
