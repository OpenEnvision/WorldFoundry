"""Cosmos3 joint :class:`~...runners.MultiStageLatentInitializer`.

Defines a single stage that returns :class:`~...contracts.ModalityState`
objects for video (required) plus optional sound and action.  Visual
context from ``request.inputs[\"image\"]`` or ``video`` is encoded with the
shared Wan2.2 48-channel VAE and written into ``clean_latent`` /
``denoise_mask``.  Action tokens use :data:`ACTION_RAW_DIMS` from
:mod:`..representations.cosmos3`.  The joint runner packs these states
for :class:`~..denoisers.cosmos3.Cosmos3JointDenoiser`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from worldfoundry.core.media.codecs.video import coerce_video_frames
from worldfoundry.core.utils import load_pil_image

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest, ModalityState
from ...runners import MultiStageLatentInitializer
from ..autoencoders.cosmos3 import load_cosmos3_video_vae
from ..autoencoders.wan.model import WanVideoVAE38
from ..representations.cosmos3 import ACTION_RAW_DIMS


class Cosmos3LatentInitializer(MultiStageLatentInitializer):
    """Create Cosmos3 modality states and anchor caller-provided visual context."""

    def __init__(
        self,
        vae: WanVideoVAE38,
        *,
        spatial_compression: int = 16,
        temporal_compression: int = 4,
        video_channels: int = 48,
        sound_channels: int = 64,
        sound_fps: float = 25.0,
        action_channels: int = 64,
    ) -> None:
        self.vae = vae
        self.spatial_compression = int(spatial_compression)
        self.temporal_compression = int(temporal_compression)
        self.video_channels = int(video_channels)
        self.sound_channels = int(sound_channels)
        self.sound_fps = float(sound_fps)
        self.action_channels = int(action_channels)

    @staticmethod
    def _randn(shape, *, generator, device, dtype) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def _image_tensor(value, *, height, width, device, dtype) -> torch.Tensor:
        image = load_pil_image(value)
        array = np.asarray(image, dtype=np.float32).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device)
        tensor = functional.interpolate(tensor, (height, width), mode="bilinear", align_corners=False)
        return tensor.div(127.5).sub(1).to(dtype=dtype)

    @staticmethod
    def _video_tensor(value, *, height, width, num_frames, device, dtype) -> torch.Tensor:
        frames = torch.from_numpy(coerce_video_frames(value, max_frames=num_frames)).permute(0, 3, 1, 2).float()
        frames = functional.interpolate(frames, (height, width), mode="bilinear", align_corners=False)
        if len(frames) < num_frames:
            frames = torch.cat((frames, frames[-1:].expand(num_frames - len(frames), -1, -1, -1)))
        frames = frames[:num_frames].div(127.5).sub(1)
        return frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype)

    def _encode_video(self, video: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.vae.encode([video[0]], device).to(device=device, dtype=dtype)

    @staticmethod
    def _frame_indexes(value: object, *, default: Sequence[int]) -> tuple[int, ...]:
        if value is None:
            return tuple(default)
        if isinstance(value, int):
            return (value,)
        if not isinstance(value, Sequence):
            raise TypeError("Cosmos3 conditioned frame indexes must be an integer sequence")
        return tuple(int(index) for index in value)

    def initialize_stage(
        self,
        request: DiffusionRequest,
        *,
        stage_index: int,
        previous_latents: Mapping[str, torch.Tensor],
        noise_scale: float,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mapping[str, ModalityState]:
        """Build the single-stage video / sound / action :class:`ModalityState` map."""
        del previous_latents, noise_scale
        if stage_index != 0:
            raise ValueError("Cosmos3 defines one denoising stage")
        if request.batch_size != 1:
            raise ValueError("Cosmos3 currently supports one sample per joint sequence")
        if request.height % self.spatial_compression or request.width % self.spatial_compression:
            raise ValueError(f"Cosmos3 height and width must be divisible by {self.spatial_compression}")
        if (request.num_frames - 1) % self.temporal_compression:
            raise ValueError(f"Cosmos3 num_frames must satisfy (num_frames - 1) % {self.temporal_compression} == 0")

        latent_frames = (request.num_frames - 1) // self.temporal_compression + 1
        video_shape = (
            1,
            self.video_channels,
            latent_frames,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
        )
        clean_video = torch.zeros(video_shape, device=device, dtype=dtype)
        video_mask = torch.ones_like(clean_video)
        image = request.inputs.get("image", request.inputs.get("images"))
        video = request.inputs.get("video", request.inputs.get("videos"))
        action_mode = request.inputs.get("action_mode")
        if image is not None and video is not None:
            raise ValueError("Cosmos3 accepts either image or video conditioning, not both")

        if video is not None:
            pixels = self._video_tensor(
                video,
                height=request.height,
                width=request.width,
                num_frames=request.num_frames,
                device=device,
                dtype=dtype,
            )
            clean_video = self._encode_video(pixels, device, dtype)
            conditioned = self._frame_indexes(
                request.inputs.get("condition_frame_indexes"),
                default=range(min(2, latent_frames)),
            )
            for index in conditioned:
                if 0 <= index < latent_frames:
                    video_mask[:, :, index] = 0
        elif image is not None and request.num_frames > 1:
            first = self._image_tensor(
                image,
                height=request.height,
                width=request.width,
                device=device,
                dtype=dtype,
            )
            pixels = first.unsqueeze(2).expand(-1, -1, request.num_frames, -1, -1).contiguous()
            clean_video = self._encode_video(pixels, device, dtype)
            video_mask[:, :, 0] = 0

        if action_mode == "inverse_dynamics":
            video_mask.zero_()
        elif action_mode in {"forward_dynamics", "policy"}:
            video_mask[:, :, 0] = 0
        video_noise = self._randn(video_shape, generator=generator, device=device, dtype=dtype)
        states: dict[str, ModalityState] = {
            "video": ModalityState(
                latent=video_noise * video_mask + clean_video * (1 - video_mask),
                denoise_mask=video_mask,
                clean_latent=clean_video,
                positions=torch.arange(latent_frames, device=device),
            )
        }

        if bool(request.inputs.get("enable_sound", False)):
            fps = float(request.inputs.get("fps", request.inputs.get("frame_rate", 24.0)))
            sound_frames = max(1, int(np.ceil(request.num_frames / fps * self.sound_fps)))
            sound_shape = (self.sound_channels, sound_frames)
            sound_mask = torch.ones(sound_shape, device=device, dtype=dtype)
            states["sound"] = ModalityState(
                latent=self._randn(sound_shape, generator=generator, device=device, dtype=dtype),
                denoise_mask=sound_mask,
                clean_latent=torch.zeros(sound_shape, device=device, dtype=dtype),
                positions=torch.arange(sound_frames, device=device),
            )

        if action_mode is not None:
            if action_mode not in {"forward_dynamics", "inverse_dynamics", "policy"}:
                raise ValueError(f"unsupported Cosmos3 action_mode: {action_mode!r}")
            chunk_size = int(request.inputs.get("action_chunk_size", request.num_frames - 1))
            if chunk_size <= 0:
                raise ValueError("Cosmos3 action_chunk_size must be positive")
            clean_action = torch.zeros((chunk_size, self.action_channels), device=device, dtype=dtype)
            action_mask = torch.ones_like(clean_action)
            domain_name = request.inputs.get("action_domain_name", request.inputs.get("domain_name"))
            inferred_dim = ACTION_RAW_DIMS.get(str(domain_name), self.action_channels)
            raw_dim = int(request.inputs.get("raw_action_dim", inferred_dim))
            if not 0 < raw_dim <= self.action_channels:
                raise ValueError("Cosmos3 raw_action_dim is outside the checkpoint action width")
            action_mask[:, raw_dim:] = 0
            if action_mode == "forward_dynamics":
                raw = request.inputs.get("raw_actions")
                if raw is None:
                    raise ValueError("Cosmos3 forward_dynamics requires raw_actions")
                raw = torch.as_tensor(raw, device=device, dtype=dtype)
                if raw.ndim != 2 or raw.shape[1] != raw_dim or raw.shape[0] < 1:
                    raise ValueError("Cosmos3 raw_actions must have shape [T, raw_action_dim]")
                if raw.shape[0] < chunk_size:
                    raw = torch.cat((raw, raw[-1:].expand(chunk_size - raw.shape[0], -1)))
                clean_action[:, :raw_dim] = raw[:chunk_size]
                action_mask.zero_()
            action_noise = self._randn(clean_action.shape, generator=generator, device=device, dtype=dtype)
            states["action"] = ModalityState(
                latent=action_noise * action_mask + clean_action * (1 - action_mask),
                denoise_mask=action_mask,
                clean_latent=clean_action,
                positions=torch.arange(chunk_size, device=device),
            )
        return states


def build_cosmos3_latent_initializer(context: ComponentBuildContext) -> Cosmos3LatentInitializer:
    return Cosmos3LatentInitializer(load_cosmos3_video_vae(context))


__all__ = ["Cosmos3LatentInitializer", "build_cosmos3_latent_initializer"]
