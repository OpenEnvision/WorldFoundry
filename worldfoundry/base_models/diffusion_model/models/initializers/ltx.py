"""LTX multi-stage :class:`~...runners.MultiStageLatentInitializer`.

The joint-multistage runner calls ``initialize_stage`` once per resolution
pass.  Each stage builds patchified :class:`~...contracts.ModalityState`
objects for video (and optionally audio) via
:mod:`..representations.ltx`.  An optional first-frame image is encoded
with the LTX video VAE and injected into the clean latent / denoise mask.

``stage_divisors`` shrink spatial size on early stages (default ``(2, 1)``
for LTX-2 AV).  The video-only LTX-Video factory uses ``(1, 1)`` plus
``first_stage_scale`` (default 2/3) to match the official two-pass ABI.
This is not the single-tensor :class:`~...contracts.LatentInitializer`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest, ModalityState
from ...runners import MultiStageLatentInitializer
from ..autoencoders.ltx import LTXVideoEncoderModule, load_ltx_video_encoder
from ..representations.ltx.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ..representations.ltx.tools import AudioLatentTools, VideoLatentTools
from ..representations.ltx.types import AudioLatentShape, VideoLatentShape, VideoPixelShape


class LTXMultiStageLatentInitializer(MultiStageLatentInitializer):
    """Create patchified joint states and inject an optional first-frame image."""

    def __init__(
        self,
        video_encoder: LTXVideoEncoderModule,
        *,
        stage_divisors: tuple[int, ...] = (2, 1),
        first_stage_scale: float | None = None,
        include_audio: bool = True,
    ) -> None:
        self.video_encoder = video_encoder
        if not stage_divisors or any(value <= 0 for value in stage_divisors):
            raise ValueError("LTX stage divisors must be positive")
        self.stage_divisors = tuple(int(value) for value in stage_divisors)
        if first_stage_scale is not None and not 0.0 < float(first_stage_scale) <= 1.0:
            raise ValueError("LTX first-stage scale must be in (0, 1]")
        self.first_stage_scale = None if first_stage_scale is None else float(first_stage_scale)
        self.include_audio = bool(include_audio)

    @staticmethod
    def _round_down_to_vae(value: int) -> int:
        result = int(value) - int(value) % 32
        if result < 32:
            raise ValueError("LTX stage dimensions must be at least 32 pixels")
        return result

    def _stage_size(
        self,
        request: DiffusionRequest,
        *,
        stage_index: int,
        previous_latents: Mapping[str, torch.Tensor],
    ) -> tuple[int, int]:
        if stage_index > 0 and "video" in previous_latents:
            previous = previous_latents["video"]
            if previous.ndim != 5:
                raise ValueError("LTX processed video latent must use BCTHW layout")
            return int(previous.shape[-2]) * 32, int(previous.shape[-1]) * 32
        divisor = self.stage_divisors[stage_index]
        if stage_index == 0 and self.first_stage_scale is not None:
            return (
                self._round_down_to_vae(int(request.height * self.first_stage_scale)),
                self._round_down_to_vae(int(request.width * self.first_stage_scale)),
            )
        return request.height // divisor, request.width // divisor

    @staticmethod
    def _image_value(request: DiffusionRequest):
        value = request.inputs.get("image", request.inputs.get("images"))
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    @staticmethod
    def _image_tensor(
        value,
        *,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, (str, Path)):
            value = Image.open(Path(value).expanduser()).convert("RGB")
        if isinstance(value, Image.Image):
            array = np.asarray(value.convert("RGB"), dtype=np.float32).copy()
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(value, torch.Tensor):
            tensor = value.detach()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                raise ValueError("LTX image tensor must be CHW, HWC, BCHW, or BHWC")
            if tensor.shape[-1] in (1, 3, 4) and tensor.shape[1] not in (1, 3, 4):
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor[:, :3]
        else:
            raise TypeError("LTX image input must be a path, PIL image, or tensor")
        tensor = tensor.to(device=device, dtype=torch.float32)
        if float(tensor.max().item()) > 1.5:
            tensor = tensor / 255.0
        tensor = functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
        tensor = tensor.mul(2).sub(1)
        if tensor.shape[0] == 1 and batch > 1:
            tensor = tensor.expand(batch, -1, -1, -1)
        if tensor.shape[0] != batch:
            raise ValueError("LTX image batch must be one or match the prompt batch")
        return tensor.to(dtype=dtype).unsqueeze(2)

    @staticmethod
    def _add_noise(
        state: ModalityState,
        *,
        scale: float,
        generator: torch.Generator,
    ) -> ModalityState:
        noise = torch.randn(
            state.latent.shape,
            generator=generator,
            device=state.latent.device,
            dtype=state.latent.dtype,
        )
        scaled_mask = state.denoise_mask * scale
        return state.with_updates(latent=noise * scaled_mask + state.latent * (1 - scaled_mask))

    @staticmethod
    def _inject_first_frame(
        state: ModalityState,
        *,
        encoded: torch.Tensor,
        tools: VideoLatentTools,
        strength: float,
    ) -> ModalityState:
        tokens = tools.patchifier.patchify(encoded)
        if tokens.shape[0] != state.latent.shape[0] or tokens.shape[2] != state.latent.shape[2]:
            raise ValueError("encoded LTX image does not match the target latent batch/channels")
        result = state.clone()
        count = tokens.shape[1]
        result.latent[:, :count] = tokens
        result.clean_latent[:, :count] = tokens
        result.denoise_mask[:, :count] = 1.0 - strength
        return result

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
        try:
            self.stage_divisors[stage_index]
        except IndexError as error:
            raise ValueError(f"LTX initializer does not define stage {stage_index}") from error
        height, width = self._stage_size(
            request,
            stage_index=stage_index,
            previous_latents=previous_latents,
        )
        if height % 32 or width % 32:
            raise ValueError("LTX stage resolutions must be divisible by 32")
        fps = float(request.inputs.get("frame_rate", 24.0))
        pixels = VideoPixelShape(
            batch=request.batch_size,
            frames=request.num_frames,
            height=height,
            width=width,
            fps=fps,
        )
        video_shape = VideoLatentShape.from_pixel_shape(pixels)
        video_tools = VideoLatentTools(VideoLatentPatchifier(1), video_shape, fps)
        video_state = video_tools.create_initial_state(
            device,
            dtype,
            previous_latents.get("video"),
        )

        image = self._image_value(request)
        if image is not None:
            image_tensor = self._image_tensor(
                image,
                batch=request.batch_size,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
            )
            encoded = self.video_encoder(image_tensor)
            video_state = self._inject_first_frame(
                video_state,
                encoded=encoded,
                tools=video_tools,
                strength=float(request.inputs.get("image_strength", 1.0)),
            )

        states = {
            "video": self._add_noise(video_state, scale=noise_scale, generator=generator),
        }
        if self.include_audio:
            audio_shape = AudioLatentShape.from_video_pixel_shape(pixels)
            audio_tools = AudioLatentTools(AudioPatchifier(1), audio_shape)
            audio_state = audio_tools.create_initial_state(
                device,
                dtype,
                previous_latents.get("audio"),
            )
            states["audio"] = self._add_noise(
                audio_state,
                scale=noise_scale,
                generator=generator,
            )
        return states


def build_ltx_multistage_latent_initializer(
    context: ComponentBuildContext,
) -> LTXMultiStageLatentInitializer:
    """Build the LTX-2 joint audio-video two-stage initializer."""
    return LTXMultiStageLatentInitializer(load_ltx_video_encoder(context))


def build_ltx_video_latent_initializer(
    context: ComponentBuildContext,
) -> LTXMultiStageLatentInitializer:
    """Build the official two-pass, video-only LTX-Video initializer."""

    return LTXMultiStageLatentInitializer(
        load_ltx_video_encoder(context),
        stage_divisors=(1, 1),
        first_stage_scale=float(context.component_options.get("first_stage_scale", 2.0 / 3.0)),
        include_audio=False,
    )


__all__ = [
    "LTXMultiStageLatentInitializer",
    "build_ltx_multistage_latent_initializer",
    "build_ltx_video_latent_initializer",
]
