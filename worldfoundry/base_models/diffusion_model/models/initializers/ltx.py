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

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from worldfoundry.core.media.processing.image_utils import center_crop_resize_geometry

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
        image_resize_mode: str = "stretch",
        split_checkpoint: bool = False,
        latent_channels: int = 128,
        default_image_crf: int = 18,
        reference_video: bool = False,
        reference_downscale_factor: int = 1,
        reference_temporal_scale_factor: int = 1,
    ) -> None:
        self.video_encoder = video_encoder
        if not stage_divisors or any(value <= 0 for value in stage_divisors):
            raise ValueError("LTX stage divisors must be positive")
        self.stage_divisors = tuple(int(value) for value in stage_divisors)
        if first_stage_scale is not None and not 0.0 < float(first_stage_scale) <= 1.0:
            raise ValueError("LTX first-stage scale must be in (0, 1]")
        self.first_stage_scale = None if first_stage_scale is None else float(first_stage_scale)
        self.include_audio = bool(include_audio)
        self.image_resize_mode = image_resize_mode
        self.split_checkpoint = split_checkpoint
        self.latent_channels = latent_channels
        self.default_image_crf = default_image_crf
        self.reference_video = reference_video
        self.reference_downscale_factor = reference_downscale_factor
        self.reference_temporal_scale_factor = reference_temporal_scale_factor

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
        resize_mode: str = "stretch",
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
        if resize_mode == "center_crop":
            source_height, source_width = tensor.shape[-2:]
            resized_height, resized_width, top, left = center_crop_resize_geometry(
                source_height,
                source_width,
                height,
                width,
            )
            tensor = functional.interpolate(
                tensor,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)
            tensor = tensor[:, :, top : top + height, left : left + width]
        else:
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
        video_shape = VideoLatentShape.from_pixel_shape(pixels, latent_channels=self.latent_channels)
        video_tools = VideoLatentTools(VideoLatentPatchifier(1), video_shape, fps)
        video_state = video_tools.create_initial_state(
            device,
            dtype,
            previous_latents.get("video"),
        )

        if self.split_checkpoint:
            # The causal first latent frame represents one pixel frame, just like
            # a generated keyframe slot, even for text-only generation.
            keyframes_mask = torch.zeros_like(video_state.denoise_mask)
            frame_tokens = video_tools.patchifier.get_token_count(video_shape._replace(frames=1))
            keyframes_mask[:, :frame_tokens] = 1.0
            video_state = video_state.with_updates(keyframes_mask=keyframes_mask)
            video_state = self._split_image_conditionings(
                video_state, video_tools, request, height, width, device, dtype
            )
            if stage_index == 0 and request.inputs.get("num_generated_keyframes", 0):
                from ..representations.ltx.conditioning import append_ltx_keyframe_slots

                video_state = append_ltx_keyframe_slots(
                    video_state,
                    video_tools,
                    count=int(request.inputs["num_generated_keyframes"]),
                    num_frames=request.num_frames,
                )
        image = None if self.split_checkpoint else self._image_value(request)
        if image is not None:
            image_tensor = self._image_tensor(
                image,
                batch=request.batch_size,
                height=height,
                width=width,
                device=device,
                dtype=dtype,
                resize_mode=self.image_resize_mode,
            )
            encoded = self.video_encoder(image_tensor)
            video_state = self._inject_first_frame(
                video_state,
                encoded=encoded,
                tools=video_tools,
                strength=float(request.inputs.get("image_strength", 1.0)),
            )

        if self.reference_video and stage_index == 0:
            from ..representations.ltx.conditioning import append_ltx_reference_video

            scale = self.reference_downscale_factor
            if height % scale or width % scale:
                raise ValueError("LTX stage-one dimensions must be divisible by reference_downscale_factor")
            reference = load_ltx_reference_video(
                request.inputs["video"],
                height=height // scale,
                width=width // scale,
                num_frames=request.num_frames,
                temporal_scale_factor=self.reference_temporal_scale_factor,
                device=device,
                dtype=dtype,
            )
            video_state = append_ltx_reference_video(
                video_state,
                video_tools,
                self.video_encoder(reference),
                strength=float(request.inputs.get("control_strength", 1.0)),
                downscale_factor=scale,
                temporal_scale_factor=self.reference_temporal_scale_factor,
            )

        noiser = self._add_noise
        if self.split_checkpoint or self.reference_video:
            from ..representations.ltx.conditioning import add_ltx_noise

            noiser = add_ltx_noise
        states = {"video": noiser(video_state, scale=noise_scale, generator=generator)}
        if self.include_audio:
            audio_shape = AudioLatentShape.from_video_pixel_shape(pixels)
            audio_tools = AudioLatentTools(AudioPatchifier(1), audio_shape)
            audio_state = audio_tools.create_initial_state(
                device,
                dtype,
                previous_latents.get("audio"),
            )
            states["audio"] = noiser(
                audio_state,
                scale=noise_scale,
                generator=generator,
            )
        return states

    def _split_image_conditionings(
        self,
        state: ModalityState,
        tools: VideoLatentTools,
        request: DiffusionRequest,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ModalityState:
        from worldfoundry.core.media.processing.image_conditioning import (
            load_srgb_image,
            recompress_image_h264,
            resize_center_crop_bilinear,
        )

        from ..representations.ltx.conditioning import inject_ltx_image

        images = request.inputs.get("images", ())
        for image in images:
            pixels = load_srgb_image(image["path"])
            crf = self.default_image_crf if image.get("crf") is None else int(image["crf"])
            pixels = recompress_image_h264(pixels, crf)
            pixels = torch.tensor(pixels, dtype=torch.float32, device=device)
            pixels = (resize_center_crop_bilinear(pixels, height, width) / 127.5 - 1.0).to(dtype)
            encoded = self.video_encoder(pixels)
            state = inject_ltx_image(
                state,
                tools,
                encoded,
                frame_index=int(image.get("frame_index", 0)),
                strength=float(image.get("strength", 1.0)),
            )
        return state


def build_ltx_multistage_latent_initializer(
    context: ComponentBuildContext,
) -> LTXMultiStageLatentInitializer:
    """Build the LTX-2 joint audio-video two-stage initializer."""
    split = bool(context.component_options.get("split_checkpoint", False))
    encoder = load_ltx_video_encoder(context)
    default_crf = 18
    if split:
        from safetensors import safe_open

        from ...loaders import NativeCheckpointResolver

        checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("weights"))
        with safe_open(checkpoint.paths[0], framework="pt", device="cpu") as handle:
            version = (handle.metadata() or {}).get("model_version", "")
        match = re.match(r"(\d+)[.-](\d+)", version)
        parsed = tuple(int(value) for value in match.groups()) if match else ()
        default_crf = 18 if parsed >= (2, 4) else 33
    reference_metadata = {}
    ic_lora = context.checkpoint("ic_lora")
    if ic_lora is not None:
        from safetensors import safe_open

        from ...loaders import NativeCheckpointResolver

        checkpoint = NativeCheckpointResolver().materialize(ic_lora)
        with safe_open(checkpoint.paths[0], framework="pt", device="cpu") as handle:
            reference_metadata = handle.metadata() or {}
    return LTXMultiStageLatentInitializer(
        encoder,
        reference_video=ic_lora is not None,
        reference_downscale_factor=int(reference_metadata.get("reference_downscale_factor", 1)),
        reference_temporal_scale_factor=int(reference_metadata.get("reference_temporal_scale_factor", 1)),
        split_checkpoint=split,
        latent_channels=encoder.encoder.per_channel_statistics.get_buffer("mean-of-means").numel() if split else 128,
        default_image_crf=int(context.component_options.get("default_image_crf", default_crf)),
        stage_divisors=tuple(context.component_options.get("stage_divisors", (2, 1))),
        image_resize_mode=str(context.component_options.get("image_resize_mode", "stretch")),
    )


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


def enhance_ltx_prompt(encoder: object, prompt: str, images: object) -> str:
    """Run instruct Gemma at the released enhancement seed and normalize its response."""
    if images:
        from worldfoundry.core.media.processing.image_conditioning import load_srgb_image, resize_center_crop_bilinear

        image = torch.tensor(load_srgb_image(images[0]["path"]))
        height, width = image.shape[:2]
        scale = 896 / max(height, width)
        image = resize_center_crop_bilinear(image, int(height * scale), int(width * scale))
        image = image[0, :, 0].permute(1, 2, 0).to(torch.uint8)
        prompt = encoder.enhance_i2v(prompt, image, seed=42)
    else:
        prompt = encoder.enhance_t2v(prompt, seed=42)
    prompt = prompt.translate(str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-"))
    return next((prompt[index:] for index, char in enumerate(prompt) if char.isalpha()), prompt)


def load_ltx_reference_video(
    path: str | Path,
    *,
    height: int,
    width: int,
    num_frames: int,
    temporal_scale_factor: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read capped SDR frames, center-crop in FP32, and normalize to LTX BCFHW pixels.

    Temporal reference scaling keeps frame zero, then frames 1, 1+S, 1+2S,
    matching the causal VAE grid used by released IC-LoRA adapters.
    """
    import av

    from worldfoundry.core.media.processing.image_conditioning import resize_center_crop_bilinear

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            pixels = torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.float32, device=device).unsqueeze(0)
            pixels = resize_center_crop_bilinear(pixels, height, width)
            frames.append((pixels / 127.5 - 1.0).to(dtype=dtype))
            if index + 1 == num_frames:
                break
    if not frames:
        raise ValueError("LTX IC-LoRA control video contains no decoded frames")
    video = torch.cat(frames, dim=2)
    if temporal_scale_factor > 1:
        video = video[:, :, [0, *range(1, video.shape[2], temporal_scale_factor)]]
    return video
