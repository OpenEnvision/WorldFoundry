"""Sana image / video / ControlNet / world-model :class:`~...contracts.LatentInitializer` roles.

:class:`SanaNoiseInitializer` draws DC-AE-shaped noise (32-channel image
latents by default; video uses temporal compression).  ControlNet, video-to-
video, and world-model subclasses implement
:class:`~...contracts.EncodedLatentInitializer` and encode
``control_image`` / source video / first-frame observations through the
recipe-bound DC-AE.  Optional ``allow_spatial_padding`` rounds H/W up to
the codec stride for Sprint / variable-resolution recipes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from ...components import ComponentBuildContext
from ...contracts import DiffusionRequest, LatentEncoder, LatentInitialization


class SanaNoiseInitializer:
    """Create image or video noise from explicit codec geometry."""

    def __init__(
        self,
        *,
        channels: int,
        spatial_compression: int,
        temporal_compression: int = 1,
        noise_scale: float = 1.0,
        allow_spatial_padding: bool = False,
        float32_noise: bool = False,
    ) -> None:
        self.channels = int(channels)
        self.spatial_compression = int(spatial_compression)
        self.temporal_compression = int(temporal_compression)
        self.noise_scale = float(noise_scale)
        self.allow_spatial_padding = bool(allow_spatial_padding)
        self.float32_noise = bool(float32_noise)
        if min(self.channels, self.spatial_compression, self.temporal_compression) <= 0:
            raise ValueError("Sana latent geometry must be positive")

    def latent_shape(self, request: DiffusionRequest) -> tuple[int, ...]:
        needs_padding = bool(
            request.height % self.spatial_compression
            or request.width % self.spatial_compression
        )
        if needs_padding and not self.allow_spatial_padding:
            raise ValueError(
                "Sana height and width must be divisible by "
                f"{self.spatial_compression}: got {request.height}x{request.width}"
            )
        height = (request.height + self.spatial_compression - 1) // self.spatial_compression
        width = (request.width + self.spatial_compression - 1) // self.spatial_compression
        if request.num_frames == 1:
            return request.batch_size, self.channels, height, width
        if (request.num_frames - 1) % self.temporal_compression:
            raise ValueError(
                "Sana num_frames must satisfy (num_frames - 1) % "
                f"{self.temporal_compression} == 0: got {request.num_frames}"
            )
        frames = (request.num_frames - 1) // self.temporal_compression + 1
        return request.batch_size, self.channels, frames, height, width

    def _noise(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(
            self.latent_shape(request),
            generator=generator,
            device=device,
            dtype=torch.float32 if self.float32_noise else dtype,
        ) * self.noise_scale

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return image ``[B,C,H,W]`` or video ``[B,C,T,H,W]`` Gaussian noise."""
        return self._noise(
            request,
            generator=generator,
            device=device,
            dtype=dtype,
        )


class SanaControlNetInitializer(SanaNoiseInitializer):
    """Encode the HED/control image with the recipe-bound DC-AE codec."""

    def initialize(self, *args, **kwargs):
        """Rejected: ControlNet must use :meth:`initialize_with_encoder`."""
        raise RuntimeError("Sana ControlNet requires its latent_encoder binding")

    @torch.no_grad()
    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: LatentEncoder,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        from worldfoundry.core import load_pil_image

        value = request.inputs.get("control_image", request.inputs.get("image", request.inputs.get("images")))
        if isinstance(value, (tuple, list)):
            value = value[0] if value else None
        if value is None:
            raise ValueError("Sana ControlNet requires request.inputs['control_image'] or ['image']")
        image = load_pil_image(value, first_sequence_item=False)
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        pixels = functional.interpolate(
            pixels,
            size=(request.height, request.width),
            mode="bilinear",
            align_corners=False,
        ).div(127.5).sub(1.0)
        if request.batch_size > 1:
            pixels = pixels.expand(request.batch_size, -1, -1, -1)
        # DC-AE encode the control image; the denoiser reads ``control_signal``.
        control = latent_encoder.encode(pixels.to(device=device, dtype=dtype))
        noise = self._noise(request, generator=generator, device=device, dtype=dtype)
        if control.shape != noise.shape:
            raise ValueError(
                "Sana ControlNet codec returned incompatible geometry: "
                f"{tuple(control.shape)} != {tuple(noise.shape)}"
            )
        return LatentInitialization(noise, {"control_signal": control.to(dtype=dtype)})


class SanaVideoToVideoInitializer(SanaNoiseInitializer):
    """Encode a source video and expose it as explicit V2V conditioning."""

    def initialize(self, *args, **kwargs):
        """Rejected: SANA-Streaming must use :meth:`initialize_with_encoder`."""
        raise RuntimeError("Sana video-to-video requires its latent_encoder binding")

    @torch.no_grad()
    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: LatentEncoder,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        from worldfoundry.core.media.codecs.video import coerce_video_frames

        value = request.inputs.get("video", request.inputs.get("videos"))
        if value is None:
            raise ValueError("SANA-Streaming requires request.inputs['video']")
        if request.batch_size != 1:
            raise ValueError("SANA-Streaming currently requires prompt batch size 1")
        source_frames = coerce_video_frames(value)
        if source_frames.shape[0] < request.num_frames:
            raise ValueError(
                "SANA-Streaming source video has "
                f"{source_frames.shape[0]} frames, but num_frames={request.num_frames}; "
                "provide a source video at least as long as the requested output"
            )
        frames = torch.from_numpy(source_frames[: request.num_frames]).permute(0, 3, 1, 2).float()
        frames = functional.interpolate(
            frames,
            size=(request.height, request.width),
            mode="bilinear",
            align_corners=False,
        )
        pixels = frames.permute(1, 0, 2, 3).unsqueeze(0).div(127.5).sub(1.0)
        source_latents = latent_encoder.encode(pixels.to(device=device, dtype=dtype))
        noise = self._noise(request, generator=generator, device=device, dtype=dtype)
        if source_latents.shape != noise.shape:
            raise ValueError(
                "SANA-Streaming source codec returned incompatible geometry: "
                f"{tuple(source_latents.shape)} != {tuple(noise.shape)}"
            )
        return LatentInitialization(
            noise,
            {"image_vae_embeds": source_latents.to(dtype=dtype)},
            {"source_latents": source_latents.to(dtype=dtype)},
        )


class SanaWorldInitializer(SanaNoiseInitializer):
    """Encode the anchor image and build explicit camera-conditioning tensors."""

    def initialize(self, *args, **kwargs):
        """Rejected: SANA-WM must use :meth:`initialize_with_encoder`."""
        raise RuntimeError("SANA-WM requires its recipe-bound LTX latent encoder")

    @staticmethod
    def _camera_conditions(request: DiffusionRequest) -> dict[str, torch.Tensor]:
        from worldfoundry.core.geometry.conditioning import pack_spatiotemporal_camera_conditioning
        from worldfoundry.core.geometry.trajectory import rollout_sana_wm_camera_actions

        direct_camera = request.inputs.get("camera_conditions")
        direct_plucker = request.inputs.get("chunk_plucker")
        if direct_camera is not None or direct_plucker is not None:
            if direct_camera is None or direct_plucker is None:
                raise ValueError("camera_conditions and chunk_plucker must be provided together")
            conditions = {
                "camera_conditions": torch.as_tensor(direct_camera),
                "chunk_plucker": torch.as_tensor(direct_plucker),
            }
            if conditions["camera_conditions"].ndim == 2:
                conditions["camera_conditions"] = conditions["camera_conditions"].unsqueeze(0)
            if conditions["chunk_plucker"].ndim == 4:
                conditions["chunk_plucker"] = conditions["chunk_plucker"].unsqueeze(0)
            return conditions

        poses = request.inputs.get("camera_to_world", request.inputs.get("camera_poses"))
        if poses is None:
            actions = request.inputs.get("camera_actions")
            if actions is None:
                actions = request.inputs.get("interactions")
            if actions is None:
                actions = request.inputs.get("camera_action")
            if actions is None:
                actions = ()
            if isinstance(actions, str) and not any(delimiter in actions for delimiter in ("-", ",", "，")):
                actions = [actions]
            poses = rollout_sana_wm_camera_actions(actions, num_frames=request.num_frames)
        poses = torch.as_tensor(poses, dtype=torch.float32)
        if poses.ndim == 4 and poses.shape[0] == 1:
            poses = poses[0]
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            raise ValueError(f"SANA-WM camera poses must be [F,4,4], got {tuple(poses.shape)}")
        if len(poses) < request.num_frames:
            poses = torch.cat((poses, poses[-1:].expand(request.num_frames - len(poses), -1, -1)))
        poses = poses[: request.num_frames]

        intrinsics = request.inputs.get("intrinsics")
        if intrinsics is None:
            focal = 0.5 * request.width / np.tan(np.deg2rad(60.0) / 2.0)
            intrinsics = torch.tensor(
                (focal, focal, request.width / 2.0, request.height / 2.0),
                dtype=torch.float32,
            ).expand(request.num_frames, -1)
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32)
        if intrinsics.ndim >= 2 and len(intrinsics) < request.num_frames:
            intrinsics = torch.cat(
                (intrinsics, intrinsics[-1:].expand(request.num_frames - len(intrinsics), *intrinsics.shape[1:]))
            )
        if intrinsics.ndim >= 2:
            intrinsics = intrinsics[: request.num_frames]
        conditions = pack_spatiotemporal_camera_conditioning(
            poses,
            intrinsics,
            image_height=request.height,
            image_width=request.width,
            temporal_stride=8,
            spatial_stride=32,
        )
        conditions["camera_conditions"] = conditions["camera_conditions"].unsqueeze(0)
        conditions["chunk_plucker"] = conditions["chunk_plucker"].unsqueeze(0)
        return conditions

    @torch.no_grad()
    def initialize_with_encoder(
        self,
        request: DiffusionRequest,
        *,
        latent_encoder: LatentEncoder,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        from PIL import ImageOps

        from worldfoundry.core import load_pil_image

        if request.batch_size != 1:
            raise ValueError("SANA-WM currently requires prompt batch size 1")
        value = request.inputs.get("image", request.inputs.get("images"))
        if isinstance(value, (tuple, list)):
            value = value[0] if value else None
        if value is None:
            raise ValueError("SANA-WM requires request.inputs['image'] or ['images']")
        image = ImageOps.fit(
            load_pil_image(value, first_sequence_item=False).convert("RGB"),
            (request.width, request.height),
        )
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        pixels = pixels.permute(2, 0, 1)[None, :, None].div(127.5).sub(1.0)
        anchor = latent_encoder.encode(pixels.to(device=device, dtype=dtype))
        noise = self._noise(request, generator=generator, device=device, dtype=dtype)
        if anchor.shape[:2] != noise.shape[:2] or anchor.shape[2] != 1 or anchor.shape[-2:] != noise.shape[-2:]:
            raise ValueError(
                "SANA-WM anchor codec returned incompatible geometry: "
                f"{tuple(anchor.shape)} for {tuple(noise.shape)}"
            )
        noise[:, :, :1] = anchor.to(dtype=dtype)
        camera = {
            name: value.to(device=device, dtype=dtype)
            for name, value in self._camera_conditions(request).items()
        }
        camera["condition_frame_info"] = {0: 0.0}
        return LatentInitialization(noise, camera, {"anchor_latent": anchor})


def build_sana_image_initializer(context: ComponentBuildContext) -> SanaNoiseInitializer:
    return SanaNoiseInitializer(
        channels=int(context.component_options.get("channels", 32)),
        spatial_compression=int(context.component_options.get("spatial_compression", 32)),
        noise_scale=float(context.component_options.get("noise_scale", 1.0)),
        float32_noise=True,
    )


def build_sana_controlnet_initializer(context: ComponentBuildContext) -> SanaControlNetInitializer:
    return SanaControlNetInitializer(channels=32, spatial_compression=32)


def build_sana_video_initializer(context: ComponentBuildContext) -> SanaNoiseInitializer:
    return SanaNoiseInitializer(
        channels=int(context.component_options["channels"]),
        spatial_compression=int(context.component_options["spatial_compression"]),
        temporal_compression=int(context.component_options["temporal_compression"]),
        allow_spatial_padding=bool(context.component_options.get("allow_spatial_padding", False)),
    )


def build_sana_video_to_video_initializer(
    context: ComponentBuildContext,
) -> SanaVideoToVideoInitializer:
    return SanaVideoToVideoInitializer(
        channels=int(context.component_options.get("channels", 128)),
        spatial_compression=int(context.component_options.get("spatial_compression", 32)),
        temporal_compression=int(context.component_options.get("temporal_compression", 8)),
    )


def build_sana_world_initializer(context: ComponentBuildContext) -> SanaWorldInitializer:
    del context
    return SanaWorldInitializer(channels=128, spatial_compression=32, temporal_compression=8)


__all__ = [
    "SanaControlNetInitializer",
    "SanaNoiseInitializer",
    "SanaVideoToVideoInitializer",
    "SanaWorldInitializer",
    "build_sana_controlnet_initializer",
    "build_sana_image_initializer",
    "build_sana_video_initializer",
    "build_sana_video_to_video_initializer",
    "build_sana_world_initializer",
]
