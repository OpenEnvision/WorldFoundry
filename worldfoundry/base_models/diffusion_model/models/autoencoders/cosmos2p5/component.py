"""Cosmos Predict 2.5 Wan-VAE :class:`~...contracts.LatentEncoder` / decoder.

One :class:`Cosmos25VideoCodec` instance serves encoded initialization
and final decode.  Encode uses
:func:`~...initializers.video_conditioning.prepare_video_conditioning_pixels`
or Canny edge frames for Transfer-2.5 control.  The inner network is
:class:`~..wan.model.WanVideoVAE` with the official 2.1 key remap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as functional

from worldfoundry.core.media.codecs.video import coerce_video_frames

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, LatentInitialization
from ....loaders import ModuleLoadSpec, NativeModuleLoader
from ...initializers.video_conditioning import prepare_video_conditioning_pixels
from ..wan.component import convert_wan21_vae_state_dict
from ..wan.model import CausalConv3d, RMS_norm, Upsample, WanVideoVAE


def _extract_canny_edges(
    frames,
    *,
    low_threshold: int = 100,
    high_threshold: int = 200,
    target_size: tuple[int, int] | None = None,
):
    """Convert uint8 RGB control frames to the official edge-control format."""

    import cv2
    import numpy as np

    low = int(low_threshold)
    high = int(high_threshold)
    # These are gradient thresholds, not pixel intensities; upstream presets
    # include 200/300 and 300/400.
    if not 0 <= low < high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    edge_frames = []
    for frame in frames:
        rgb = np.asarray(frame)[..., :3]
        if target_size is not None:
            rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
        # Match upstream: resize RGB first, then extract multichannel edges.
        edges = cv2.Canny(rgb, low, high)
        edge_frames.append(np.repeat(edges[..., None], 3, axis=-1))
    return np.stack(edge_frames, axis=0)


class Cosmos25VideoCodec:
    """One VAE instance serving both latent initialization and final decoding."""

    def __init__(
        self,
        vae: WanVideoVAE,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
        minimum_frames: int = 1,
        zero_pad_image: bool = True,
        conditional_frame_timestep: float = -1.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> None:
        self.vae = vae
        self.device = device
        self.dtype = dtype
        self.minimum_frames = int(minimum_frames)
        self.zero_pad_image = bool(zero_pad_image)
        self.conditional_frame_timestep = float(conditional_frame_timestep)
        if self.minimum_frames < 1 or (self.minimum_frames - 1) % 4:
            raise ValueError("Cosmos minimum_frames must be positive and satisfy 4k + 1")
        self.tiled = bool(tiled)
        self.tile_size = tile_size
        self.tile_stride = tile_stride

    @staticmethod
    def _resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return functional.interpolate(frames.float(), (height, width), mode="bilinear", align_corners=False)

    def _control_pixels(
        self,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        control = None
        for key in ("control_video", "control", "controls"):
            if request.inputs.get(key) is not None:
                control = request.inputs[key]
                break
        control_is_mapping = isinstance(control, Mapping)
        variant = str(request.inputs.get("controlnet_variant", "edge")).lower()
        if control_is_mapping:
            control = control.get(variant, next(iter(control.values()), None))
        if control is None:
            return None
        if request.batch_size != 1:
            raise ValueError("Cosmos Transfer2.5 control video currently requires batch size 1")
        frame_array = coerce_video_frames(control)
        if not len(frame_array):
            raise ValueError("Cosmos Transfer2.5 control video cannot be empty")
        control_is_preprocessed = bool(
            request.inputs.get("control_is_preprocessed", control_is_mapping)
        )
        if variant == "edge" and not control_is_preprocessed:
            frame_array = _extract_canny_edges(
                frame_array,
                low_threshold=int(request.inputs.get("canny_low_threshold", 100)),
                high_threshold=int(request.inputs.get("canny_high_threshold", 200)),
                target_size=(request.width, request.height),
            )
        frames = torch.from_numpy(frame_array).permute(0, 3, 1, 2)
        frames = self._resize(frames, request.height, request.width)
        if len(frames) < request.num_frames:
            frames = torch.cat((frames, frames[-1:].expand(request.num_frames - len(frames), -1, -1, -1)))
        frames = frames[: request.num_frames]
        return frames.permute(1, 0, 2, 3).unsqueeze(0).div(127.5).sub(1).to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Expose the shared Wan encoder through the canonical latent role."""

        if images.ndim == 4:
            images = images.unsqueeze(2)
        if images.ndim != 5:
            raise ValueError("Cosmos Predict2 Wan encoder expects BCHW or BCTHW pixels")
        return self.vae.encode(
            [images[index] for index in range(len(images))],
            self.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LatentInitialization:
        if request.height % 16 or request.width % 16:
            raise ValueError("Cosmos2.5 height and width must be divisible by 16")
        if (request.num_frames - 1) % 4:
            raise ValueError("Cosmos2.5 num_frames must satisfy (num_frames - 1) % 4 == 0")
        # Transfer weights require their full temporal context for stable
        # generation. Decode still receives the original requested length.
        request = replace(request, num_frames=max(self.minimum_frames, request.num_frames))
        latent_frames = (request.num_frames - 1) // 4 + 1
        shape = (request.batch_size, 16, latent_frames, request.height // 8, request.width // 8)
        # The runner seeds this generator from request.sampling.seed. Use its
        # initial seed so direct initializer callers can still override it,
        # while matching NVIDIA's NumPy stream instead of torch.randn.
        noise_seed = int(generator.initial_seed()) & 0xFFFFFFFF
        noise = torch.from_numpy(np.random.RandomState(noise_seed).standard_normal(shape).astype(np.float32)).to(device)
        pixels, conditioned_frames = prepare_video_conditioning_pixels(
            request,
            device=device,
            dtype=dtype,
            temporal_compression=4,
            owner="Cosmos2.5",
            zero_pad_image=self.zero_pad_image,
        )
        if pixels is None:
            condition_latents = torch.zeros_like(noise)
            condition_count = 0
        else:
            condition_latents = self.vae.encode(
                [pixels[index] for index in range(len(pixels))],
                self.device,
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            ).to(device=device, dtype=dtype)
            condition_count = (conditioned_frames - 1) // 4 + 1
        indicator = torch.zeros((request.batch_size, 1, latent_frames, 1, 1), device=device, dtype=dtype)
        indicator[:, :, :condition_count] = 1
        condition_mask = indicator.expand(-1, -1, -1, shape[-2], shape[-1])
        conditioning: dict[str, object] = {
            "condition_latents": condition_latents,
            "condition_mask": condition_mask,
            "condition_indicator": indicator,
            "initial_noise": noise.clone(),
            "padding_mask": torch.zeros(
                (request.batch_size, 1, request.height, request.width),
                device=device,
                dtype=dtype,
            ),
            "conditional_frame_timestep": float(
                request.inputs.get("conditional_frame_timestep", self.conditional_frame_timestep)
            ),
        }
        control_pixels = self._control_pixels(request, device=device, dtype=dtype)
        if control_pixels is not None:
            conditioning["latent_control_input"] = self.vae.encode(
                [control_pixels[index] for index in range(len(control_pixels))],
                self.device,
                tiled=self.tiled,
                tile_size=self.tile_size,
                tile_stride=self.tile_stride,
            ).to(device=device, dtype=dtype)
            conditioning["control_context_scale"] = float(request.inputs.get("control_context_scale", 1.0))
        return LatentInitialization(latents=noise, conditioning=conditioning)

    def decode(self, latents: torch.Tensor, request: DiffusionRequest) -> torch.Tensor:
        if bool(request.inputs.get("return_latent", False)):
            return latents
        # Sampling keeps its state in FP32; cast only at the VAE boundary.
        dtype = self.dtype or next(self.vae.parameters(), latents).dtype
        video = self.vae.decode(
            latents.to(dtype=dtype),
            self.device,
            tiled=self.tiled,
            tile_size=self.tile_size,
            tile_stride=self.tile_stride,
        )
        return video[:, :, : request.num_frames]


def build_cosmos25_video_codec(context: ComponentBuildContext) -> Cosmos25VideoCodec:
    from worldfoundry.core.vram import AutoWrappedLinear, AutoWrappedModule

    vae = NativeModuleLoader().load(
        ModuleLoadSpec(
            module_class=WanVideoVAE,
            state_dict_converter=convert_wan21_vae_state_dict,
            vram_module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
        ),
        context.require_checkpoint("weights"),
        context.policy,
    )
    if not isinstance(vae, WanVideoVAE):
        raise TypeError(f"expected WanVideoVAE, got {type(vae).__name__}")
    tile_size = tuple(context.component_options.get("tile_size", (34, 34)))
    tile_stride = tuple(context.component_options.get("tile_stride", (18, 16)))
    if len(tile_size) != 2 or len(tile_stride) != 2:
        raise ValueError("Cosmos2.5 VAE tile sizes must contain two integers")
    return Cosmos25VideoCodec(
        vae,
        device=context.policy.device,
        dtype=context.policy.dtype,
        minimum_frames=int(context.component_options.get("minimum_frames", 1)),
        zero_pad_image=bool(context.component_options.get("zero_pad_image", True)),
        conditional_frame_timestep=float(context.component_options.get("conditional_frame_timestep", -1.0)),
        tiled=bool(context.component_options.get("tiled", False)),
        tile_size=(int(tile_size[0]), int(tile_size[1])),
        tile_stride=(int(tile_stride[0]), int(tile_stride[1])),
    )


__all__ = ["Cosmos25VideoCodec", "build_cosmos25_video_codec"]
