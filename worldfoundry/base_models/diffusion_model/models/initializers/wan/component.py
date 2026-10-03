"""Wan family :class:`~...contracts.LatentInitializer` implementations.

All roles share Wan 4n+1 geometry: ``T_lat = (num_frames-1)//4 + 1``,
default 16 channels and 8× spatial compression.  The runner calls
``initialize(DiffusionRequest, ...)`` for text-to-video; image / reference /
VACE roles also implement
:meth:`~...contracts.EncodedLatentInitializer.initialize_with_encoder`
so they reuse the recipe-bound :class:`~...contracts.LatentEncoder`.

Roles
-----
- :class:`WanTextToVideoLatentInitializer`: Gaussian noise only.
- :class:`WanReferenceLatentInitializer`: encode 1–4 reference stills and
  append them as extra time tokens.
- :class:`WanImageToVideoLatentInitializer`: first-frame VAE condition + mask
  for Wan2.1 I2V (``in_dim=36``).
- :class:`WanTextImageToVideoLatentInitializer`: Wan2.2 TI2V first-frame
  condition without CLIP.
- :class:`WanVaceLatentInitializer`: VACE control video / mask packing.

``LatentInitialization.conditioning`` keys become
``DenoiserInput.conditioning`` on every denoise step.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, LatentEncoder, LatentInitialization

# Internal semantic proof consumed by the Wan TI2V denoiser.  The initializer
# knows whether an image was supplied without inspecting a CUDA tensor; the
# denoiser can therefore retain a scalar timestep for all-noise T2V requests
# without a synchronizing ``denoise_mask.all().item()`` check.
WAN_DENOISE_MASK_IS_ALL_ONES = "_worldfoundry_wan_denoise_mask_is_all_ones"


class WanTextToVideoLatentInitializer:
    """Create the canonical Wan noise tensor without model-owned runtime state."""

    def __init__(
        self,
        *,
        channels: int = 16,
        spatial_compression: int = 8,
        temporal_compression: int = 4,
        latent_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.channels = int(channels)
        self.spatial_compression = int(spatial_compression)
        self.temporal_compression = int(temporal_compression)
        self.latent_dtype = latent_dtype
        if min(self.channels, self.spatial_compression, self.temporal_compression) <= 0:
            raise ValueError("Wan latent dimensions and compression factors must be positive")
        if not isinstance(self.latent_dtype, torch.dtype) or not self.latent_dtype.is_floating_point:
            raise TypeError("Wan latent dtype must be a floating-point torch.dtype")

    @staticmethod
    def _codec_dtype(latent_encoder: LatentEncoder) -> torch.dtype:
        dtype = getattr(latent_encoder, "dtype", torch.float32)
        return dtype if isinstance(dtype, torch.dtype) else torch.float32

    def _noise(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        extra_latent_frames: int = 0,
    ) -> torch.Tensor:
        del dtype
        if extra_latent_frames < 0:
            raise ValueError("extra_latent_frames must be non-negative")
        if request.height % self.spatial_compression or request.width % self.spatial_compression:
            raise ValueError(
                f"Wan height and width must be divisible by {self.spatial_compression}: "
                f"got {request.height}x{request.width}"
            )
        if (request.num_frames - 1) % self.temporal_compression:
            raise ValueError(
                f"Wan num_frames must satisfy (num_frames - 1) % {self.temporal_compression} == 0: "
                f"got {request.num_frames}"
            )
        latent_frames = (
            (request.num_frames - 1) // self.temporal_compression
            + 1
            + int(extra_latent_frames)
        )
        return torch.randn(
            request.batch_size,
            self.channels,
            latent_frames,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
            generator=generator,
            device=device,
            dtype=self.latent_dtype,
        )

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[B, C, T_lat, H/s, W/s]`` Gaussian noise (Wan T2V)."""
        return self._noise(
            request,
            generator=generator,
            device=device,
            dtype=dtype,
        )


class WanReferenceLatentInitializer(WanTextToVideoLatentInitializer):
    """Encode one to four references and append them as fixed Wan time tokens."""

    def __init__(self, *, max_reference_images: int = 4, **kwargs: int) -> None:
        super().__init__(**kwargs)
        self.max_reference_images = int(max_reference_images)
        if self.max_reference_images <= 0:
            raise ValueError("max_reference_images must be positive")

    @staticmethod
    def _references(request: DiffusionRequest) -> list[object]:
        value = request.inputs.get("images", request.inputs.get("image"))
        if value is None:
            raise ValueError("Wan reference-to-video inference requires request.inputs['images']")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            references = list(value)
        else:
            references = [value]
        if not references:
            raise ValueError("Wan reference-to-video inference requires at least one image")
        return references

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del request, generator, device, dtype
        raise RuntimeError("Wan reference-to-video recipe requires its latent_encoder binding")

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
        from worldfoundry.core import load_pil_image, resize_and_letterbox

        noise = self._noise(
            request,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        references = self._references(request)
        if len(references) > self.max_reference_images:
            raise ValueError(
                f"Wan reference-to-video accepts at most {self.max_reference_images} images, "
                f"got {len(references)}"
            )
        encoded_references = []
        codec_dtype = self._codec_dtype(latent_encoder)
        for reference in references:
            image = resize_and_letterbox(
                load_pil_image(reference, first_sequence_item=False),
                request.width,
                request.height,
            )
            array = np.asarray(image, dtype=np.float32)
            pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
            pixels = pixels.div_(127.5).sub_(1.0).unsqueeze(0).unsqueeze(2)
            pixels = pixels.repeat(request.batch_size, 1, 1, 1, 1).to(
                device=device,
                dtype=codec_dtype,
            )
            encoded = latent_encoder.encode(pixels).to(device=device, dtype=noise.dtype)
            if encoded.shape[2] != 1:
                encoded = encoded[:, :, :1]
            if encoded.shape[:2] != noise.shape[:2] or encoded.shape[-2:] != noise.shape[-2:]:
                raise ValueError(
                    "encoded Wan reference geometry does not match the denoiser: "
                    f"{tuple(encoded.shape)} vs {tuple(noise.shape)}"
                )
            encoded_references.append(encoded)
        while len(encoded_references) < self.max_reference_images:
            encoded_references.append(torch.zeros_like(encoded_references[0]))
        reference_latents = torch.cat(encoded_references, dim=2)
        return LatentInitialization(noise, {"reference_latents": reference_latents})


class WanImageToVideoLatentInitializer(WanTextToVideoLatentInitializer):
    """Encode Wan2.1's first-frame video condition and four-channel mask."""

    @staticmethod
    def _image(request: DiffusionRequest) -> object:
        value = request.inputs.get("images", request.inputs.get("image"))
        if value is None:
            raise ValueError("Wan image-to-video inference requires request.inputs['images']")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                raise ValueError("Wan image-to-video inference requires a non-empty image sequence")
            return value[0]
        return value

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del request, generator, device, dtype
        raise RuntimeError("Wan image-to-video recipe requires its latent_encoder binding")

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

        noise = self._noise(
            request,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        image = load_pil_image(self._image(request), first_sequence_item=False)
        array = np.asarray(image, dtype=np.float32)
        pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        pixels = pixels.div_(127.5).sub_(1.0).unsqueeze(0)
        pixels = torch.nn.functional.interpolate(
            pixels,
            size=(request.height, request.width),
            mode="bicubic",
            align_corners=False,
        )
        pixels = pixels.unsqueeze(2).repeat(request.batch_size, 1, 1, 1, 1)
        codec_dtype = self._codec_dtype(latent_encoder)
        video = torch.zeros(
            request.batch_size,
            3,
            request.num_frames,
            request.height,
            request.width,
            device=device,
            dtype=codec_dtype,
        )
        video[:, :, :1] = pixels.to(device=device, dtype=codec_dtype)
        image_latents = latent_encoder.encode(video).to(device=device, dtype=noise.dtype)
        if image_latents.shape != noise.shape:
            raise ValueError(
                "encoded Wan image condition must match generated latent geometry: "
                f"{tuple(image_latents.shape)} vs {tuple(noise.shape)}"
            )

        latent_frames = int(noise.shape[2])
        latent_height, latent_width = int(noise.shape[-2]), int(noise.shape[-1])
        mask = torch.zeros(
            request.batch_size,
            request.num_frames,
            latent_height,
            latent_width,
            device=device,
            dtype=noise.dtype,
        )
        mask[:, :1] = 1.0
        mask = torch.cat(
            (
                mask[:, :1].repeat(1, self.temporal_compression, 1, 1),
                mask[:, 1:],
            ),
            dim=1,
        )
        target_frames = latent_frames * self.temporal_compression
        if mask.shape[1] < target_frames:
            mask = torch.cat(
                (
                    mask,
                    mask.new_zeros(
                        request.batch_size,
                        target_frames - mask.shape[1],
                        latent_height,
                        latent_width,
                    ),
                ),
                dim=1,
            )
        elif mask.shape[1] > target_frames:
            mask = mask[:, :target_frames]
        mask = mask.view(
            request.batch_size,
            latent_frames,
            self.temporal_compression,
            latent_height,
            latent_width,
        ).transpose(1, 2)
        condition_latents = torch.cat((mask, image_latents), dim=1)
        return LatentInitialization(
            noise,
            {
                "condition_latents": condition_latents,
                "reference_pixels": pixels.to(device=device, dtype=noise.dtype),
            },
        )


class WanCausalFirstFrameLatentInitializer(WanTextToVideoLatentInitializer):
    """Encode one image latent for causal Wan's clean first-frame cache seed."""

    def initialize(self, request, *, generator, device, dtype):
        del request, generator, device, dtype
        raise RuntimeError("causal Wan I2V requires its latent_encoder binding")

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

        noise = self._noise(request, generator=generator, device=device, dtype=dtype)
        image = load_pil_image(WanImageToVideoLatentInitializer._image(request), first_sequence_item=False)
        pixels = torch.from_numpy(np.ascontiguousarray(np.asarray(image, dtype=np.float32)))
        pixels = pixels.permute(2, 0, 1).div_(127.5).sub_(1.0).unsqueeze(0)
        pixels = torch.nn.functional.interpolate(
            pixels, size=(request.height, request.width), mode="bicubic", align_corners=False
        ).unsqueeze(2).repeat(request.batch_size, 1, 1, 1, 1)
        first = latent_encoder.encode(
            pixels.to(device=device, dtype=self._codec_dtype(latent_encoder))
        ).to(device=device, dtype=noise.dtype)
        if first.shape != noise[:, :, :1].shape:
            raise ValueError(
                "causal Wan first-frame latent geometry must match generated noise: "
                f"{tuple(first.shape)} vs {tuple(noise[:, :, :1].shape)}"
            )
        return LatentInitialization(noise, {"first_frame_latent": first})


class WanTextImageToVideoLatentInitializer(WanTextToVideoLatentInitializer):
    """Initialize Wan2.2 TI2V with an optional frozen first-frame latent."""

    @staticmethod
    def _optional_image(request: DiffusionRequest) -> object | None:
        value = request.inputs.get("images", request.inputs.get("image"))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return value[0] if value else None
        return value

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del request, generator, device, dtype
        raise RuntimeError("Wan text-image-to-video recipe requires its latent_encoder binding")

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
        noise = self._noise(
            request,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        denoise_mask = torch.ones(
            noise.shape[0],
            1,
            noise.shape[2],
            noise.shape[3],
            noise.shape[4],
            device=device,
            dtype=noise.dtype,
        )
        image = self._optional_image(request)
        if image is None:
            return LatentInitialization(
                noise,
                {
                    "clean_latents": torch.zeros_like(noise[:, :, :1]),
                    "denoise_mask": denoise_mask,
                    WAN_DENOISE_MASK_IS_ALL_ONES: True,
                },
            )

        from worldfoundry.core import load_pil_image
        from worldfoundry.core.media.processing.image_utils import resize_and_center_crop

        array = np.array(
            load_pil_image(image, first_sequence_item=False),
            copy=True,
        )
        array = resize_and_center_crop(
            array,
            target_width=request.width,
            target_height=request.height,
        )
        pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
        pixels = pixels.div_(127.5).sub_(1.0).unsqueeze(0).unsqueeze(2)
        pixels = pixels.repeat(request.batch_size, 1, 1, 1, 1).to(
            device=device,
            dtype=self._codec_dtype(latent_encoder),
        )
        clean = latent_encoder.encode(pixels).to(device=device, dtype=noise.dtype)
        if clean.shape[0] != noise.shape[0] or clean.shape[1] != noise.shape[1]:
            raise ValueError(
                "encoded Wan2.2 image channels do not match generated latents: "
                f"{tuple(clean.shape)} vs {tuple(noise.shape)}"
            )
        if clean.shape[2] != 1 or clean.shape[-2:] != noise.shape[-2:]:
            raise ValueError(
                "encoded Wan2.2 image geometry does not match generated latents: "
                f"{tuple(clean.shape)} vs {tuple(noise.shape)}"
            )
        denoise_mask[:, :, :1] = 0.0
        latents = noise * denoise_mask + clean * (1.0 - denoise_mask)
        return LatentInitialization(
            latents,
            {
                "clean_latents": clean,
                "denoise_mask": denoise_mask,
                "reference_pixels": pixels,
                WAN_DENOISE_MASK_IS_ALL_ONES: False,
            },
        )


class WanVaceLatentInitializer(WanTextToVideoLatentInitializer):
    """Prepare VACE video, mask, and reference conditions with the shared Wan codec."""

    @staticmethod
    def _fit_frames(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        if len(frames) < num_frames:
            frames = torch.cat((frames, frames[-1:].expand(num_frames - len(frames), -1, -1, -1)))
        return frames[:num_frames]

    @staticmethod
    def _references(request: DiffusionRequest) -> list[object]:
        value = request.inputs.get("images", request.inputs.get("image"))
        if value is None:
            return []
        if isinstance(value, str) and "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
        return [value]

    @staticmethod
    def _video_pixels(
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        from worldfoundry.core.media.codecs.video import coerce_video_frames

        value = request.inputs.get("video", request.inputs.get("videos"))
        if value is None:
            return torch.zeros(
                request.batch_size,
                3,
                request.num_frames,
                request.height,
                request.width,
                device=device,
                dtype=dtype,
            )
        if request.batch_size != 1:
            raise ValueError("Wan VACE media conditioning currently requires batch size one")
        frames = torch.from_numpy(
            coerce_video_frames(value, max_frames=request.num_frames)[..., :3]
        ).permute(0, 3, 1, 2).float()
        frames = WanVaceLatentInitializer._fit_frames(frames, request.num_frames)
        frames = torch.nn.functional.interpolate(
            frames,
            size=(request.height, request.width),
            mode="bilinear",
            align_corners=False,
        )
        return frames.permute(1, 0, 2, 3).unsqueeze(0).div(127.5).sub(1.0).to(
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _mask_pixels(
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        from worldfoundry.core import load_pil_image
        from worldfoundry.core.media.codecs.video import coerce_video_frames
        from worldfoundry.core.media.types import IMAGE_EXTENSIONS

        value = request.inputs.get("vace_mask", request.inputs.get("mask"))
        if value is None:
            return torch.ones(
                request.batch_size,
                1,
                request.num_frames,
                request.height,
                request.width,
                device=device,
                dtype=dtype,
            )
        if request.batch_size != 1:
            raise ValueError("Wan VACE mask conditioning currently requires batch size one")
        is_image_path = isinstance(value, (str, Path)) and Path(value).suffix.lower() in IMAGE_EXTENSIONS
        if is_image_path or not isinstance(value, (str, Path, np.ndarray, torch.Tensor, list, tuple)):
            array = np.asarray(load_pil_image(value, first_sequence_item=False), dtype=np.uint8)
            arrays = array[None]
        else:
            arrays = coerce_video_frames(value, max_frames=request.num_frames)
        frames = torch.from_numpy(np.ascontiguousarray(arrays)).float()
        if frames.shape[-1] > 1:
            frames = frames[..., :3].mean(dim=-1, keepdim=True)
        frames = frames.permute(0, 3, 1, 2)
        frames = WanVaceLatentInitializer._fit_frames(frames, request.num_frames)
        frames = torch.nn.functional.interpolate(
            frames,
            size=(request.height, request.width),
            mode="nearest",
        )
        return frames.permute(1, 0, 2, 3).unsqueeze(0).div(255.0).clamp_(0.0, 1.0).to(
            device=device,
            dtype=dtype,
        )

    def initialize(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del request, generator, device, dtype
        raise RuntimeError("Wan VACE recipe requires its latent_encoder binding")

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
        references = self._references(request)
        declared_reference_count = int(
            request.inputs.get("vace_reference_count", len(references))
        )
        if declared_reference_count != len(references):
            raise ValueError(
                "vace_reference_count must match the number of VACE reference images: "
                f"{declared_reference_count} != {len(references)}"
            )
        noise = self._noise(
            request,
            generator=generator,
            device=device,
            dtype=dtype,
            extra_latent_frames=declared_reference_count,
        )
        codec_dtype = self._codec_dtype(latent_encoder)
        direct_context = request.inputs.get("vace_context")
        if direct_context is not None:
            if not isinstance(direct_context, torch.Tensor):
                raise TypeError("vace_context must be a tensor")
            direct_context = direct_context.to(device=device, dtype=noise.dtype)
            expected = (noise.shape[0], 96, *noise.shape[2:])
            if tuple(direct_context.shape) != expected:
                raise ValueError(
                    f"vace_context must have shape {expected}, got {tuple(direct_context.shape)}"
                )
            return LatentInitialization(
                noise,
                {
                    "vace_context": direct_context,
                    "vace_context_scale": float(request.inputs.get("vace_context_scale", 1.0)),
                },
            )

        pixels = self._video_pixels(request, device=device, dtype=codec_dtype)
        mask = self._mask_pixels(request, device=device, dtype=codec_dtype)
        inactive = latent_encoder.encode(pixels * (1.0 - mask)).to(
            device=device,
            dtype=noise.dtype,
        )
        reactive = latent_encoder.encode(pixels * mask).to(device=device, dtype=noise.dtype)
        generated_noise = noise[:, :, declared_reference_count:]
        if inactive.shape != generated_noise.shape or reactive.shape != generated_noise.shape:
            raise ValueError(
                "encoded VACE video geometry must match generated latents: "
                f"{tuple(inactive.shape)}, {tuple(reactive.shape)} vs "
                f"{tuple(generated_noise.shape)}"
            )
        video_latents = torch.cat((inactive, reactive), dim=1)

        batch, _, frames, height, width = mask.shape
        spatial = self.spatial_compression
        if height % spatial or width % spatial:
            raise ValueError("Wan VACE mask dimensions must be divisible by the codec compression")
        mask_latents = (
            mask[:, 0]
            .reshape(batch, frames, height // spatial, spatial, width // spatial, spatial)
            .permute(0, 3, 5, 1, 2, 4)
            .reshape(batch, spatial * spatial, frames, height // spatial, width // spatial)
        )
        mask_latents = torch.nn.functional.interpolate(
            mask_latents,
            size=generated_noise.shape[2:],
            mode="nearest",
        ).to(dtype=noise.dtype)

        if len(references) > generated_noise.shape[2]:
            raise ValueError(
                f"Wan VACE accepts at most {generated_noise.shape[2]} references "
                "at this frame count"
            )
        reference_latents = []
        if references:
            from worldfoundry.core import load_pil_image, resize_and_letterbox

            for reference in references:
                image = resize_and_letterbox(
                    load_pil_image(reference, first_sequence_item=False),
                    request.width,
                    request.height,
                )
                array = np.asarray(image, dtype=np.float32)
                reference_pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
                reference_pixels = reference_pixels.div_(127.5).sub_(1.0).unsqueeze(0).unsqueeze(2)
                reference_pixels = reference_pixels.repeat(request.batch_size, 1, 1, 1, 1).to(
                    device=device,
                    dtype=codec_dtype,
                )
                reference_latent = latent_encoder.encode(reference_pixels).to(
                    device=device,
                    dtype=noise.dtype,
                )
                if reference_latent.shape[2] != 1 or reference_latent.shape[-2:] != noise.shape[-2:]:
                    raise ValueError("encoded VACE reference geometry does not match generated latents")
                reference_latents.append(
                    torch.cat(
                        (reference_latent, torch.zeros_like(reference_latent)),
                        dim=1,
                    )
                )
            video_latents = torch.cat((*reference_latents, video_latents), dim=2)
            mask_latents = torch.cat(
                (
                    torch.zeros_like(mask_latents[:, :, :declared_reference_count]),
                    mask_latents,
                ),
                dim=2,
            )

        expected_context_shape = (noise.shape[0], 96, *noise.shape[2:])
        context = torch.cat((video_latents, mask_latents), dim=1)
        if tuple(context.shape) != expected_context_shape:
            raise ValueError(
                "encoded VACE context geometry must match reference-prefixed noise: "
                f"{tuple(context.shape)} vs {expected_context_shape}"
            )

        return LatentInitialization(
            noise,
            {
                "vace_context": context,
                "vace_context_scale": float(request.inputs.get("vace_context_scale", 1.0)),
            },
        )


def build_wan_t2v_latent_initializer(context: ComponentBuildContext) -> WanTextToVideoLatentInitializer:
    """Build Wan latent geometry from declarative component options."""

    return WanTextToVideoLatentInitializer(
        channels=int(context.component_options.get("channels", 16)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
        temporal_compression=int(context.component_options.get("temporal_compression", 4)),
    )


def build_wan_reference_latent_initializer(
    context: ComponentBuildContext,
) -> WanReferenceLatentInitializer:
    """Build shared Wan reference-to-video geometry from recipe options."""

    return WanReferenceLatentInitializer(
        channels=int(context.component_options.get("channels", 16)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
        temporal_compression=int(context.component_options.get("temporal_compression", 4)),
        max_reference_images=int(context.component_options.get("max_reference_images", 4)),
    )


def build_wan_i2v_latent_initializer(
    context: ComponentBuildContext,
) -> WanImageToVideoLatentInitializer:
    """Build Wan2.1 image-to-video latent conditioning from recipe options."""

    return WanImageToVideoLatentInitializer(
        channels=int(context.component_options.get("channels", 16)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
        temporal_compression=int(context.component_options.get("temporal_compression", 4)),
    )


def build_wan_causal_i2v_latent_initializer(
    context: ComponentBuildContext,
) -> WanCausalFirstFrameLatentInitializer:
    """Build the first-frame-only condition used by FastVideo CausalWan I2V."""

    return WanCausalFirstFrameLatentInitializer(
        channels=int(context.component_options.get("channels", 16)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
        temporal_compression=int(context.component_options.get("temporal_compression", 4)),
    )


def build_wan_ti2v_latent_initializer(
    context: ComponentBuildContext,
) -> WanTextImageToVideoLatentInitializer:
    """Build Wan2.2 text/image-to-video latent initialization."""

    return WanTextImageToVideoLatentInitializer(
        channels=int(context.component_options.get("channels", 48)),
        spatial_compression=int(context.component_options.get("spatial_compression", 16)),
        temporal_compression=int(context.component_options.get("temporal_compression", 4)),
    )


def build_wan_vace_latent_initializer(
    context: ComponentBuildContext,
) -> WanVaceLatentInitializer:
    """Build Wan VACE condition preparation on the shared latent contract."""

    return WanVaceLatentInitializer(
        channels=int(context.component_options.get("channels", 16)),
        spatial_compression=int(context.component_options.get("spatial_compression", 8)),
        temporal_compression=int(context.component_options.get("temporal_compression", 4)),
    )


__all__ = [
    "WanCausalFirstFrameLatentInitializer",
    "WanImageToVideoLatentInitializer",
    "WanReferenceLatentInitializer",
    "WanTextToVideoLatentInitializer",
    "WanTextImageToVideoLatentInitializer",
    "WanVaceLatentInitializer",
    "build_wan_causal_i2v_latent_initializer",
    "build_wan_i2v_latent_initializer",
    "build_wan_reference_latent_initializer",
    "build_wan_t2v_latent_initializer",
    "build_wan_ti2v_latent_initializer",
    "build_wan_vace_latent_initializer",
]
