"""HunyuanVideo T2V / I2V :class:`~...contracts.LatentInitializer`.

T2V (``image_to_video=False``) returns Gaussian noise, optionally with a
zero ``condition_latents`` concat channel for graphs that always expect
it.  I2V requires the recipe-bound VAE: the first frame is encoded and
either concatenated (``concat_condition``) or written into latent frame 0
(``freeze_first_frame``).  Keys land in
``LatentInitialization.conditioning`` for
:class:`~...denoisers.hunyuan_video` adapters.
"""

from __future__ import annotations

import numpy as np
import torch

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, LatentEncoder, LatentInitialization


def _reference_image(request: DiffusionRequest) -> object:
    for key in ("image", "images", "reference_image", "first_frame"):
        value = request.inputs.get(key)
        if value is not None:
            return value
    raise ValueError("HunyuanVideo I2V requires request.inputs['image']")


def _normalized_image_tensor(
    value: object,
    *,
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    from worldfoundry.core import load_pil_image
    from worldfoundry.core.media.processing.image_utils import resize_and_center_crop

    image = load_pil_image(value)
    array = resize_and_center_crop(np.asarray(image), target_width=width, target_height=height)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
    tensor = tensor.div_(127.5).sub_(1.0).unsqueeze(0).unsqueeze(2)
    return tensor.repeat(batch_size, 1, 1, 1, 1).to(device=device, dtype=dtype)


class HunyuanVideoLatentInitializer:
    """Create original or 1.5 latent trajectories through one native contract."""

    def __init__(
        self,
        *,
        channels: int,
        spatial_compression: int,
        temporal_compression: int,
        image_to_video: bool = False,
        concat_condition: bool = False,
        freeze_first_frame: bool = False,
        stabilize_i2v: bool = False,
    ) -> None:
        self.channels = int(channels)
        self.spatial_compression = int(spatial_compression)
        self.temporal_compression = int(temporal_compression)
        self.image_to_video = bool(image_to_video)
        self.concat_condition = bool(concat_condition)
        self.freeze_first_frame = bool(freeze_first_frame)
        self.stabilize_i2v = bool(stabilize_i2v)

    def _noise(
        self,
        request: DiffusionRequest,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if request.height % self.spatial_compression or request.width % self.spatial_compression:
            raise ValueError(
                "HunyuanVideo height and width must be divisible by "
                f"{self.spatial_compression}: got {request.height}x{request.width}"
            )
        if (request.num_frames - 1) % self.temporal_compression:
            raise ValueError(
                "HunyuanVideo num_frames must satisfy "
                f"(num_frames - 1) % {self.temporal_compression} == 0: got {request.num_frames}"
            )
        latent_frames = (request.num_frames - 1) // self.temporal_compression + 1
        return torch.randn(
            request.batch_size,
            self.channels,
            latent_frames,
            request.height // self.spatial_compression,
            request.width // self.spatial_compression,
            generator=generator,
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
    ) -> torch.Tensor | LatentInitialization:
        """T2V noise, or raise when I2V must use :meth:`initialize_with_encoder`."""
        if self.image_to_video:
            raise RuntimeError("HunyuanVideo I2V recipe requires its latent_encoder binding")
        noise = self._noise(request, generator=generator, device=device, dtype=dtype)
        if not self.concat_condition:
            return noise
        condition = torch.zeros_like(noise)
        mask = torch.zeros(
            noise.shape[0], 1, noise.shape[2], noise.shape[3], noise.shape[4],
            device=device, dtype=dtype,
        )
        return LatentInitialization(noise, {"condition_latents": torch.cat((condition, mask), dim=1)})

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
        """Encode the reference still and pack I2V condition tensors."""
        noise = self._noise(request, generator=generator, device=device, dtype=dtype)
        pixels = _normalized_image_tensor(
            _reference_image(request),
            height=request.height,
            width=request.width,
            batch_size=request.batch_size,
            device=device,
            dtype=dtype,
        )
        image_latents = latent_encoder.encode(pixels).to(device=device, dtype=dtype)
        if image_latents.shape[2] != 1:
            image_latents = image_latents[:, :, :1]
        if image_latents.shape[1] != self.channels or image_latents.shape[-2:] != noise.shape[-2:]:
            raise ValueError(
                "encoded HunyuanVideo image latent geometry does not match the denoiser: "
                f"{tuple(image_latents.shape)} vs {tuple(noise.shape)}"
            )

        if self.stabilize_i2v:
            # The released I2V stable recipe (flow shift 7) adds a small copy
            # of the reference latent to every initial noise frame before
            # token replacement freezes the first frame during denoising.
            stability_t = torch.tensor([0.999], device=device, dtype=torch.float32)
            noise = (
                noise * stability_t
                + image_latents.expand_as(noise) * (1.0 - stability_t)
            ).to(dtype=dtype)

        if self.concat_condition:
            condition = torch.zeros_like(noise)
            condition[:, :, :1] = image_latents
            mask = torch.zeros(
                noise.shape[0], 1, noise.shape[2], noise.shape[3], noise.shape[4],
                device=device, dtype=dtype,
            )
            mask[:, :, :1] = 1.0
            return LatentInitialization(
                noise,
                {"condition_latents": torch.cat((condition, mask), dim=1), "reference_pixels": pixels},
            )

        if self.freeze_first_frame:
            noise = noise.clone()
            noise[:, :, :1] = image_latents
        return LatentInitialization(noise, {"first_frame_latents": image_latents, "reference_pixels": pixels})


def build_hunyuan_video_latent_initializer(context: ComponentBuildContext) -> HunyuanVideoLatentInitializer:
    """Build from recipe ``latent_channels`` / compression and I2V flags."""
    options = context.component_options
    return HunyuanVideoLatentInitializer(
        channels=int(options.get("channels", context.recipe_options.get("latent_channels", 16))),
        spatial_compression=int(
            options.get("spatial_compression", context.recipe_options.get("spatial_compression", 8))
        ),
        temporal_compression=int(
            options.get("temporal_compression", context.recipe_options.get("temporal_compression", 4))
        ),
        image_to_video=bool(options.get("image_to_video", False)),
        concat_condition=bool(options.get("concat_condition", False)),
        freeze_first_frame=bool(options.get("freeze_first_frame", False)),
        stabilize_i2v=bool(options.get("stabilize_i2v", False)),
    )


__all__ = ["HunyuanVideoLatentInitializer", "build_hunyuan_video_latent_initializer"]
