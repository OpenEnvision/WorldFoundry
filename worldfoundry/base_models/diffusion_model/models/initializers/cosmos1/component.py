"""GEN3C :class:`~...contracts.EncodedLatentInitializer` for Predict1 Video2World.

Implements ``initialize_with_encoder``: the recipe-bound Tokenize1 VAE
encodes the reference image; a depth network plus
:func:`~worldfoundry.core.geometry.warp.forward_warp_indexed_frames` build
the 3D cache.  Returned :class:`~...contracts.LatentInitialization`
conditioning keys (``condition_latents``, ``condition_indicator``,
``condition_video_input_mask``, ``condition_video_pose``,
``condition_noise``) map onto :class:`~...denoisers.cosmos1.Cosmos1Gen3CDenoiser`.

``initialize`` without an encoder always raises.  Camera trajectories come
from ``request.inputs`` or :func:`named_camera_trajectory_tensors`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from worldfoundry.core.geometry.trajectory import named_camera_trajectory_tensors
from worldfoundry.core.geometry.warp import forward_warp_indexed_frames

from ....components import ComponentBuildContext
from ....contracts import DiffusionRequest, LatentEncoder, LatentInitialization
from ....loaders import NativeCheckpointResolver


def _reference_image(request: DiffusionRequest) -> object:
    value = request.inputs.get("image", request.inputs.get("images"))
    if value is None:
        raise ValueError("GEN3C requires request.inputs['image']")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ValueError("GEN3C requires a non-empty image sequence")
        return value[0]
    return value


def _image_tensor(
    value: object,
    *,
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    from worldfoundry.core import load_pil_image

    image = load_pil_image(value, first_sequence_item=False)
    array = np.asarray(image, dtype=np.float32)
    pixels = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
    pixels = functional.interpolate(pixels, (height, width), mode="bicubic", align_corners=False)
    return pixels.div_(127.5).sub_(1.0).repeat(batch_size, 1, 1, 1).to(device=device, dtype=dtype)


def _as_warp_tensor(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
    channels: int,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 5:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 6:
        raise ValueError("GEN3C rendered warps must use B,T,N,C,H,W layout")
    if tensor.shape[3] != channels:
        raise ValueError(f"GEN3C rendered warp tensor requires {channels} channels")
    return tensor


class Cosmos1Gen3CInitializer:
    """Build GEN3C's image, mask, and encoded 3D cache conditions."""

    def __init__(
        self,
        *,
        depth_model: torch.nn.Module,
        sigma_max: float = 80.0,
        augment_sigma: float = 0.001,
        offload_depth_model: bool = True,
    ) -> None:
        self.depth_model = depth_model.eval()
        self.sigma_max = float(sigma_max)
        self.augment_sigma = float(augment_sigma)
        self.offload_depth_model = bool(offload_depth_model)

    @torch.no_grad()
    def _render_camera_warps(
        self,
        first_frame: torch.Tensor,
        request: DiffusionRequest,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.depth_model.to(device)
        depth_input = functional.interpolate(
            first_frame.float().add(1.0).div(2.0),
            (720, 1280),
            mode="bilinear",
            align_corners=False,
        )
        prediction = self.depth_model.infer(depth_input[0])
        depth = prediction["depth"].float().unsqueeze(0).unsqueeze(0)
        valid = prediction["mask"].float().unsqueeze(0).unsqueeze(0)
        depth = functional.interpolate(
            depth,
            (request.height, request.width),
            mode="bilinear",
            align_corners=False,
        )
        valid = functional.interpolate(valid, (request.height, request.width), mode="nearest")
        depth = torch.where(valid > 0.5, depth, torch.zeros_like(depth))
        intrinsic = prediction["intrinsics"].float().clone()
        intrinsic[0, 0] *= request.width
        intrinsic[1, 1] *= request.height
        intrinsic[0, 2] *= request.width
        intrinsic[1, 2] *= request.height
        explicit_cameras = request.inputs.get("camera_to_world")
        explicit_intrinsics = request.inputs.get("camera_intrinsics")
        if (explicit_cameras is None) != (explicit_intrinsics is None):
            raise ValueError("GEN3C camera_to_world and camera_intrinsics must be provided together")
        if explicit_cameras is not None:
            camera_to_world = torch.as_tensor(
                explicit_cameras, device=device, dtype=torch.float32
            )
            intrinsics = torch.as_tensor(
                explicit_intrinsics, device=device, dtype=torch.float32
            )
            if camera_to_world.ndim == 3:
                camera_to_world = camera_to_world.unsqueeze(0)
            if intrinsics.ndim == 3:
                intrinsics = intrinsics.unsqueeze(0)
            if camera_to_world.shape != (request.batch_size, request.num_frames, 4, 4):
                raise ValueError(
                    "GEN3C explicit cameras must be [B,T,4,4], got "
                    f"{tuple(camera_to_world.shape)}"
                )
            if intrinsics.shape != (request.batch_size, request.num_frames, 3, 3):
                raise ValueError(
                    "GEN3C explicit intrinsics must be [B,T,3,3], got "
                    f"{tuple(intrinsics.shape)}"
                )
        else:
            initial_view = torch.eye(4, device=device, dtype=torch.float32)
            center_depth = float(request.inputs.get("center_depth", 1.0))
            if bool(request.inputs.get("center_depth_quantile", False)):
                valid_depth = depth[(valid > 0.5) & torch.isfinite(depth) & (depth > 0)]
                if valid_depth.numel():
                    quantile = min(
                        max(float(request.inputs.get("center_depth_quantile_value", 0.5)), 0.0),
                        1.0,
                    )
                    center_depth = float(torch.quantile(valid_depth, quantile).item())
            views, intrinsics = named_camera_trajectory_tensors(
                str(request.inputs.get("trajectory", "left")),
                initial_world_to_camera=initial_view,
                initial_intrinsic=intrinsic.to(device),
                num_frames=request.num_frames,
                movement_distance=float(request.inputs.get("movement_distance", 0.3)),
                camera_rotation=str(request.inputs.get("camera_rotation", "center_facing")),
                center_depth=center_depth,
            )
            camera_to_world = torch.linalg.inv(views)
        rendered = forward_warp_indexed_frames(
            source_pixels=first_frame.unsqueeze(2),
            source_indices=[0],
            source_camera_indices=[0],
            target_camera_indices=list(range(request.num_frames)),
            camera_to_world=camera_to_world,
            intrinsic=intrinsics,
            source_depths={0: depth.to(device)},
            height=request.height,
            width=request.width,
            fill_value=-1.0,
        )
        if self.offload_depth_model and device.type == "cuda":
            self.depth_model.to("cpu")
            torch.cuda.empty_cache()
        if rendered is None:
            raise RuntimeError("GEN3C camera warp rendering produced no frames")
        images, masks = rendered
        return (
            images.permute(0, 2, 1, 3, 4).unsqueeze(2).to(dtype=dtype),
            masks.permute(0, 2, 1, 3, 4).unsqueeze(2).to(dtype=dtype),
            camera_to_world,
            intrinsics,
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
        raise RuntimeError("GEN3C recipe requires its latent_encoder binding")

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
        if request.batch_size != 1:
            raise ValueError("GEN3C currently supports batch size 1")
        if request.num_frames != 121:
            raise ValueError("GEN3C native inference currently generates one 121-frame chunk")
        if request.height % 16 or request.width % 16:
            raise ValueError("GEN3C height and width must be divisible by 16")

        latent_frames = 16
        shape = (request.batch_size, 16, latent_frames, request.height // 8, request.width // 8)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype) * self.sigma_max
        first_frame = _image_tensor(
            _reference_image(request),
            height=request.height,
            width=request.width,
            batch_size=request.batch_size,
            device=device,
            dtype=dtype,
        )

        condition_video = torch.zeros(
            request.batch_size,
            3,
            request.num_frames,
            request.height,
            request.width,
            device=device,
            dtype=dtype,
        )
        condition_video[:, :, :1] = first_frame.unsqueeze(2)
        condition_latents = latent_encoder.encode(condition_video).to(device=device, dtype=dtype)
        if condition_latents.shape != noise.shape:
            raise ValueError(
                "GEN3C condition latent geometry does not match denoiser noise: "
                f"{tuple(condition_latents.shape)} vs {tuple(noise.shape)}"
            )

        warp_images_value = request.inputs.get("rendered_warp_images")
        warp_masks_value = request.inputs.get("rendered_warp_masks")
        if (warp_images_value is None) != (warp_masks_value is None):
            raise ValueError("GEN3C rendered_warp_images and rendered_warp_masks must be provided together")
        if warp_images_value is None:
            warp_images, warp_masks, camera_to_world, camera_intrinsics = self._render_camera_warps(
                first_frame,
                request,
                device=device,
                dtype=dtype,
            )
            artifacts: dict[str, object] = {
                "camera_to_world": camera_to_world,
                "camera_intrinsics": camera_intrinsics,
            }
        else:
            warp_images = _as_warp_tensor(warp_images_value, device=device, dtype=dtype, channels=3)
            warp_masks = _as_warp_tensor(warp_masks_value, device=device, dtype=dtype, channels=1)
            if warp_images.shape[:3] != warp_masks.shape[:3]:
                raise ValueError("GEN3C rendered image and mask batch/time/buffer dimensions must match")
            if warp_images.shape[1] != request.num_frames:
                raise ValueError("GEN3C rendered warps must contain exactly 121 frames")
            if warp_images.shape[2] > 2:
                raise ValueError("GEN3C supports at most two rendered 3D-cache buffers")
            if warp_images.shape[2] < 1:
                raise ValueError("GEN3C requires at least one rendered 3D-cache buffer")
            batch, frames, buffers = warp_images.shape[:3]
            flat_images = warp_images.reshape(batch * frames * buffers, 3, *warp_images.shape[-2:])
            flat_masks = warp_masks.reshape(batch * frames * buffers, 1, *warp_masks.shape[-2:])
            flat_images = functional.interpolate(
                flat_images.float(), (request.height, request.width), mode="bilinear", align_corners=False
            ).to(dtype)
            flat_masks = functional.interpolate(
                flat_masks.float(), (request.height, request.width), mode="nearest"
            ).to(dtype)
            if flat_images.max() > 1.5:
                flat_images = flat_images.div(127.5).sub(1.0)
            elif flat_images.min() >= 0:
                flat_images = flat_images.mul(2.0).sub(1.0)
            warp_images = flat_images.reshape(batch, frames, buffers, 3, request.height, request.width)
            warp_masks = flat_masks.reshape(batch, frames, buffers, 1, request.height, request.width).clamp(0, 1)
            artifacts = {
                key: request.inputs[key]
                for key in ("camera_to_world", "camera_intrinsics")
                if key in request.inputs
            }

        pose_parts: list[torch.Tensor] = []
        for index in range(warp_images.shape[2]):
            rendered_video = warp_images[:, :, index].permute(0, 2, 1, 3, 4).contiguous()
            rendered_mask = warp_masks[:, :, index].permute(0, 2, 1, 3, 4).contiguous()
            rendered_mask = rendered_mask.repeat(1, 3, 1, 1, 1).mul(2.0).sub(1.0)
            pose_parts.extend(
                (
                    latent_encoder.encode(rendered_video).to(device=device, dtype=dtype),
                    latent_encoder.encode(rendered_mask).to(device=device, dtype=dtype),
                )
            )
        while len(pose_parts) < 4:
            pose_parts.append(torch.zeros_like(pose_parts[0]))
        pose = torch.cat(pose_parts, dim=1)
        if pose.shape != (request.batch_size, 64, *shape[2:]):
            raise ValueError(f"GEN3C encoded 3D-cache geometry is invalid: {tuple(pose.shape)}")

        indicator = torch.zeros(
            request.batch_size, 1, latent_frames, 1, 1, device=device, dtype=dtype
        )
        indicator[:, :, :1] = 1.0
        input_mask = indicator.expand(-1, -1, -1, shape[-2], shape[-1])
        condition_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return LatentInitialization(
            noise,
            {
                "condition_latents": condition_latents,
                "condition_indicator": indicator,
                "condition_video_input_mask": input_mask,
                "condition_video_pose": pose,
                "condition_noise": condition_noise,
                "condition_augment_sigma": float(
                    request.inputs.get("condition_augment_sigma", self.augment_sigma)
                ),
                "padding_mask": torch.zeros(
                    request.batch_size,
                    1,
                    request.height,
                    request.width,
                    device=device,
                    dtype=dtype,
                ),
                "fps": float(request.inputs.get("fps", 24.0)),
            },
            artifacts,
        )


def build_cosmos1_gen3c_initializer(context: ComponentBuildContext) -> Cosmos1Gen3CInitializer:
    from worldfoundry.base_models.three_dimensions.depth.moge.model.v1 import MoGeModel

    depth_checkpoint = NativeCheckpointResolver().materialize(context.require_checkpoint("depth"))
    depth_model = MoGeModel.from_pretrained(depth_checkpoint.paths[0])
    return Cosmos1Gen3CInitializer(
        depth_model=depth_model,
        sigma_max=float(context.component_options.get("sigma_max", 80.0)),
        augment_sigma=float(context.component_options.get("augment_sigma", 0.001)),
        offload_depth_model=bool(context.component_options.get("offload_depth_model", True)),
    )


__all__ = ["Cosmos1Gen3CInitializer", "build_cosmos1_gen3c_initializer"]
