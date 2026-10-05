"""AlayaWorld spatial-memory orchestration using canonical core geometry and DA3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from worldfoundry.core.geometry.warp import (
    Sparse3DCache,
    forward_warp_indexed_frames,
    pixel_intrinsics,
    safe_inverse,
)


@dataclass
class SpatialBank:
    pixels: list[torch.Tensor]
    camera_frame_indices: list[int]
    depths: list[torch.Tensor | None]
    world_points: dict[int, torch.Tensor] = field(default_factory=dict)


class AlayaSpatialMemory:
    """Maintain prior RGBD frames and produce target-camera warped LTX latents."""

    def __init__(
        self,
        *,
        video_encoder,
        video_decoder,
        encoder_tiling=None,
        device: torch.device,
        dtype: torch.dtype,
        height: int,
        width: int,
        temporal_stride: int = 8,
        sink_latent_frames: int = 1,
        num_context_frames: int = 10,
        retrieval_views: int = 1,
        downsample: int = 4,
        maximum_coverage: bool = True,
        retrieval_depth_threshold: float = 0.1,
        constant_depth: float = 1.0,
        include_sink: bool = False,
        require_full_context: bool = True,
        depth_backend: str = "da3",
        da3_path: str | None = None,
        da3_process_res: int = 504,
        da3_process_res_method: str = "upper_bound_resize",
        da3_align_to_input_scale: bool = True,
    ) -> None:
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.encoder_tiling = encoder_tiling
        self.device = device
        self.dtype = dtype
        self.height = int(height)
        self.width = int(width)
        self.temporal_stride = int(temporal_stride)
        self.sink_latent_frames = int(sink_latent_frames)
        self.num_context_frames = max(1, int(num_context_frames))
        self.retrieval_views = max(1, int(retrieval_views))
        self.downsample = max(1, int(downsample))
        self.maximum_coverage = bool(maximum_coverage)
        self.retrieval_depth_threshold = float(retrieval_depth_threshold)
        self.constant_depth = float(constant_depth)
        self.include_sink = bool(include_sink)
        self.require_full_context = bool(require_full_context)
        self.depth_backend = str(depth_backend).lower()
        self.da3_path = da3_path
        self.da3_process_res = int(da3_process_res)
        self.da3_process_res_method = str(da3_process_res_method)
        self.da3_align_to_input_scale = bool(da3_align_to_input_scale)
        self._da3 = None
        if self.depth_backend not in {"da3", "constant"}:
            raise ValueError(f"depth_backend must be 'da3' or 'constant', got {depth_backend!r}")

    def _load_da3(self):
        if self.depth_backend != "da3":
            return None
        if self._da3 is not None:
            return self._da3
        if not self.da3_path:
            raise ValueError("da3_path is required when depth_backend='da3'")
        path = Path(self.da3_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"DA3 checkpoint does not exist: {path}")
        from worldfoundry.base_models.three_dimensions.depth.depth_anything.depth_anything_v3.api import (
            DepthAnything3,
        )

        if path.is_file():
            self._da3 = DepthAnything3.from_pretrained(
                "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                model_name="da3nested-giant-large",
                weights_path=str(path),
            ).to(self.device).eval()
        else:
            self._da3 = DepthAnything3.from_pretrained(str(path)).to(self.device).eval()
        return self._da3

    @staticmethod
    def _as_bcfhw(video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 5:
            if video.shape[1] == 3:
                return video
            if video.shape[2] == 3:
                return video.permute(0, 2, 1, 3, 4).contiguous()
        if video.ndim == 4:
            if video.shape[0] == 3:
                return video.unsqueeze(0)
            if video.shape[1] == 3:
                return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        raise ValueError(f"unexpected video shape {tuple(video.shape)}")

    def _intrinsic_at(self, intrinsic: torch.Tensor, index: int, batch: int) -> torch.Tensor:
        value = intrinsic.to(device=self.device, dtype=torch.float32)
        if value.ndim == 4:
            value = value[:, min(max(0, index), value.shape[1] - 1)]
        elif value.ndim == 2:
            value = value.unsqueeze(0)
        value = pixel_intrinsics(value, height=self.height, width=self.width)
        return value.expand(batch, -1, -1).contiguous() if value.shape[0] == 1 and batch > 1 else value

    @staticmethod
    def _frame_uint8(frame: torch.Tensor) -> np.ndarray:
        value = frame.detach().float().cpu()
        if float(value.min()) < -0.05:
            value = value * 0.5 + 0.5
        elif float(value.max()) > 2.0:
            value = value / 255.0
        return np.clip(value.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def _depths(
        self,
        pixels: torch.Tensor,
        *,
        camera_to_world: torch.Tensor,
        intrinsic: torch.Tensor,
        camera_indices: list[int],
    ) -> list[torch.Tensor | None]:
        if self.depth_backend == "constant":
            return [None] * pixels.shape[2]
        if pixels.shape[0] != 1:
            raise ValueError("DA3 spatial memory currently supports batch size 1")
        model = self._load_da3()
        images = [self._frame_uint8(pixels[0, :, index]) for index in range(pixels.shape[2])]
        camera = camera_to_world
        if camera.ndim == 3:
            camera = camera.unsqueeze(0)
        camera = camera.detach().cpu().float()
        extrinsics = np.stack(
            [safe_inverse(camera[:, min(max(0, index), camera.shape[1] - 1)])[0].numpy() for index in camera_indices]
        )
        intrinsic_pixels = np.stack(
            [self._intrinsic_at(intrinsic, index, 1)[0].cpu().numpy() for index in camera_indices]
        )
        try:
            prediction = model.inference(
                image=images,
                extrinsics=extrinsics,
                intrinsics=intrinsic_pixels,
                align_to_input_ext_scale=self.da3_align_to_input_scale and len(images) >= 2,
                align_to_input_pose=len(images) >= 2,
                infer_gs=False,
                process_res=self.da3_process_res,
                process_res_method=self.da3_process_res_method,
                export_dir=None,
                export_format="mini_npz",
            )
        except Exception:
            prediction = model.inference(
                image=images,
                extrinsics=None,
                intrinsics=None,
                align_to_input_ext_scale=False,
                align_to_input_pose=False,
                infer_gs=False,
                process_res=self.da3_process_res,
                process_res_method=self.da3_process_res_method,
                export_dir=None,
                export_format="mini_npz",
            )
        depths = np.asarray(prediction.depth, dtype=np.float32)
        if depths.ndim == 2:
            depths = depths[None]
        if len(depths) != len(images):
            raise RuntimeError(f"DA3 returned {len(depths)} maps for {len(images)} frames")
        output: list[torch.Tensor] = []
        for value in depths:
            tensor = torch.from_numpy(value).to(device=self.device, dtype=torch.float32)[None, None]
            if tensor.shape[-2:] != (self.height, self.width):
                tensor = F.interpolate(tensor, size=(self.height, self.width), mode="bilinear", align_corners=False)
            output.append(torch.nan_to_num(tensor, nan=1e4, posinf=1e4, neginf=0.0).clamp_(0.0, 1e4))
        return output

    @torch.no_grad()
    def initialize(
        self,
        *,
        video_pixels: torch.Tensor,
        camera_to_world: torch.Tensor,
        intrinsic: torch.Tensor,
        target_latent_start: int,
    ) -> SpatialBank | None:
        # Keep the replicated image/video on CPU until the small source window
        # has been selected.  A long camera trajectory can contain thousands of
        # pixel frames; moving the whole seed clip to the GPU would waste several
        # gigabytes even though the spatial bank only consumes the last few.
        video = self._as_bcfhw(video_pixels)
        camera_frames = camera_to_world.shape[-3]
        pixel_start = int(target_latent_start) * self.temporal_stride
        source_floor = 0 if self.include_sink else self.sink_latent_frames * self.temporal_stride
        source_indices = list(range(max(source_floor, pixel_start - self.num_context_frames), pixel_start))
        source_indices = [index for index in source_indices if index < video.shape[2] and index < camera_frames]
        if not source_indices or (self.require_full_context and len(source_indices) < self.num_context_frames):
            return None
        index_tensor = torch.tensor(source_indices, device=video.device, dtype=torch.long)
        pixels = video.index_select(2, index_tensor).to(device=self.device, dtype=self.dtype).contiguous()
        depths = self._depths(
            pixels,
            camera_to_world=camera_to_world,
            intrinsic=intrinsic,
            camera_indices=source_indices,
        )
        return SpatialBank(
            pixels=[pixels[:, :, index].detach() for index in range(pixels.shape[2])],
            camera_frame_indices=source_indices,
            depths=depths,
        )

    def _select_sources(
        self,
        bank: SpatialBank,
        *,
        camera_to_world: torch.Tensor,
        intrinsic: torch.Tensor,
        target_indices: list[int],
    ) -> list[int]:
        camera = camera_to_world
        if camera.ndim == 3:
            camera = camera.unsqueeze(0)
        camera = camera.to(device=self.device, dtype=torch.float32)
        candidates = list(range(len(bank.pixels)))
        cache = Sparse3DCache(downsample=self.downsample)
        for local_index in candidates:
            frame_index = bank.camera_frame_indices[local_index]
            points = bank.world_points.get(local_index)
            if points is None:
                depth = bank.depths[local_index]
                if depth is None:
                    depth = torch.full(
                        (camera.shape[0], 1, self.height, self.width),
                        self.constant_depth,
                        device=self.device,
                    )
                points = Sparse3DCache.compute_points(
                    depth=depth,
                    world_to_camera=safe_inverse(camera[:, frame_index]),
                    intrinsic=self._intrinsic_at(intrinsic, frame_index, camera.shape[0]),
                    downsample=self.downsample,
                )
                bank.world_points[local_index] = points
            cache.add_precomputed(points=points, latent_index=local_index, frame_id=frame_index)

        if self.retrieval_views == 1:
            view_indices = [target_indices[-1]]
        else:
            offsets = torch.linspace(0, len(target_indices) - 1, self.retrieval_views)
            view_indices = [target_indices[int(round(float(offset)))] for offset in offsets]
        target_world_to_camera = torch.stack([safe_inverse(camera[:, index]) for index in view_indices], dim=1)
        target_intrinsic = torch.stack(
            [self._intrinsic_at(intrinsic, index, camera.shape[0]) for index in view_indices],
            dim=1,
        )
        retrieved = cache.retrieve(
            target_world_to_camera=target_world_to_camera,
            target_intrinsic=target_intrinsic,
            target_hw=(self.height, self.width),
            count=self.num_context_frames,
            maximum_coverage=self.maximum_coverage,
            depth_threshold=self.retrieval_depth_threshold,
        )
        selected = [local_index for local_index, _ in retrieved]
        if len(selected) < self.num_context_frames:
            seen = set(selected)
            for local_index in reversed(candidates):
                if local_index not in seen:
                    selected.append(local_index)
                    seen.add(local_index)
                if len(selected) >= self.num_context_frames:
                    break
        return selected[: self.num_context_frames]

    @torch.no_grad()
    def build_context(
        self,
        bank: SpatialBank | None,
        *,
        camera_to_world: torch.Tensor,
        intrinsic: torch.Tensor,
        target_latent_start: int,
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if bank is None or not bank.pixels:
            return None
        camera = camera_to_world
        if camera.ndim == 3:
            camera = camera.unsqueeze(0)
        pixel_start = int(target_latent_start) * self.temporal_stride
        target_pixel_count = 1 + (int(latent_frames) - 1) * self.temporal_stride
        targets = list(range(pixel_start, pixel_start + target_pixel_count))
        if targets[-1] >= camera.shape[1]:
            return None
        selected = self._select_sources(
            bank,
            camera_to_world=camera,
            intrinsic=intrinsic,
            target_indices=targets,
        )
        if not selected or (self.require_full_context and len(selected) < self.num_context_frames):
            return None
        source_video = torch.stack(bank.pixels, dim=2).to(device=self.device, dtype=self.dtype)
        depths = {index: bank.depths[index] for index in selected if bank.depths[index] is not None}
        warped = forward_warp_indexed_frames(
            source_pixels=source_video,
            source_indices=selected,
            source_camera_indices=bank.camera_frame_indices,
            target_camera_indices=targets,
            camera_to_world=camera,
            intrinsic=intrinsic,
            source_depths=depths,
            height=self.height,
            width=self.width,
            constant_depth=self.constant_depth,
            depth_threshold=min(self.retrieval_depth_threshold, 1e-3),
        )
        if warped is None:
            return None
        pixels, coverage = warped
        pixels = pixels.to(device=self.device, dtype=self.dtype)
        latent = (
            self.video_encoder.tiled_encode(pixels, self.encoder_tiling)
            if self.encoder_tiling is not None
            else self.video_encoder(pixels)
        ).contiguous()
        if latent.shape[2] != latent_frames:
            raise RuntimeError(f"spatial VAE returned {latent.shape[2]} latents, expected {latent_frames}")
        valid = F.adaptive_avg_pool3d(coverage.float(), output_size=latent.shape[2:]).gt(0.5)
        return latent, valid[:, 0].reshape(latent.shape[0], -1)

    @torch.no_grad()
    def append(
        self,
        bank: SpatialBank | None,
        *,
        latent: torch.Tensor,
        camera_to_world: torch.Tensor,
        intrinsic: torch.Tensor,
        target_latent_start: int,
    ) -> None:
        if bank is None:
            return
        pixels = self.video_decoder(latent.to(device=self.device, dtype=self.dtype)).detach()
        start = int(target_latent_start) * self.temporal_stride
        camera_indices = list(range(start, start + pixels.shape[2]))
        camera_frames = camera_to_world.shape[-3]
        keep = [index for index, frame in enumerate(camera_indices) if frame < camera_frames]
        if not keep:
            return
        pixels = pixels[:, :, keep].contiguous()
        camera_indices = [camera_indices[index] for index in keep]
        depths = self._depths(
            pixels,
            camera_to_world=camera_to_world,
            intrinsic=intrinsic,
            camera_indices=camera_indices,
        )
        for index, frame in enumerate(camera_indices):
            bank.pixels.append(pixels[:, :, index].detach())
            bank.camera_frame_indices.append(frame)
            bank.depths.append(depths[index])

    def offload_depth_model(self) -> None:
        """Keep DA3 available for later spatial updates without GPU residency."""

        if self._da3 is None:
            return
        self._da3.to("cpu")
        if hasattr(self._da3, "device"):
            self._da3.device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release(self) -> None:
        if self._da3 is not None:
            self._da3.to("meta")
            self._da3 = None


__all__ = ["AlayaSpatialMemory", "SpatialBank"]
