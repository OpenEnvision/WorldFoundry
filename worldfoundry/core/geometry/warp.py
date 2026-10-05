"""Reusable camera-space retrieval and differentiable-free RGBD forward warping.

Memory-conditioned world models retrieve previously generated RGBD
frames that cover a new viewpoint, then optionally warp color into
that view. This module is the differentiable-free geometry for that
path (no autograd through the warp):

- :func:`unproject_depth` / :func:`pixel_intrinsics` /
  :func:`safe_inverse` — OpenCV unprojection. Inverse runs on CPU to
  avoid cuSOLVER spikes on tiny 4x4 batches.
- :class:`Sparse3DCache` — FIFO store of world points; ranks cached
  frames by how many points fall into a target frustum.
- :func:`forward_warp_rgbd` (and helpers below it) — splat source
  RGB(+depth) into a target camera with z-buffering.

Used by mosaic / retrieval adapters; not a renderer and not a
training loss.
"""

from __future__ import annotations

import torch


def safe_inverse(matrix: torch.Tensor) -> torch.Tensor:
    """Invert small camera matrices on CPU to avoid cuSOLVER allocation spikes."""

    return torch.linalg.inv(matrix.float().cpu()).to(device=matrix.device, dtype=matrix.dtype)


def pixel_intrinsics(intrinsic: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Convert normalized camera intrinsics to pixels, preserving pixel inputs."""

    result = intrinsic.clone().to(dtype=torch.float32)
    if result.ndim == 2:
        result = result.unsqueeze(0)
    if float(result[..., 0, 2].abs().max()) <= 1.5 and float(result[..., 1, 2].abs().max()) <= 1.5:
        result[..., 0, 0] *= float(width)
        result[..., 1, 1] *= float(height)
        result[..., 0, 2] *= float(width)
        result[..., 1, 2] *= float(height)
    return result


def unproject_depth(
    depth: torch.Tensor,
    *,
    world_to_camera: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Unproject ``[B,1,H,W]`` depth into ``[B,H,W,3]`` world points."""

    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [B,1,H,W], got {tuple(depth.shape)}")
    batch, _, height, width = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth[:, 0]
    intrinsic = intrinsic.to(device=depth.device, dtype=depth.dtype)
    fx = intrinsic[:, 0, 0].view(batch, 1, 1)
    fy = intrinsic[:, 1, 1].view(batch, 1, 1)
    cx = intrinsic[:, 0, 2].view(batch, 1, 1)
    cy = intrinsic[:, 1, 2].view(batch, 1, 1)
    x = (xs.view(1, height, width) - cx) / fx.clamp(min=1e-6) * z
    y = (ys.view(1, height, width) - cy) / fy.clamp(min=1e-6) * z
    homogeneous = torch.stack((x, y, z, torch.ones_like(z)), dim=-1)
    camera_to_world = safe_inverse(world_to_camera.to(device=depth.device, dtype=depth.dtype))
    return torch.matmul(camera_to_world[:, None, None], homogeneous.unsqueeze(-1))[..., :3, 0]


class Sparse3DCache:
    """Rank candidate RGBD frames by visible coverage in target camera views.

    Args:
        downsample: Spatial downsampling factor applied to cached world points.
        max_entries: Optional capacity bound. ``None`` (default) keeps the
            historical unbounded behavior; when set, the oldest cached frames
            are evicted FIFO once the limit is exceeded, keeping memory and
            retrieval cost bounded for long streaming sessions (CC-36).
    """

    def __init__(
        self,
        *,
        downsample: int = 4,
        max_entries: int | None = None,
        projection_batch_size: int = 8,
    ) -> None:
        """Store capacity and projection-batch limits; ``downsample`` is at least 1."""
        self.downsample = max(1, int(downsample))
        if max_entries is not None and int(max_entries) <= 0:
            raise ValueError(f"max_entries must be positive or None, got {max_entries!r}")
        self.max_entries = None if max_entries is None else int(max_entries)
        if int(projection_batch_size) <= 0:
            raise ValueError("projection_batch_size must be positive")
        self.projection_batch_size = int(projection_batch_size)
        self._world_points: list[torch.Tensor] = []
        self._latent_indices: list[int] = []
        self._frame_ids: list[int] = []

    def __len__(self) -> int:
        """Number of cached frames still held after FIFO eviction."""
        return len(self._world_points)

    def clear(self) -> None:
        """Drop all cached frames (and their device tensors)."""

        self._world_points.clear()
        self._latent_indices.clear()
        self._frame_ids.clear()

    def _evict_to_capacity(self) -> None:
        """Drop the oldest frames when ``max_entries`` is set; unbounded caches skip this."""
        if self.max_entries is None:
            return
        excess = len(self._world_points) - self.max_entries
        if excess > 0:
            del self._world_points[:excess]
            del self._latent_indices[:excess]
            del self._frame_ids[:excess]

    @staticmethod
    def _scale_intrinsics(intrinsic: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale fx/fy/cx/cy together so a spatially-strided depth map keeps the same FOV."""
        result = intrinsic.clone()
        result[:, 0] *= scale
        result[:, 1] *= scale
        return result

    @staticmethod
    def compute_points(
        *,
        depth: torch.Tensor,
        world_to_camera: torch.Tensor,
        intrinsic: torch.Tensor,
        downsample: int,
    ) -> torch.Tensor:
        """Unproject a spatially-strided depth map into world points."""
        factor = max(1, int(downsample))
        depth = depth[:, :, ::factor, ::factor].to(torch.float32)
        intrinsic = Sparse3DCache._scale_intrinsics(intrinsic.to(torch.float32), 1.0 / factor)
        return unproject_depth(depth, world_to_camera=world_to_camera.to(torch.float32), intrinsic=intrinsic)

    def add_precomputed(self, *, points: torch.Tensor, latent_index: int, frame_id: int | None = None) -> None:
        """Insert already-unprojected world points; evict FIFO if over capacity."""
        self._world_points.append(points.detach())
        self._latent_indices.append(int(latent_index))
        self._frame_ids.append(int(latent_index) if frame_id is None else int(frame_id))
        self._evict_to_capacity()

    def add(
        self,
        *,
        depth: torch.Tensor,
        world_to_camera: torch.Tensor,
        intrinsic: torch.Tensor,
        latent_index: int,
        frame_id: int | None = None,
    ) -> None:
        """Unproject *depth* and cache the resulting world points."""
        self.add_precomputed(
            points=self.compute_points(
                depth=depth,
                world_to_camera=world_to_camera,
                intrinsic=intrinsic,
                downsample=self.downsample,
            ),
            latent_index=latent_index,
            frame_id=frame_id,
        )

    @torch.no_grad()
    def retrieve(
        self,
        *,
        target_world_to_camera: torch.Tensor,
        target_intrinsic: torch.Tensor,
        target_hw: tuple[int, int],
        count: int,
        maximum_coverage: bool = True,
        depth_threshold: float = 0.1,
    ) -> list[tuple[int, int]]:
        """Rank cached frames by target-frustum coverage; return ``(latent, frame)`` ids."""
        if not self._world_points or count <= 0:
            return []
        device = target_world_to_camera.device
        factor = self.downsample
        target_height = (target_hw[0] + factor - 1) // factor
        target_width = (target_hw[1] + factor - 1) // factor
        if target_world_to_camera.ndim == 4:
            views = target_world_to_camera.shape[1]
            world_to_camera = target_world_to_camera.to(device=device, dtype=torch.float32)
            intrinsics = target_intrinsic.to(device=device, dtype=torch.float32)
        else:
            views = 1
            world_to_camera = target_world_to_camera[:, None].to(device=device, dtype=torch.float32)
            intrinsics = target_intrinsic[:, None].to(device=device, dtype=torch.float32)
        intrinsics = torch.stack(
            [self._scale_intrinsics(intrinsics[:, index], 1.0 / factor) for index in range(views)],
            dim=1,
        )

        candidates = len(self._world_points)
        point_shape = self._world_points[0].shape
        if len(point_shape) != 4 or point_shape[-1] != 3:
            raise ValueError(f"cached points must have shape [B,H,W,3], got {tuple(point_shape)}")
        if any(value.shape != point_shape for value in self._world_points[1:]):
            raise ValueError("all cached point tensors must have the same shape")
        batch, _height, _width, _coordinates = point_shape
        world_to_camera = world_to_camera.permute(1, 0, 2, 3).contiguous()
        intrinsics = intrinsics.permute(1, 0, 2, 3).contiguous()
        pixels_per_view = batch * target_height * target_width
        key_count = views * pixels_per_view
        minimum_depth = torch.full((key_count,), float("inf"), device=device)

        # Two bounded passes preserve the exact global z-buffer semantics while
        # avoiding the former [all_candidates, B, H, W, 3] stack and its much
        # larger per-view projection intermediates (CC-36).
        for start in range(0, candidates, self.projection_batch_size):
            stop = min(start + self.projection_batch_size, candidates)
            points = torch.stack(
                [value.to(device=device, dtype=torch.float32) for value in self._world_points[start:stop]]
            )
            _candidate_ids, keys, depths = self._project_chunk(
                points,
                world_to_camera=world_to_camera,
                intrinsics=intrinsics,
                target_height=target_height,
                target_width=target_width,
                pixels_per_view=pixels_per_view,
            )
            minimum_depth.scatter_reduce_(0, keys, depths, reduce="amin", include_self=True)

        coverage = torch.zeros((candidates, key_count), dtype=torch.bool, device="cpu")
        for start in range(0, candidates, self.projection_batch_size):
            stop = min(start + self.projection_batch_size, candidates)
            points = torch.stack(
                [value.to(device=device, dtype=torch.float32) for value in self._world_points[start:stop]]
            )
            candidate_ids, keys, depths = self._project_chunk(
                points,
                world_to_camera=world_to_camera,
                intrinsics=intrinsics,
                target_height=target_height,
                target_width=target_width,
                pixels_per_view=pixels_per_view,
            )
            visible = depths <= minimum_depth[keys] + float(depth_threshold)
            flat_keys = candidate_ids[visible].long() * key_count + keys[visible]
            chunk_coverage = torch.zeros((stop - start) * key_count, device=device, dtype=torch.bool)
            chunk_coverage.scatter_(0, flat_keys, True)
            coverage[start:stop].copy_(chunk_coverage.view(stop - start, key_count).cpu())
        if not bool(coverage.any()):
            return []
        take = min(int(count), candidates)

        if maximum_coverage:
            covered = torch.zeros(key_count, device="cpu", dtype=torch.bool)
            selected: list[int] = []
            for _ in range(take):
                additional = (coverage & ~covered).sum(dim=1)
                if selected:
                    additional[selected] = -1
                best = int(additional.argmax().item())
                if int(additional[best].item()) <= 0:
                    break
                selected.append(best)
                covered |= coverage[best]
        else:
            scores = coverage.sum(dim=1)
            selected = torch.topk(scores, k=take).indices.tolist() if int(scores.max().item()) > 0 else []
        return [(self._latent_indices[index], self._frame_ids[index]) for index in reversed(selected)]

    @staticmethod
    def _project_chunk(
        points: torch.Tensor,
        *,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        target_height: int,
        target_width: int,
        pixels_per_view: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project one bounded candidate chunk and return sparse valid pixels."""

        homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1).unsqueeze(-1)
        camera = torch.matmul(world_to_camera[:, None, :, None, None], homogeneous[None])[..., :3, :]
        projected = torch.matmul(intrinsics[:, None, :, None, None], camera)[..., 0]
        z = camera[..., 2, 0]
        x = torch.round(projected[..., 0] / projected[..., 2].clamp(min=1e-6)).long()
        y = torch.round(projected[..., 1] / projected[..., 2].clamp(min=1e-6)).long()
        valid = (z > 0) & (x >= 0) & (x < target_width) & (y >= 0) & (y < target_height)
        view_ids, candidate_ids, batch_ids, _source_y, _source_x = valid.nonzero(as_tuple=True)
        keys = (
            view_ids * pixels_per_view
            + batch_ids * target_height * target_width
            + y[valid] * target_width
            + x[valid]
        )
        return candidate_ids, keys, z[valid].to(torch.float32)


def _video_to_bcfhw(video: torch.Tensor) -> torch.Tensor:
    """Normalize common video layouts to ``[B, C, F, H, W]``; reject anything else."""
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
    raise ValueError(f"expected video in BCFHW/BFCHW/CFHW/FCHW layout, got {tuple(video.shape)}")


def _prepare_intrinsics(intrinsic: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Promote per-frame or per-batch 3x3 K to pixels; reject unexpected ranks."""
    if intrinsic.ndim == 3:
        return pixel_intrinsics(intrinsic, height=height, width=width)
    if intrinsic.ndim == 4:
        batch, frames = intrinsic.shape[:2]
        return pixel_intrinsics(
            intrinsic.reshape(batch * frames, 3, 3),
            height=height,
            width=width,
        ).reshape(batch, frames, 3, 3)
    raise ValueError(f"unexpected intrinsic shape {tuple(intrinsic.shape)}")


def _select_intrinsic(intrinsic: torch.Tensor, index: int) -> torch.Tensor:
    """Pick frame *index* from ``[B,F,3,3]``; clamp so a short K sequence still covers later frames."""
    if intrinsic.ndim == 4:
        return intrinsic[:, min(max(0, index), intrinsic.shape[1] - 1)]
    return intrinsic


def _depth_for_source(
    depths: dict[int, torch.Tensor] | None,
    index: int,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    constant_depth: float,
) -> torch.Tensor:
    """Use a per-source depth map when present; otherwise a constant plane so color can still splat."""
    depth = None if depths is None else depths.get(index)
    if depth is None:
        return torch.full((batch, 1, height, width), constant_depth, device=device, dtype=torch.float32)
    depth = depth.to(device=device, dtype=torch.float32)
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.shape[-2:] != (height, width):
        depth = torch.nn.functional.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
    return depth


def _warp_sources_to_target(
    *,
    points: torch.Tensor,
    source_valid: torch.Tensor,
    rgb: torch.Tensor,
    target_world_to_camera: torch.Tensor,
    target_intrinsic: torch.Tensor,
    depth_threshold: float,
    fill: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-buffer multiple source RGBD clouds into one target view; uncovered pixels stay *fill*."""
    sources, batch, height, width, _ = points.shape
    channels = rgb.shape[2]
    pixel_count = batch * height * width
    homogeneous = torch.cat(
        (points, torch.ones(sources, batch, height, width, 1, device=rgb.device, dtype=points.dtype)),
        dim=-1,
    ).unsqueeze(-1)
    camera = torch.matmul(target_world_to_camera[None, :, None, None], homogeneous)[..., :3, 0]
    z = camera[..., 2]
    projected = torch.matmul(target_intrinsic[None, :, None, None], camera.unsqueeze(-1))[..., 0]
    x = torch.round(projected[..., 0] / projected[..., 2].clamp(min=1e-6)).long()
    y = torch.round(projected[..., 1] / projected[..., 2].clamp(min=1e-6)).long()
    valid = source_valid & (z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)

    fused = torch.full((pixel_count, channels), fill, device=rgb.device, dtype=torch.float32)
    covered = torch.zeros(pixel_count, device=rgb.device, dtype=torch.bool)
    if not bool(valid.any()):
        return fused.view(batch, height, width, channels).permute(0, 3, 1, 2), covered.view(batch, height, width)
    source_ids, batch_ids, source_y, source_x = valid.nonzero(as_tuple=True)
    target_pixel = batch_ids * height * width + y[valid] * width + x[valid]
    source_key = source_ids * pixel_count + target_pixel
    z_valid = z[valid].to(torch.float32)
    source_pixel_count = sources * pixel_count
    minimum_depth = torch.full((source_pixel_count,), float("inf"), device=rgb.device)
    minimum_depth.scatter_reduce_(0, source_key, z_valid, reduce="amin", include_self=True)
    keep = z_valid <= minimum_depth[source_key] + float(depth_threshold)
    if not bool(keep.any()):
        return fused.view(batch, height, width, channels).permute(0, 3, 1, 2), covered.view(batch, height, width)
    ordinal = keep.nonzero(as_tuple=False).flatten()
    owner = torch.full(
        (source_pixel_count,),
        torch.iinfo(torch.long).max,
        device=rgb.device,
        dtype=torch.long,
    )
    owner.scatter_reduce_(0, source_key[ordinal], ordinal.long(), reduce="amin", include_self=True)
    assigned = owner != torch.iinfo(torch.long).max
    candidate_depth = torch.full((source_pixel_count,), float("inf"), device=rgb.device)
    candidate_depth[assigned] = z_valid[owner[assigned]]
    gathered_rgb = rgb.permute(0, 1, 3, 4, 2)[source_ids, batch_ids, source_y, source_x]
    candidate_rgb = torch.full((source_pixel_count, channels), fill, device=rgb.device)
    candidate_rgb[assigned] = gathered_rgb[owner[assigned]]
    best_depth, best_source = candidate_depth.view(sources, pixel_count).min(dim=0)
    covered = torch.isfinite(best_depth)
    best_key = best_source * pixel_count + torch.arange(pixel_count, device=rgb.device)
    fused[covered] = candidate_rgb[best_key[covered]]
    return (
        fused.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous(),
        covered.view(batch, height, width),
    )


@torch.no_grad()
def forward_warp_indexed_frames(
    *,
    source_pixels: torch.Tensor,
    source_indices: list[int],
    source_camera_indices: list[int],
    target_camera_indices: list[int],
    camera_to_world: torch.Tensor,
    intrinsic: torch.Tensor,
    source_depths: dict[int, torch.Tensor] | None,
    height: int,
    width: int,
    constant_depth: float = 1.0,
    depth_threshold: float = 1e-4,
    fill_value: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Forward-warp selected bank frames into target camera frames with z-buffering."""

    if not source_indices or not target_camera_indices:
        return None
    video = _video_to_bcfhw(source_pixels).to(camera_to_world.device)
    if camera_to_world.ndim == 3:
        camera_to_world = camera_to_world.unsqueeze(0)
    if intrinsic.ndim == 2:
        intrinsic = intrinsic.unsqueeze(0)
    camera_to_world = camera_to_world.to(device=video.device, dtype=torch.float32)
    intrinsic = _prepare_intrinsics(intrinsic.to(video.device), height=height, width=width)
    batch = video.shape[0]
    if camera_to_world.shape[0] == 1 and batch > 1:
        camera_to_world = camera_to_world.expand(batch, -1, -1, -1)
    if len(source_camera_indices) < video.shape[2]:
        raise ValueError("source_camera_indices must describe every bank pixel frame")

    payloads = []
    for raw_source in source_indices:
        source = min(max(0, int(raw_source)), video.shape[2] - 1)
        camera_index = min(max(0, int(source_camera_indices[source])), camera_to_world.shape[1] - 1)
        depth = _depth_for_source(
            source_depths,
            source,
            batch=batch,
            height=height,
            width=width,
            device=video.device,
            constant_depth=float(constant_depth),
        )
        source_world_to_camera = safe_inverse(camera_to_world[:, camera_index])
        points = unproject_depth(
            depth,
            world_to_camera=source_world_to_camera,
            intrinsic=_select_intrinsic(intrinsic, camera_index),
        )
        payloads.append((video[:, :, source].float(), points, depth[:, 0] > 0))
    rgb = torch.stack([value[0] for value in payloads])
    points = torch.stack([value[1] for value in payloads])
    source_valid = torch.stack([value[2] for value in payloads])
    fill = float(video.amin().item()) if fill_value is None else float(fill_value)
    warped, coverage = [], []
    for raw_target in target_camera_indices:
        target = min(max(0, int(raw_target)), camera_to_world.shape[1] - 1)
        frame, covered = _warp_sources_to_target(
            points=points,
            source_valid=source_valid,
            rgb=rgb,
            target_world_to_camera=safe_inverse(camera_to_world[:, target]),
            target_intrinsic=_select_intrinsic(intrinsic, target),
            depth_threshold=depth_threshold,
            fill=fill,
        )
        warped.append(frame)
        coverage.append(covered[:, None].float())
    return torch.stack(warped, dim=2).to(source_pixels.dtype), torch.stack(coverage, dim=2)


__all__ = [
    "Sparse3DCache",
    "forward_warp_indexed_frames",
    "pixel_intrinsics",
    "safe_inverse",
    "unproject_depth",
]
