"""MosaicMem-inspired spatial patch memory for 3D-aware retrieval.

MosaicMem stores 3D patches on a latent canvas keyed by camera pose.
A later viewpoint queries the store and gets the patches that project
into its frustum, which a memory-conditioned DiT can attend to.

This module owns the geometry and the in-process store — not a
training loop:

- :class:`CameraIntrinsics` / :class:`CameraPose` — pinhole project /
  unproject and camera-to-world.
- :class:`Patch3D` / :class:`LatentCanvas` / :class:`MosaicFrame` —
  patch payload and the 2D canvas they land on.
- :class:`MosaicMemoryStore` / :class:`MemoryRetriever` — insert
  frames and retrieve patches for a query pose.
- :class:`MosaicMemoryConfig` — capacity, patch size, and score
  thresholds shared by adapters.

Deterministic: no RNG in ranking. Torch is optional and only needed
when callers pass tensors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, floor, isfinite, sqrt
from typing import Any, Iterable, Mapping, Sequence

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]

# ──────────────────────────────────────────────────────────────────────────
# Pinhole camera — project / unproject; z <= 0 or non-finite yields None
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Camera intrinsic parameters modeling pinhole camera projections.

    Attributes:
        width: Image resolution width in pixels.
        height: Image resolution height in pixels.
        fx: Focal length along the X-axis in pixels.
        fy: Focal length along the Y-axis in pixels.
        cx: Optical center (principal point) X-coordinate in pixels.
        cy: Optical center (principal point) Y-coordinate in pixels.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_size(cls, width: int, height: int, focal: float | None = None) -> "CameraIntrinsics":
        """Factory creating symmetric intrinsics with principal point set to image centers."""
        focal_value = float(focal or max(width, height))
        return cls(
            width=int(width), height=int(height), fx=focal_value, fy=focal_value, cx=width / 2.0, cy=height / 2.0
        )

    def project(self, point: Point3) -> Point2 | None:
        """Projects a 3D point in the camera reference frame onto the 2D image plane.

        Returns None if the point lies behind the camera lens (z <= 0) or contains non-finite values.
        """
        x, y, z = point
        if z <= 0.0 or not all(isfinite(value) for value in point):
            return None
        return (self.fx * x / z + self.cx, self.fy * y / z + self.cy)

    def unproject(self, pixel: Point2, depth: float) -> Point3:
        """Unprojects a 2D pixel coordinate back into a 3D point in camera space given its depth."""
        u, v = pixel
        return ((u - self.cx) * depth / self.fx, ((v - self.cy) * depth) / self.fy, depth)

    def in_bounds(self, pixel: Point2) -> bool:
        """Returns True if the 2D pixel lies strictly within the image boundaries."""
        u, v = pixel
        return 0.0 <= u < float(self.width) and 0.0 <= v < float(self.height)


# ──────────────────────────────────────────────────────────────────────────
# World ↔ camera — rotation is camera-from-world after +translation
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Represents the world-to-camera coordinate system transformations.

    Attributes:
        rotation: 3x3 orthonormal rotation matrix.
        translation: 3D translation vector.
        timestamp: Simulation or video playback timestamp.
    """

    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation: Point3 = (0.0, 0.0, 0.0)
    timestamp: float = 0.0

    @classmethod
    def identity(cls, timestamp: float = 0.0) -> "CameraPose":
        """Factory producing a zero-translation, identity-rotation pose."""
        return cls(timestamp=float(timestamp))

    def world_to_camera(self, point: Point3) -> Point3:
        """Applies the coordinate transformation to map a 3D point from World to Camera frame."""
        shifted = (
            point[0] + self.translation[0],
            point[1] + self.translation[1],
            point[2] + self.translation[2],
        )
        return _matvec(self.rotation, shifted)

    def camera_to_world(self, point: Point3) -> Point3:
        """Applies the inverse coordinate transformation to map a 3D point from Camera to World frame."""
        inverse_rotation = _transpose(self.rotation)
        rotated = _matvec(inverse_rotation, point)
        return (
            rotated[0] - self.translation[0],
            rotated[1] - self.translation[1],
            rotated[2] - self.translation[2],
        )


# ──────────────────────────────────────────────────────────────────────────
# Patch payload and retrieval row — latent tokens keyed by world center
# ──────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Patch3D:
    """Represents a localized 3D latent patch capturing spatial-temporal features.

    Stores localized representation vectors (latent token maps) generated by visual encoders,
    associating them with a camera viewpoint and physical coordinates for 3D mosaic queries.
    """

    id: int
    center: Point3
    source_pose: CameraPose
    source_frame: int
    source_timestamp: float
    source_depth: float
    source_rect: tuple[float, float, float, float]
    latent: list[float]
    latent_height: int
    latent_width: int
    token_coords: list[Point2] = field(default_factory=list)
    depth_tile: list[float] | None = None
    source_intrinsics: CameraIntrinsics | None = None
    latent_shape: tuple[int, int, int] = (0, 0, 0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_latent_shape(self) -> tuple[int, int, int]:
        """Infers the shape format of the latent vector as (Channels, Height, Width)."""
        if self.latent_shape != (0, 0, 0):
            return self.latent_shape
        token_count = self.latent_height * self.latent_width
        channels = len(self.latent) // token_count if token_count else 0
        return (channels, self.latent_height, self.latent_width)

    def channels(self) -> int:
        """Channel count from ``latent_shape``, or inferred from latent length."""
        return self.resolved_latent_shape()[0]

    def token_count(self) -> int:
        """Spatial token count ``H * W`` after shape resolution."""
        _, height, width = self.resolved_latent_shape()
        return height * width


@dataclass(slots=True)
class RetrievedPatch:
    """Pairs a 3D patch with projection parameters retrieved for a target camera viewpoint."""

    patch: Patch3D
    target_position: Point2
    projected_footprint: list[Point2]
    target_depth: float
    visibility_score: float


# ──────────────────────────────────────────────────────────────────────────
# Capacity / ranking knobs — shared by store and retriever; no RNG
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MosaicMemoryConfig:
    """Hyperparameters configuring the capacity, density, and diversity of the Mosaic Memory."""

    max_patches: int = 10000
    top_k: int = 64
    near_clip: float = 0.1
    far_clip: float = 100.0
    patch_size: int = 16
    latent_patch_size: int = 2
    temporal_decay_half_life: float = 0.0
    diversity_radius: float = 0.0
    diversity_penalty: float = 0.5
    min_visibility: float = 0.01


# ──────────────────────────────────────────────────────────────────────────
# Dense canvas — THWC storage; stack requires matching H/W/C
# ──────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class LatentCanvas:
    """A flattened, multi-dimensional grid holding dense 3D visual tokens.

    Acts as the dense rendering target where scattered memory patches are re-projected
    and composited into a standard layout compatible with standard convolutional or
    Vision Transformer (ViT) feature processing.
    """

    data: list[float]
    frames: int
    height: int
    width: int
    channels: int

    @classmethod
    def empty(cls, frames: int, height: int, width: int, channels: int) -> "LatentCanvas":
        """Zero-filled canvas; negative extents clamp to zero so size cannot go negative."""
        size = max(frames, 0) * max(height, 0) * max(width, 0) * max(channels, 0)
        return cls(
            [0.0] * size, frames=max(frames, 0), height=max(height, 0), width=max(width, 0), channels=max(channels, 0)
        )

    @classmethod
    def stack(cls, canvases: Sequence["LatentCanvas"]) -> "LatentCanvas":
        """Concatenate along time; :exc:`ValueError` if H/W/C disagree."""
        if not canvases:
            return cls.empty(0, 0, 0, 0)
        first = canvases[0]
        data: list[float] = []
        frames = 0
        for canvas in canvases:
            if (canvas.height, canvas.width, canvas.channels) != (first.height, first.width, first.channels):
                raise ValueError("All latent canvases must share height, width, and channels.")
            data.extend(canvas.data)
            frames += canvas.frames
        return cls(data, frames=frames, height=first.height, width=first.width, channels=first.channels)

    def is_empty(self) -> bool:
        """True when there are no frames or the flat buffer has no values."""
        return self.frames == 0 or not self.data

    def to_tokens(self) -> list[list[float]]:
        """Converts the flat buffer into a list of per-token channel vectors."""
        if self.channels <= 0:
            return []
        return [self.data[index : index + self.channels] for index in range(0, len(self.data), self.channels)]

    def to_cthw(self) -> list[float]:
        """Transposes the native block memory layout (T, H, W, C) to a PyTorch-friendly contiguous (C, T, H, W) layout."""
        if self.frames <= 0 or self.channels <= 0 or self.height <= 0 or self.width <= 0:
            return []
        frame_size = self.height * self.width
        latent = [0.0] * (self.channels * self.frames * frame_size)
        for c in range(self.channels):
            for t in range(self.frames):
                for y in range(self.height):
                    for x in range(self.width):
                        src = (((t * self.height + y) * self.width + x) * self.channels) + c
                        dst = ((c * self.frames + t) * self.height + y) * self.width + x
                        latent[dst] = self.data[src]
        return latent


# ──────────────────────────────────────────────────────────────────────────
# Query result — retrieved patches plus a coarse occupancy grid
# ──────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class MosaicFrame:
    """A synthesized query result representing a virtual camera viewpoint.

    Contains the set of optimal historical 3D patches retrieved and projected into
    the target camera's FOV, alongside coverage metadata.
    """

    target_pose: CameraPose
    patches: list[RetrievedPatch]
    coverage_mask: list[list[bool]]
    width: int
    height: int

    def num_patches(self) -> int:
        """Count of retrieved patches in this viewpoint (not token count)."""
        return len(self.patches)

    def has_coverage(self) -> bool:
        """True when any coarse occupancy cell is marked covered."""
        return any(any(row) for row in self.coverage_mask)

    def coverage_ratio(self) -> float:
        """Calculates the ratio of successfully retrieved pixels versus total image pixels."""
        total = sum(len(row) for row in self.coverage_mask)
        if total == 0:
            return 0.0
        covered = sum(1 for row in self.coverage_mask for value in row if value)
        return covered / total

    def compose_tokens(self) -> tuple[list[list[float]], list[Point3]]:
        """Extracts unordered spatial-temporal point cloud tokens for sparse attention architectures."""
        tokens: list[list[float]] = []
        positions: list[Point3] = []
        for retrieved in self.patches:
            channels = retrieved.patch.channels()
            if channels <= 0:
                continue
            for token_index in range(retrieved.patch.token_count()):
                start = token_index * channels
                end = start + channels
                if end > len(retrieved.patch.latent):
                    continue
                tokens.append(retrieved.patch.latent[start:end])
                footprint = (
                    retrieved.projected_footprint[token_index]
                    if token_index < len(retrieved.projected_footprint)
                    else retrieved.target_position
                )
                positions.append((footprint[0], footprint[1], retrieved.target_depth))
        return tokens, positions

    def compose_latent_canvas(self, latent_height: int, latent_width: int, channels: int) -> LatentCanvas:
        """Renders overlapping scattered retrieved patches down onto a dense rectangular pixel canvas.

        Uses Z-buffer or visibility-score based blending to seamlessly mosaic
        different historical perspectives together.
        """
        canvas = LatentCanvas.empty(1, latent_height, latent_width, channels)
        if latent_height <= 0 or latent_width <= 0 or channels <= 0 or self.width <= 0 or self.height <= 0:
            return canvas

        weights = [0.0] * (latent_height * latent_width)
        for retrieved in self.patches:
            patch = retrieved.patch
            if patch.token_count() <= 0 or len(patch.latent) < patch.token_count() * channels:
                continue
            weight = max(retrieved.visibility_score, 1e-6)
            center_x = retrieved.target_position[0] / self.width * latent_width
            center_y = retrieved.target_position[1] / self.height * latent_height
            half_w = patch.latent_width / 2.0
            half_h = patch.latent_height / 2.0

            for py in range(patch.latent_height):
                for px in range(patch.latent_width):
                    token_index = py * patch.latent_width + px
                    start = token_index * channels
                    token = patch.latent[start : start + channels]
                    if token_index < len(retrieved.projected_footprint):
                        point = retrieved.projected_footprint[token_index]
                        local_x = point[0] / self.width * latent_width
                        local_y = point[1] / self.height * latent_height
                    else:
                        local_x = center_x + px + 0.5 - half_w
                        local_y = center_y + py + 0.5 - half_h
                    _splat_token(
                        canvas.data, weights, token, local_x, local_y, latent_width, latent_height, channels, weight
                    )

        for cell, weight in enumerate(weights):
            if weight <= 0.0:
                continue
            start = cell * channels
            for offset in range(channels):
                canvas.data[start + offset] /= weight
        return canvas


# ──────────────────────────────────────────────────────────────────────────
# In-process store — insert / retrieve / evict oldest when over max_patches
# ──────────────────────────────────────────────────────────────────────────


class MosaicMemoryStore:
    """MosaicMem-inspired spatial patch store for long-horizon visual memory.

    This persistent memory structure caches local feature representations (patches) encoded
    during interaction histories, indexing them by 3D world coordinates.

    When querying from a new camera angle, this store dynamically synthesizes a pseudo-observation
    (a `MosaicFrame`) by querying the top-K relevant patches via inverse-projection, handling
    depth ordering, occlusion clipping, and memory budget pruning automatically.
    """

    def __init__(self, config: MosaicMemoryConfig | None = None) -> None:
        """Start empty; ``_next_id`` is the next unused patch identifier."""
        self.config = config or MosaicMemoryConfig()
        self.patches: list[Patch3D] = []
        self._next_id = 0

    def snapshot(self) -> dict[str, Any]:
        """Serializes the entire memory state to dictionary payload."""
        return {
            "config": asdict(self.config),
            "next_id": self._next_id,
            "patches": [_patch_to_dict(patch) for patch in self.patches],
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "MosaicMemoryStore":
        """Deserializes and reconstructs the memory store from a saved snapshot."""
        store = cls(MosaicMemoryConfig(**dict(snapshot.get("config") or {})))
        store._next_id = int(snapshot.get("next_id", 0))
        store.patches = [_patch_from_dict(row) for row in snapshot.get("patches", [])]
        return store

    def insert_patch(self, patch: Patch3D) -> int:
        """Inserts a single pre-computed 3D latent patch and trims memory if over budget."""
        if patch.id < 0:
            patch.id = self._next_id
        self._next_id = max(self._next_id, patch.id + 1)
        self.patches.append(patch)
        self._enforce_budget()
        return patch.id

    def insert_keyframe(
        self,
        *,
        frame_index: int,
        timestamp: float,
        latents: Sequence[float],
        latent_height: int,
        latent_width: int,
        channels: int,
        depth_map: Sequence[Sequence[float]],
        intrinsics: CameraIntrinsics,
        pose: CameraPose,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[int]:
        """Ingests an entire encoded image, splitting its visual tokens into localized 3D patches.

        Unprojects every latent token (or sub-grid) into 3D world space using the frame's
        CameraIntrinsics and depth map, and stores them in memory.
        """
        patch_h = max(int(self.config.latent_patch_size), 1)
        patch_w = patch_h
        image_h = len(depth_map)
        image_w = len(depth_map[0]) if image_h else intrinsics.width
        ids: list[int] = []

        for py in range(0, latent_height, patch_h):
            for px in range(0, latent_width, patch_w):
                ph = min(patch_h, latent_height - py)
                pw = min(patch_w, latent_width - px)
                latent = _extract_latent_patch(latents, latent_height, latent_width, channels, px, py, pw, ph)
                rect = (
                    px / max(latent_width, 1) * image_w,
                    py / max(latent_height, 1) * image_h,
                    pw / max(latent_width, 1) * image_w,
                    ph / max(latent_height, 1) * image_h,
                )
                token_coords = _token_coords(rect, pw, ph)
                depth_tile = [_sample_depth(depth_map, coord) for coord in token_coords]
                depth = _sample_depth(depth_map, (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0))
                center_camera = intrinsics.unproject((rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0), depth)
                center_world = pose.camera_to_world(center_camera)
                patch = Patch3D(
                    id=self._next_id,
                    center=center_world,
                    source_pose=pose,
                    source_frame=frame_index,
                    source_timestamp=timestamp,
                    source_depth=depth,
                    source_rect=rect,
                    latent=latent,
                    latent_height=ph,
                    latent_width=pw,
                    token_coords=token_coords,
                    depth_tile=depth_tile,
                    source_intrinsics=intrinsics,
                    latent_shape=(channels, ph, pw),
                    metadata=dict(metadata or {}),
                )
                ids.append(self.insert_patch(patch))
        return ids

    def retrieve(
        self, target_pose: CameraPose, intrinsics: CameraIntrinsics, *, query_time: float | None = None
    ) -> list[RetrievedPatch]:
        """Queries the memory store to retrieve optimal 3D patches for a novel target camera viewpoint."""
        return MemoryRetriever(self.config).retrieve_patches(
            self.patches, target_pose, intrinsics, query_time=query_time
        )

    def retrieve_mosaic(
        self, target_pose: CameraPose, intrinsics: CameraIntrinsics, *, query_time: float | None = None
    ) -> MosaicFrame:
        """Retrieves and synthesizes a full pseudo-observation (MosaicFrame) for a novel viewpoint."""
        retrieved = self.retrieve(target_pose, intrinsics, query_time=query_time)
        return MosaicFrame(
            target_pose=target_pose,
            patches=retrieved,
            coverage_mask=coverage_mask(retrieved, intrinsics),
            width=intrinsics.width,
            height=intrinsics.height,
        )

    def delete_frame(self, frame_index: int) -> None:
        """Removes all patches originating from a specific frame index (e.g. for windowed context sliding)."""
        self.patches = [patch for patch in self.patches if patch.source_frame != frame_index]

    def delete_patch(self, patch_id: int) -> None:
        """Removes a specific patch by its unique identifier."""
        self.patches = [patch for patch in self.patches if patch.id != patch_id]

    def num_patches(self) -> int:
        """Current patch count after budget eviction."""
        return len(self.patches)

    def total_tokens(self) -> int:
        """Sum of spatial tokens across every stored patch."""
        return sum(patch.token_count() for patch in self.patches)

    def query_nearest(self, position: Point3, k: int) -> list[Patch3D]:
        """Performs a spatial k-nearest-neighbors search based on 3D Euclidean distance."""
        ranked = sorted(self.patches, key=lambda patch: _distance2(patch.center, position))
        return ranked[: max(k, 0)]

    def _enforce_budget(self) -> None:
        """Enforces a strict upper bound on memory capacity.

        If the number of patches exceeds the `max_patches` limit, evicts the oldest
        historical patches (based on `source_timestamp`) until the budget is satisfied.
        """
        if len(self.patches) <= self.config.max_patches:
            return
        self.patches.sort(key=lambda patch: patch.source_timestamp, reverse=True)
        del self.patches[self.config.max_patches :]


# ──────────────────────────────────────────────────────────────────────────
# Inverse projection ranker — frustum clip, visibility, optional spatial NMS
# ──────────────────────────────────────────────────────────────────────────


class MemoryRetriever:
    """Executes inverse projection and visibility ranking for Mosaic queries."""

    def __init__(self, config: MosaicMemoryConfig | None = None, *, depth_sort: bool = True) -> None:
        """``depth_sort`` orders the final top-k far-to-near for painter-style blend."""
        self.config = config or MosaicMemoryConfig()
        self.depth_sort = depth_sort

    def retrieve_patches(
        self,
        patches: Iterable[Patch3D],
        target_pose: CameraPose,
        intrinsics: CameraIntrinsics,
        *,
        query_time: float | None = None,
    ) -> list[RetrievedPatch]:
        """Projects available 3D patches into the target camera plane and selects the top-K optimal.

        Filters out patches falling outside the view frustum (clipping) or failing
        visibility bounds, then scores and optionally diversifies them.
        """
        candidates: list[RetrievedPatch] = []
        for patch in patches:
            camera_point = target_pose.world_to_camera(patch.center)
            # Z-axis near/far plane clipping
            if camera_point[2] < self.config.near_clip or camera_point[2] > self.config.far_clip:
                continue
            footprint = project_patch_footprint(patch, target_pose, intrinsics)
            pixel = _average_point(footprint) if footprint else intrinsics.project(camera_point)
            if pixel is None or not intrinsics.in_bounds(pixel):
                continue
            visibility = self._visibility_score(pixel, camera_point[2], patch, footprint, intrinsics, query_time)
            if visibility < self.config.min_visibility:
                continue
            candidates.append(
                RetrievedPatch(
                    patch=patch,
                    target_position=pixel,
                    projected_footprint=footprint or [pixel],
                    target_depth=camera_point[2],
                    visibility_score=visibility,
                )
            )

        candidates.sort(key=lambda item: item.visibility_score, reverse=True)
        if self.config.diversity_radius > 0.0:
            candidates = self._diversity_select(candidates)
        candidates = candidates[: self.config.top_k]
        if self.depth_sort:
            # Depth sorting facilitates back-to-front Alpha Compositing or Painters Algorithm blending
            candidates.sort(key=lambda item: item.target_depth)
        return candidates

    def _visibility_score(
        self,
        pixel: Point2,
        depth: float,
        patch: Patch3D,
        footprint: Sequence[Point2],
        intrinsics: CameraIntrinsics,
        query_time: float | None,
    ) -> float:
        """Computes a heuristic quality score for a projected patch.

        Prefers patches that:
        1. Are centered in the camera's FOV (avoids lens distortion / edge cutoff).
        2. Are closer to the camera (higher visual fidelity / resolution).
        3. Have more of their constituent tokens successfully projected within bounds.
        4. (Optional) Are newer / less affected by temporal decay.
        """
        center_distance = sqrt(
            ((pixel[0] - intrinsics.cx) / max(intrinsics.width, 1)) ** 2
            + ((pixel[1] - intrinsics.cy) / max(intrinsics.height, 1)) ** 2
        )
        depth_score = 1.0 / (1.0 + depth * 0.1)
        footprint_score = max(len(footprint), 1) / max(patch.token_count(), 1)
        score = max(1.0 - center_distance, 0.0) * depth_score * max(footprint_score, 0.25)

        # Temporal fading
        if query_time is not None and self.config.temporal_decay_half_life > 0.0:
            age = max(float(query_time) - patch.source_timestamp, 0.0)
            score *= exp(-age / self.config.temporal_decay_half_life)
        return score

    def _diversity_select(self, candidates: list[RetrievedPatch]) -> list[RetrievedPatch]:
        """Applies spatial Non-Maximum Suppression (NMS) to avoid overlapping identical patches."""
        selected: list[RetrievedPatch] = []
        scores = [candidate.visibility_score for candidate in candidates]
        used = [False] * len(candidates)
        radius2 = self.config.diversity_radius * self.config.diversity_radius

        for _ in candidates:
            best_idx = None
            best_score = self.config.min_visibility
            for idx, score in enumerate(scores):
                if not used[idx] and score >= best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                break
            used[best_idx] = True
            chosen = candidates[best_idx]
            chosen.visibility_score = best_score
            selected.append(chosen)
            for idx, candidate in enumerate(candidates):
                if used[idx]:
                    continue
                # Spatially penalize neighbors
                if _distance2_2d(candidate.target_position, chosen.target_position) < radius2:
                    scores[idx] *= self.config.diversity_penalty
        return selected


# ──────────────────────────────────────────────────────────────────────────
# Geometry helpers — footprint, occupancy grid, latent extract, bilinear splat
# ──────────────────────────────────────────────────────────────────────────


def project_patch_footprint(patch: Patch3D, target_pose: CameraPose, intrinsics: CameraIntrinsics) -> list[Point2]:
    """Re-projects individual tokens of a patch to establish a 2D bounding polygon (footprint) on the target view."""
    source_intrinsics = patch.source_intrinsics or intrinsics
    coords = patch.token_coords or _token_coords(
        patch.source_rect, max(patch.latent_width, 1), max(patch.latent_height, 1)
    )
    depths = patch.depth_tile or [patch.source_depth] * len(coords)
    footprint: list[Point2] = []
    for coord, depth in zip(coords, depths):
        if depth <= 0.0 or not isfinite(depth):
            continue
        source_camera = source_intrinsics.unproject(coord, depth)
        world = patch.source_pose.camera_to_world(source_camera)
        target_camera = target_pose.world_to_camera(world)
        pixel = intrinsics.project(target_camera)
        if pixel is not None and intrinsics.in_bounds(pixel):
            footprint.append(pixel)
    return footprint


def coverage_mask(patches: Sequence[RetrievedPatch], intrinsics: CameraIntrinsics) -> list[list[bool]]:
    """Computes a coarse boolean occupancy grid over the image plane to evaluate projection density.

    Useful for tracking whether the mosaiced pseudo-observation has enough visual features
    covering the new camera view, avoiding blind spots or overly sparse representations.
    """
    grid_h = max(intrinsics.height // 8, 1)
    grid_w = max(intrinsics.width // 8, 1)
    coverage = [[False for _ in range(grid_w)] for _ in range(grid_h)]
    for patch in patches:
        cells = []
        for point in patch.projected_footprint or [patch.target_position]:
            gx = int(floor(point[0] / max(intrinsics.width, 1) * grid_w))
            gy = int(floor(point[1] / max(intrinsics.height, 1) * grid_h))
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                cells.append((gx, gy))
        if not cells:
            continue
        min_x, max_x = min(cell[0] for cell in cells), max(cell[0] for cell in cells)
        min_y, max_y = min(cell[1] for cell in cells), max(cell[1] for cell in cells)
        for gy in range(min_y, max_y + 1):
            for gx in range(min_x, max_x + 1):
                coverage[gy][gx] = True
    return coverage


def _extract_latent_patch(
    latents: Sequence[float],
    latent_height: int,
    latent_width: int,
    channels: int,
    px: int,
    py: int,
    pw: int,
    ph: int,
) -> list[float]:
    """Extracts a localized sub-grid (patch) of channel features from a flattened 2D dense latent map."""
    values: list[float] = []
    for row in range(py, py + ph):
        for col in range(px, px + pw):
            start = (row * latent_width + col) * channels
            values.extend(float(value) for value in latents[start : start + channels])
    return values


def _token_coords(rect: tuple[float, float, float, float], latent_width: int, latent_height: int) -> list[Point2]:
    """Generates continuous 2D coordinates for the centers of tokens within a geometric patch rectangle."""
    x, y, width, height = rect
    if latent_width <= 0 or latent_height <= 0:
        return []
    step_x = width / latent_width
    step_y = height / latent_height
    return [
        (x + (col + 0.5) * step_x, y + (row + 0.5) * step_y)
        for row in range(latent_height)
        for col in range(latent_width)
    ]


def _sample_depth(depth_map: Sequence[Sequence[float]], coord: Point2) -> float:
    """Safely samples nearest-neighbor depth from a 2D depth map given continuous UV coordinates."""
    if not depth_map or not depth_map[0]:
        return 1.0
    x = max(0, min(int(round(coord[0])), len(depth_map[0]) - 1))
    y = max(0, min(int(round(coord[1])), len(depth_map) - 1))
    depth = float(depth_map[y][x])
    return depth if depth > 0.0 and isfinite(depth) else 1.0


def _splat_token(
    data: list[float],
    weights: list[float],
    token: Sequence[float],
    local_x: float,
    local_y: float,
    width: int,
    height: int,
    channels: int,
    patch_weight: float,
) -> None:
    """Performs bilinear splatting (forward mapping) of a continuous 2D feature token onto a discrete 2D grid.

    Distributes the multi-channel token vector onto the 4 nearest discrete pixel/grid neighbors
    weighted by bilinear interpolation factors multiplied by the patch's visibility score.
    """
    x0 = floor(local_x)
    y0 = floor(local_y)
    fx = local_x - x0
    fy = local_y - y0
    for weight, cx, cy in (
        ((1.0 - fx) * (1.0 - fy), x0, y0),
        ((1.0 - fx) * fy, x0, y0 + 1),
        (fx * (1.0 - fy), x0 + 1, y0),
        (fx * fy, x0 + 1, y0 + 1),
    ):
        if weight <= 0.0 or cx < 0 or cy < 0 or cx >= width or cy >= height:
            continue
        cell = cy * width + cx
        cell_weight = patch_weight * weight
        weights[cell] += cell_weight
        start = cell * channels
        for channel in range(channels):
            data[start + channel] += float(token[channel]) * cell_weight


# ──────────────────────────────────────────────────────────────────────────
# Tiny linear algebra and snapshot codecs — no torch; JSON-safe tuples
# ──────────────────────────────────────────────────────────────────────────


def _matvec(matrix: Sequence[Sequence[float]], vector: Point3) -> Point3:
    """3×3 matrix-vector product used by pose transforms (no torch dependency)."""
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _transpose(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Transpose a 3×3 rotation so camera-to-world can invert without a solver."""
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _average_point(points: Sequence[Point2]) -> Point2:
    """Centroid of a footprint; caller must pass a non-empty sequence."""
    return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))


def _distance2(left: Point3, right: Point3) -> float:
    """Squared Euclidean distance in world space (avoids ``sqrt`` in k-NN ranking)."""
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2


def _distance2_2d(left: Point2, right: Point2) -> float:
    """Squared pixel distance for diversity NMS."""
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _patch_to_dict(patch: Patch3D) -> dict[str, Any]:
    """``dataclasses.asdict`` snapshot; explicit ``None`` keeps missing intrinsics stable."""
    row = asdict(patch)
    if row["source_intrinsics"] is None:
        row["source_intrinsics"] = None
    return row


def _patch_from_dict(row: Mapping[str, Any]) -> Patch3D:
    """Rebuild nested pose/intrinsics from mapping rows produced by :func:`_patch_to_dict`."""
    payload = dict(row)
    pose = payload.get("source_pose")
    if isinstance(pose, Mapping):
        payload["source_pose"] = CameraPose(
            rotation=tuple(
                tuple(float(value) for value in item) for item in pose.get("rotation", CameraPose.identity().rotation)
            ),
            translation=tuple(float(value) for value in pose.get("translation", (0.0, 0.0, 0.0))),
            timestamp=float(pose.get("timestamp", 0.0)),
        )
    intrinsics = payload.get("source_intrinsics")
    if isinstance(intrinsics, Mapping):
        payload["source_intrinsics"] = CameraIntrinsics(
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
        )
    payload["center"] = tuple(float(value) for value in payload["center"])
    payload["source_rect"] = tuple(float(value) for value in payload["source_rect"])
    payload["latent_shape"] = tuple(int(value) for value in payload.get("latent_shape", (0, 0, 0)))
    payload["token_coords"] = [tuple(float(value) for value in coord) for coord in payload.get("token_coords", [])]
    return Patch3D(**payload)


__all__ = [
    "CameraIntrinsics",
    "CameraPose",
    "LatentCanvas",
    "MemoryRetriever",
    "MosaicFrame",
    "MosaicMemoryConfig",
    "MosaicMemoryStore",
    "Patch3D",
    "RetrievedPatch",
    "coverage_mask",
    "project_patch_footprint",
]
