"""3D geometry consistency rewards for offline candidate scoring."""

from __future__ import annotations

import math
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

POINTCLOUD_FIELDS = (
    "pointcloud_npz",
    "point_cloud_npz",
    "geometry_pointcloud_npz",
    "pointcloud_cache_path",
)


def safe_video_id(video_id: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video_id)).strip("._") or "video"


def _require_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        raise ValueError("geometry scorer requires an explicit config")
    return config


def _require_config_value(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"geometry scorer config.{key} is required")
    return config[key]


def _require_str(config: dict[str, Any], key: str) -> str:
    value = _require_config_value(config, key)
    if value in (None, ""):
        raise ValueError(f"geometry scorer config.{key} is required")
    return str(value)


def _require_float(config: dict[str, Any], key: str) -> float:
    value = _require_config_value(config, key)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"geometry scorer config.{key} must be a number") from exc


def _require_positive_int(config: dict[str, Any], key: str) -> int:
    value = _require_config_value(config, key)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"geometry scorer config.{key} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"geometry scorer config.{key} must be a positive integer")
    return parsed


def _require_bool(config: dict[str, Any], key: str) -> bool:
    value = _require_config_value(config, key)
    if not isinstance(value, bool):
        raise ValueError(f"geometry scorer config.{key} must be boolean")
    return value


def _is_enabled(config: dict[str, Any]) -> bool:
    return _require_bool(config, "enabled")


def _is_static_row(row: dict[str, Any]) -> bool:
    camera = row.get("lar_type") or row.get("camera_type")
    return str(camera).lower() == "static"


def _video_path(row: dict[str, Any]) -> str | None:
    value = row.get("path") or row.get("video_path")
    return str(value) if value else None


def _with_geometry_status(
    row: dict[str, Any],
    status: str,
    *,
    static_gt_video_id: Any | None = None,
) -> dict[str, Any]:
    enriched = dict(row)
    enriched["geometry_status"] = status
    if static_gt_video_id is not None or status == "no_static_gt":
        enriched["static_gt_video_id"] = static_gt_video_id
    if status != "ok":
        enriched.pop("geometry_reward", None)
        enriched.pop("geometry_error", None)
        components = dict(enriched.get("reward_components") or {})
        components.pop("geometry", None)
        if components:
            enriched["reward_components"] = components
        else:
            enriched.pop("reward_components", None)
    return enriched


def _sample_indices(length: int, sample_frames: Any) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.int64)
    if sample_frames is None:
        return np.arange(length, dtype=np.int64)
    if isinstance(sample_frames, (list, tuple)):
        return np.array([idx for idx in sample_frames if 0 <= int(idx) < length], dtype=np.int64)
    count = int(sample_frames)
    if count <= 0 or count >= length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, count).round().astype(np.int64)


def _expand_poses(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 4:
        if poses.shape[0] != 1:
            raise ValueError(f"expected singleton pose batch, got {poses.shape}")
        poses = poses[0]
    if poses.ndim == 2:
        poses = poses[None]
    if poses.shape[-2:] == (3, 4):
        bottom = np.zeros((*poses.shape[:-2], 1, 4), dtype=poses.dtype)
        bottom[..., 0, 3] = 1.0
        poses = np.concatenate([poses, bottom], axis=-2)
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"expected pose shape T x 4 x 4 or T x 3 x 4, got {poses.shape}")
    return poses


def _collapse_singleton_batch(array: np.ndarray, *, expected_ndim: int, name: str) -> np.ndarray:
    if array.ndim == expected_ndim + 1:
        if array.shape[0] != 1:
            raise ValueError(f"expected singleton {name} batch, got {array.shape}")
        return array[0]
    return array


def _array_for_frame(array: np.ndarray, index: int) -> np.ndarray:
    if array.ndim >= 3 and array.shape[0] > 1:
        return array[index]
    if array.ndim >= 3:
        return array[0]
    return array


def _unproject_depths(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    c2ws: np.ndarray,
    *,
    sample_frames: Any = None,
) -> np.ndarray:
    depths = np.asarray(depths, dtype=np.float64)
    depths = _collapse_singleton_batch(depths, expected_ndim=3, name="depth")
    if depths.ndim == 2:
        depths = depths[None]
    if depths.ndim != 3:
        raise ValueError(f"expected depth shape T x H x W or H x W, got {depths.shape}")
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    intrinsics = _collapse_singleton_batch(intrinsics, expected_ndim=3, name="intrinsics")
    c2ws = _expand_poses(c2ws)
    frame_count = min(depths.shape[0], c2ws.shape[0])
    points: list[np.ndarray] = []
    for frame_idx in _sample_indices(frame_count, sample_frames):
        depth = depths[frame_idx]
        k = _array_for_frame(intrinsics, int(frame_idx))
        pose = c2ws[int(frame_idx)]
        if k.shape != (3, 3):
            raise ValueError(f"expected intrinsics shape 3 x 3, got {k.shape}")
        h, w = depth.shape
        ys, xs = np.mgrid[0:h, 0:w]
        z = depth.reshape(-1)
        valid = np.isfinite(z) & (z > 0)
        if not np.any(valid):
            continue
        x = (xs.reshape(-1)[valid] - k[0, 2]) / k[0, 0] * z[valid]
        y = (ys.reshape(-1)[valid] - k[1, 2]) / k[1, 1] * z[valid]
        cam = np.stack([x, y, z[valid], np.ones_like(x)], axis=1)
        world = (pose @ cam.T).T[:, :3]
        points.append(world)
    if not points:
        raise ValueError("depth stack produced no valid points")
    return np.concatenate(points, axis=0)


def load_point_cloud_npz(path: str | Path, *, sample_frames: Any = None) -> np.ndarray:
    """Load accepted NPZ point-cloud or depth-stack formats into Nx3 world points."""
    with np.load(path, allow_pickle=False) as payload:
        if "points" in payload:
            points = payload["points"]
        elif "point_cloud" in payload:
            points = payload["point_cloud"]
        else:
            depth_key = "depth" if "depth" in payload else "depths" if "depths" in payload else None
            pose_key = next((key for key in ("c2w", "poses", "c2ws", "w2c", "w2cs") if key in payload), None)
            if depth_key is None or "intrinsics" not in payload or pose_key is None:
                raise ValueError("missing points or depth/intrinsics/pose arrays")
            poses = payload[pose_key]
            if pose_key in {"w2c", "w2cs"}:
                poses = np.linalg.inv(_expand_poses(poses))
            points = _unproject_depths(
                payload[depth_key],
                payload["intrinsics"],
                poses,
                sample_frames=sample_frames,
            )
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected Nx3 point cloud, got {points.shape}")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError("point cloud has no finite points")
    return points


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).round().astype(np.int64)
    return points[indices]


def _nearest_distances(source: np.ndarray, target: np.ndarray, chunk_size: int) -> np.ndarray:
    nearest = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        diff = source[start:stop, None, :] - target[None, :, :]
        nearest[start:stop] = np.linalg.norm(diff, axis=2).min(axis=1)
    return nearest


def _symmetric_nearest_error(reference: np.ndarray, candidate: np.ndarray, *, chunk_size: int) -> float:
    ref_to_candidate = _nearest_distances(reference, candidate, chunk_size)
    candidate_to_ref = _nearest_distances(candidate, reference, chunk_size)
    return float((ref_to_candidate.mean() + candidate_to_ref.mean()) * 0.5)


def score_pointcloud_geometry(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    *,
    error_scale: float,
    max_points: int,
    chunk_size: int,
) -> dict[str, Any]:
    reference = np.asarray(reference_points, dtype=np.float64)
    candidate = np.asarray(candidate_points, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 3 or candidate.ndim != 2 or candidate.shape[1] != 3:
        return {"geometry_status": "invalid_pointcloud"}
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    candidate = candidate[np.all(np.isfinite(candidate), axis=1)]
    if len(reference) == 0 or len(candidate) == 0:
        return {"geometry_status": "invalid_pointcloud"}
    max_points = int(max_points)
    chunk_size = int(chunk_size)
    if max_points <= 0 or chunk_size <= 0:
        raise ValueError("max_points and chunk_size must be positive")
    reference = _downsample_points(reference, max_points)
    candidate = _downsample_points(candidate, max_points)
    error = _symmetric_nearest_error(reference, candidate, chunk_size=chunk_size)
    if not math.isfinite(error):
        return {"geometry_status": "invalid_pointcloud"}
    scale = max(float(error_scale), 1e-8)
    reward = math.exp(-error / scale)
    return {
        "geometry_status": "ok",
        "geometry_error": error,
        "geometry_reward": max(0.0, min(1.0, reward)),
    }


def _cached_npz_path(row: dict[str, Any], config: dict[str, Any]) -> Path:
    cache_root = Path(_require_str(config, "cache_root"))
    return cache_root / "geometry" / f"{safe_video_id(row.get('video_id'))}.npz"


def _run_extractor(row: dict[str, Any], output_npz: Path, config: dict[str, Any]) -> str | None:
    template = _require_str(config, "extractor_command")
    video_path = _video_path(row)
    if not video_path:
        return "missing_video"
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    command = template.format(
        video_path=shlex.quote(str(video_path)),
        video_id=shlex.quote(str(row.get("video_id"))),
        output_npz=shlex.quote(str(output_npz)),
        gen3c_code_dir=shlex.quote(_require_str(config, "gen3c_code_dir")),
        moge_checkpoint=shlex.quote(_require_str(config, "moge_checkpoint")),
        sample_frames=shlex.quote(str(_require_config_value(config, "sample_frames"))),
        device=shlex.quote(_require_str(config, "device")),
    )
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=_require_config_value(config, "timeout"),
        )
    except FileNotFoundError:
        return "extractor_unavailable"
    except subprocess.TimeoutExpired:
        return "extractor_failed"
    if result.returncode != 0:
        return "extractor_failed"
    if not output_npz.exists():
        return "missing_pointcloud"
    return None


def _resolve_pointcloud_path(row: dict[str, Any], config: dict[str, Any]) -> tuple[Path | None, str | None]:
    for field in POINTCLOUD_FIELDS:
        value = row.get(field)
        if value is None or (isinstance(value, str) and value == ""):
            continue
        try:
            path = Path(value)
            exists = path.exists()
        except (TypeError, ValueError, OSError):
            return None, "invalid_pointcloud"
        if exists:
            return path, None
        return None, "missing_pointcloud"
    cache_path = _cached_npz_path(row, config)
    if cache_path.exists():
        return cache_path, None
    status = _run_extractor(row, cache_path, config)
    if status:
        return None, status
    return cache_path, None


def _score_row_against_reference(
    row: dict[str, Any],
    reference_points: np.ndarray,
    static_gt_video_id: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(row)
    enriched["static_gt_video_id"] = static_gt_video_id
    npz_path, status = _resolve_pointcloud_path(row, config)
    if status:
        return _with_geometry_status(row, status, static_gt_video_id=static_gt_video_id)
    try:
        candidate_points = load_point_cloud_npz(
            npz_path,
            sample_frames=_require_config_value(config, "sample_frames"),
        )
        score = score_pointcloud_geometry(
            reference_points,
            candidate_points,
            error_scale=_require_float(config, "error_scale"),
            max_points=_require_positive_int(config, "max_points"),
            chunk_size=_require_positive_int(config, "chunk_size"),
        )
    except ValueError:
        return _with_geometry_status(row, "invalid_pointcloud", static_gt_video_id=static_gt_video_id)
    except Exception:
        return _with_geometry_status(row, "error", static_gt_video_id=static_gt_video_id)
    enriched.update(score)
    if score.get("geometry_status") == "ok":
        components = dict(enriched.get("reward_components") or {})
        components["geometry"] = score["geometry_reward"]
        enriched["reward_components"] = components
    return enriched


def score_geometry_group(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Score a prompt group against its static camera reference row."""
    cfg = dict(_require_config(config))
    if not _is_enabled(cfg):
        return [_with_geometry_status(row, "disabled") for row in rows]
    reference = next((row for row in rows if _is_static_row(row)), None)
    if reference is None:
        return [_with_geometry_status(row, "no_static_gt", static_gt_video_id=None) for row in rows]
    reference_id = reference.get("video_id")
    ref_path, ref_status = _resolve_pointcloud_path(reference, cfg)
    if ref_status:
        return [_with_geometry_status(row, ref_status, static_gt_video_id=reference_id) for row in rows]
    try:
        reference_points = load_point_cloud_npz(
            ref_path,
            sample_frames=_require_config_value(cfg, "sample_frames"),
        )
    except ValueError:
        return [_with_geometry_status(row, "invalid_pointcloud", static_gt_video_id=reference_id) for row in rows]
    except Exception:
        return [_with_geometry_status(row, "error", static_gt_video_id=reference_id) for row in rows]
    return [_score_row_against_reference(row, reference_points, reference_id, cfg) for row in rows]
