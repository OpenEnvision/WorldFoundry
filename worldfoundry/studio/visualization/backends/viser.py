"""Hosts a localhost Viser viewer for Studio point-cloud inspection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import socket
import threading
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

POINT_CLOUD_NODE = "worldfoundry_studio_point_cloud"
AUXILIARY_MESH_NODE = "worldfoundry_studio_auxiliary_mesh"
CAMERA_NODE_ROOT = "worldfoundry_studio_cameras"
DEFAULT_MAX_POINTS = 400_000
DEFAULT_POINT_SIZE = 0.02
DEFAULT_CAMERA_SIZE = 0.08
DEFAULT_POINT_SHAPE = "circle"
DEFAULT_VISER_PORT_BASE = 18590
DEFAULT_VISER_PORT_COUNT = 8
VISER_POINT_SHAPES = frozenset({"square", "diamond", "circle", "rounded", "sparkle"})
VISER_COORDINATE_PRESETS = frozenset({"asset-native", "opencv-colmap", "opencv-to-opengl"})
VISER_ALIGNMENT_PRESETS = frozenset({"auto", "none", "first-camera", "first-camera-opengl"})


@dataclass(frozen=True)
class ViserPresentation:
    """Iframe markup plus a short human caption for embedded viewers."""

    html: str
    caption: str
    url: str = ""


def viser_importable() -> bool:
    """Return True when optional ``viser`` is installed."""

    return importlib.util.find_spec("viser") is not None


def _pick_free_port(host: str = "127.0.0.1") -> int:
    """Reserve an ephemeral TCP port on ``host``."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port = int(sock.getsockname()[1])
    return port


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Parse a positive integer env var with a bounded fallback."""

    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError:
        value = default
    return max(value, minimum)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Parse a finite positive float env var with a bounded fallback."""

    import math

    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        value = default
    if not math.isfinite(value):
        value = default
    return max(value, minimum)


def _env_choice(name: str, default: str, choices: frozenset[str]) -> str:
    """Parse an env var constrained to a known choice set."""

    value = (os.getenv(name) or default).strip().lower()
    return value if value in choices else default


def _env_bool(name: str, default: bool) -> bool:
    """Parse a common boolean env var."""

    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_coordinate_preset(value: str | None) -> str:
    """Normalize user-facing coordinate preset aliases."""

    preset = (value or "asset-native").strip().lower().replace("_", "-")
    aliases = {
        "native": "asset-native",
        "asset": "asset-native",
        "opencv": "opencv-colmap",
        "colmap": "opencv-colmap",
        "opencv/opengl": "opencv-to-opengl",
        "opengl": "opencv-to-opengl",
        "vggt": "opencv-to-opengl",
    }
    preset = aliases.get(preset, preset)
    return preset if preset in VISER_COORDINATE_PRESETS else "asset-native"


def _normalize_alignment_preset(value: str | None) -> str:
    """Normalize user-facing alignment preset aliases."""

    preset = (value or "auto").strip().lower().replace("_", "-")
    aliases = {
        "off": "none",
        "disabled": "none",
        "native": "none",
        "canonical": "first-camera",
        "canonical-first-camera": "first-camera",
        "first": "first-camera",
        "first-camera-frame": "first-camera",
        "vggt": "first-camera-opengl",
        "opengl": "first-camera-opengl",
    }
    preset = aliases.get(preset, preset)
    return preset if preset in VISER_ALIGNMENT_PRESETS else "auto"


def _default_up_direction_for_preset(preset: str) -> str:
    """Return a Viser up direction matching the selected coordinate preset."""

    if preset == "opencv-colmap":
        return "-y"
    if preset == "opencv-to-opengl":
        return "+y"
    return "+z"


def viser_orientation_defaults(model_id: str, geometry_path: str | Path) -> dict[str, str]:
    """Return verified model/asset orientation defaults for every Viser entrypoint."""

    defaults = {
        "coordinate_preset": "asset-native",
        "up_direction": "+z",
        "alignment": "auto",
    }
    if model_id in {"vggt", "vggt-omega", "depth-anything-v3"}:
        defaults.update(
            coordinate_preset=(
                "opencv-to-opengl" if Path(geometry_path).suffix.lower() == ".npz" else "asset-native"
            ),
            up_direction="+y",
            alignment="none",
        )
    elif model_id in {"unidepth-v2-prior", "unik3d-prior"}:
        defaults.update(
            coordinate_preset=(
                "opencv-to-opengl" if Path(geometry_path).suffix.lower() == ".npz" else "asset-native"
            ),
            up_direction="+y",
            alignment="none",
        )
    elif model_id in {"dust3r", "dust3r-base-model", "loger", "pi3"}:
        defaults.update(
            coordinate_preset="asset-native",
            up_direction="+y",
            alignment="first-camera-opengl",
        )
    return defaults


def _parse_up_direction(value: str | None, *, default: str) -> str | tuple[float, float, float]:
    """Parse a Viser up-direction string or comma/space-separated vector."""

    text = (value or default).strip().lower()
    if text in {"+x", "+y", "+z", "-x", "-y", "-z"}:
        return text
    parts = [part for part in text.replace(",", " ").split() if part]
    if len(parts) != 3:
        return default
    try:
        vector = tuple(float(part) for part in parts)
    except ValueError:
        return default
    if sum(abs(part) for part in vector) == 0:
        return default
    return vector


def _coordinate_transform_matrix(preset: str) -> Any | None:
    """Return a 4x4 transform matrix for the selected coordinate preset."""

    import numpy as np

    if preset == "opencv-to-opengl":
        matrix = np.eye(4)
        matrix[1, 1] = -1.0
        matrix[2, 2] = -1.0
        return matrix
    return None


def _transform_points_for_preset(points: Any, preset: str) -> Any:
    """Apply the selected coordinate transform to Nx3 points."""

    matrix = _coordinate_transform_matrix(preset)
    return _transform_points(points, matrix)


def _apply_mesh_transform_for_preset(mesh: Any, preset: str) -> None:
    """Apply the selected coordinate transform to a Trimesh object in-place."""

    matrix = _coordinate_transform_matrix(preset)
    _apply_mesh_transform(mesh, matrix)


def _transform_points(points: Any, matrix: Any | None) -> Any:
    """Apply a 4x4 transform to Nx3 points."""

    if matrix is None:
        return points
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _apply_mesh_transform(mesh: Any, matrix: Any | None) -> None:
    """Apply an arbitrary 4x4 transform to a Trimesh object in-place."""

    if matrix is not None:
        mesh.apply_transform(matrix)


def _as_4x4_pose(value: Any) -> Any | None:
    """Return a homogeneous 4x4 pose matrix when ``value`` is pose-like."""

    import numpy as np

    pose = np.asarray(value, dtype=np.float64)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = pose
        return out
    return None


def _load_camera_poses(geometry_path: Path) -> list[Any]:
    """Load camera-to-world sidecar poses for a Studio run when available."""

    import numpy as np

    for root in (geometry_path.parent, geometry_path.parent.parent):
        stem = geometry_path.stem
        for suffix in ("_point_cloud", "_points", "_cloud"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        candidates = [
            root / "raw_data" / "camera_poses.npy",
            root / "camera_poses.npy",
            root / f"{stem}_camera_poses.npy",
        ]
        named_sidecars = sorted(root.glob("*_camera_poses.npy"))
        if len(named_sidecars) == 1:
            candidates.extend(named_sidecars)
        for path in dict.fromkeys(candidates):
            if not path.exists():
                continue
            try:
                arr = np.load(str(path), allow_pickle=False)
            except Exception:
                continue
            arr = np.asarray(arr)
            if arr.ndim >= 3 and arr.shape[-2:] in {(3, 4), (4, 4)}:
                arr = arr.reshape(-1, *arr.shape[-2:])
                poses = [_as_4x4_pose(pose) for pose in arr]
                return [pose for pose in poses if pose is not None]

    for pose_dir in (geometry_path.parent / "camera_poses", geometry_path.parent.parent / "camera_poses"):
        if not pose_dir.is_dir():
            continue
        poses = []
        for path in sorted(pose_dir.glob("pose_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pose_value = payload.get("camera_to_world") or payload.get("c2w") or payload.get("pose")
            pose = _as_4x4_pose(pose_value)
            if pose is not None:
                poses.append(pose)
        if poses:
            return poses
    return []


def _alignment_transform_matrix(poses: list[Any], preset: str) -> Any | None:
    """Return an optional canonical scene alignment transform."""

    import numpy as np

    if preset == "none":
        return None
    if preset == "auto":
        preset = "first-camera" if poses else "none"
    if preset not in {"first-camera", "first-camera-opengl"} or not poses:
        return None

    transform = np.linalg.inv(poses[0])
    if preset == "first-camera-opengl":
        opengl = _coordinate_transform_matrix("opencv-to-opengl")
        if opengl is not None:
            transform = opengl @ transform
    return transform


def _compose_transforms(*matrices: Any | None) -> Any | None:
    """Compose 4x4 scene transforms in the same order they are applied to points."""

    import numpy as np

    transform = np.eye(4, dtype=np.float64)
    used = False
    for matrix in matrices:
        if matrix is None:
            continue
        transform = matrix @ transform
        used = True
    return transform if used else None


def _transform_camera_poses(poses: list[Any], transform: Any | None) -> list[Any]:
    """Apply a scene transform to camera-to-world poses."""

    if transform is None:
        return poses
    return [transform @ pose for pose in poses]


def _stable_port_offset(run_id: str, count: int) -> int:
    """Return a deterministic pool offset for ``run_id``."""

    digest = hashlib.sha256(run_id.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big") % count


def _port_is_free(host: str, port: int) -> bool:
    """Return True when a TCP port can be bound on ``host``."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError:
        return False
    return True


def _pick_pool_port(run_id: str, *, host: str = "127.0.0.1") -> int:
    """Pick a Viser port from the configured stable pool."""

    base = _env_int("WORLDFOUNDRY_STUDIO_VISER_PORT_BASE", DEFAULT_VISER_PORT_BASE)
    count = _env_int("WORLDFOUNDRY_STUDIO_VISER_PORT_COUNT", DEFAULT_VISER_PORT_COUNT)
    start = _stable_port_offset(run_id or "studio", count)
    for step in range(count):
        port = base + ((start + step) % count)
        if _port_is_free(host, port):
            return port
    last = base + count - 1
    raise OSError(f"No free Viser ports in pool {base}-{last}; increase WORLDFOUNDRY_STUDIO_VISER_PORT_COUNT.")


def _resolve_viser_port(run_id: str, *, host: str, requested_port: int | None) -> int:
    """Honor explicit ports, otherwise allocate from the stable Viser pool."""

    if requested_port is not None:
        return int(requested_port)
    return _pick_pool_port(run_id, host=host)


def _geometry_imports_available() -> bool:
    """Return True when point-cloud parsing dependencies are installed."""

    return importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("trimesh") is not None


def _scene_geometries(scene: Any) -> list[Any]:
    """Return scene geometries with node transforms applied when possible."""

    dumped = scene.dump(concatenate=False)
    if isinstance(dumped, (list, tuple)):
        return list(dumped)
    return [dumped]


def _geometry_xyz_rgb(geometry: Any) -> tuple[Any, Any | None] | None:
    """Extract vertex positions and optional vertex colors from a Trimesh geometry."""

    import numpy as np

    vertices = getattr(geometry, "vertices", None)
    if vertices is None:
        vertices = getattr(geometry, "points", None)
    if vertices is None:
        return None

    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None
    points = points[:, :3]

    colors = getattr(geometry, "colors", None)
    if colors is None:
        visual = getattr(geometry, "visual", None)
        colors = getattr(visual, "vertex_colors", None) if visual is not None else None
    if colors is None:
        return points, None

    colors_array = np.asarray(colors)
    if colors_array.ndim != 2 or colors_array.shape[0] != points.shape[0] or colors_array.shape[1] < 3:
        return points, None
    return points, colors_array[:, :3]


def _coerce_rgb(colors: Any, row_count: int) -> Any:
    """Normalize optional color arrays to uint8 RGB triplets."""

    import numpy as np

    if colors is None or colors.shape[0] != row_count:
        return np.full((row_count, 3), 200, dtype=np.uint8)
    if colors.dtype != np.dtype("uint8"):
        scale = 255.0 if np.nanmax(colors) <= 1.0 else 1.0
        colors = np.clip(colors * scale, 0, 255).astype(np.uint8)
    return colors


_NPZ_POINT_KEYS = (
    "xyz",
    "points",
    "world_points",
    "world_points_from_depth",
    "points3d",
    "vertices",
)
_NPZ_IMAGE_KEYS = ("rgb", "colors", "color", "images", "input_images", "image", "imgs")


def _npz_array_has_xyz_shape(array: Any) -> bool:
    """Return True when an NPZ array can be flattened to XYZ rows."""

    import numpy as np

    arr = np.asarray(array)
    return arr.ndim >= 2 and arr.shape[-1] >= 3


def npz_has_supported_geometry(path: Path) -> bool:
    """Return True when an NPZ file contains geometry Viser can interpret."""

    import numpy as np

    try:
        with np.load(str(path), allow_pickle=False) as data:
            keys = set(data.files)
            if any(key in keys and _npz_array_has_xyz_shape(data[key]) for key in _NPZ_POINT_KEYS):
                return True
            return "depth" in keys and bool(keys & {"intrinsic", "intrinsics"})
    except Exception:
        return False


def _points_from_npz_arrays(data: Any) -> Any | None:
    """Load explicit XYZ-like arrays from known NPZ keys."""

    extracted = _points_and_finite_mask_from_npz_arrays(data)
    return extracted[0] if extracted is not None else None


def _points_and_finite_mask_from_npz_arrays(data: Any) -> tuple[Any, Any] | None:
    """Load XYZ rows and retain the source-row mask for aligned color filtering."""

    import numpy as np

    for key in _NPZ_POINT_KEYS:
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float64)
        if not _npz_array_has_xyz_shape(arr):
            continue
        points = arr.reshape(-1, arr.shape[-1])[:, :3]
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.shape[0] > 0:
            return points, finite
    return None


def _depth_points_from_npz(data: Any) -> Any | None:
    """Project depth plus camera calibration NPZ records into world-space XYZ rows.

    The conventional ``extrinsic(s)`` keys are world-to-camera matrices.  NPZ
    producers that already expose camera-to-world poses can opt out of that
    inversion with an explicit ``camera_to_world`` or ``c2w`` key.
    """

    import numpy as np

    if "depth" not in data:
        return None
    intrinsic_key = "intrinsics" if "intrinsics" in data else "intrinsic" if "intrinsic" in data else ""
    if not intrinsic_key:
        return None

    depth = np.asarray(data["depth"], dtype=np.float64)
    if depth.ndim >= 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 2:
        depth = depth[None, ...]
    if depth.ndim != 3:
        return None

    intrinsics = np.asarray(data[intrinsic_key], dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None, ...]
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        return None

    c2w_key = next(
        (key for key in ("camera_to_world", "c2w", "camera_to_worlds", "c2ws") if key in data),
        "",
    )
    w2c_key = "extrinsics" if "extrinsics" in data else "extrinsic" if "extrinsic" in data else ""
    pose_key = c2w_key or w2c_key
    camera_to_world = np.asarray(data[pose_key], dtype=np.float64) if pose_key else None
    if camera_to_world is not None and camera_to_world.ndim == 2:
        camera_to_world = camera_to_world[None, ...]

    segments = []
    for frame_idx in range(depth.shape[0]):
        z = depth[frame_idx]
        valid = np.isfinite(z) & (z > 0)
        if not valid.any():
            continue
        intr = intrinsics[min(frame_idx, intrinsics.shape[0] - 1)]
        fy = intr[1, 1] if abs(intr[1, 1]) > 1e-8 else 1.0
        fx = intr[0, 0] if abs(intr[0, 0]) > 1e-8 else fy
        cx = intr[0, 2]
        cy = intr[1, 2]
        yy, xx = np.indices(z.shape)
        z_valid = z[valid]
        camera_points = np.stack(
            [
                (xx[valid] - cx) * z_valid / fx,
                (yy[valid] - cy) * z_valid / fy,
                z_valid,
            ],
            axis=1,
        )
        if camera_to_world is not None and camera_to_world.ndim == 3:
            pose = _as_4x4_pose(camera_to_world[min(frame_idx, camera_to_world.shape[0] - 1)])
            if pose is not None:
                if w2c_key and not c2w_key:
                    try:
                        pose = np.linalg.inv(pose)
                    except np.linalg.LinAlgError:
                        pose = None
                if pose is not None:
                    camera_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
        segments.append(camera_points)

    if not segments:
        return None
    return np.concatenate(segments, axis=0)


def _colors_from_npz(data: Any, row_count: int) -> Any | None:
    """Return flattened RGB rows from known NPZ image/color arrays."""

    import numpy as np

    for key in _NPZ_IMAGE_KEYS:
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[-1] >= 3:
            colors = arr[:, :3]
        elif arr.ndim >= 4 and arr.shape[-3] == 3:
            colors = np.moveaxis(arr, -3, -1).reshape(-1, 3)
        elif arr.ndim >= 3 and arr.shape[-1] >= 3:
            colors = arr.reshape(-1, arr.shape[-1])[:, :3]
        else:
            continue
        if colors.shape[0] == row_count:
            return colors
    return None


def _load_xyz_rgb(path: Path) -> tuple[Any, Any]:
    """Load XYZ positions and uint8 RGB triplets from supported geometry files."""

    import numpy as np
    import trimesh

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(str(path), allow_pickle=False) as data:
            explicit = _points_and_finite_mask_from_npz_arrays(data)
            if explicit is None:
                points = _depth_points_from_npz(data)
                colors = _colors_from_npz(data, points.shape[0]) if points is not None else None
            else:
                points, finite = explicit
                colors = _colors_from_npz(data, finite.shape[0])
                if colors is not None:
                    colors = colors[finite]
            if points is None:
                raise ValueError(f"{path.name} does not contain supported geometry NPZ keys.")
    else:
        loaded: Any = trimesh.load(str(path), process=False, validate=False)
        geometries = _scene_geometries(loaded) if isinstance(loaded, trimesh.Scene) else [loaded]
        point_geometries = [geom for geom in geometries if isinstance(geom, trimesh.points.PointCloud)]
        geometries = point_geometries or geometries

        point_segments = []
        color_segments = []
        for geometry in geometries:
            extracted = _geometry_xyz_rgb(geometry)
            if extracted is None:
                continue
            segment_points, segment_colors = extracted
            point_segments.append(segment_points)
            color_segments.append(_coerce_rgb(segment_colors, segment_points.shape[0]))

        if not point_segments:
            raise ValueError(f"{path.name} exposes zero vertices.")
        points = np.concatenate(point_segments, axis=0)
        colors = np.concatenate(color_segments, axis=0)

    row_count = points.shape[0]
    if row_count == 0:
        raise ValueError(f"{path.name} exposes zero vertices.")

    colors = _coerce_rgb(colors, row_count)

    return points.astype(np.float32), colors


def _load_mesh(path: Path) -> Any | None:
    """Load a mesh-like asset for Viser when faces are available."""

    import numpy as np
    import trimesh

    if path.suffix.lower() == ".npz":
        return None
    loaded: Any = trimesh.load(str(path), process=False, validate=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = _scene_geometries(loaded)
        geometries = [geom for geom in geometries if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            return None
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        return None
    faces = np.asarray(getattr(loaded, "faces", None))
    vertices = np.asarray(getattr(loaded, "vertices", None))
    if faces.size == 0 or vertices.size == 0:
        return None
    return loaded


def _has_explicit_point_cloud(path: Path) -> bool:
    """Return whether a geometry asset contains a dedicated point-cloud layer."""

    import trimesh

    if path.suffix.lower() in {".pcd", ".xyz", ".npz"}:
        return True
    loaded: Any = trimesh.load(str(path), process=False, validate=False)
    geometries = _scene_geometries(loaded) if isinstance(loaded, trimesh.Scene) else [loaded]
    return any(isinstance(geometry, trimesh.points.PointCloud) for geometry in geometries)


def _subsample(points: Any, colors: Any, limit: int) -> tuple[Any, Any]:
    """Uniformly subsample gigantic clouds for realtime Viser meshes."""

    import numpy as np

    if points.shape[0] <= limit:
        return points, colors
    rng = np.random.default_rng(seed=1337)
    idx = rng.choice(points.shape[0], size=limit, replace=False)
    return points[idx], colors[idx]


def _adaptive_point_size(points: Any) -> float:
    """Choose a scale-aware world-space radius for dense reconstruction points."""

    import numpy as np

    vertices = np.asarray(points, dtype=np.float64)
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if vertices.shape[0] == 0:
        return DEFAULT_POINT_SIZE
    if vertices.shape[0] >= 100:
        low, high = np.percentile(vertices, (5.0, 95.0), axis=0)
    else:
        low, high = vertices.min(axis=0), vertices.max(axis=0)
    return max(float(np.linalg.norm(high - low)) / 400.0, 0.00001)


def _initial_camera_pose(points: Any | None, mesh: Any | None, up_direction: Any) -> tuple[Any, Any, Any]:
    """Fit a readable initial camera to the primary finite geometry."""

    import numpy as np

    vertices = points if points is not None else getattr(mesh, "vertices", None)
    if vertices is None:
        return np.array([3.0, 3.0, 3.0]), np.zeros(3), np.array([0.0, 0.0, 1.0])
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return np.array([3.0, 3.0, 3.0]), np.zeros(3), np.array([0.0, 0.0, 1.0])
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if vertices.shape[0] == 0:
        return np.array([3.0, 3.0, 3.0]), np.zeros(3), np.array([0.0, 0.0, 1.0])
    if vertices.shape[0] >= 100:
        low, high = np.percentile(vertices, (5.0, 95.0), axis=0)
    else:
        low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float(np.linalg.norm(high - low) * 0.5), 1e-3)

    if isinstance(up_direction, str):
        sign = -1.0 if up_direction.startswith("-") else 1.0
        axis = {"x": 0, "y": 1, "z": 2}.get(up_direction[-1:], 2)
        up = np.zeros(3)
        up[axis] = sign
    else:
        up = np.asarray(up_direction, dtype=np.float64)
        up /= max(float(np.linalg.norm(up)), 1e-8)
    reference = np.array([0.0, 0.0, 1.0]) if abs(float(up[2])) < 0.9 else np.array([0.0, 1.0, 0.0])
    right = np.cross(reference, up)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    forward = np.cross(up, right)
    direction = forward + right + 0.65 * up
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    distance = radius * 2.15
    return center + direction * distance, center, up


def _add_camera_frustums(server: Any, poses: list[Any], *, scale: float) -> int:
    """Add lightweight sidecar camera frustums to a Viser scene."""

    import numpy as np
    import viser.transforms as vt

    if not poses:
        return 0
    server.scene.add_frame(f"/{CAMERA_NODE_ROOT}", show_axes=False)
    count = 0
    denom = max(len(poses) - 1, 1)
    for idx, pose in enumerate(poses):
        T_cam = vt.SE3.from_matrix(pose)
        hue = idx / denom
        color = (
            int(255 * hue),
            int(180 * (1.0 - hue)),
            int(255 * (1.0 - 0.5 * hue)),
        )
        server.scene.add_frame(
            f"/{CAMERA_NODE_ROOT}/frame_{idx:04d}",
            wxyz=T_cam.rotation().wxyz,
            position=T_cam.translation(),
            axes_length=scale * 0.7,
            axes_radius=scale * 0.025,
            origin_radius=scale * 0.04,
        )
        server.scene.add_camera_frustum(
            f"/{CAMERA_NODE_ROOT}/frame_{idx:04d}/frustum",
            fov=np.deg2rad(60.0),
            aspect=16.0 / 9.0,
            scale=scale,
            color=color,
            line_width=2.0,
        )
        count += 1
    return count


class StudioViserService:
    """Serializes Viser launches so each run replaces the hosted cloud safely."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server = None

    def present_point_cloud(
        self,
        *,
        run_id: str,
        cloud_path: Path,
        model_id: str = "",
    ) -> ViserPresentation:
        """Boot or refresh Viser with ``cloud_path`` and return iframe markup."""

        return self.present_geometry(run_id=run_id, geometry_path=cloud_path, model_id=model_id)

    def present_geometry(
        self,
        *,
        run_id: str,
        geometry_path: Path,
        model_id: str = "",
        host: str | None = None,
        port: int | None = None,
        embed: bool = True,
    ) -> ViserPresentation:
        """Boot or refresh Viser with a point cloud or mesh asset."""

        _ = run_id
        if not viser_importable():
            return ViserPresentation(
                html=_fallback_card("Viser is not installed", "pip install viser or worldfoundry[studio_pointcloud]."),
                caption="Viser optional dependency missing.",
            )
        if not geometry_path.exists():
            return ViserPresentation(
                html=_fallback_card("Geometry missing", escape(geometry_path.name)),
                caption="Expected geometry file is not on disk.",
            )
        if not _geometry_imports_available():
            return ViserPresentation(
                html=_fallback_card(
                    "Point-cloud dependencies are not installed",
                    "pip install worldfoundry[studio_pointcloud].",
                ),
                caption="Point-cloud parsing dependencies missing.",
            )
        mesh = None
        points = None
        colors = None
        try:
            host = (
                host
                or os.getenv("WORLDFOUNDRY_STUDIO_VISER_HOST")
                or "127.0.0.1"
            ).strip() or "127.0.0.1"
            if host not in {"127.0.0.1", "localhost"}:
                return ViserPresentation(
                    html=_fallback_card("Viser host rejected", "Only loopback hosts are permitted."),
                    caption="Refusing non-loopback Viser bind.",
                )
            limit = int(
                os.getenv("WORLDFOUNDRY_STUDIO_VISER_MAX_POINTS")
                or str(DEFAULT_MAX_POINTS)
            )
            limit = max(limit, 1024)
            point_size_override = os.getenv("WORLDFOUNDRY_STUDIO_VISER_POINT_SIZE")
            point_size = (
                _env_float("WORLDFOUNDRY_STUDIO_VISER_POINT_SIZE", DEFAULT_POINT_SIZE, minimum=0.000001)
                if point_size_override is not None
                else None
            )
            point_shape = _env_choice("WORLDFOUNDRY_STUDIO_VISER_POINT_SHAPE", DEFAULT_POINT_SHAPE, VISER_POINT_SHAPES)
            orientation_defaults = viser_orientation_defaults(model_id, geometry_path)
            coordinate_preset = _normalize_coordinate_preset(
                os.getenv("WORLDFOUNDRY_STUDIO_VISER_COORDINATE_PRESET")
                or orientation_defaults["coordinate_preset"]
            )
            alignment_preset = _normalize_alignment_preset(
                os.getenv("WORLDFOUNDRY_STUDIO_VISER_ALIGNMENT")
                or orientation_defaults["alignment"]
            )
            camera_size = _env_float("WORLDFOUNDRY_STUDIO_VISER_CAMERA_SIZE", DEFAULT_CAMERA_SIZE, minimum=0.000001)
            camera_poses = _load_camera_poses(geometry_path)
            effective_alignment = (
                "first-camera" if alignment_preset == "auto" and camera_poses else
                "none" if alignment_preset == "auto" else
                alignment_preset
            )
            show_cameras = _env_bool("WORLDFOUNDRY_STUDIO_VISER_SHOW_CAMERAS", bool(camera_poses))
            alignment_matrix = _alignment_transform_matrix(camera_poses, alignment_preset)
            full_transform = _compose_transforms(alignment_matrix, _coordinate_transform_matrix(coordinate_preset))
            transformed_camera_poses = _transform_camera_poses(camera_poses, full_transform)
            up_direction = _parse_up_direction(
                os.getenv("WORLDFOUNDRY_STUDIO_VISER_UP_DIRECTION"),
                default=orientation_defaults.get(
                    "up_direction", _default_up_direction_for_preset(coordinate_preset)
                ),
            )
            has_point_cloud = _has_explicit_point_cloud(geometry_path)
            if geometry_path.suffix.lower() not in {".pcd", ".xyz", ".npz"}:
                try:
                    mesh = _load_mesh(geometry_path)
                except Exception:
                    mesh = None
                if mesh is not None:
                    _apply_mesh_transform(mesh, full_transform)
            if mesh is None or has_point_cloud:
                points, colors = _load_xyz_rgb(geometry_path)
                points = _transform_points(points, full_transform)
                points, colors = _subsample(points, colors, limit)
                if point_size is None:
                    point_size = _adaptive_point_size(points)
        except Exception as exc:
            return ViserPresentation(
                html=_fallback_card("Geometry load failed", escape(str(exc))),
                caption="Could not interpret the geometry asset.",
            )

        import viser

        with self._lock:
            if self._server is not None:
                self._server.stop()
                self._server = None
            port = _resolve_viser_port(run_id, host=host, requested_port=port)
            server = viser.ViserServer(host=host, port=port, verbose=False)
            port = int(server.get_port())
            server.scene.set_up_direction(up_direction)
            camera_position, camera_target, camera_up = _initial_camera_pose(points, mesh, up_direction)
            server.initial_camera.position = camera_position
            server.initial_camera.look_at = camera_target
            server.initial_camera.up = camera_up
            server.initial_camera.fov = math.radians(60.0)
            server.scene.remove_by_name(POINT_CLOUD_NODE)
            server.scene.remove_by_name(AUXILIARY_MESH_NODE)
            server.scene.remove_by_name(f"/{CAMERA_NODE_ROOT}")
            geometry_parts = []
            if points is not None and colors is not None:
                assert points is not None and colors is not None
                assert point_size is not None
                server.scene.add_point_cloud(
                    POINT_CLOUD_NODE,
                    points,
                    colors,
                    point_size=point_size,
                    point_shape=point_shape,
                )
                geometry_parts.append(f"{points.shape[0]:,} points · point_size={point_size:g} · {point_shape}")
            if mesh is not None:
                mesh_node = AUXILIARY_MESH_NODE if geometry_parts else POINT_CLOUD_NODE
                server.scene.add_mesh_trimesh(mesh_node, mesh=mesh)
                geometry_parts.append(f"{len(mesh.vertices):,} mesh verts · {len(mesh.faces):,} faces")
            geometry_caption = " · ".join(geometry_parts)
            camera_caption = "cameras=off"
            if show_cameras:
                camera_count = _add_camera_frustums(server, transformed_camera_poses, scale=camera_size)
                camera_caption = f"cameras={camera_count} · camera_size={camera_size:g}"
            self._server = server

        url = f"http://{host}:{port}/"
        orientation_caption = (
            f"coords={coordinate_preset} · alignment={effective_alignment} · "
            f"up={up_direction} · {camera_caption}"
        )
        if not embed:
            return ViserPresentation(
                html=url,
                caption=f"Viser ready · {geometry_path.name} · {geometry_caption} · {orientation_caption}",
                url=url,
            )
        iframe = (
            f'<section class="wa-points-viewport" data-wa-viser="1">'
            f'<iframe class="wa-viser-frame" title="Viser point cloud" src="{escape(url)}" '
            f'sandbox="allow-scripts allow-same-origin allow-forms" loading="lazy" '
            f'referrerpolicy="no-referrer"></iframe>'
            f'<div class="wa-points-caption">Loopback Viser · {escape(geometry_path.name)} · '
            f"{geometry_caption} · {orientation_caption}</div>"
            f"</section>"
        )
        return ViserPresentation(
            html=iframe,
            caption=f"Viser ready · {geometry_path.name} · {geometry_caption} · {orientation_caption}",
            url=url,
        )

    def shutdown(self) -> None:
        """Stop any active Viser server (used on model unload)."""

        with self._lock:
            if self._server is None:
                return
            self._server.stop()
            self._server = None


def _fallback_card(title: str, detail: str) -> str:
    """Render a lightweight status card when Viser cannot start."""

    return (
        f'<section class="wa-points-viewport wa-points-viewport--idle">'
        f'<div class="wa-points-fallback-title">{escape(title)}</div>'
        f'<div class="wa-points-fallback-detail">{detail}</div>'
        f"</section>"
    )


STUDIO_VISER = StudioViserService()
