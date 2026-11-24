"""Hosts a localhost Viser viewer for Studio point-cloud inspection."""

from __future__ import annotations

import contextlib
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
from typing import Any, Literal, TypeAlias, cast, get_args

from worldfoundry.studio.serving import is_loopback_host

ViserPointShape: TypeAlias = Literal["square", "diamond", "circle", "rounded", "sparkle"]
ViserUpAxis: TypeAlias = Literal["+x", "+y", "+z", "-x", "-y", "-z"]
ViserUpDirection: TypeAlias = ViserUpAxis | tuple[float, float, float]

POINT_CLOUD_NODE = "worldfoundry_studio_point_cloud"
AUXILIARY_MESH_NODE = "worldfoundry_studio_auxiliary_mesh"
CAMERA_NODE_ROOT = "worldfoundry_studio_cameras"
DEFAULT_MAX_POINTS = 400_000
DEFAULT_POINT_SIZE = 0.02
DEFAULT_CAMERA_SIZE = 0.08
DEFAULT_POINT_SHAPE: ViserPointShape = "circle"
DEFAULT_VISER_PORT_BASE = 18590
DEFAULT_VISER_PORT_COUNT = 8
VISER_POINT_SHAPES = frozenset(get_args(ViserPointShape))
VISER_COORDINATE_PRESETS = frozenset({"asset-native", "opencv-colmap", "opencv-to-opengl"})
VISER_ALIGNMENT_PRESETS = frozenset({"auto", "none", "first-camera", "first-camera-opengl"})
_VISER_UP_AXES = frozenset(get_args(ViserUpAxis))


@dataclass(frozen=True)
class ViserPresentation:
    """Iframe markup plus a short human caption for embedded viewers."""

    html: str
    caption: str
    url: str = ""


def viser_importable() -> bool:
    """Return True when optional ``viser`` is installed."""

    return importlib.util.find_spec("viser") is not None


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
    if model_id in {"vggt", "vggt-omega", "depth-anything-v3"} or model_id == "unidepth-v2-prior":
        defaults.update(
            coordinate_preset=("opencv-to-opengl" if Path(geometry_path).suffix.lower() == ".npz" else "asset-native"),
            up_direction="+y",
            alignment="none",
        )
    elif model_id in {"loger", "pi3"}:
        defaults.update(
            coordinate_preset="asset-native",
            up_direction="+y",
            alignment="first-camera-opengl",
        )
    return defaults


def _parse_up_direction(value: str | None, *, default: str) -> ViserUpDirection:
    """Parse a Viser up-direction axis or comma/space-separated vector, falling back to ``default``."""

    for candidate in (value, default):
        parsed = _parse_up_direction_text(candidate or "")
        if parsed is not None:
            return parsed
    raise ValueError(f"Invalid default Viser up direction: {default!r}")


def _parse_up_direction_text(value: str) -> ViserUpDirection | None:
    """Return a signed axis or a non-zero 3-vector, or ``None`` when ``value`` is neither."""

    text = value.strip().lower()
    if text in _VISER_UP_AXES:
        return cast(ViserUpAxis, text)
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        x, y, z = (float(part) for part in parts)
    except ValueError:
        return None
    if abs(x) + abs(y) + abs(z) == 0:
        return None
    return (x, y, z)


def _coordinate_transform_matrix(preset: str) -> Any | None:
    """Return a 4x4 transform matrix for the selected coordinate preset."""

    import numpy as np

    if preset == "opencv-to-opengl":
        matrix = np.eye(4)
        matrix[1, 1] = -1.0
        matrix[2, 2] = -1.0
        return matrix
    return None


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


def _strip_host_brackets(host: str) -> str:
    """Strip IPv6 brackets from a host literal such as ``[::1]``."""

    value = host.strip()
    if value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    return value


def _port_is_free(host: str, port: int) -> bool:
    """Return True when a TCP port can be bound on ``host``."""

    bind_host = _strip_host_brackets(host)
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
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


def _depth_points_from_npz(data: Any) -> tuple[Any, Any] | None:
    """Project depth plus camera calibration NPZ records into world-space XYZ rows and a valid-pixel mask.

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
    valid_masks = []
    for frame_idx in range(depth.shape[0]):
        z = depth[frame_idx]
        valid = np.isfinite(z) & (z > 0)
        valid_masks.append(valid.reshape(-1))
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
    return np.concatenate(segments, axis=0), np.concatenate(valid_masks, axis=0)


def _colors_from_npz(data: Any, row_count: int) -> Any | None:
    """Return flattened RGB rows from known NPZ image/color arrays."""

    import numpy as np

    for key in _NPZ_IMAGE_KEYS:
        if key not in data:
            continue
        arr = np.asarray(data[key])
        candidates: list[Any] = []
        if arr.ndim == 2 and arr.shape[-1] >= 3:
            candidates.append(arr[:, :3])
        elif arr.ndim >= 4:
            if arr.shape[-3] in {3, 4}:
                candidates.append(np.moveaxis(arr, -3, -1).reshape(-1, arr.shape[-3])[:, :3])
            if arr.shape[-1] >= 3:
                candidates.append(arr.reshape(-1, arr.shape[-1])[:, :3])
        elif arr.ndim == 3:
            if arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
                candidates.append(np.moveaxis(arr, 0, -1).reshape(-1, arr.shape[0])[:, :3])
            elif arr.shape[-1] in {3, 4}:
                candidates.append(arr.reshape(-1, arr.shape[-1])[:, :3])
                if arr.shape[0] in {3, 4}:
                    candidates.append(np.moveaxis(arr, 0, -1).reshape(-1, arr.shape[0])[:, :3])
            elif arr.shape[-1] >= 3:
                candidates.append(arr.reshape(-1, arr.shape[-1])[:, :3])
        for colors in candidates:
            if colors.shape[0] == row_count:
                return colors
    return None


def _unpack_pcd_rgb_values(raw_values: Any, field_type: str) -> Any:
    """Unpack PCL packed ``rgb`` / ``rgba`` fields into uint8 RGB triplets."""

    import numpy as np

    if field_type == "F":
        packed = np.asarray(raw_values, dtype=np.float32).view(np.uint32)
    else:
        packed = np.asarray(raw_values, dtype=np.uint32)
    return np.stack(
        [
            ((packed >> 16) & 0xFF).astype(np.uint8),
            ((packed >> 8) & 0xFF).astype(np.uint8),
            (packed & 0xFF).astype(np.uint8),
        ],
        axis=1,
    )


def _parse_pcd_xyz_rgb(path: Path) -> tuple[Any, Any | None]:
    """Parse ASCII or uncompressed binary PCD point clouds without optional dependencies."""

    import numpy as np

    header: dict[str, list[str]] = {}
    with path.open("rb") as handle:
        while True:
            raw_line = handle.readline()
            if not raw_line:
                break
            line = raw_line.decode("ascii", errors="replace").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                break
        payload = handle.read()

    fields = [name.lower() for name in header.get("FIELDS", ())]
    if not fields or not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"{path.name} is missing x/y/z PCD fields.")
    counts = [int(value) for value in header.get("COUNT", ())] or [1] * len(fields)
    if len(counts) != len(fields):
        raise ValueError(f"{path.name} has mismatched PCD FIELD/COUNT header entries.")
    types = [value.upper() for value in header.get("TYPE", ())] or ["F"] * len(fields)
    sizes = [int(value) for value in header.get("SIZE", ())] or [4] * len(fields)
    data_mode = (header.get("DATA") or [""])[0].lower()
    points_header = int(header["POINTS"][0]) if header.get("POINTS") else 0
    if not points_header and header.get("WIDTH") and header.get("HEIGHT"):
        points_header = int(header["WIDTH"][0]) * int(header["HEIGHT"][0])

    if data_mode == "ascii":
        col_offsets: dict[str, int] = {}
        field_types: dict[str, str] = {}
        offset = 0
        for name, count, ftype in zip(fields, counts, types, strict=False):
            if name not in col_offsets:
                col_offsets[name] = offset
                field_types[name] = ftype
            offset += count
        total_cols = offset
        rows: list[list[float]] = []
        for line in payload.decode("utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            if len(tokens) < total_cols:
                continue
            rows.append([float(token) for token in tokens[:total_cols]])
        if not rows:
            raise ValueError(f"{path.name} exposes zero vertices.")
        table = np.asarray(rows, dtype=np.float64)
        points = table[:, [col_offsets["x"], col_offsets["y"], col_offsets["z"]]]
        colors: Any | None = None
        if {"r", "g", "b"}.issubset(col_offsets):
            colors = table[:, [col_offsets["r"], col_offsets["g"], col_offsets["b"]]]
        elif "rgb" in col_offsets or "rgba" in col_offsets:
            rgb_key = "rgb" if "rgb" in col_offsets else "rgba"
            colors = _unpack_pcd_rgb_values(table[:, col_offsets[rgb_key]], field_types[rgb_key])
    elif data_mode == "binary":
        if len(types) != len(fields) or len(sizes) != len(fields):
            raise ValueError(f"{path.name} has incomplete binary PCD header metadata.")
        dtype_map = {
            "F": {4: "<f4", 8: "<f8"},
            "U": {1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"},
            "I": {1: "<i1", 2: "<i2", 4: "<i4", 8: "<i8"},
        }
        dtype_fields: list[tuple[str, Any]] = []
        seen_names: set[str] = set()
        field_types = {}
        for idx, (name, size, ftype, count) in enumerate(zip(fields, sizes, types, counts, strict=True)):
            base_dtype = dtype_map.get(ftype, {}).get(size)
            if base_dtype is None:
                raise ValueError(f"{path.name} uses unsupported PCD field type {ftype}{size}.")
            unique_name = name if name not in seen_names and name != "_" else f"__field_{idx}"
            seen_names.add(unique_name)
            if unique_name == name:
                field_types[name] = ftype
            dtype_fields.append((unique_name, base_dtype) if count == 1 else (unique_name, (base_dtype, count)))
        record_dtype = np.dtype(dtype_fields)
        max_records = len(payload) // record_dtype.itemsize if record_dtype.itemsize else 0
        record_count = min(points_header, max_records) if points_header else max_records
        if record_count <= 0:
            raise ValueError(f"{path.name} exposes zero vertices.")
        records = np.frombuffer(payload, dtype=record_dtype, count=record_count)
        points = np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float64)
        colors = None
        if {"r", "g", "b"}.issubset(seen_names):
            colors = np.column_stack([records["r"], records["g"], records["b"]]).astype(np.float64)
        elif "rgb" in seen_names or "rgba" in seen_names:
            rgb_key = "rgb" if "rgb" in seen_names else "rgba"
            colors = _unpack_pcd_rgb_values(records[rgb_key], field_types[rgb_key])
    elif data_mode == "binary_compressed":
        raise ValueError(f"{path.name} uses binary_compressed PCD format, which requires open3d.")
    else:
        raise ValueError(f"{path.name} has unsupported PCD DATA mode {data_mode!r}.")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    return points, colors


def _load_pcd_xyz_rgb(path: Path) -> tuple[Any, Any | None]:
    """Load XYZ positions and optional RGB colors from a ``.pcd`` point cloud."""

    import numpy as np

    try:
        import open3d as o3d
    except Exception:
        o3d = None

    if o3d is not None:
        with contextlib.suppress(Exception):
            cloud = o3d.io.read_point_cloud(path)
            points = np.asarray(cloud.points, dtype=np.float64)
            if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] >= 3:
                points = points[:, :3]
                colors = np.asarray(cloud.colors, dtype=np.float64) if cloud.has_colors() else None
                if colors is not None and (
                    colors.ndim != 2 or colors.shape[0] != points.shape[0] or colors.shape[1] < 3
                ):
                    colors = None
                finite = np.isfinite(points).all(axis=1)
                points = points[finite]
                if colors is not None:
                    colors = colors[finite, :3]
                if points.shape[0] > 0:
                    return points, colors

    return _parse_pcd_xyz_rgb(path)


def _load_xyz_rgb(path: Path) -> tuple[Any, Any]:
    """Load XYZ positions and uint8 RGB triplets from supported geometry files."""

    import numpy as np

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(str(path), allow_pickle=False) as data:
            explicit = _points_and_finite_mask_from_npz_arrays(data)
            if explicit is None:
                depth_result = _depth_points_from_npz(data)
                if depth_result is None:
                    points = None
                    colors = None
                else:
                    points, valid_mask = depth_result
                    colors = _colors_from_npz(data, valid_mask.shape[0])
                    colors = colors[valid_mask] if colors is not None else _colors_from_npz(data, points.shape[0])
            else:
                points, finite = explicit
                colors = _colors_from_npz(data, finite.shape[0])
                colors = colors[finite] if colors is not None else _colors_from_npz(data, points.shape[0])
            if points is None:
                raise ValueError(f"{path.name} does not contain supported geometry NPZ keys.")
    elif suffix == ".pcd":
        points, colors = _load_pcd_xyz_rgb(path)
    else:
        import trimesh

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

    if path.suffix.lower() in {".npz", ".pcd", ".xyz"}:
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


def _set_initial_camera(server: Any, position: Any, look_at: Any, up: Any, *, fov: float) -> None:
    """Point every new Viser client at the geometry.

    viser>=1.0 configures ``server.initial_camera``. viser 0.2 has no such
    attribute, so each client's camera is set once its first camera state
    arrives. Its ``position`` setter moves ``look_at`` by the same offset, so
    ``look_at`` is assigned last.
    """

    initial_camera = getattr(server, "initial_camera", None)
    if initial_camera is not None:
        initial_camera.position = position
        initial_camera.look_at = look_at
        initial_camera.up = up
        initial_camera.fov = fov
        return

    def point_client_camera(client: Any) -> None:
        client.camera.up_direction = up
        client.camera.position = position
        client.camera.look_at = look_at
        client.camera.fov = fov

    server.on_client_connect(point_client_camera)


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
            host = (host or os.getenv("WORLDFOUNDRY_STUDIO_VISER_HOST") or "127.0.0.1").strip() or "127.0.0.1"
            if not is_loopback_host(host):
                return ViserPresentation(
                    html=_fallback_card("Viser host rejected", "Only loopback hosts are permitted."),
                    caption="Refusing non-loopback Viser bind.",
                )
            bind_host = _strip_host_brackets(host)
            limit = int(os.getenv("WORLDFOUNDRY_STUDIO_VISER_MAX_POINTS") or str(DEFAULT_MAX_POINTS))
            limit = max(limit, 1024)
            point_size_override = os.getenv("WORLDFOUNDRY_STUDIO_VISER_POINT_SIZE")
            point_size = (
                _env_float("WORLDFOUNDRY_STUDIO_VISER_POINT_SIZE", DEFAULT_POINT_SIZE, minimum=0.000001)
                if point_size_override is not None
                else None
            )
            point_shape = cast(
                ViserPointShape,
                _env_choice("WORLDFOUNDRY_STUDIO_VISER_POINT_SHAPE", DEFAULT_POINT_SHAPE, VISER_POINT_SHAPES),
            )
            orientation_defaults = viser_orientation_defaults(model_id, geometry_path)
            coordinate_preset_env = (os.getenv("WORLDFOUNDRY_STUDIO_VISER_COORDINATE_PRESET") or "").strip()
            coordinate_preset = _normalize_coordinate_preset(
                coordinate_preset_env or orientation_defaults["coordinate_preset"]
            )
            alignment_preset = _normalize_alignment_preset(
                os.getenv("WORLDFOUNDRY_STUDIO_VISER_ALIGNMENT") or orientation_defaults["alignment"]
            )
            camera_size = _env_float("WORLDFOUNDRY_STUDIO_VISER_CAMERA_SIZE", DEFAULT_CAMERA_SIZE, minimum=0.000001)
            camera_poses = _load_camera_poses(geometry_path)
            effective_alignment = (
                "first-camera"
                if alignment_preset == "auto" and camera_poses
                else "none"
                if alignment_preset == "auto"
                else alignment_preset
            )
            show_cameras = _env_bool("WORLDFOUNDRY_STUDIO_VISER_SHOW_CAMERAS", bool(camera_poses))
            alignment_matrix = _alignment_transform_matrix(camera_poses, alignment_preset)
            full_transform = _compose_transforms(alignment_matrix, _coordinate_transform_matrix(coordinate_preset))
            transformed_camera_poses = _transform_camera_poses(camera_poses, full_transform)
            default_up_direction = (
                _default_up_direction_for_preset(coordinate_preset)
                if coordinate_preset_env and coordinate_preset != orientation_defaults["coordinate_preset"]
                else orientation_defaults.get("up_direction", _default_up_direction_for_preset(coordinate_preset))
            )
            up_direction = _parse_up_direction(
                os.getenv("WORLDFOUNDRY_STUDIO_VISER_UP_DIRECTION"),
                default=default_up_direction,
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

        server: Any | None = None
        try:
            import viser

            with self._lock:
                if self._server is not None:
                    self._server.stop()
                    self._server = None
                resolved_port = _resolve_viser_port(run_id, host=bind_host, requested_port=port)
                server = viser.ViserServer(host=bind_host, port=resolved_port, verbose=False)
                resolved_port = int(server.get_port())
                server.scene.set_up_direction(up_direction)
                camera_position, camera_target, camera_up = _initial_camera_pose(points, mesh, up_direction)
                _set_initial_camera(server, camera_position, camera_target, camera_up, fov=math.radians(60.0))
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
                port = resolved_port
        except Exception as exc:
            if server is not None:
                with contextlib.suppress(Exception):
                    server.stop()
            return ViserPresentation(
                html=_fallback_card("Viser failed to start", escape(str(exc))),
                caption=f"Viser startup failed: {exc}",
            )

        url_host = f"[{bind_host}]" if ":" in bind_host else bind_host
        url = f"http://{url_host}:{port}/"
        orientation_caption = (
            f"coords={coordinate_preset} · alignment={effective_alignment} · up={up_direction} · {camera_caption}"
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
