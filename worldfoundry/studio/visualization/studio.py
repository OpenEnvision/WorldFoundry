"""Optional run-record presentation: viewer conversion and viewport manifests.

Only UI/visualization entrypoints install this adapter. Model execution and
subprocess workers persist their raw artifacts without importing it.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from worldfoundry.studio.inference.catalog import CatalogEntry, find_entry
from worldfoundry.studio.inference.execution import (
    MODEL_EXTS,
    RERUN_PREVIEW_ENV,
    RunRecord,
    _env_flag,
    _existing_artifact_paths,
    _is_gaussian_splat_ply,
    _viewport_artifact_paths,
    collect_artifact_paths,
)

from .core.manifest import build_studio_viewports_payload


def _load_trimesh() -> Any:
    try:
        return importlib.import_module("trimesh")
    except Exception:
        return None


def convert_model_for_preview(model_path: Optional[str], output_dir: str) -> Optional[str]:
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    if path.suffix.lower() in {".glb", ".ply", ".pcd", ".xyz"}:
        return str(path)
    trimesh = _load_trimesh()
    if trimesh is None:
        return str(path)
    try:
        mesh = trimesh.load(path, force="mesh")
        glb_path = Path(output_dir) / "model_preview.glb"
        mesh.export(glb_path)
        return str(glb_path)
    except Exception:
        return str(path)


def _geometry_points_and_colors(geometry: Any) -> tuple[np.ndarray, np.ndarray | None] | None:
    vertices = getattr(geometry, "vertices", None)
    if vertices is None:
        vertices = getattr(geometry, "points", None)
    if vertices is None:
        return None
    points = np.asarray(vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None
    points = points[:, :3]

    colors = getattr(geometry, "colors", None)
    if colors is None:
        visual = getattr(geometry, "visual", None)
        colors = getattr(visual, "vertex_colors", None) if visual is not None else None
    if colors is None:
        return points, None
    color_array = np.asarray(colors)
    if color_array.ndim != 2 or color_array.shape[0] != points.shape[0] or color_array.shape[1] < 3:
        return points, None
    return points, color_array[:, :3]


def _scene_points_and_colors(scene_or_geometry: Any, *, max_points: int = 400_000) -> tuple[np.ndarray, np.ndarray | None] | None:
    geometries = []
    if hasattr(scene_or_geometry, "geometry"):
        try:
            dumped = scene_or_geometry.dump(concatenate=False)
            geometries = list(dumped) if isinstance(dumped, (list, tuple)) else [dumped]
        except Exception:
            geometries = list(scene_or_geometry.geometry.values())
    else:
        geometries = [scene_or_geometry]

    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    missing_color = False
    for geometry in geometries:
        extracted = _geometry_points_and_colors(geometry)
        if extracted is None:
            continue
        points, colors = extracted
        point_chunks.append(points)
        if colors is None:
            missing_color = True
        else:
            color_chunks.append(colors)
    if not point_chunks:
        return None

    points = np.concatenate(point_chunks, axis=0)
    colors = None if missing_color or len(color_chunks) != len(point_chunks) else np.concatenate(color_chunks, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    if points.shape[0] == 0:
        return None

    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    if colors is not None and colors.dtype != np.uint8:
        scale = 255.0 if np.nanmax(colors) <= 1.0 else 1.0
        colors = np.clip(colors * scale, 0, 255).astype(np.uint8)
    return points, colors


def maybe_build_rerun_rrd(output_dir: str) -> Optional[str]:
    try:
        import rerun as rr  # type: ignore
        import rerun.blueprint as rrb  # type: ignore
    except Exception:
        return None

    model_candidates = [
        path
        for path in collect_artifact_paths(output_dir)
        if Path(path).suffix.lower() in MODEL_EXTS
    ]
    if not model_candidates:
        return None
    trimesh = _load_trimesh()
    if trimesh is None:
        return None

    try:
        extracted = None
        for model_path in model_candidates:
            scene_or_geometry = trimesh.load(model_path)
            extracted = _scene_points_and_colors(scene_or_geometry)
            if extracted is not None:
                break
        if extracted is None:
            return None
        points, colors = extracted

        recording_path = Path(output_dir) / "scene.rrd"
        rr.init("worldfoundry_studio")
        rr.save(str(recording_path), default_blueprint=rrb.Spatial3DView(origin="scene"))
        rr.log("scene/points", rr.Points3D(points, colors=colors))
        rr.disconnect()
        return str(recording_path)
    except Exception:
        return None


def prepare_run_record(record: RunRecord, entry: CatalogEntry | None = None) -> RunRecord:
    """Add presentation metadata on the UI side, preserving raw model outputs."""
    if isinstance(record.metadata.get("studio_viewports"), dict):
        return record
    entry = entry or find_entry(record.model_id)
    started = time.perf_counter()
    record.preview_model = convert_model_for_preview(record.preview_model, record.output_dir)
    if not record.rrd_path and _env_flag(RERUN_PREVIEW_ENV):
        record.rrd_path = maybe_build_rerun_rrd(record.output_dir)
    record.artifacts = sorted(_existing_artifact_paths([
        *record.artifacts, record.preview_model, record.rrd_path,
    ]))
    converted = time.perf_counter()
    record.metadata["studio_viewports"] = build_studio_viewports_payload(
        entry=entry, output_dir=record.output_dir,
        previews={"preview_video": record.preview_video, "preview_image": record.preview_image,
                  "preview_splat": record.preview_splat, "preview_model": record.preview_model,
                  "rrd_path": record.rrd_path},
        artifact_paths=_viewport_artifact_paths(record.artifacts),
        gaussian_ply_predicate=_is_gaussian_splat_ply,
        result_metadata=record.metadata.get("result", {}).get("metadata")
        if isinstance(record.metadata.get("result"), dict) else None,
    )
    finished = time.perf_counter()
    timings = record.metadata.setdefault("studio_performance", {})
    timings["materialize_preview_convert_ms"] = round((converted - started) * 1000, 3)
    timings["materialize_viewports_ms"] = round((finished - converted) * 1000, 3)
    timings["materialize_total_ms"] = round(timings.get("materialize_total_ms", 0) + (finished - started) * 1000, 3)
    return record
