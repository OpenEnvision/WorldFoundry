"""Geometry artifact provider for point clouds, meshes, splats, and cameras."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
from worldfoundry.studio.visualization.core.scene import Layer, VisualizationScene

GEOMETRY_KINDS = {"point_cloud", "mesh", "gaussian_splat", "camera", "trajectory", "depth"}
CAMERA_SUFFIXES = {".json", ".yaml", ".yml"}


class GeometryProvider:
    provider_id = "geometry"

    def discover(self, source) -> VisualizationScene | None:
        layers = []
        for path in _source_paths(source):
            artifact = infer_visualization_artifact(path)
            kind = artifact.kind
            lowered = path.name.lower()
            if kind == "artifact" and path.suffix.lower() in CAMERA_SUFFIXES and "camera" in lowered:
                kind = "camera"
            if kind not in GEOMETRY_KINDS:
                continue
            layers.append(Layer(layer_id=path.stem, kind=kind, uri=path.as_posix(), metadata={"format": artifact.format_hint}))
        if not layers:
            return None
        backend = "spark" if any(layer.kind == "gaussian_splat" for layer in layers) else "points"
        return VisualizationScene(scene_id="geometry/scene", title="Geometry Scene", layers=tuple(layers), recommended_backend=backend)


def _source_paths(source) -> list[Path]:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            return sorted(item for item in path.rglob("*") if item.is_file())
        return [path]
    if isinstance(source, Iterable):
        return [Path(item) for item in source]
    return []
