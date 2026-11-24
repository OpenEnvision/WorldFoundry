"""Artifact references and URI normalization for Studio visualization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StudioVisualizationArtifact:
    """Normalized artifact reference that a frontend can present."""

    path: str
    kind: str
    format_hint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


KIND_BY_SUFFIX = {
    ".ply": "point_cloud",
    ".pcd": "point_cloud",
    ".xyz": "point_cloud",
    ".npz": "point_cloud",
    ".glb": "mesh",
    ".gltf": "mesh",
    ".obj": "mesh",
    ".spz": "gaussian_splat",
    ".splat": "gaussian_splat",
    ".ksplat": "gaussian_splat",
    ".sog": "gaussian_splat",
    ".rrd": "timeline",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".webm": "video",
    ".mkv": "video",
    ".avi": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
    ".ogg": "audio",
}


def infer_visualization_artifact(
    path: str | Path, *, metadata: Mapping[str, Any] | None = None
) -> StudioVisualizationArtifact:
    artifact_path = Path(path)
    suffix = artifact_path.suffix.lower()
    return StudioVisualizationArtifact(
        path=str(artifact_path),
        kind=KIND_BY_SUFFIX.get(suffix, "artifact"),
        format_hint=suffix[1:],
        metadata=dict(metadata or {}),
    )
