"""Viewport manifest payload shared by Studio manifests, routing, and viewers."""


from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ViewportKind(str, Enum):
    """High-level preview lane for Studio runs."""

    WORLD = "world"
    SPLAT = "splat"
    POINTS = "points"
    EMBODIED = "embodied"


@dataclass(frozen=True)
class WorldViewportAssets:
    """Pointers to media shown in the world viewport."""

    preview_video: str | None = None
    preview_image: str | None = None
    rrd_path: str | None = None


@dataclass(frozen=True)
class SplatViewportAssets:
    """Gaussian splat paths and URLs surfaced by Studio viewers."""

    primary_path: str | None = None
    primary_url: str | None = None
    format_hint: str | None = None


@dataclass(frozen=True)
class PointsViewportAssets:
    """Artifacts intended for geometric inspection (non-GS ply, etc.)."""

    point_cloud_path: str | None = None
    mesh_path: str | None = None
    camera_path: str | None = None
    coordinate_frame: str = "world"


@dataclass(frozen=True)
class EmbodiedViewportAssets:
    """Artifacts intended for action-policy and simulator inspection."""

    action_trace_path: str | None = None
    simulator_video_path: str | None = None
    episode_metadata_path: str | None = None
    simulator_hint: str | None = None


@dataclass(frozen=True)
class ViewportCapabilities:
    """Boolean switches derived from CatalogEntry + persisted assets."""

    has_streaming: bool = False
    has_gaussian_splat: bool = False
    has_points_cloud: bool = False
    has_viser: bool = False
    has_rrd: bool = False
    has_embodied_trace: bool = False
    has_simulator_replay: bool = False


@dataclass
class StudioViewportsPayload:
    """Serializable `studio_viewports` block persisted under run metadata."""

    recommended: ViewportKind
    schema_version: int = 1
    assets_world: WorldViewportAssets = field(default_factory=WorldViewportAssets)
    assets_splat: SplatViewportAssets = field(default_factory=SplatViewportAssets)
    assets_points: PointsViewportAssets = field(default_factory=PointsViewportAssets)
    assets_embodied: EmbodiedViewportAssets = field(default_factory=EmbodiedViewportAssets)
    capabilities: ViewportCapabilities = field(default_factory=ViewportCapabilities)

    def asdict(self) -> dict[str, Any]:
        """Project the payload into JSON-friendly primitives."""
        caps = self.capabilities
        assets_w = self.assets_world
        assets_s = self.assets_splat
        assets_p = self.assets_points
        assets_e = self.assets_embodied
        return {
            "schema_version": self.schema_version,
            "recommended": self.recommended.value,
            "assets": {
                "world": {
                    "preview_video": assets_w.preview_video,
                    "preview_image": assets_w.preview_image,
                    "rrd_path": assets_w.rrd_path,
                },
                "splat": {
                    "primary_path": assets_s.primary_path,
                    "primary_url": assets_s.primary_url,
                    "format": assets_s.format_hint,
                },
                "points": {
                    "point_cloud_path": assets_p.point_cloud_path,
                    "mesh_path": assets_p.mesh_path,
                    "camera_path": assets_p.camera_path,
                    "coordinate_frame": assets_p.coordinate_frame,
                },
                "embodied": {
                    "action_trace_path": assets_e.action_trace_path,
                    "simulator_video_path": assets_e.simulator_video_path,
                    "episode_metadata_path": assets_e.episode_metadata_path,
                    "simulator_hint": assets_e.simulator_hint,
                },
            },
            "capabilities": {
                "has_streaming": caps.has_streaming,
                "has_gaussian_splat": caps.has_gaussian_splat,
                "has_points_cloud": caps.has_points_cloud,
                "has_viser": caps.has_viser,
                "has_rrd": caps.has_rrd,
                "has_embodied_trace": caps.has_embodied_trace,
                "has_simulator_replay": caps.has_simulator_replay,
            },
        }


def viewport_payload_from_metadata(metadata: Mapping[str, Any] | None) -> StudioViewportsPayload | None:
    """Hydrate ``StudioViewportsPayload`` when ``studio_viewports`` is present."""

    if metadata is None:
        return None
    blob = metadata.get("studio_viewports")
    if not isinstance(blob, dict):
        return None
    assets = blob.get("assets") or {}
    caps = blob.get("capabilities") or {}

    try:
        rec = ViewportKind(str(blob.get("recommended") or ViewportKind.WORLD.value))
    except ValueError:
        rec = ViewportKind.WORLD

    capabilities = ViewportCapabilities(
        has_streaming=bool(caps.get("has_streaming")),
        has_gaussian_splat=bool(caps.get("has_gaussian_splat")),
        has_points_cloud=bool(caps.get("has_points_cloud") or caps.get("has_viser")),
        has_viser=bool(caps.get("has_viser") or caps.get("has_points_cloud")),
        has_rrd=bool(caps.get("has_rrd")),
        has_embodied_trace=bool(caps.get("has_embodied_trace")),
        has_simulator_replay=bool(caps.get("has_simulator_replay")),
    )
    world = assets.get("world") or {}
    splat = assets.get("splat") or {}
    points = assets.get("points") or {}
    embodied = assets.get("embodied") or {}
    return StudioViewportsPayload(
        recommended=rec,
        schema_version=_int_or_default(blob.get("schema_version"), 1),
        assets_world=WorldViewportAssets(
            preview_video=_str_or_none(world.get("preview_video")),
            preview_image=_str_or_none(world.get("preview_image")),
            rrd_path=_str_or_none(world.get("rrd_path")),
        ),
        assets_splat=SplatViewportAssets(
            primary_path=_str_or_none(splat.get("primary_path") or splat.get("primary_url")),
            primary_url=_str_or_none(splat.get("primary_url")),
            format_hint=_str_or_none(splat.get("format")),
        ),
        assets_points=PointsViewportAssets(
            point_cloud_path=_str_or_none(points.get("point_cloud_path")),
            mesh_path=_str_or_none(points.get("mesh_path")),
            camera_path=_str_or_none(points.get("camera_path")),
            coordinate_frame=str(points.get("coordinate_frame") or "world"),
        ),
        assets_embodied=EmbodiedViewportAssets(
            action_trace_path=_str_or_none(embodied.get("action_trace_path")),
            simulator_video_path=_str_or_none(embodied.get("simulator_video_path")),
            episode_metadata_path=_str_or_none(embodied.get("episode_metadata_path")),
            simulator_hint=_str_or_none(embodied.get("simulator_hint")),
        ),
        capabilities=capabilities,
    )


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# --- studio_viewports payload builder ---

from pathlib import Path
from typing import Callable, Mapping, Sequence

from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.visualization.providers.run_record import (
    first_embodied_trace_candidate,
    first_episode_metadata_candidate,
    first_geometry_point_candidate,
    first_simulator_replay_candidate,
    first_splat_asset,
)


def _relative_to_run(path_text: str | None, output_dir: str) -> str | None:
    """Store run-scoped paths relative to ``output_dir`` when possible."""

    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    resolved = path.resolve()
    root = Path(output_dir).resolve()
    if resolved == root:
        return "."
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return resolved.as_posix()


def build_studio_viewports_payload(
    *,
    entry: CatalogEntry,
    output_dir: str,
    previews: Mapping[str, str | None],
    artifact_paths: Sequence[str],
    gaussian_ply_predicate: Callable[[Path], bool],
    result_metadata: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Return a manifest-ready ``studio_viewports`` dict."""

    preview_video = previews.get("preview_video")
    preview_image = previews.get("preview_image")
    preview_splat = previews.get("preview_splat")
    preview_model = previews.get("preview_model")
    rrd_path = previews.get("rrd_path")

    point_rel = first_geometry_point_candidate(
        list(artifact_paths),
        output_dir,
        gs_ply_predicate=gaussian_ply_predicate,
    )

    splat_source = ""
    splat_fmt = ""
    for candidate in (preview_splat or "",):
        if candidate and Path(candidate).exists():
            splat_source = candidate
            break
    if not splat_source:
        guessed_path, guessed_fmt = first_splat_asset(
            list(artifact_paths),
            gs_ply_predicate=gaussian_ply_predicate,
        )
        if guessed_path and Path(guessed_path).exists():
            splat_source = guessed_path
            splat_fmt = guessed_fmt or ""
    if splat_source and not splat_fmt:
        _, splat_fmt = first_splat_asset([splat_source], gs_ply_predicate=gaussian_ply_predicate)

    world_assets = WorldViewportAssets(
        preview_video=_relative_to_run(str(preview_video) if preview_video else None, output_dir),
        preview_image=_relative_to_run(str(preview_image) if preview_image else None, output_dir),
        rrd_path=_relative_to_run(str(rrd_path) if rrd_path else None, output_dir),
    )
    splat_assets = SplatViewportAssets(
        primary_path=_relative_to_run(splat_source, output_dir),
        format_hint=splat_fmt or None,
    )
    mesh_rel = _relative_to_run(str(preview_model) if preview_model else None, output_dir)

    coordinate_frame = _str_or_none(
        result_metadata.get("coordinate_frame") if result_metadata is not None else None
    ) or "world"
    points_assets = PointsViewportAssets(
        point_cloud_path=point_rel,
        mesh_path=mesh_rel,
        coordinate_frame=coordinate_frame,
    )
    action_trace_rel = first_embodied_trace_candidate(list(artifact_paths), output_dir)
    simulator_replay_rel = first_simulator_replay_candidate(list(artifact_paths), output_dir)
    episode_metadata_rel = first_episode_metadata_candidate(list(artifact_paths), output_dir)
    embodied_assets = EmbodiedViewportAssets(
        action_trace_path=action_trace_rel,
        simulator_video_path=simulator_replay_rel,
        episode_metadata_path=episode_metadata_rel,
        simulator_hint=_simulator_hint_for_entry(entry),
    )

    caps = ViewportCapabilities(
        has_streaming=entry.supports_stream,
        has_gaussian_splat=bool(splat_assets.primary_path),
        has_points_cloud=bool(point_rel or mesh_rel),
        has_viser=bool(point_rel or mesh_rel),
        has_rrd=bool(world_assets.rrd_path),
        has_embodied_trace=bool(embodied_assets.action_trace_path),
        has_simulator_replay=bool(embodied_assets.simulator_video_path),
    )

    has_world_video = bool(world_assets.preview_video and Path(output_dir, world_assets.preview_video).exists())
    has_world_image = bool(world_assets.preview_image and Path(output_dir, world_assets.preview_image).exists())

    from worldfoundry.studio.visualization.core.capabilities import recommend_viewport

    viewport_rec = recommend_viewport(
        caps=caps,
        has_preview_video=has_world_video,
        has_preview_image=has_world_image,
    )

    payload = StudioViewportsPayload(
        recommended=viewport_rec,
        assets_world=world_assets,
        assets_splat=splat_assets,
        assets_points=points_assets,
        assets_embodied=embodied_assets,
        capabilities=caps,
    )
    return payload.asdict()


def _simulator_hint_for_entry(entry: CatalogEntry) -> str | None:
    """Return a compact simulator family hint from catalog metadata."""

    if entry.category != "Embodied Action":
        return None
    tag_blob = " ".join((*entry.tags, entry.family, entry.model_id)).lower()
    if "robotwin" in tag_blob or entry.model_id in {"openvla", "openpi", "diffusion-policy", "act"}:
        return "RoboTwin / LIBERO-style episode"
    if "dreamzero" in tag_blob:
        return "RoboArena websocket rollout"
    if "gr00t" in tag_blob or "humanoid" in tag_blob:
        return "Humanoid simulator episode"
    if "giga" in tag_blob:
        return "Giga-style multi-view robot episode"
    return "Embodied simulator episode"
