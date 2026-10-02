"""Unified Studio visualization framework.

This module defines the lightweight protocol shared by the interactive world
frontend, Viser geometry frontend, and Spark Gaussian-splat frontend. It does
not import optional UI or rendering dependencies; concrete launch functions are
bound by ``studio.frontends``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.ui.interfaces import StudioInterfaceSpec, interface_spec_for_entry

from .artifacts import StudioVisualizationArtifact, infer_visualization_artifact
from .scene import VisualizationScene

if TYPE_CHECKING:
    from worldfoundry.studio.ui.launch_config import StudioLaunchConfig


INTERACTIVE_WORLD_VISUALIZATION = "world"
VISER_VISUALIZATION = "points"
SPARK_VISUALIZATION = "spark"
MEDIA_VISUALIZATION = "media"
RERUN_VISUALIZATION = "rerun"
EMBODIED_VISUALIZATION = "embodied"
AUTO_VISUALIZATION = "auto"

ARTIFACT_DOMAIN_WORLD = "interactive_world"
ARTIFACT_DOMAIN_GEOMETRY = "geometry"
ARTIFACT_DOMAIN_GAUSSIAN_SPLAT = "gaussian_splat"
ARTIFACT_DOMAIN_ACTION = "embodied_action"
ARTIFACT_DOMAIN_TIMELINE = "timeline"
ARTIFACT_DOMAIN_MEDIA = "media"
ARTIFACT_DOMAIN_UI = "ui"


@dataclass(frozen=True)
class StudioVisualizationEvent:
    """Normalized user event passed by interactive viewers."""

    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float | None = None



@dataclass(frozen=True)
class StudioModelVisualizationProfile:
    """Model-agnostic visualization decision for a catalog entry.

    The profile is intentionally based on interface/runtime/artifact domains,
    not on ``worldfoundry.studio.<model>`` modules. Model-specific helper code can
    exist as an implementation detail, but Studio routing should resolve through
    these profiles.
    """

    mode: str
    artifact_domain: str
    title: str
    reason: str = ""
    accepted_artifact_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class StudioVisualizationRequest:
    """One resolved Studio frontend launch request."""

    entry: CatalogEntry
    launch_config: StudioLaunchConfig
    mode: str
    interface_spec: StudioInterfaceSpec
    artifact: StudioVisualizationArtifact | None = None


@dataclass(frozen=True)
class StudioVisualizationLaunch:
    """Result metadata returned by non-blocking visualization backends."""

    mode: str
    url: str = ""
    caption: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendCapabilities:
    """Layer compatibility summary for a visualization backend."""

    layer_kinds: frozenset[str] = field(default_factory=frozenset)
    partial: bool = True
    score: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderPlan:
    """Backend decision returned by ``can_render``."""

    backend_id: str
    supported: bool
    score: int = 0
    unsupported_layers: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class RenderRequest:
    """Backend-neutral render request."""

    backend: str = AUTO_VISUALIZATION
    output_path: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderResult:
    """Metadata returned after a scene render."""

    backend_id: str
    url: str = ""
    output_path: str = ""
    caption: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VisualizationProvider(Protocol):
    provider_id: str

    def discover(self, source: Any) -> VisualizationScene | None: ...


class VisualizationBackend(Protocol):
    backend_id: str
    capabilities: BackendCapabilities

    def can_render(self, scene: VisualizationScene) -> RenderPlan: ...
    def render(self, scene: VisualizationScene, request: RenderRequest) -> RenderResult: ...
    def shutdown(self) -> None: ...


MatchFn = Callable[[CatalogEntry, StudioInterfaceSpec], bool]
ServeFn = Callable[[StudioVisualizationRequest], StudioVisualizationLaunch | None]


@dataclass(frozen=True)
class StudioVisualizationBackend:
    """Registered visualization backend.

    Args:
        mode: Stable frontend mode used in CLI/env options.
        title: Human-readable backend name.
        default_port: Preferred port when no explicit port is supplied.
        aliases: Additional mode tokens accepted by the registry.
        native: Whether the backend exposes a standalone server.
        match: Predicate used when resolving ``auto``.
        serve: Blocking or non-blocking launcher for the backend.
    """

    mode: str
    title: str
    default_port: int
    aliases: tuple[str, ...] = ()
    native: bool = True
    match: MatchFn = lambda entry, spec: False
    serve: ServeFn | None = None
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)

    @property
    def backend_id(self) -> str:
        return self.mode

    def can_render(self, scene: VisualizationScene) -> RenderPlan:
        supported_kinds = self.capabilities.layer_kinds
        if not supported_kinds:
            return RenderPlan(backend_id=self.mode, supported=False, reason="no scene capability declared")
        unsupported = tuple(sorted(scene.layer_kinds() - supported_kinds))
        supported = bool(scene.layers) and (not unsupported or self.capabilities.partial)
        score = self.capabilities.score + max(0, len(scene.layers) - len(unsupported))
        return RenderPlan(backend_id=self.mode, supported=supported, score=score, unsupported_layers=unsupported)

    def render(self, scene: VisualizationScene, request: RenderRequest) -> RenderResult:
        plan = self.can_render(scene)
        if not plan.supported:
            raise ValueError(f"Visualization backend `{self.mode}` cannot render scene `{scene.scene_id}`: {plan.reason or plan.unsupported_layers}")
        return RenderResult(backend_id=self.mode, metadata={"scene_id": scene.scene_id})

    def shutdown(self) -> None:
        return None

    def accepts(self, requested: str) -> bool:
        token = normalize_visualization_mode(requested)
        return token == self.mode or token in self.aliases

    def supports(self, entry: CatalogEntry, spec: StudioInterfaceSpec | None = None) -> bool:
        resolved_spec = spec or interface_spec_for_entry(entry)
        profile = model_visualization_profile(entry, resolved_spec)
        return profile.mode == self.mode or bool(self.match(entry, resolved_spec))

    def launch(self, request: StudioVisualizationRequest) -> StudioVisualizationLaunch | None:
        if self.serve is None:
            raise ValueError(f"Visualization backend `{self.mode}` has no serve function.")
        return self.serve(request)


class StudioVisualizationRegistry:
    """Mode registry and auto-router for Studio visualization backends."""

    def __init__(self, backends: Iterable[StudioVisualizationBackend] = ()) -> None:
        self._ordered: list[StudioVisualizationBackend] = []
        self._by_token: dict[str, StudioVisualizationBackend] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: StudioVisualizationBackend) -> None:
        tokens = (backend.mode, *backend.aliases)
        for token in tokens:
            normalized = normalize_visualization_mode(token)
            if not normalized:
                raise ValueError("Visualization backend mode/alias cannot be empty.")
            if normalized in self._by_token and self._by_token[normalized].mode != backend.mode:
                raise ValueError(f"Duplicate Studio visualization mode `{normalized}`.")
            self._by_token[normalized] = backend
        self._ordered = [item for item in self._ordered if item.mode != backend.mode]
        self._ordered.append(backend)

    @property
    def modes(self) -> frozenset[str]:
        return frozenset(backend.mode for backend in self._ordered)

    @property
    def native_modes(self) -> frozenset[str]:
        return frozenset(backend.mode for backend in self._ordered if backend.native)

    @property
    def default_ports(self) -> dict[str, int]:
        return {backend.mode: backend.default_port for backend in self._ordered}

    def backend_for(self, mode: str) -> StudioVisualizationBackend:
        normalized = normalize_visualization_mode(mode)
        try:
            return self._by_token[normalized]
        except KeyError as exc:
            known = ", ".join(sorted(self._by_token))
            raise ValueError(f"Unsupported Studio visualization mode `{mode}`. Known modes: {known}.") from exc

    def resolve_mode(
        self,
        entry: CatalogEntry,
        requested: str | None,
        artifact: StudioVisualizationArtifact | None = None,
    ) -> str:
        mode = normalize_visualization_mode(requested or AUTO_VISUALIZATION) or AUTO_VISUALIZATION
        if mode != AUTO_VISUALIZATION:
            return self.backend_for(mode).mode
        artifact_mode = _visualization_mode_for_artifact(artifact)
        if artifact_mode in self.modes:
            return artifact_mode
        profile = model_visualization_profile(entry)
        if profile.mode in self.modes:
            return profile.mode
        spec = interface_spec_for_entry(entry)
        for backend in self._ordered:
            if backend.supports(entry, spec):
                return backend.mode
        return INTERACTIVE_WORLD_VISUALIZATION

    def request_for(
        self,
        *,
        entry: CatalogEntry,
        launch_config: StudioLaunchConfig,
        mode: str,
        artifact: StudioVisualizationArtifact | None = None,
    ) -> StudioVisualizationRequest:
        resolved_mode = self.backend_for(mode).mode
        return StudioVisualizationRequest(
            entry=entry,
            launch_config=launch_config,
            mode=resolved_mode,
            interface_spec=interface_spec_for_entry(entry),
            artifact=artifact,
        )

    def serve(
        self,
        *,
        entry: CatalogEntry,
        launch_config: StudioLaunchConfig,
        mode: str,
        artifact: StudioVisualizationArtifact | None = None,
    ) -> StudioVisualizationLaunch | None:
        backend = self.backend_for(mode)
        return backend.launch(
            self.request_for(
                entry=entry,
                launch_config=launch_config,
                mode=backend.mode,
                artifact=artifact,
            )
        )


def normalize_visualization_mode(value: str | None) -> str:
    return (value or "").strip().lower().replace("_", "-")


def _visualization_mode_for_artifact(artifact: StudioVisualizationArtifact | None) -> str:
    """Choose a native viewer from the artifact before consulting model metadata."""

    if artifact is None:
        return ""
    path = Path(artifact.path).expanduser()
    suffix = path.suffix.lower()
    kind = (artifact.kind or "").strip().lower()

    if suffix == ".rrd" or kind == "timeline":
        return RERUN_VISUALIZATION
    if kind == "gaussian_splat" or suffix in {".spz", ".splat", ".ksplat", ".sog"}:
        return SPARK_VISUALIZATION
    if suffix == ".ply":
        return SPARK_VISUALIZATION if _is_gaussian_splat_ply(path) else VISER_VISUALIZATION
    if kind in {"image", "video", "audio", "optical_flow"} or suffix in {
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
        ".mp4", ".mov", ".webm", ".mkv", ".avi",
        ".wav", ".mp3", ".flac", ".ogg",
    }:
        return MEDIA_VISUALIZATION
    if kind in {"point_cloud", "mesh", "camera", "trajectory", "depth"} or suffix in {
        ".pcd", ".xyz", ".npz", ".glb", ".gltf", ".obj",
    }:
        return VISER_VISUALIZATION
    return ""


def _is_gaussian_splat_ply(path: Path) -> bool:
    """Identify Gaussian-splat PLY headers without loading the potentially huge asset."""

    try:
        with path.open("rb") as handle:
            header = handle.read(16384).decode("latin-1", errors="ignore")
    except OSError:
        return False
    header = header.split("end_header", maxsplit=1)[0].lower()
    return header.lstrip().startswith("ply") and all(
        marker in header
        for marker in (
            "property float opacity",
            "property float scale_0",
            "property float f_dc_0",
            "property float rot_0",
        )
    )


def model_visualization_profile(
    entry: CatalogEntry,
    spec: StudioInterfaceSpec | None = None,
) -> StudioModelVisualizationProfile:
    """Resolve a Studio catalog entry to one visualization profile.

    This is the single routing policy for model visualization. It is deliberately
    expressed in terms of category/runtime/interface contracts so model-specific
    folders are not the abstraction boundary.
    """

    resolved_spec = spec or interface_spec_for_entry(entry)
    if resolved_spec.template_id == "interactive-world":
        return StudioModelVisualizationProfile(
            mode=INTERACTIVE_WORLD_VISUALIZATION,
            artifact_domain=ARTIFACT_DOMAIN_WORLD,
            title="Interactive World Model",
            reason="interactive-world interface template",
            accepted_artifact_kinds=("state", "video", "image"),
        )
    if (
        entry.runtime_kind in {"pointcloud_nav", "worldfm"}
        or resolved_spec.template_id == "depth-geometry"
        or entry.category == "Depth / Geometry"
    ):
        return StudioModelVisualizationProfile(
            mode=VISER_VISUALIZATION,
            artifact_domain=ARTIFACT_DOMAIN_GEOMETRY,
            title="Geometry Viewer",
            reason="depth/geometry artifact contract",
            accepted_artifact_kinds=("point_cloud", "mesh", "depth", "geometry"),
        )
    if (
        entry.runtime_kind == "two_stage_3dgs"
        or resolved_spec.template_id == "scene-3d"
        or entry.category == "3D Scene"
    ):
        return StudioModelVisualizationProfile(
            mode=SPARK_VISUALIZATION,
            artifact_domain=ARTIFACT_DOMAIN_GAUSSIAN_SPLAT,
            title="3D Scene Viewer",
            reason="3D scene artifact contract",
            accepted_artifact_kinds=("gaussian_splat", "point_cloud", "mesh", "scene"),
        )
    if resolved_spec.template_id == "embodied-policy" or entry.category == "Embodied Action":
        return StudioModelVisualizationProfile(
            mode=EMBODIED_VISUALIZATION,
            artifact_domain=ARTIFACT_DOMAIN_ACTION,
            title="Embodied Policy Console",
            reason="embodied action contract",
            accepted_artifact_kinds=("action_trace", "robot_action", "policy_state"),
        )
    if entry.runtime_kind in {"rerun", "rrd"}:
        return StudioModelVisualizationProfile(
            mode=RERUN_VISUALIZATION,
            artifact_domain=ARTIFACT_DOMAIN_TIMELINE,
            title="Timeline Viewer",
            reason="Rerun timeline runtime",
            accepted_artifact_kinds=("timeline", "rrd"),
        )
    if entry.category in {"Video", "Image", "Audio / Video", "Video Generation", "Video-to-Video", "Visual Action"}:
        return StudioModelVisualizationProfile(
            mode=MEDIA_VISUALIZATION,
            artifact_domain=ARTIFACT_DOMAIN_MEDIA,
            title="Media Preview",
            reason="media artifact contract",
            accepted_artifact_kinds=("video", "image", "audio", "gallery"),
        )
    return StudioModelVisualizationProfile(
        mode=INTERACTIVE_WORLD_VISUALIZATION,
        artifact_domain=ARTIFACT_DOMAIN_UI,
        title="Studio UI",
        reason="fallback Studio visualization",
        accepted_artifact_kinds=("artifact",),
    )


__all__ = [
    "ARTIFACT_DOMAIN_ACTION",
    "ARTIFACT_DOMAIN_GAUSSIAN_SPLAT",
    "ARTIFACT_DOMAIN_GEOMETRY",
    "ARTIFACT_DOMAIN_MEDIA",
    "ARTIFACT_DOMAIN_TIMELINE",
    "ARTIFACT_DOMAIN_UI",
    "ARTIFACT_DOMAIN_WORLD",
    "AUTO_VISUALIZATION",
    "EMBODIED_VISUALIZATION",
    "INTERACTIVE_WORLD_VISUALIZATION",
    "MEDIA_VISUALIZATION",
    "RERUN_VISUALIZATION",
    "SPARK_VISUALIZATION",
    "VISER_VISUALIZATION",
    "StudioVisualizationArtifact",
    "StudioVisualizationBackend",
    "StudioVisualizationEvent",
    "StudioVisualizationLaunch",
    "StudioModelVisualizationProfile",
    "StudioVisualizationRegistry",
    "StudioVisualizationRequest",
    "VisualizationProvider",
    "VisualizationBackend",
    "RenderResult",
    "RenderRequest",
    "RenderPlan",
    "BackendCapabilities",
    "infer_visualization_artifact",
    "model_visualization_profile",
    "normalize_visualization_mode",
]
