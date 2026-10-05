"""Choose default Studio preview lanes based on catalog metadata and persisted assets."""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet

from worldfoundry.studio.inference.catalog import CatalogEntry, find_entry
from worldfoundry.studio.visualization.core.manifest import ViewportCapabilities, ViewportKind

if TYPE_CHECKING:
    from worldfoundry.studio.inference.execution import RunRecord


def available_viewport_kinds(
    *,
    caps: ViewportCapabilities,
    entry: CatalogEntry | None,
) -> FrozenSet[ViewportKind]:
    """Return viewport tabs that deserve activation for this run/catalog."""

    modes: set[ViewportKind] = {ViewportKind.WORLD}
    if caps.has_gaussian_splat:
        modes.add(ViewportKind.SPLAT)
    if caps.has_points_cloud or caps.has_viser:
        modes.add(ViewportKind.POINTS)
    if caps.has_embodied_trace or caps.has_simulator_replay:
        modes.add(ViewportKind.EMBODIED)
    driver = entry.runtime_kind if entry else ""
    if driver in {"pointcloud_nav", "two_stage_3dgs", "worldfm"}:
        modes.add(ViewportKind.POINTS)
    if caps.has_gaussian_splat or (entry is not None and entry.category == "3D Scene"):
        modes.add(ViewportKind.SPLAT)
    if entry is not None and entry.category in {"Embodied Action", "Visual Action"}:
        modes.add(ViewportKind.EMBODIED)
    return frozenset(modes)


def recommend_viewport(
    *,
    caps: ViewportCapabilities,
    has_preview_video: bool,
    has_preview_image: bool,
    user_viewport_override: str | ViewportKind | None = None,
    available: FrozenSet[ViewportKind] | None = None,
) -> ViewportKind:
    """Prefer explicit tabs, simulator replay, streaming media, splats, then points."""

    override = _coerce_viewport_kind(user_viewport_override)
    if override is not None and (available is None or override in available):
        return override

    if caps.has_simulator_replay or caps.has_embodied_trace:
        return ViewportKind.EMBODIED
    if caps.has_streaming and (has_preview_video or has_preview_image):
        return ViewportKind.WORLD
    if caps.has_gaussian_splat:
        return ViewportKind.SPLAT
    if caps.has_points_cloud:
        return ViewportKind.POINTS
    if has_preview_video or has_preview_image:
        return ViewportKind.WORLD
    return ViewportKind.WORLD


def _coerce_viewport_kind(value: str | ViewportKind | None) -> ViewportKind | None:
    if value is None:
        return None
    if isinstance(value, ViewportKind):
        return value
    try:
        return ViewportKind(str(value).strip().lower())
    except ValueError:
        return None


def summarize_routing_hints(record: RunRecord | None) -> str | None:
    """Return compact plain text describing manifest routing guidance."""

    if record is None:
        return None
    entry: CatalogEntry | None = None
    try:
        entry = find_entry(record.model_id)
    except KeyError:
        entry = None
    from worldfoundry.studio.visualization.core.manifest import viewport_payload_from_metadata

    payload = viewport_payload_from_metadata(dict(record.metadata or {}))
    if payload is None:
        return None
    avail = sorted(mode.value for mode in available_viewport_kinds(caps=payload.capabilities, entry=entry))
    return f"viewport focus={payload.recommended.value}; tabs={'/'.join(avail)}"
