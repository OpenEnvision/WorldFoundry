from __future__ import annotations

import json
from html import escape
from pathlib import Path

from worldfoundry.studio.inference.execution import RunRecord

from ..visualization.backends.viser import STUDIO_VISER
from ..visualization.core.manifest import viewport_payload_from_metadata
from .urls import file_url as _file_url


def _spatial_stage_html(
    *,
    title: str = "WorldFoundry",
    splat_path: str | None = None,
    poster_path: str | None = None,
    note: str = "",
    has_mesh_fallback: bool = False,
) -> str:
    splat_url = _file_url(splat_path)
    poster_url = _file_url(poster_path)
    world_title = escape(title or "WorldFoundry", quote=True)
    shell_state = "splat" if splat_url else ("mesh" if has_mesh_fallback else "empty")
    body_note = note or (
        "Load a Gaussian Splat export to continue the generated world in 3D."
        if not splat_url
        else "Spark viewer is loading the current world."
    )
    return f"""
<div
  class="wa-spatial-shell {'is-empty' if not splat_url else 'is-loading'}"
  id="wa-spatial-shell"
  data-kind="{shell_state}"
  data-splat-url="{escape(splat_url, quote=True)}"
  data-poster-url="{escape(poster_url, quote=True)}"
  data-world-title="{world_title}"
  data-note="{escape(body_note, quote=True)}"
>
  <canvas class="wa-spark-canvas" id="wa-spark-canvas"></canvas>
  <div class="wa-spark-overlay" id="wa-spark-overlay">
    <div class="wa-spark-copy-stack">
      <span class="wa-spark-kicker">Spatial Continuation</span>
      <strong class="wa-spark-title">{escape(title or "WorldFoundry")}</strong>
      <span class="wa-spark-copy" id="wa-spark-copy">{escape(body_note)}</span>
    </div>
    <div class="wa-spark-hud">
      <span>Drag to look</span>
      <span>Wheel to dolly</span>
      <span>WASD to move</span>
    </div>
  </div>
  <div class="wa-spark-loading" id="wa-spark-loading">Loading 3DGS…</div>
</div>
"""


def _spatial_stage_for_record(record: RunRecord | None = None) -> tuple[str, str]:
    if record is None:
        return (
            _spatial_stage_html(note="Run a model or import a Gaussian Splat asset to unlock the 3D world stage."),
            "Waiting for a 3DGS asset or a run with spatial output.",
        )

    poster = record.preview_image or (record.gallery[0] if record.gallery else None)
    if record.preview_splat:
        return (
            _spatial_stage_html(
                title=record.display_name,
                splat_path=record.preview_splat,
                poster_path=poster,
                note="The latest run exported a Gaussian Splat. Explore it with drag, wheel, and WASD controls.",
                has_mesh_fallback=bool(record.preview_model),
            ),
            f"3DGS asset attached · {Path(record.preview_splat).name}",
        )
    if record.preview_model:
        return (
            _spatial_stage_html(
                title=record.display_name,
                poster_path=poster,
                note="This run has a mesh or point-cloud preview. Import a Gaussian Splat to continue in Spark.",
                has_mesh_fallback=True,
            ),
            "This run only has a mesh / point-cloud preview. Import a 3DGS asset to switch to Spark automatically.",
        )
    return (
        _spatial_stage_html(
            title=record.display_name,
            poster_path=poster,
            note="This run has no spatial artifact yet. Import a Gaussian Splat to continue the world in 3D.",
        ),
        "This run has no spatial asset yet. Import a VIPE / 3DGS export to keep exploring.",
    )


def _points_viewport_idle_html() -> str:
    """Return baseline copy for the Viser tab before a point cloud is routed."""

    return (
        '<section class="wa-points-viewport wa-points-viewport--idle">'
        '<div class="wa-points-fallback-title">Point cloud inspector</div>'
        '<div class="wa-points-fallback-detail">'
        "Install <code>worldfoundry[studio_pointcloud]</code>, run a geometry-heavy model, "
        "and Studio will stream a loopback Viser session for this tab."
        "</div>"
        "</section>"
    )


def _points_viewport_for_record(record: RunRecord | None) -> str:
    """Launch or describe the Viser host using manifest routing metadata."""

    if record is None:
        return _points_viewport_idle_html()
    payload = viewport_payload_from_metadata(dict(record.metadata or {}))
    if payload is None:
        return _points_viewport_idle_html()
    geometry_path = payload.assets_points.point_cloud_path or payload.assets_points.mesh_path
    if not geometry_path:
        return _points_viewport_idle_html()
    path = (Path(record.output_dir) / geometry_path).resolve()
    if not path.exists():
        return _points_viewport_idle_html()
    presentation = STUDIO_VISER.present_geometry(
        run_id=record.run_id,
        geometry_path=path,
        model_id=record.model_id,
    )
    return presentation.html


def _embodied_viewport_idle_html() -> str:
    """Return baseline copy for the embodied simulator tab before a trace is routed."""

    return (
        '<section class="wa-embodied-viewport wa-embodied-viewport--idle">'
        '<div class="wa-embodied-head">'
        '<span>Embodied Sim</span>'
        '<strong>Policy replay surface</strong>'
        '</div>'
        '<div class="wa-embodied-copy">'
        "Run a VLA, VA, WAM, or robot policy model. Studio will route simulator video, action traces, "
        "and episode metadata here instead of leaving them buried in artifacts."
        "</div>"
        '<div class="wa-embodied-pill-row">'
        "<span>observation</span><span>action trace</span><span>sim replay</span>"
        "</div>"
        "</section>"
    )


def _record_asset_path(record: RunRecord, relative_or_abs: str | None) -> Path | None:
    if not relative_or_abs:
        return None
    path = Path(relative_or_abs)
    if not path.is_absolute():
        path = Path(record.output_dir) / path
    resolved = path.expanduser().resolve()
    return resolved if resolved.exists() else None


def _json_file_summary(path: Path | None) -> tuple[str, str]:
    if path is None or path.suffix.lower() not in {".json", ".jsonl"}:
        return "", ""
    try:
        if path.stat().st_size > 2_000_000:
            return f"{path.stat().st_size / 1_000_000:.1f} MB", "large JSON trace"
        if path.suffix.lower() == ".jsonl":
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return f"{len(lines)} jsonl rows", "JSONL trace"
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    if isinstance(payload, dict):
        keys = list(payload.keys())
        preview = ", ".join(str(key) for key in keys[:8])
        return f"{len(keys)} top-level keys", preview
    if isinstance(payload, list):
        return f"{len(payload)} items", "list trace"
    return type(payload).__name__, "JSON payload"


def _embodied_viewport_for_record(record: RunRecord | None) -> str:
    """Render simulator replay and action trace details from viewport metadata."""

    if record is None:
        return _embodied_viewport_idle_html()
    payload = viewport_payload_from_metadata(dict(record.metadata or {}))
    if payload is None:
        return _embodied_viewport_idle_html()

    assets = payload.assets_embodied
    trace_path = _record_asset_path(record, assets.action_trace_path)
    replay_path = _record_asset_path(record, assets.simulator_video_path)
    meta_path = _record_asset_path(record, assets.episode_metadata_path)
    if trace_path is None and replay_path is None and meta_path is None:
        return _embodied_viewport_idle_html()

    replay_html = ""
    if replay_path is not None:
        replay_html = (
            f'<video class="wa-embodied-video" src="{escape(_file_url(str(replay_path)), quote=True)}" '
            'controls playsinline preload="metadata"></video>'
        )

    trace_summary, trace_preview = _json_file_summary(trace_path)
    meta_summary, meta_preview = _json_file_summary(meta_path)
    rows = []
    for label, path, detail in (
        ("Action Trace", trace_path, trace_summary),
        ("Simulator Replay", replay_path, "video replay" if replay_path else ""),
        ("Episode Metadata", meta_path, meta_summary),
    ):
        if path is None:
            continue
        rows.append(
            '<a class="wa-embodied-artifact" '
            f'href="{escape(_file_url(str(path)), quote=True)}" target="_blank" rel="noreferrer">'
            f'<span>{escape(label)}</span>'
            f'<strong>{escape(path.name)}</strong>'
            f'<em>{escape(detail or "artifact")}</em>'
            "</a>"
        )
    preview_lines = []
    if assets.simulator_hint:
        preview_lines.append(f"Simulator: {assets.simulator_hint}")
    if trace_preview:
        preview_lines.append(f"Trace: {trace_preview}")
    if meta_preview:
        preview_lines.append(f"Episode: {meta_preview}")
    if not preview_lines:
        preview_lines.append("Trace artifacts are ready for the simulator or policy debugger.")

    return (
        '<section class="wa-embodied-viewport" data-wa-embodied="1">'
        '<div class="wa-embodied-head">'
        '<span>Embodied Sim</span>'
        f'<strong>{escape(record.display_name)}</strong>'
        "</div>"
        f"{replay_html}"
        '<div class="wa-embodied-copy">'
        + "<br>".join(escape(line) for line in preview_lines)
        + "</div>"
        '<div class="wa-embodied-artifacts">'
        + "".join(rows)
        + "</div>"
        "</section>"
    )

__all__ = [
    "_embodied_viewport_for_record",
    "_embodied_viewport_idle_html",
    "_json_file_summary",
    "_points_viewport_for_record",
    "_points_viewport_idle_html",
    "_record_asset_path",
    "_spatial_stage_for_record",
    "_spatial_stage_html",
]
