from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.visualization.providers.run_record import first_geometry_point_candidate
from worldfoundry.studio.visualization.core.manifest import build_studio_viewports_payload
from worldfoundry.studio.visualization.core.capabilities import available_viewport_kinds, recommend_viewport, summarize_routing_hints
from worldfoundry.studio.visualization.core.manifest import ViewportCapabilities, ViewportKind


def test_first_geometry_point_candidate_resolves_plain_ply(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    ply = root / "plain.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n", encoding="ascii")
    assert (
        first_geometry_point_candidate([], str(root), gs_ply_predicate=lambda _p: False) == "plain.ply"
    )


def test_first_geometry_point_candidate_skips_gs_classified_ply(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    ply = root / "maybe_gs.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n", encoding="ascii")
    assert first_geometry_point_candidate([str(ply)], str(root), gs_ply_predicate=lambda _p: True) is None


def test_build_studio_viewports_payload_registers_points_capability(tmp_path: Path) -> None:
    outp = tmp_path / "mat"
    outp.mkdir()
    ply = outp / "scene.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n", encoding="ascii")
    entry = find_entry("pi3")
    blob = build_studio_viewports_payload(
        entry=entry,
        output_dir=str(outp),
        previews={},
        artifact_paths=[str(ply)],
        gaussian_ply_predicate=lambda _p: False,
    )
    assert blob["schema_version"] == 1
    assert blob["capabilities"]["has_points_cloud"] is True
    assert blob["capabilities"]["has_viser"] is True
    assert blob["assets"]["points"]["point_cloud_path"] == "scene.ply"


def test_build_studio_viewports_payload_registers_embodied_replay(tmp_path: Path) -> None:
    outp = tmp_path / "mat"
    outp.mkdir()
    trace = outp / "action_trace.json"
    trace.write_text('{"actions": [[0, 0, 1]], "task": "pick"}', encoding="utf-8")
    replay = outp / "sim_rollout.mp4"
    replay.write_bytes(b"placeholder")
    entry = find_entry("openvla")
    blob = build_studio_viewports_payload(
        entry=entry,
        output_dir=str(outp),
        previews={},
        artifact_paths=[str(trace), str(replay)],
        gaussian_ply_predicate=lambda _p: False,
    )
    assert blob["recommended"] == "embodied"
    assert blob["capabilities"]["has_embodied_trace"] is True
    assert blob["capabilities"]["has_simulator_replay"] is True
    assert blob["assets"]["embodied"]["action_trace_path"] == "action_trace.json"
    assert blob["assets"]["embodied"]["simulator_video_path"] == "sim_rollout.mp4"


def test_recommend_viewport_prioritizes_streaming_lane() -> None:
    caps = ViewportCapabilities(
        has_streaming=True,
        has_gaussian_splat=True,
        has_points_cloud=True,
        has_rrd=False,
    )
    assert recommend_viewport(caps=caps, has_preview_video=True, has_preview_image=False) is ViewportKind.WORLD
    assert recommend_viewport(caps=caps, has_preview_video=False, has_preview_image=False) is ViewportKind.SPLAT


def test_recommend_viewport_keeps_non_streaming_3d_ahead_of_static_preview() -> None:
    splat_caps = ViewportCapabilities(
        has_streaming=False,
        has_gaussian_splat=True,
        has_points_cloud=True,
    )
    points_caps = ViewportCapabilities(
        has_streaming=False,
        has_gaussian_splat=False,
        has_points_cloud=True,
    )

    assert recommend_viewport(caps=splat_caps, has_preview_video=False, has_preview_image=True) is ViewportKind.SPLAT
    assert recommend_viewport(caps=points_caps, has_preview_video=False, has_preview_image=True) is ViewportKind.POINTS


def test_recommend_viewport_honors_user_override_when_available() -> None:
    caps = ViewportCapabilities(
        has_streaming=True,
        has_gaussian_splat=True,
        has_points_cloud=True,
    )
    available = frozenset({ViewportKind.WORLD, ViewportKind.SPLAT, ViewportKind.POINTS})

    assert (
        recommend_viewport(
            caps=caps,
            has_preview_video=True,
            has_preview_image=False,
            user_viewport_override="points",
            available=available,
        )
        is ViewportKind.POINTS
    )
    assert (
        recommend_viewport(
            caps=caps,
            has_preview_video=True,
            has_preview_image=False,
            user_viewport_override="embodied",
            available=available,
        )
        is ViewportKind.WORLD
    )


def test_recommend_viewport_prioritizes_embodied_replay() -> None:
    caps = ViewportCapabilities(
        has_streaming=False,
        has_gaussian_splat=True,
        has_points_cloud=True,
        has_rrd=False,
        has_embodied_trace=True,
        has_simulator_replay=True,
    )
    assert recommend_viewport(caps=caps, has_preview_video=True, has_preview_image=True) is ViewportKind.EMBODIED


def test_available_viewport_kinds_includes_expected_tabs() -> None:
    caps = ViewportCapabilities(
        has_streaming=True,
        has_gaussian_splat=True,
        has_points_cloud=True,
        has_rrd=False,
    )
    modes = available_viewport_kinds(caps=caps, entry=find_entry("pi3"))
    assert ViewportKind.WORLD in modes
    assert ViewportKind.SPLAT in modes
    assert ViewportKind.POINTS in modes


def test_available_viewport_kinds_accepts_viser_capability_alias() -> None:
    modes = available_viewport_kinds(
        caps=ViewportCapabilities(has_viser=True),
        entry=None,
    )
    assert ViewportKind.POINTS in modes


def test_available_viewport_kinds_includes_embodied_for_action_models() -> None:
    modes = available_viewport_kinds(caps=ViewportCapabilities(), entry=find_entry("openvla"))
    assert ViewportKind.EMBODIED in modes


def test_build_studio_viewports_payload_keeps_external_preview_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    external = tmp_path / "seed.png"
    Image.new("RGB", (4, 4), "red").save(external)
    entry = find_entry("pi3")
    blob = build_studio_viewports_payload(
        entry=entry,
        output_dir=str(run_dir),
        previews={
            "preview_video": None,
            "preview_image": str(external),
            "preview_splat": None,
            "preview_model": None,
            "rrd_path": None,
        },
        artifact_paths=[],
        gaussian_ply_predicate=lambda _p: False,
    )
    assert blob["assets"]["world"]["preview_image"] == external.resolve().as_posix()


def test_summarize_routing_hints_emits_compact_line() -> None:
    record = SimpleNamespace(
        model_id="pi3",
        metadata={
            "studio_viewports": {
                "recommended": "points",
                "schema_version": 1,
                "assets": {
                    "world": {},
                    "splat": {},
                    "points": {
                        "point_cloud_path": "cloud.ply",
                        "mesh_path": None,
                        "camera_path": "cameras.json",
                        "coordinate_frame": "world",
                    },
                },
                "capabilities": {
                    "has_streaming": True,
                    "has_gaussian_splat": False,
                    "has_points_cloud": True,
                    "has_viser": True,
                    "has_rrd": False,
                },
            }
        },
    )
    text = summarize_routing_hints(record)
    assert text is not None
    assert "viewport focus=points" in text
    assert "world" in text and "points" in text


def test_viewport_payload_schema_accepts_documented_aliases() -> None:
    from worldfoundry.studio.visualization.core.manifest import viewport_payload_from_metadata

    payload = viewport_payload_from_metadata(
        {
            "studio_viewports": {
                "schema_version": 2,
                "recommended": "splat",
                "assets": {
                    "splat": {"primary_url": "scene.spz", "format": "spz"},
                    "points": {"camera_path": "cameras.json"},
                },
                "capabilities": {"has_viser": True},
            }
        }
    )

    assert payload is not None
    assert payload.schema_version == 2
    assert payload.assets_splat.primary_path == "scene.spz"
    assert payload.assets_splat.primary_url == "scene.spz"
    assert payload.assets_points.camera_path == "cameras.json"
    assert payload.capabilities.has_points_cloud is True
    assert payload.asdict()["assets"]["splat"]["primary_url"] == "scene.spz"


def test_viser_port_pool_uses_stable_run_slot_and_linear_probe(monkeypatch) -> None:
    from worldfoundry.studio.visualization.backends import viser as viser_host

    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_VISER_PORT_BASE", "19500")
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_VISER_PORT_COUNT", "4")
    first_slot = viser_host._stable_port_offset("run-001", 4)
    blocked_port = 19500 + first_slot
    monkeypatch.setattr(
        viser_host,
        "_port_is_free",
        lambda host, port: host == "127.0.0.1" and port != blocked_port,
    )

    assert viser_host._pick_pool_port("run-001", host="127.0.0.1") == 19500 + ((first_slot + 1) % 4)


def test_viser_explicit_port_bypasses_pool() -> None:
    from worldfoundry.studio.visualization.backends import viser as viser_host

    assert viser_host._resolve_viser_port("run-001", host="127.0.0.1", requested_port=19999) == 19999
