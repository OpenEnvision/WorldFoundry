from __future__ import annotations

import base64
import json
import os
import socket
import time
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1.  backends/__init__.py — imports and __all__
# ---------------------------------------------------------------------------


class TestBackendsInit:
    """Verify the backends package __init__.py exports."""

    def test_all_contains_expected_submodules(self):
        from worldfoundry.studio.visualization.backends import __all__ as all_exports

        assert "frontends" in all_exports
        assert "viser" in all_exports
        assert "world" in all_exports
        assert len(all_exports) == 3

    def test_submodules_importable(self):
        from worldfoundry.studio.visualization.backends import frontends, viser, world

        assert frontends is not None
        assert viser is not None
        assert world is not None


# ---------------------------------------------------------------------------
# 2.  viser.py — ViserPresentation dataclass
# ---------------------------------------------------------------------------


class TestViserPresentation:
    """Tests for ViserPresentation frozen dataclass."""

    def test_construction_with_all_fields(self):
        from worldfoundry.studio.visualization.backends.viser import ViserPresentation

        v = ViserPresentation(html="<iframe>", caption="Ready", url="http://localhost:1234/")
        assert v.html == "<iframe>"
        assert v.caption == "Ready"
        assert v.url == "http://localhost:1234/"

    def test_construction_url_default_empty(self):
        from worldfoundry.studio.visualization.backends.viser import ViserPresentation

        v = ViserPresentation(html="test", caption="cap")
        assert v.url == ""

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.backends.viser import ViserPresentation

        v = ViserPresentation(html="test", caption="cap")
        with pytest.raises(FrozenInstanceError):
            setattr(v, "html", "changed")
        with pytest.raises(FrozenInstanceError):
            setattr(v, "caption", "changed")
        with pytest.raises(FrozenInstanceError):
            setattr(v, "url", "http://changed")

    def test_field_count(self):
        from worldfoundry.studio.visualization.backends.viser import ViserPresentation

        assert len(fields(ViserPresentation)) == 3


# ---------------------------------------------------------------------------
# 3.  viser.py — Constants
# ---------------------------------------------------------------------------


class TestViserConstants:
    """Tests for viser module constants."""

    def test_point_cloud_node(self):
        from worldfoundry.studio.visualization.backends.viser import POINT_CLOUD_NODE

        assert POINT_CLOUD_NODE == "worldfoundry_studio_point_cloud"
        assert isinstance(POINT_CLOUD_NODE, str)

    def test_default_max_points(self):
        from worldfoundry.studio.visualization.backends.viser import DEFAULT_MAX_POINTS

        assert DEFAULT_MAX_POINTS == 400_000

    def test_default_viser_port_base(self):
        from worldfoundry.studio.visualization.backends.viser import DEFAULT_VISER_PORT_BASE

        assert DEFAULT_VISER_PORT_BASE == 18590

    def test_default_viser_port_count(self):
        from worldfoundry.studio.visualization.backends.viser import DEFAULT_VISER_PORT_COUNT

        assert DEFAULT_VISER_PORT_COUNT == 8


# ---------------------------------------------------------------------------
# 4.  viser.py — Helper functions
# ---------------------------------------------------------------------------


class TestViserFunctions:
    """Tests for viser module standalone functions."""

    def test_viser_importable(self):
        from worldfoundry.studio.visualization.backends.viser import viser_importable

        result = viser_importable()
        assert isinstance(result, bool)
        # Result depends on whether 'viser' is installed, either is acceptable.

    def test_env_int_no_env_var(self):
        from worldfoundry.studio.visualization.backends.viser import _env_int

        result = _env_int("NONEXISTENT_VISER_VAR", 42)
        assert result == 42

    def test_env_int_with_env_var(self):
        from worldfoundry.studio.visualization.backends.viser import _env_int

        with patch.dict(os.environ, {"TEST_VISER_INT": "200"}):
            result = _env_int("TEST_VISER_INT", 42)
            assert result == 200

    def test_env_int_invalid_value(self):
        from worldfoundry.studio.visualization.backends.viser import _env_int

        with patch.dict(os.environ, {"TEST_VISER_INT": "not_a_number"}):
            result = _env_int("TEST_VISER_INT", 42)
            assert result == 42

    def test_env_int_minimum_clamp(self):
        from worldfoundry.studio.visualization.backends.viser import _env_int

        result = _env_int("NONEXISTENT_VISER_VAR", -5, minimum=1)
        assert result == 1

    def test_env_int_minimum_passes_valid(self):
        from worldfoundry.studio.visualization.backends.viser import _env_int

        result = _env_int("NONEXISTENT_VISER_VAR", 10, minimum=1)
        assert result == 10

    def test_stable_port_offset_deterministic(self):
        from worldfoundry.studio.visualization.backends.viser import _stable_port_offset

        offset1 = _stable_port_offset("test-run-id", 8)
        offset2 = _stable_port_offset("test-run-id", 8)
        assert offset1 == offset2

    def test_stable_port_offset_different_ids(self):
        from worldfoundry.studio.visualization.backends.viser import _stable_port_offset

        offset1 = _stable_port_offset("run-a", 8)
        offset2 = _stable_port_offset("run-b", 8)
        # Different IDs may or may not produce different offsets, but the
        # function should return a valid int.
        assert isinstance(offset1, int)
        assert isinstance(offset2, int)
        assert 0 <= offset1 < 8
        assert 0 <= offset2 < 8

    def test_stable_port_offset_range(self):
        from worldfoundry.studio.visualization.backends.viser import _stable_port_offset

        for i in range(20):
            offset = _stable_port_offset(f"id-{i}", 8)
            assert 0 <= offset < 8

    def test_port_is_free_high_port(self):
        from worldfoundry.studio.visualization.backends.viser import _port_is_free

        # High ephemeral port should be free.
        assert _port_is_free("127.0.0.1", 59999) is True

    def test_port_is_free_bound_port(self):
        from worldfoundry.studio.visualization.backends.viser import _port_is_free

        # Bind a socket, then check that the port is no longer free.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            bound_port = sock.getsockname()[1]
            # Socket is still held, so port should appear not free.
            assert _port_is_free("127.0.0.1", bound_port) is False

    def test_resolve_viser_port_explicit(self):
        from worldfoundry.studio.visualization.backends.viser import _resolve_viser_port

        result = _resolve_viser_port("run-id", host="127.0.0.1", requested_port=12345)
        assert result == 12345

    def test_resolve_viser_port_pool(self):
        from worldfoundry.studio.visualization.backends.viser import _resolve_viser_port

        with patch("worldfoundry.studio.visualization.backends.viser._pick_pool_port", return_value=18593):
            result = _resolve_viser_port("run-id", host="127.0.0.1", requested_port=None)
        assert isinstance(result, int)
        assert result == 18593

    def test_geometry_imports_available(self):
        from worldfoundry.studio.visualization.backends.viser import _geometry_imports_available

        result = _geometry_imports_available()
        assert isinstance(result, bool)

    def test_fallback_card(self):
        from worldfoundry.studio.visualization.backends.viser import _fallback_card

        html = _fallback_card("Title Here", "Detail Here")
        assert "Title Here" in html
        assert "Detail Here" in html
        assert html.startswith("<section")
        assert html.endswith("</section>")

    def test_fallback_card_escapes_html(self):
        from worldfoundry.studio.visualization.backends.viser import _fallback_card

        html = _fallback_card("<script>alert('x')</script>", "normal detail")
        # The title is passed through escape(), so <script> should be escaped.
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_pick_pool_port_returns_valid_port(self):
        from worldfoundry.studio.visualization.backends.viser import _pick_pool_port

        with patch("worldfoundry.studio.visualization.backends.viser._port_is_free", return_value=True):
            port = _pick_pool_port("test-pick-pool")
        assert isinstance(port, int)
        assert 18590 <= port < 18598

    def test_pick_pool_port_env_override(self):
        from worldfoundry.studio.visualization.backends.viser import _pick_pool_port

        with patch.dict(
            os.environ,
            {
                "WORLDFOUNDRY_STUDIO_VISER_PORT_BASE": "30000",
                "WORLDFOUNDRY_STUDIO_VISER_PORT_COUNT": "4",
            },
        ):
            port = _pick_pool_port("test-env-override")
            assert 30000 <= port < 30004

    def test_env_float_rejects_invalid_values(self):
        from worldfoundry.studio.visualization.backends.viser import _env_float

        with patch.dict(os.environ, {"WF_TEST_FLOAT": "nan"}):
            assert _env_float("WF_TEST_FLOAT", 0.02, minimum=0.001) == 0.02
        with patch.dict(os.environ, {"WF_TEST_FLOAT": "-10"}):
            assert _env_float("WF_TEST_FLOAT", 0.02, minimum=0.001) == 0.001

    def test_coordinate_preset_aliases(self):
        from worldfoundry.studio.visualization.backends.viser import (
            _default_up_direction_for_preset,
            _normalize_alignment_preset,
            _normalize_coordinate_preset,
        )

        assert _normalize_coordinate_preset("opencv") == "opencv-colmap"
        assert _default_up_direction_for_preset("opencv-colmap") == "-y"
        assert _normalize_coordinate_preset("vggt") == "opencv-to-opengl"
        assert _default_up_direction_for_preset("opencv-to-opengl") == "+y"
        assert _normalize_coordinate_preset("unknown") == "asset-native"
        assert _normalize_alignment_preset("canonical") == "first-camera"
        assert _normalize_alignment_preset("off") == "none"
        assert _normalize_alignment_preset("unknown") == "auto"

    def test_verified_vggt_orientation_defaults_are_asset_aware(self):
        from worldfoundry.studio.visualization.backends.viser import viser_orientation_defaults

        glb = viser_orientation_defaults("vggt", "scene.glb")
        assert glb == {
            "coordinate_preset": "asset-native",
            "up_direction": "+y",
            "alignment": "none",
        }
        npz = viser_orientation_defaults("vggt-omega", "predictions.npz")
        assert npz["coordinate_preset"] == "opencv-to-opengl"
        assert npz["up_direction"] == "+y"
        unidepth = viser_orientation_defaults("unidepth-v2-prior", "points.npz")
        assert unidepth == {
            "coordinate_preset": "opencv-to-opengl",
            "up_direction": "+y",
            "alignment": "none",
        }
        exported = viser_orientation_defaults("unidepth-v2-prior", "point_cloud.ply")
        assert exported == {
            "coordinate_preset": "asset-native",
            "up_direction": "+y",
            "alignment": "none",
        }
        camera_aligned = {
            "coordinate_preset": "asset-native",
            "up_direction": "+y",
            "alignment": "first-camera-opengl",
        }
        assert viser_orientation_defaults("loger", "point_cloud/result.ply") == camera_aligned
        assert viser_orientation_defaults("pi3", "point_cloud/result.ply") == camera_aligned
        assert viser_orientation_defaults("unknown", "scene.ply")["up_direction"] == "+z"

    def test_depth_npz_extrinsics_are_w2c_and_explicit_c2w_takes_precedence(self):
        import numpy as np

        from worldfoundry.studio.visualization.backends.viser import _depth_points_from_npz

        depth = np.array([[[2.0]]])
        intrinsics = np.eye(3)[None]
        w2c = np.eye(4)[None]
        w2c[0, :3, 3] = [-3.0, 1.0, -4.0]

        result = _depth_points_from_npz({"depth": depth, "intrinsics": intrinsics, "extrinsics": w2c})
        # The camera center is (3, -1, 4), so the optical-axis sample at
        # camera z=2 must land at (3, -1, 6), not at the w2c translation.
        assert result is not None
        points, valid_mask = result
        np.testing.assert_allclose(points, [[3.0, -1.0, 6.0]], atol=1e-8)
        np.testing.assert_array_equal(valid_mask, [True])

        c2w = np.eye(4)[None]
        c2w[0, :3, 3] = [7.0, 8.0, 9.0]
        explicit_result = _depth_points_from_npz(
            {
                "depth": depth,
                "intrinsics": intrinsics,
                "extrinsics": w2c,
                "camera_to_world": c2w,
            }
        )
        assert explicit_result is not None
        explicit_points, explicit_mask = explicit_result
        np.testing.assert_allclose(explicit_points, [[7.0, 8.0, 11.0]], atol=1e-8)
        np.testing.assert_array_equal(explicit_mask, [True])

    def test_parse_up_direction_vector_and_fallback(self):
        from worldfoundry.studio.visualization.backends.viser import _parse_up_direction

        assert _parse_up_direction("-Y", default="+z") == "-y"
        assert _parse_up_direction("0, -1, 0", default="+z") == (0.0, -1.0, 0.0)
        assert _parse_up_direction("0 0 0", default="+z") == "+z"

    def test_load_camera_poses_and_first_camera_alignment(self, tmp_path):
        import numpy as np

        from worldfoundry.studio.visualization.backends.viser import (
            _alignment_transform_matrix,
            _load_camera_poses,
            _transform_camera_poses,
            _transform_points,
        )

        cloud_dir = tmp_path / "point_cloud"
        pose_dir = tmp_path / "camera_poses"
        cloud_dir.mkdir()
        pose_dir.mkdir()
        cloud_path = cloud_dir / "result.ply"
        cloud_path.write_text("ply\n", encoding="utf-8")

        pose0 = np.eye(4)
        pose0[:3, 3] = [1.0, 2.0, 3.0]
        pose1 = np.eye(4)
        pose1[:3, 3] = [2.0, 2.0, 3.0]
        (pose_dir / "pose_0000.json").write_text(json.dumps({"camera_to_world": pose0.tolist()}), encoding="utf-8")
        (pose_dir / "pose_0001.json").write_text(json.dumps({"camera_to_world": pose1.tolist()}), encoding="utf-8")

        poses = _load_camera_poses(cloud_path)
        assert len(poses) == 2
        alignment = _alignment_transform_matrix(poses, "first-camera")
        aligned_poses = _transform_camera_poses(poses, alignment)
        np.testing.assert_allclose(aligned_poses[0], np.eye(4), atol=1e-8)
        np.testing.assert_allclose(aligned_poses[1][:3, 3], [1.0, 0.0, 0.0], atol=1e-8)
        points = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
        np.testing.assert_allclose(_transform_points(points, alignment), [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_load_camera_poses_from_model_named_sidecar(self, tmp_path):
        import numpy as np

        from worldfoundry.studio.visualization.backends.viser import _load_camera_poses

        cloud_path = tmp_path / "scene_point_cloud.ply"
        cloud_path.write_text("ply\n", encoding="utf-8")
        expected = np.stack([np.eye(4), np.eye(4)])
        expected[1, 0, 3] = 1.0
        np.save(tmp_path / "scene_camera_poses.npy", expected)

        poses = _load_camera_poses(cloud_path)
        assert len(poses) == 2
        np.testing.assert_allclose(poses, expected)

    def test_initial_camera_pose_focuses_primary_geometry(self):
        import numpy as np

        from worldfoundry.studio.visualization.backends.viser import _initial_camera_pose

        points = np.stack(
            [
                np.linspace(-2.0, 2.0, 200),
                np.linspace(-1.0, 1.0, 200),
                np.linspace(-4.0, -2.0, 200),
            ],
            axis=1,
        )
        position, target, up = _initial_camera_pose(points, None, "+z")

        np.testing.assert_allclose(target, [0.0, 0.0, -3.0], atol=1e-8)
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0])
        assert 4.0 < np.linalg.norm(position - target) < 6.0

    def test_initial_camera_pose_and_point_size_ignore_sparse_outliers(self):
        import numpy as np

        from worldfoundry.studio.visualization.backends.viser import _adaptive_point_size, _initial_camera_pose

        rng = np.random.default_rng(7)
        primary = rng.normal(0.0, 0.2, size=(1000, 3))
        outliers = np.full((20, 3), 100.0)
        points = np.concatenate([primary, outliers], axis=0)

        position, target, _ = _initial_camera_pose(points, None, "+y")
        assert np.linalg.norm(target) < 0.1
        assert np.linalg.norm(position - target) < 3.0
        assert _adaptive_point_size(points) < 0.01


# ---------------------------------------------------------------------------
# 5.  viser.py — StudioViserService
# ---------------------------------------------------------------------------


class TestStudioViserService:
    """Tests for StudioViserService class."""

    def test_shutdown_when_no_server(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService

        svc = StudioViserService()
        # shutdown should work gracefully when no server is running.
        svc.shutdown()
        assert svc._server is None

    def test_init(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService

        svc = StudioViserService()
        assert svc._lock is not None
        assert svc._server is None

    def test_present_geometry_no_viser(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService

        svc = StudioViserService()
        # When viser is not importable, it returns a fallback ViserPresentation.
        with patch("worldfoundry.studio.visualization.backends.viser.viser_importable", return_value=False):
            result = svc.present_geometry(
                run_id="test-run",
                geometry_path=Path("/nonexistent/test.ply"),
            )
            assert result.html.startswith("<section")
            assert "not installed" in result.caption.lower() or "not installed" in result.html.lower()

    def test_present_geometry_missing_file(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService, ViserPresentation

        svc = StudioViserService()
        with patch("worldfoundry.studio.visualization.backends.viser.viser_importable", return_value=True):
            result = svc.present_geometry(
                run_id="test-run",
                geometry_path=Path("/nonexistent/missing.ply"),
            )
            assert isinstance(result, ViserPresentation)
            assert "missing" in result.html.lower() or "not on disk" in result.caption.lower()

    def test_present_geometry_missing_dependencies(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService, ViserPresentation

        svc = StudioViserService()
        with patch("worldfoundry.studio.visualization.backends.viser.viser_importable", return_value=True):
            with patch(
                "worldfoundry.studio.visualization.backends.viser._geometry_imports_available", return_value=False
            ):
                # Create a temp file so .exists() returns True.
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    result = svc.present_geometry(
                        run_id="test-run",
                        geometry_path=tmp_path,
                    )
                    assert isinstance(result, ViserPresentation)
                    assert "dependencies" in result.html.lower() or "dependencies" in result.caption.lower()
                finally:
                    tmp_path.unlink(missing_ok=True)

    def test_present_geometry_non_loopback_host_rejected(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService, ViserPresentation

        svc = StudioViserService()
        with patch("worldfoundry.studio.visualization.backends.viser.viser_importable", return_value=True):
            with patch(
                "worldfoundry.studio.visualization.backends.viser._geometry_imports_available", return_value=True
            ):
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    result = svc.present_geometry(
                        run_id="test-run",
                        geometry_path=tmp_path,
                        host="10.0.0.1",
                    )
                    assert isinstance(result, ViserPresentation)
                    assert "rejected" in result.html.lower() or "loopback" in result.caption.lower()
                finally:
                    tmp_path.unlink(missing_ok=True)

    def test_studio_viser_global_instance(self):
        from worldfoundry.studio.visualization.backends.viser import STUDIO_VISER, StudioViserService

        assert isinstance(STUDIO_VISER, StudioViserService)


class TestWorkspaceVisualizerParams:
    """Tests for Workspace visualizer parameter plumbing."""

    def test_points_env_overrides_are_whitelisted(self):
        from worldfoundry.studio.workspace.app import _visualizer_env_overrides

        overrides = _visualizer_env_overrides(
            "points",
            {
                "coordinate_preset": "opencv-colmap",
                "point_size": 0.003,
                "max_points": 1234,
                "alignment": "first-camera",
                "show_cameras": True,
                "camera_size": 0.05,
                "ignored": "value",
            },
        )
        assert overrides == {
            "WORLDFOUNDRY_STUDIO_VISER_ALIGNMENT": "first-camera",
            "WORLDFOUNDRY_STUDIO_VISER_CAMERA_SIZE": "0.05",
            "WORLDFOUNDRY_STUDIO_VISER_COORDINATE_PRESET": "opencv-colmap",
            "WORLDFOUNDRY_STUDIO_VISER_MAX_POINTS": "1234",
            "WORLDFOUNDRY_STUDIO_VISER_POINT_SIZE": "0.003",
            "WORLDFOUNDRY_STUDIO_VISER_SHOW_CAMERAS": "1",
        }
        assert _visualizer_env_overrides("spark", {"point_size": 0.003}) == {}

    def test_vggt_visualizer_defaults_preserve_official_glb_alignment(self):
        from worldfoundry.studio.workspace.app import _resolved_visualizer_params

        params = _resolved_visualizer_params(
            "points",
            model_id="vggt",
            asset_path="scene.glb",
            requested={"coordinate_preset": "auto", "up_direction": "auto"},
        )
        assert params["coordinate_preset"] == "asset-native"
        assert params["up_direction"] == "+y"
        assert params["alignment"] == "none"

    def test_vggt_raw_predictions_are_converted_from_opencv(self):
        from worldfoundry.studio.workspace.app import _resolved_visualizer_params

        params = _resolved_visualizer_params(
            "points",
            model_id="vggt-omega",
            asset_path="predictions.npz",
            requested={},
        )
        assert params["coordinate_preset"] == "opencv-to-opengl"
        assert params["up_direction"] == "+y"

    def test_visualizer_reuse_requires_same_asset_and_model(self):
        from worldfoundry.studio.workspace.app import ManagedVisualizer, _visualizer_reusable

        record = ManagedVisualizer(
            mode="points",
            title="Viser",
            url="http://127.0.0.1:18590/",
            health_url="http://127.0.0.1:18590/",
            host="127.0.0.1",
            port=18590,
            model_id="vggt",
            asset_path="/tmp/a.glb",
            command=[],
            log_path=None,
            started_at=time.time(),
            params={"up_direction": "+y"},
            process=None,
            external=True,
        )
        assert _visualizer_reusable(
            record,
            model_id="vggt",
            asset_path="/tmp/a.glb",
            params={"up_direction": "+y"},
        )
        assert not _visualizer_reusable(
            record,
            model_id="vggt",
            asset_path="/tmp/b.glb",
            params={"up_direction": "+y"},
        )
        assert not _visualizer_reusable(
            record,
            model_id="vggt-omega",
            asset_path="/tmp/a.glb",
            params={"up_direction": "+y"},
        )

    def test_rerun_renderer_prefers_webgpu_with_explicit_webgl_fallback(self):
        from worldfoundry.studio.workspace.app import _rerun_renderer

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORLDFOUNDRY_STUDIO_RERUN_RENDERER", None)
            assert _rerun_renderer() == "webgpu"
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_RERUN_RENDERER": "webgl"}):
            assert _rerun_renderer() == "webgl"
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_RERUN_RENDERER": "invalid"}):
            assert _rerun_renderer() == "webgpu"

    def test_visualizer_status_includes_params(self):
        from worldfoundry.studio.workspace.app import ManagedVisualizer, _visualizer_status

        record = ManagedVisualizer(
            mode="points",
            title="Viser Point Cloud",
            url="http://127.0.0.1:18590/",
            health_url="http://127.0.0.1:18590/",
            host="127.0.0.1",
            port=18590,
            model_id="vggt-omega",
            asset_path="/tmp/cloud.ply",
            command=[],
            log_path=None,
            started_at=time.time(),
            params={"coordinate_preset": "opencv-colmap"},
            process=None,
            external=True,
        )
        status = _visualizer_status(record)
        assert status["params"] == {"coordinate_preset": "opencv-colmap"}

    def test_adaptive_point_size_tracks_scene_scale(self):
        import numpy as np

        from worldfoundry.studio.visualization.backends.viser import _adaptive_point_size

        unit_scene = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert _adaptive_point_size(unit_scene * 10.0) == pytest.approx(_adaptive_point_size(unit_scene) * 10.0)

    def test_depth_to_world_points_uses_camera_to_world_pose(self):
        import numpy as np

        from worldfoundry.core.geometry.transforms import depth_to_world_points

        intrinsics = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        pose = np.eye(4)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        points = depth_to_world_points(np.ones((1, 2)), intrinsics, pose)
        np.testing.assert_allclose(points, [[[1.0, 2.0, 4.0], [1.5, 2.0, 4.0]]])


# ---------------------------------------------------------------------------
# 6.  frontends.py — Constants
# ---------------------------------------------------------------------------


class TestFrontendsConstants:
    """Tests for frontends module constants."""

    def test_frontend_mode_strings(self):
        from worldfoundry.studio.visualization.backends.frontends import (
            EMBODIED_FRONTEND,
            MEDIA_FRONTEND,
            POINTS_FRONTEND,
            RERUN_FRONTEND,
            SPARK_FRONTEND,
            WORLD_FRONTEND,
        )

        assert WORLD_FRONTEND == "world"
        assert POINTS_FRONTEND == "points"
        assert EMBODIED_FRONTEND == "embodied"
        assert SPARK_FRONTEND == "spark"
        assert MEDIA_FRONTEND == "media"
        assert RERUN_FRONTEND == "rerun"

    def test_default_rerun_command_assigns_distinct_data_ports(self):
        from worldfoundry.studio.visualization.backends.frontends import _default_rerun_command_template

        command = _default_rerun_command_template().format(
            asset="scene.rrd",
            bind_host="127.0.0.1",
            grpc_port=9878,
            ws_port=9877,
            port=9876,
        )
        assert "--serve-web" in command
        assert "--web-viewer " not in command
        assert "--bind 127.0.0.1" in command
        assert "--port 9878" in command
        assert "--ws-server-port 9877" in command
        assert "--web-viewer-port 9876" in command

    def test_spark_asset_exts(self):
        from worldfoundry.studio.visualization.backends.frontends import SPARK_ASSET_EXTS

        assert isinstance(SPARK_ASSET_EXTS, set)
        assert ".ply" in SPARK_ASSET_EXTS
        assert ".spz" in SPARK_ASSET_EXTS
        assert ".splat" in SPARK_ASSET_EXTS
        assert ".ksplat" in SPARK_ASSET_EXTS
        assert ".sog" in SPARK_ASSET_EXTS

    def test_media_asset_exts(self):
        from worldfoundry.studio.visualization.backends.frontends import MEDIA_ASSET_EXTS

        assert isinstance(MEDIA_ASSET_EXTS, set)
        # Image formats
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]:
            assert ext in MEDIA_ASSET_EXTS
        # Video formats
        for ext in [".mp4", ".mov", ".webm", ".mkv", ".avi"]:
            assert ext in MEDIA_ASSET_EXTS
        # Audio formats
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            assert ext in MEDIA_ASSET_EXTS

    def test_path_constants(self):
        from worldfoundry.studio.visualization.backends.frontends import (
            SPARK_MODULE_PATH,
            THREE_CORE_MODULE_PATH,
            THREE_MODULE_PATH,
        )

        assert isinstance(SPARK_MODULE_PATH, Path)
        assert isinstance(THREE_MODULE_PATH, Path)
        assert isinstance(THREE_CORE_MODULE_PATH, Path)
        # Parent-child relationships.

    def test_studio_visualizations_registry(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS
        from worldfoundry.studio.visualization.registry import StudioVisualizationRegistry

        assert isinstance(STUDIO_VISUALIZATIONS, StudioVisualizationRegistry)
        assert "world" in STUDIO_VISUALIZATIONS.modes
        assert "points" in STUDIO_VISUALIZATIONS.modes
        assert "spark" in STUDIO_VISUALIZATIONS.modes
        assert "embodied" in STUDIO_VISUALIZATIONS.modes
        assert "rerun" in STUDIO_VISUALIZATIONS.modes
        assert "media" in STUDIO_VISUALIZATIONS.modes

    def test_native_frontends(self):
        from worldfoundry.studio.visualization.backends.frontends import NATIVE_FRONTENDS

        assert isinstance(NATIVE_FRONTENDS, frozenset)
        assert "world" in NATIVE_FRONTENDS
        assert "points" in NATIVE_FRONTENDS

    def test_default_frontend_ports(self):
        from worldfoundry.studio.visualization.backends.frontends import DEFAULT_FRONTEND_PORTS

        assert isinstance(DEFAULT_FRONTEND_PORTS, dict)
        assert DEFAULT_FRONTEND_PORTS["world"] == 7868
        assert DEFAULT_FRONTEND_PORTS["points"] == 18590
        assert DEFAULT_FRONTEND_PORTS["spark"] == 8765
        assert DEFAULT_FRONTEND_PORTS["embodied"] == 18610
        assert DEFAULT_FRONTEND_PORTS["rerun"] == 9876
        assert DEFAULT_FRONTEND_PORTS["media"] == 18720


# ---------------------------------------------------------------------------
# 7.  frontends.py — StudioVisualizationBackend instances in registry
# ---------------------------------------------------------------------------


class TestFrontendsRegistryBackendInstances:
    """Verify each registered backend's attributes."""

    def test_world_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS

        backend = STUDIO_VISUALIZATIONS.backend_for("world")
        assert backend.mode == "world"
        assert backend.title == "Interactive World Model"
        assert backend.default_port == 7868
        assert "interactive-world" in backend.aliases
        assert "world-model" in backend.aliases
        assert backend.native is True
        assert backend.serve is not None

    def test_points_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS

        backend = STUDIO_VISUALIZATIONS.backend_for("points")
        assert backend.mode == "points"
        assert backend.title == "Viser Geometry Viewer"
        assert backend.default_port == 18590
        assert "viser" in backend.aliases
        assert "geometry" in backend.aliases
        assert "pointcloud" in backend.aliases
        assert "point-cloud" in backend.aliases
        assert backend.native is True

    def test_spark_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS

        backend = STUDIO_VISUALIZATIONS.backend_for("spark")
        assert backend.mode == "spark"
        assert backend.title == "Spark Gaussian Splat Viewer"
        assert backend.default_port == 8765
        assert "3dgs" in backend.aliases
        assert "splat" in backend.aliases
        assert "gaussian-splat" in backend.aliases
        assert backend.native is True

    def test_embodied_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS

        backend = STUDIO_VISUALIZATIONS.backend_for("embodied")
        assert backend.mode == "embodied"
        assert backend.title == "Embodied Simulator Bridge"
        assert backend.default_port == 18610
        assert "sim" in backend.aliases
        assert "simulator" in backend.aliases
        assert backend.native is True

    def test_rerun_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS

        backend = STUDIO_VISUALIZATIONS.backend_for("rerun")
        assert backend.mode == "rerun"
        assert backend.default_port == 9876
        assert "rrd" in backend.aliases
        assert backend.native is True

    def test_media_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS

        backend = STUDIO_VISUALIZATIONS.backend_for("media")
        assert backend.mode == "media"
        assert backend.title == "Media Preview"
        assert backend.default_port == 18720
        assert "preview" in backend.aliases
        assert "video" in backend.aliases
        assert "image" in backend.aliases
        assert backend.native is True


# ---------------------------------------------------------------------------
# 8.  frontends.py — Functions
# ---------------------------------------------------------------------------


class TestFrontendsFunctions:
    """Tests for frontends module standalone functions."""

    def test_profile_mode_lambda(self):
        from worldfoundry.studio.visualization.backends.frontends import _profile_mode

        fn = _profile_mode("world")
        assert callable(fn)

    def test_host_for_frontend_defaults(self):
        from worldfoundry.studio.visualization.backends.frontends import host_for_frontend
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model")
        result = host_for_frontend(lc)
        assert result == "127.0.0.1"

    def test_host_for_frontend_explicit_host(self):
        from worldfoundry.studio.visualization.backends.frontends import host_for_frontend
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model", host="0.0.0.0")
        result = host_for_frontend(lc)
        assert result == "0.0.0.0"

    def test_host_for_frontend_env_override(self):
        from worldfoundry.studio.visualization.backends.frontends import host_for_frontend
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model")
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_HOST": "10.0.0.5"}):
            result = host_for_frontend(lc)
            assert result == "10.0.0.5"

    def test_host_for_frontend_blank_host_falls_to_default(self):
        from worldfoundry.studio.visualization.backends.frontends import host_for_frontend
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model", host="   ")
        result = host_for_frontend(lc)
        assert result == "127.0.0.1"

    def test_port_for_frontend_explicit_port(self):
        from worldfoundry.studio.visualization.backends.frontends import port_for_frontend
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model", port=9999)
        result = port_for_frontend(lc, "world")
        assert result == 9999

    def test_port_for_frontend_default(self):
        from worldfoundry.studio.visualization.backends.frontends import port_for_frontend
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model")
        result = port_for_frontend(lc, "world")
        assert result == 7868

    def test_port_for_frontend_env_override(self):
        from worldfoundry.studio.visualization.backends.frontends import port_for_frontend
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model")
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_WORLD_PORT": "5555"}):
            result = port_for_frontend(lc, "world")
            assert result == 5555

    def test_port_from_url_valid(self):
        from worldfoundry.studio.visualization.backends.frontends import _port_from_url

        assert _port_from_url("http://localhost:1234/") == 1234

    def test_port_from_url_no_port(self):
        from worldfoundry.studio.visualization.backends.frontends import _port_from_url

        assert _port_from_url("http://localhost/") is None

    def test_port_from_url_invalid_port(self):
        from worldfoundry.studio.visualization.backends.frontends import _port_from_url

        assert _port_from_url("http://localhost:abc/") is None

    def test_js_string_simple(self):
        from worldfoundry.studio.visualization.backends.frontends import _js_string

        result = _js_string("hello")
        assert result == '"hello"'

    def test_js_string_empty(self):
        from worldfoundry.studio.visualization.backends.frontends import _js_string

        result = _js_string("")
        assert result == '""'

    def test_js_string_backslash(self):
        from worldfoundry.studio.visualization.backends.frontends import _js_string

        result = _js_string("C:\\Users")
        assert result == '"C:\\\\Users"'

    def test_js_string_quotes(self):
        from worldfoundry.studio.visualization.backends.frontends import _js_string

        result = _js_string('say "hi"')
        assert result == '"say \\"hi\\""'

    def test_js_string_newline(self):
        from worldfoundry.studio.visualization.backends.frontends import _js_string

        result = _js_string("line1\nline2")
        assert result == '"line1\\nline2"'

    def test_strip_html_with_tags(self):
        from worldfoundry.studio.visualization.backends.frontends import _strip_html

        result = _strip_html("<b>bold</b> text")
        assert result == "bold text"

    def test_strip_html_no_tags(self):
        from worldfoundry.studio.visualization.backends.frontends import _strip_html

        result = _strip_html("plain text")
        assert result == "plain text"

    def test_strip_html_empty(self):
        from worldfoundry.studio.visualization.backends.frontends import _strip_html

        result = _strip_html("")
        assert result == ""

    def test_strip_html_none(self):
        from worldfoundry.studio.visualization.backends.frontends import _strip_html

        invalid_value: Any = None
        result = _strip_html(invalid_value)
        assert result == ""

    def test_strip_html_multiple_tags(self):
        from worldfoundry.studio.visualization.backends.frontends import _strip_html

        result = _strip_html("<p>Hello <em>world</em></p>")
        assert result == "Hello world"

    def test_optional_asset_none(self):
        from worldfoundry.studio.visualization.backends.frontends import _optional_asset

        invalid_value: Any = None
        result = _optional_asset(invalid_value)
        assert result is None

    def test_optional_asset_empty_string(self):
        from worldfoundry.studio.visualization.backends.frontends import _optional_asset

        result = _optional_asset("")
        assert result is None

    def test_optional_asset_whitespace(self):
        from worldfoundry.studio.visualization.backends.frontends import _optional_asset

        result = _optional_asset("   ")
        assert result is None

    def test_optional_asset_nonexistent_path(self):
        from worldfoundry.studio.visualization.backends.frontends import _optional_asset

        with pytest.raises(SystemExit):
            _optional_asset("/nonexistent/path/to/file.ply")

    def test_optional_asset_existing_file(self):
        import tempfile

        from worldfoundry.studio.visualization.backends.frontends import _optional_asset

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = _optional_asset(tmp_path)
            assert result is not None
            assert result.exists()
        finally:
            os.unlink(tmp_path)

    def test_required_existing_asset_none(self):
        from worldfoundry.studio.visualization.backends.frontends import _required_existing_asset

        invalid_value: Any = None
        with pytest.raises(SystemExit):
            _required_existing_asset(invalid_value, frontend="test", hint="pass --asset path")

    def test_required_existing_asset_existing_file(self):
        import tempfile

        from worldfoundry.studio.visualization.backends.frontends import _required_existing_asset

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = _required_existing_asset(tmp_path, frontend="test", hint="pass --asset path")
            assert result is not None
            assert result.exists()
        finally:
            os.unlink(tmp_path)

    def test_spark_allowed_roots_no_asset(self):
        from worldfoundry.studio.visualization.backends.frontends import _spark_allowed_roots

        roots = _spark_allowed_roots(None)
        assert isinstance(roots, tuple)
        assert len(roots) >= 1  # At least cwd.

    def test_spark_allowed_roots_with_asset(self):
        import tempfile

        from worldfoundry.studio.visualization.backends.frontends import _spark_allowed_roots

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            roots = _spark_allowed_roots(tmp_path)
            assert isinstance(roots, tuple)
            # Should include the asset's parent dir.
            assert tmp_path.parent.resolve() in roots
        finally:
            os.unlink(tmp_path)

    def test_explicit_points_port_none(self):
        from worldfoundry.studio.visualization.backends.frontends import _explicit_points_port
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model")
        result = _explicit_points_port(lc)
        assert result is None

    def test_explicit_points_port_with_port(self):
        from worldfoundry.studio.visualization.backends.frontends import _explicit_points_port
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model", port=18591)
        result = _explicit_points_port(lc)
        assert result == 18591

    def test_explicit_points_port_env_override(self):
        from worldfoundry.studio.visualization.backends.frontends import _explicit_points_port
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        lc = StudioLaunchConfig(model_id="test-model")
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_POINTS_PORT": "18595"}):
            result = _explicit_points_port(lc)
            assert result == 18595

    def test_print_remote_access_is_callable(self):
        from worldfoundry.studio.visualization.backends.frontends import print_remote_access

        assert callable(print_remote_access)

    def test_resolve_frontend_mode_callable(self):
        from worldfoundry.studio.visualization.backends.frontends import resolve_frontend_mode

        assert callable(resolve_frontend_mode)

    def test_print_tunnel_for_url_valid(self):
        from worldfoundry.studio.visualization.backends.frontends import _print_tunnel_for_url

        # Should print without error.
        _print_tunnel_for_url("http://127.0.0.1:7868")

    def test_print_tunnel_for_url_missing_hostname(self):
        from worldfoundry.studio.visualization.backends.frontends import _print_tunnel_for_url

        # Should gracefully handle URL without hostname/port.
        _print_tunnel_for_url("no-url")

    def test_wait_forever_callable(self):
        from worldfoundry.studio.visualization.backends.frontends import _wait_forever

        assert callable(_wait_forever)


# ---------------------------------------------------------------------------
# 9.  frontends.py — HTML generators
# ---------------------------------------------------------------------------


class TestFrontendsHTMLGenerators:
    """Tests for spark_viewer_html and media_viewer_html."""

    def test_spark_viewer_html_structure(self):
        from worldfoundry.studio.visualization.backends.frontends import spark_viewer_html

        html = spark_viewer_html(title="My Scene", default_asset="/tmp/scene.splat")
        assert html.startswith("<!doctype")
        assert "My Scene" in html
        assert "/tmp/scene.splat" in html
        assert "<canvas" in html
        assert "spark" in html.lower()

    def test_spark_viewer_html_escapes_title(self):
        from worldfoundry.studio.visualization.backends.frontends import spark_viewer_html

        html = spark_viewer_html(title="<script>alert(1)</script>", default_asset="test.ply")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_media_viewer_html_video(self):
        from worldfoundry.studio.visualization.backends.frontends import media_viewer_html

        html = media_viewer_html(title="My Video", asset_path=Path("preview.mp4"))
        assert html.startswith("<!doctype")
        assert "My Video" in html
        assert "<video" in html

    def test_media_viewer_html_image(self):
        from worldfoundry.studio.visualization.backends.frontends import media_viewer_html

        html = media_viewer_html(title="My Image", asset_path=Path("preview.png"))
        assert "<img" in html

    def test_media_viewer_html_audio(self):
        from worldfoundry.studio.visualization.backends.frontends import media_viewer_html

        html = media_viewer_html(title="My Audio", asset_path=Path("preview.wav"))
        assert "<audio" in html

    def test_media_viewer_html_escapes_title(self):
        from worldfoundry.studio.visualization.backends.frontends import media_viewer_html

        html = media_viewer_html(title="<b>bold</b>", asset_path=Path("test.mp4"))
        assert "<b>" not in html
        assert "&lt;b&gt;" in html

    def test_media_viewer_html_escapes_filename(self):
        from worldfoundry.studio.visualization.backends.frontends import media_viewer_html

        html = media_viewer_html(title="Test", asset_path=Path("<script>.mp4"))
        assert "<script>.mp4" not in html


# ---------------------------------------------------------------------------
# 10.  frontends.py — Handler classes (structural checks)
# ---------------------------------------------------------------------------


class TestFrontendsHandlerClasses:
    """Tests for MediaViewerHandler and SparkViewerHandler class structures."""

    def test_media_viewer_handler_class_exists(self):
        from worldfoundry.studio.visualization.backends.frontends import MediaViewerHandler

        assert MediaViewerHandler.server_version == "WorldFoundryMediaFrontend/1.0"

    def test_spark_viewer_handler_class_exists(self):
        from worldfoundry.studio.visualization.backends.frontends import SparkViewerHandler

        assert SparkViewerHandler.server_version == "WorldFoundrySparkFrontend/1.0"


# ---------------------------------------------------------------------------
# 12.  world.py — Constants
# ---------------------------------------------------------------------------


class TestWorldConstants:
    """Tests for world module constants."""

    def test_default_fps(self):
        from worldfoundry.studio.visualization.backends.world import DEFAULT_FPS

        assert DEFAULT_FPS == 16


# ---------------------------------------------------------------------------
# 14.  world.py — Helper functions
# ---------------------------------------------------------------------------


class TestWorldHelperFunctions:
    """Tests for world module standalone helper functions."""

    def test_html_escape_amp(self):
        from worldfoundry.studio.visualization.backends.world import _html_escape

        assert _html_escape("test&value") == "test&amp;value"

    def test_html_escape_lt(self):
        from worldfoundry.studio.visualization.backends.world import _html_escape

        assert _html_escape("<script>") == "&lt;script&gt;"

    def test_html_escape_gt(self):
        from worldfoundry.studio.visualization.backends.world import _html_escape

        assert _html_escape("a>b") == "a&gt;b"

    def test_html_escape_quot(self):
        from worldfoundry.studio.visualization.backends.world import _html_escape

        assert _html_escape('say"hello"') == "say&quot;hello&quot;"

    def test_html_escape_empty(self):
        from worldfoundry.studio.visualization.backends.world import _html_escape

        assert _html_escape("") == ""

    def test_html_escape_combined(self):
        from worldfoundry.studio.visualization.backends.world import _html_escape

        result = _html_escape("<b>bold &amp; 'quoted'</b>")
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_decode_data_url_with_header(self):
        from worldfoundry.studio.visualization.backends.world import _decode_data_url

        data = base64.b64encode(b"hello world").decode()
        result, mime = _decode_data_url(f"data:image/png;base64,{data}")
        assert result == b"hello world"
        assert mime == "image/png"

    def test_decode_data_url_no_header(self):
        from worldfoundry.studio.visualization.backends.world import _decode_data_url

        data = base64.b64encode(b"raw data").decode()
        result, mime = _decode_data_url(data)
        assert result == b"raw data"
        assert mime == "application/octet-stream"

    def test_decode_data_url_with_mime_but_no_base64_prefix(self):
        from worldfoundry.studio.visualization.backends.world import _decode_data_url

        data = base64.b64encode(b"test").decode()
        result, mime = _decode_data_url(f"data:video/mp4;base64,{data}")
        assert mime == "video/mp4"


# ---------------------------------------------------------------------------
# 15.  world.py — HTML generators
# ---------------------------------------------------------------------------


def _make_catalog_entry(**overrides):
    """Create a minimal CatalogEntry for testing."""
    defaults: dict[str, Any] = {
        "model_id": "test-model",
        "display_name": "Test Display",
        "module_path": "test.module",
        "class_name": "TestClass",
        "family": "test-family",
        "category": "Video",
        "summary": "Test summary",
        "call_params": (),
        "stream_params": (),
        "load_params": (),
        "default_interactions": (),
        "default_load_kwargs": {},
        "default_call_kwargs": {},
        "extra_variants": (),
        "suggested_task_types": (),
        "aliases": (),
        "tags": (),
        "env_hints": (),
    }
    defaults.update(overrides)
    from worldfoundry.studio.inference.catalog import CatalogEntry

    return CatalogEntry(**defaults)


class TestWorldHTMLGenerators:
    """Tests for world_frontend_html, CSS, JS, favicon."""

    def test_world_frontend_html_contains_title(self):
        from worldfoundry.studio.visualization.backends.world import world_frontend_html
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        entry = _make_catalog_entry(model_id="test-model", display_name="Test Display")
        lc = StudioLaunchConfig(model_id="test-model")
        html = world_frontend_html(entry, lc)
        assert "Test Display" in html
        assert "test-model" in html

    def test_world_frontend_html_structure(self):
        from worldfoundry.studio.visualization.backends.world import world_frontend_html
        from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

        entry = _make_catalog_entry(model_id="m1", display_name="M1")
        lc = StudioLaunchConfig(model_id="m1")
        html = world_frontend_html(entry, lc)
        assert html.startswith("<!doctype")
        assert "<html" in html
        assert "</html>" in html

    def test_world_frontend_css_non_empty(self):
        from worldfoundry.studio.visualization.backends.world import world_frontend_css

        css = world_frontend_css()
        assert len(css) > 100
        assert ":root" in css

    def test_world_frontend_js_non_empty(self):
        from worldfoundry.studio.visualization.backends.world import world_frontend_js

        js = world_frontend_js()
        assert len(js) > 100
        assert "state" in js

    def test_world_controls_do_not_transform_the_viewport(self):
        from worldfoundry.studio.visualization.backends.world import (
            world_frontend_css,
            world_frontend_js,
        )

        js = world_frontend_js()
        css = world_frontend_css()

        assert "startPreviewMotion" not in js
        assert "--preview-offset" not in js
        assert "--preview-offset" not in css
        assert "event.preventDefault()" in js

    def test_world_favicon_svg(self):
        from worldfoundry.studio.visualization.backends.world import world_favicon_svg

        svg = world_favicon_svg()
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_world_frontend_css_cached(self):
        from worldfoundry.studio.visualization.backends.world import world_frontend_css

        css1 = world_frontend_css()
        css2 = world_frontend_css()
        assert css1 == css2

    def test_world_frontend_js_cached(self):
        from worldfoundry.studio.visualization.backends.world import world_frontend_js

        js1 = world_frontend_js()
        js2 = world_frontend_js()
        assert js1 == js2

    def test_world_favicon_svg_cached(self):
        from worldfoundry.studio.visualization.backends.world import world_favicon_svg

        svg1 = world_favicon_svg()
        svg2 = world_favicon_svg()
        assert svg1 == svg2


# ---------------------------------------------------------------------------
# 19.  core/registry.py — StudioVisualizationBackend
# ---------------------------------------------------------------------------


class TestStudioVisualizationBackend:
    """Tests for StudioVisualizationBackend dataclass and methods."""

    def test_construction_defaults(self):
        from worldfoundry.studio.visualization.registry import StudioVisualizationBackend

        b = StudioVisualizationBackend(mode="test", title="Test Backend", default_port=1234)
        assert b.mode == "test"
        assert b.title == "Test Backend"
        assert b.default_port == 1234
        assert b.aliases == ()
        assert b.native is True
        assert callable(b.match)
        assert b.serve is None

    def test_construction_with_aliases(self):
        from worldfoundry.studio.visualization.registry import StudioVisualizationBackend

        b = StudioVisualizationBackend(
            mode="test",
            title="Test",
            default_port=1234,
            aliases=("t1", "t2"),
        )
        assert b.aliases == ("t1", "t2")

    def test_launch_no_serve_raises(self):
        from worldfoundry.studio.visualization.registry import StudioVisualizationBackend, StudioVisualizationRequest

        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        with pytest.raises(ValueError, match="no serve function"):
            b.launch(MagicMock(spec=StudioVisualizationRequest))

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.registry import StudioVisualizationBackend

        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        with pytest.raises(FrozenInstanceError):
            setattr(b, "mode", "new")


# ---------------------------------------------------------------------------
# 21.  core/registry.py — StudioVisualizationRegistry
# ---------------------------------------------------------------------------


class TestStudioVisualizationRegistry:
    """Tests for StudioVisualizationRegistry class."""

    def test_empty_registry(self):
        from worldfoundry.studio.visualization.registry import StudioVisualizationRegistry

        reg = StudioVisualizationRegistry()
        assert reg.modes == frozenset()
        assert reg.native_modes == frozenset()
        assert reg.default_ports == {}

    def test_register_and_modes(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationBackend,
            StudioVisualizationRegistry,
        )

        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        reg = StudioVisualizationRegistry([b])
        assert "test" in reg.modes

    def test_register_with_aliases(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationBackend,
            StudioVisualizationRegistry,
        )

        b = StudioVisualizationBackend(mode="test", title="T", default_port=1, aliases=("t1", "t-2"))
        reg = StudioVisualizationRegistry([b])
        assert reg.backend_for("test").mode == "test"
        assert reg.backend_for("t1").mode == "test"
        assert reg.backend_for("t-2").mode == "test"

    def test_register_duplicate_mode_raises(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationBackend,
            StudioVisualizationRegistry,
        )

        b1 = StudioVisualizationBackend(mode="test", title="T1", default_port=1)
        b2 = StudioVisualizationBackend(mode="other", title="T2", default_port=2, aliases=("test",))
        with pytest.raises(ValueError, match="Duplicate"):
            StudioVisualizationRegistry([b1, b2])

    def test_register_empty_token_raises(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationBackend,
            StudioVisualizationRegistry,
        )

        b = StudioVisualizationBackend(mode="", title="T", default_port=1)
        with pytest.raises(ValueError, match="cannot be empty"):
            StudioVisualizationRegistry([b])

    def test_backend_for_unknown_raises(self):
        from worldfoundry.studio.visualization.registry import StudioVisualizationRegistry

        reg = StudioVisualizationRegistry()
        with pytest.raises(ValueError, match="Unsupported"):
            reg.backend_for("unknown")

    def test_native_modes(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationBackend,
            StudioVisualizationRegistry,
        )

        b1 = StudioVisualizationBackend(mode="native1", title="N1", default_port=1, native=True)
        b2 = StudioVisualizationBackend(mode="non-native", title="NN", default_port=2, native=False)
        reg = StudioVisualizationRegistry([b1, b2])
        assert "native1" in reg.native_modes
        assert "non-native" not in reg.native_modes

    def test_default_ports(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationBackend,
            StudioVisualizationRegistry,
        )

        b = StudioVisualizationBackend(mode="test", title="T", default_port=5555)
        reg = StudioVisualizationRegistry([b])
        assert reg.default_ports == {"test": 5555}

    def test_register_replaces_same_mode(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationBackend,
            StudioVisualizationRegistry,
        )

        b1 = StudioVisualizationBackend(mode="test", title="Old", default_port=1)
        b2 = StudioVisualizationBackend(mode="test", title="New", default_port=2)
        reg = StudioVisualizationRegistry([b1])
        reg.register(b2)
        assert reg.backend_for("test").title == "New"
        assert reg.default_ports["test"] == 2

    def test_resolve_mode_explicit(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationBackend,
            StudioVisualizationRegistry,
        )

        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        reg = StudioVisualizationRegistry([b])
        entry = _make_catalog_entry(model_id="m1")
        result = reg.resolve_mode(entry, "test")
        assert result == "test"

    def test_resolve_mode_auto_fallback(self):
        from worldfoundry.studio.visualization.registry import (
            StudioVisualizationRegistry,
        )

        # Empty registry with no matching backends falls back to "world".
        reg = StudioVisualizationRegistry()
        entry = _make_catalog_entry(model_id="m1")
        result = reg.resolve_mode(entry, "auto")
        assert result == "world"  # INTERACTIVE_WORLD_VISUALIZATION


# ---------------------------------------------------------------------------
# 22.  core/registry.py — Other frozen dataclasses
# ---------------------------------------------------------------------------


class TestRegistryFrozenDataclasses:
    """Tests for StudioVisualizationEvent, StudioModelVisualizationProfile,
    StudioVisualizationRequest, StudioVisualizationLaunch, RenderPlan,
    RenderRequest, RenderResult."""

    def test_studio_model_visualization_profile(self):
        from worldfoundry.studio.visualization.registry import StudioModelVisualizationProfile

        p = StudioModelVisualizationProfile(
            mode="world",
            artifact_domain="interactive_world",
            title="World",
        )
        assert p.mode == "world"
        assert p.artifact_domain == "interactive_world"
        assert p.title == "World"
        assert p.reason == ""
        assert p.accepted_artifact_kinds == ()
        with pytest.raises(FrozenInstanceError):
            setattr(p, "mode", "new")

    def test_studio_visualization_launch(self):
        from worldfoundry.studio.visualization.registry import StudioVisualizationLaunch

        launch = StudioVisualizationLaunch(mode="world")
        assert launch.mode == "world"
        assert launch.url == ""
        assert launch.caption == ""
        assert launch.metadata == {}
        with pytest.raises(FrozenInstanceError):
            setattr(launch, "mode", "new")

    def test_studio_visualization_launch_with_all_fields(self):
        from worldfoundry.studio.visualization.registry import StudioVisualizationLaunch

        launch = StudioVisualizationLaunch(
            mode="points", url="http://localhost/", caption="Ready", metadata={"fps": 16}
        )
        assert launch.url == "http://localhost/"
        assert launch.caption == "Ready"
        assert launch.metadata == {"fps": 16}


# ---------------------------------------------------------------------------
# 23.  core/registry.py — normalize_visualization_mode
# ---------------------------------------------------------------------------


class TestNormalizeVisualizationMode:
    """Tests for normalize_visualization_mode function."""

    def test_none(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode(None) == ""

    def test_empty(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode("") == ""

    def test_lowercase(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode("world") == "world"

    def test_uppercase(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode("WORLD") == "world"

    def test_mixed_case(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode("InteractiveWorld") == "interactiveworld"

    def test_underscore_to_hyphen(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode("interactive_world") == "interactive-world"

    def test_whitespace_stripped(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode("  points  ") == "points"

    def test_combined(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode(" INTERACTIVE_WORLD ") == "interactive-world"


# ---------------------------------------------------------------------------
# 25.  core/registry.py — Mode string constants
# ---------------------------------------------------------------------------


class TestRegistryModeConstants:
    """Tests for visualization mode string constants."""

    def test_mode_strings(self):
        from worldfoundry.studio.visualization.registry import (
            AUTO_VISUALIZATION,
            EMBODIED_VISUALIZATION,
            INTERACTIVE_WORLD_VISUALIZATION,
            MEDIA_VISUALIZATION,
            RERUN_VISUALIZATION,
            SPARK_VISUALIZATION,
            VISER_VISUALIZATION,
        )

        assert INTERACTIVE_WORLD_VISUALIZATION == "world"
        assert VISER_VISUALIZATION == "points"
        assert SPARK_VISUALIZATION == "spark"
        assert MEDIA_VISUALIZATION == "media"
        assert RERUN_VISUALIZATION == "rerun"
        assert EMBODIED_VISUALIZATION == "embodied"
        assert AUTO_VISUALIZATION == "auto"

    def test_artifact_domain_strings(self):
        from worldfoundry.studio.visualization.registry import (
            ARTIFACT_DOMAIN_ACTION,
            ARTIFACT_DOMAIN_GAUSSIAN_SPLAT,
            ARTIFACT_DOMAIN_GEOMETRY,
            ARTIFACT_DOMAIN_MEDIA,
            ARTIFACT_DOMAIN_TIMELINE,
            ARTIFACT_DOMAIN_UI,
            ARTIFACT_DOMAIN_WORLD,
        )

        assert ARTIFACT_DOMAIN_WORLD == "interactive_world"
        assert ARTIFACT_DOMAIN_GEOMETRY == "geometry"
        assert ARTIFACT_DOMAIN_GAUSSIAN_SPLAT == "gaussian_splat"
        assert ARTIFACT_DOMAIN_ACTION == "embodied_action"
        assert ARTIFACT_DOMAIN_TIMELINE == "timeline"
        assert ARTIFACT_DOMAIN_MEDIA == "media"
        assert ARTIFACT_DOMAIN_UI == "ui"


# ---------------------------------------------------------------------------
# 26.  core/registry.py — __all__ completeness
# ---------------------------------------------------------------------------


class TestRegistryAllExports:
    """Verify registry __all__ is complete and importable."""

    def test_all_exports_importable(self):
        import worldfoundry.studio.visualization.registry as mod
        from worldfoundry.studio.visualization.registry import __all__

        for name in __all__:
            assert hasattr(mod, name), f"{name} not found in registry module"


# ---------------------------------------------------------------------------
# 27.  core/artifacts.py — StudioVisualizationArtifact
# ---------------------------------------------------------------------------


class TestStudioVisualizationArtifact:
    """Tests for StudioVisualizationArtifact frozen dataclass."""

    def test_construction(self):
        from worldfoundry.studio.visualization.artifacts import StudioVisualizationArtifact

        a = StudioVisualizationArtifact(path="/tmp/test.ply", kind="point_cloud")
        assert a.path == "/tmp/test.ply"
        assert a.kind == "point_cloud"
        assert a.format_hint == ""
        assert a.metadata == {}

    def test_construction_with_all_fields(self):
        from worldfoundry.studio.visualization.artifacts import StudioVisualizationArtifact

        a = StudioVisualizationArtifact(
            path="/tmp/test.ply",
            kind="point_cloud",
            format_hint="ply",
            metadata={"size": 1024},
        )
        assert a.format_hint == "ply"
        assert a.metadata == {"size": 1024}

    def test_resolved_path(self):
        from worldfoundry.studio.visualization.artifacts import StudioVisualizationArtifact

        a = StudioVisualizationArtifact(path="/tmp/test.ply", kind="point_cloud")
        resolved = a.resolved_path
        assert isinstance(resolved, Path)
        assert resolved.is_absolute()

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.artifacts import StudioVisualizationArtifact

        a = StudioVisualizationArtifact(path="/tmp/test.ply", kind="point_cloud")
        with pytest.raises(FrozenInstanceError):
            setattr(a, "path", "/new")


# ---------------------------------------------------------------------------
# 28.  core/artifacts.py — infer_visualization_artifact
# ---------------------------------------------------------------------------


class TestInferVisualizationArtifact:
    """Tests for infer_visualization_artifact function."""

    def test_ply_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.ply")
        assert a.kind == "point_cloud"
        assert a.format_hint == "ply"

    def test_mp4_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.mp4")
        assert a.kind == "video"

    def test_png_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.png")
        assert a.kind == "image"

    def test_splat_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.splat")
        assert a.kind == "gaussian_splat"

    def test_rrd_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.rrd")
        assert a.kind == "timeline"

    def test_glb_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.glb")
        assert a.kind == "mesh"

    def test_xyz_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.xyz")
        assert a.kind == "point_cloud"
        assert a.format_hint == "xyz"

    def test_unknown_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.unknown")
        assert a.kind == "artifact"

    def test_with_metadata(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.ply", metadata={"size": 1024})
        assert a.metadata == {"size": 1024}

    def test_case_insensitive_suffix(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("/tmp/test.PLY")
        assert a.kind == "point_cloud"

    def test_path_input(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact(Path("/tmp/test.ply"))
        assert a.kind == "point_cloud"


# ---------------------------------------------------------------------------
# 30.  core/artifacts.py — KIND_BY_SUFFIX
# ---------------------------------------------------------------------------


class TestKindBySuffix:
    """Tests for KIND_BY_SUFFIX mapping."""

    def test_mapping_coverage(self):
        from worldfoundry.studio.visualization.artifacts import KIND_BY_SUFFIX

        # Point cloud formats
        assert ".ply" in KIND_BY_SUFFIX
        assert ".pcd" in KIND_BY_SUFFIX
        assert ".xyz" in KIND_BY_SUFFIX
        assert ".npz" in KIND_BY_SUFFIX
        # Mesh formats
        assert ".glb" in KIND_BY_SUFFIX
        assert ".gltf" in KIND_BY_SUFFIX
        assert ".obj" in KIND_BY_SUFFIX
        # Gaussian splat formats
        assert ".spz" in KIND_BY_SUFFIX
        assert ".splat" in KIND_BY_SUFFIX
        assert ".ksplat" in KIND_BY_SUFFIX
        assert ".sog" in KIND_BY_SUFFIX
        # Timeline
        assert ".rrd" in KIND_BY_SUFFIX
        # Image formats
        assert ".png" in KIND_BY_SUFFIX
        assert ".jpg" in KIND_BY_SUFFIX
        # Video formats
        assert ".mp4" in KIND_BY_SUFFIX
        # Audio formats
        assert ".wav" in KIND_BY_SUFFIX
        assert ".mp3" in KIND_BY_SUFFIX

    def test_all_values_are_strings(self):
        from worldfoundry.studio.visualization.artifacts import KIND_BY_SUFFIX

        for key, value in KIND_BY_SUFFIX.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


# ---------------------------------------------------------------------------
# 35.  Cross-module integration — frontends registry uses core registry
# ---------------------------------------------------------------------------


class TestCrossModuleIntegration:
    """Verify frontends.py properly integrates with core/registry."""

    def test_studio_visualizations_backend_for_world(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS

        backend = STUDIO_VISUALIZATIONS.backend_for("world")
        assert backend.mode == "world"

    def test_studio_visualizations_backend_for_alias(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS

        # "interactive-world" is an alias for "world".
        backend = STUDIO_VISUALIZATIONS.backend_for("interactive-world")
        assert backend.mode == "world"

    def test_frontends_constants_match_core_registry(self):
        from worldfoundry.studio.visualization.backends.frontends import (
            EMBODIED_FRONTEND,
            MEDIA_FRONTEND,
            POINTS_FRONTEND,
            RERUN_FRONTEND,
            SPARK_FRONTEND,
            WORLD_FRONTEND,
        )
        from worldfoundry.studio.visualization.registry import (
            EMBODIED_VISUALIZATION,
            INTERACTIVE_WORLD_VISUALIZATION,
            MEDIA_VISUALIZATION,
            RERUN_VISUALIZATION,
            SPARK_VISUALIZATION,
            VISER_VISUALIZATION,
        )

        assert WORLD_FRONTEND == INTERACTIVE_WORLD_VISUALIZATION
        assert POINTS_FRONTEND == VISER_VISUALIZATION
        assert EMBODIED_FRONTEND == EMBODIED_VISUALIZATION
        assert SPARK_FRONTEND == SPARK_VISUALIZATION
        assert MEDIA_FRONTEND == MEDIA_VISUALIZATION
        assert RERUN_FRONTEND == RERUN_VISUALIZATION


# ---------------------------------------------------------------------------
# 36.  Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests across all modules."""

    def test_normalize_visualization_mode_whitespace_only(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode("   ") == ""

    def test_normalize_visualization_mode_hyphens_and_underscores(self):
        from worldfoundry.studio.visualization.registry import normalize_visualization_mode

        assert normalize_visualization_mode("interactive_world-model") == "interactive-world-model"

    def test_infer_artifact_empty_path(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact("")
        assert a.kind == "artifact"  # No suffix => unknown kind.

    def test_infer_artifact_path_object(self):
        from worldfoundry.studio.visualization.artifacts import infer_visualization_artifact

        a = infer_visualization_artifact(Path("test.wav"))
        assert a.kind == "audio"

    def test_viser_presentation_empty_html_caption(self):
        from worldfoundry.studio.visualization.backends.viser import ViserPresentation

        v = ViserPresentation(html="", caption="")
        assert v.html == ""
        assert v.caption == ""

    def test_decode_data_url_data_with_semicolon_charset(self):
        from worldfoundry.studio.visualization.backends.world import _decode_data_url

        data = base64.b64encode(b"test").decode()
        result, mime = _decode_data_url(f"data:text/plain;charset=utf-8;base64,{data}")
        assert mime == "text/plain"

    def test_js_string_special_chars(self):
        from worldfoundry.studio.visualization.backends.frontends import _js_string

        result = _js_string("a\tb")
        assert isinstance(result, str)
        assert result.startswith('"')

    def test_stable_port_offset_empty_string(self):
        from worldfoundry.studio.visualization.backends.viser import _stable_port_offset

        offset = _stable_port_offset("", 8)
        assert isinstance(offset, int)
        assert 0 <= offset < 8

    def test_env_int_empty_string_env(self):
        from worldfoundry.studio.visualization.backends.viser import _env_int

        with patch.dict(os.environ, {"TEST_VISER_INT": ""}):
            result = _env_int("TEST_VISER_INT", 42)
            # Empty string can't be parsed as int => fallback.
            assert result == 42


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


# ---------------------------------------------------------------------------
# Regression tests for standalone viewers, presentation, paths, launch config,
# and workspace launcher
# ---------------------------------------------------------------------------


def test_spark_viewer_html_supports_splat_formats_progress_error_handling_and_input_isolation() -> None:
    from worldfoundry.studio.visualization.backends.frontends import spark_viewer_html

    html = spark_viewer_html(title="Spark Test", default_asset="/tmp/world.ksplat")
    assert "new Spark.SplatLoader()" in html
    assert "Spark.SplatFileType.SPLAT" in html
    assert "Spark.SplatFileType.KSPLAT" in html
    assert "/__worldfoundry__/asset/${name}?path=" in html
    assert "let loadSeq = 0;" in html
    assert "if (seq !== loadSeq)" in html
    assert "catch (error)" in html
    assert "Failed to load ${path}" in html
    assert html.count("new Spark.SparkControls(") == 1
    assert "controls.fpsMovement.enable = false;" in html
    assert "controls.fpsMovement.enable = true;" in html
    assert "isEditableTarget(event.target)" in html
    assert "event.ctrlKey || event.metaKey || event.altKey" in html


def test_spark_handler_serves_named_and_legacy_asset_routes_without_double_unquote(tmp_path: Path) -> None:
    import threading
    from functools import partial
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.request import urlopen

    from worldfoundry.studio.serving import StudioServiceTelemetry, StudioThreadingHTTPServer
    from worldfoundry.studio.visualization.backends.frontends import SparkViewerHandler

    percent_file = tmp_path / "scene%20final.splat"
    percent_file.write_bytes(b"splat-bytes")
    subdir = tmp_path / "nested_dir"
    subdir.mkdir()

    handler = partial(
        SparkViewerHandler,
        default_asset=str(percent_file),
        title="Spark",
        allowed_roots=(tmp_path.resolve(),),
        telemetry=StudioServiceTelemetry("spark:test"),
    )
    server = StudioThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        encoded_path = quote(str(percent_file), safe="")
        with urlopen(f"http://127.0.0.1:{port}/__worldfoundry__/asset/scene.splat?path={encoded_path}") as resp:
            assert resp.status == 200
            assert resp.read() == b"splat-bytes"

        with urlopen(f"http://127.0.0.1:{port}/__worldfoundry__/asset?path={encoded_path}") as resp:
            assert resp.status == 200
            assert resp.read() == b"splat-bytes"

        with pytest.raises(HTTPError) as dir_err:
            urlopen(f"http://127.0.0.1:{port}/__worldfoundry__/asset/dir?path={quote(str(subdir), safe='')}")
        assert dir_err.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_spark_allowed_roots_includes_workspace_root_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.studio.inference.paths import studio_workspace_root
    from worldfoundry.studio.visualization.backends.frontends import _spark_allowed_roots

    for key in (
        "WORLDFOUNDRY_STUDIO_WORKSPACE_DIR",
        "WORLDFOUNDRY_ARTIFACT_DIR",
        "WORLDFOUNDRY_GENERATED_ARTIFACT_DIR",
        "WORLDFOUNDRY_MODEL_DIR",
    ):
        monkeypatch.delenv(key, raising=False)

    roots = _spark_allowed_roots(None)
    assert studio_workspace_root().expanduser().resolve() in roots


def test_viser_depth_npz_preserves_rgb_with_invalid_depths_and_3d_chw(tmp_path: Path) -> None:
    import numpy as np

    from worldfoundry.studio.visualization.backends.viser import _load_xyz_rgb

    depth = np.array([[0.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)
    hwc_rgb = np.array(
        [
            [[10, 20, 30], [255, 0, 0]],
            [[40, 50, 60], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )
    hwc_path = tmp_path / "depth_hwc.npz"
    np.savez(hwc_path, depth=depth, intrinsics=intrinsics, rgb=hwc_rgb)
    points, colors = _load_xyz_rgb(hwc_path)
    assert points.shape == (2, 3)
    np.testing.assert_array_equal(colors, [[255, 0, 0], [0, 0, 255]])

    chw_rgb = np.moveaxis(hwc_rgb, -1, 0)
    assert chw_rgb.shape == (3, 2, 2)
    chw_path = tmp_path / "depth_chw.npz"
    np.savez(chw_path, depth=depth, intrinsics=intrinsics, image=chw_rgb)
    chw_points, chw_colors = _load_xyz_rgb(chw_path)
    assert chw_points.shape == (2, 3)
    np.testing.assert_array_equal(chw_colors, [[255, 0, 0], [0, 0, 255]])


def test_viser_loads_ascii_and_binary_pcd_without_open3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import struct

    import numpy as np

    from worldfoundry.studio.visualization.backends.viser import _load_xyz_rgb

    real_import = builtins.__import__

    def _block_open3d(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "open3d" or name.startswith("open3d."):
            raise ImportError("open3d unavailable in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_open3d)

    ascii_pcd = tmp_path / "cloud_ascii.pcd"
    ascii_pcd.write_text(
        "\n".join(
            [
                "# .PCD v0.7",
                "VERSION 0.7",
                "FIELDS x y z rgb",
                "SIZE 4 4 4 4",
                "TYPE F F F U",
                "COUNT 1 1 1 1",
                "WIDTH 2",
                "HEIGHT 1",
                "POINTS 2",
                "DATA ascii",
                f"1.0 2.0 3.0 {(255 << 16) | (10 << 8) | 20}",
                f"4.0 5.0 6.0 {(0 << 16) | (128 << 8) | 255}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    points, colors = _load_xyz_rgb(ascii_pcd)
    np.testing.assert_allclose(points, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(colors, [[255, 10, 20], [0, 128, 255]])

    binary_pcd = tmp_path / "cloud_binary.pcd"
    header = (
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH 2\n"
        "HEIGHT 1\n"
        "POINTS 2\n"
        "DATA binary\n"
    ).encode("ascii")
    packed_red = struct.unpack("<f", struct.pack("<I", (250 << 16) | (20 << 8) | 10))[0]
    packed_green = struct.unpack("<f", struct.pack("<I", (5 << 16) | (240 << 8) | 15))[0]
    payload = struct.pack("<ffff", 1.0, 0.0, -1.0, packed_red) + struct.pack("<ffff", 2.0, 3.0, 4.0, packed_green)
    binary_pcd.write_bytes(header + payload)

    bin_points, bin_colors = _load_xyz_rgb(binary_pcd)
    np.testing.assert_allclose(bin_points, [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(bin_colors, [[250, 20, 10], [5, 240, 15]])


def test_viser_coordinate_preset_override_updates_default_up_direction_and_handles_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    import numpy as np

    from worldfoundry.studio.visualization.backends import viser as viser_mod

    cloud_path = tmp_path / "cloud.npz"
    np.savez(cloud_path, xyz=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32))

    recorded_up: list[Any] = []
    recorded_hosts: list[str] = []
    stopped: list[bool] = []
    should_fail_scene = {"flag": False}

    class FakeScene:
        def set_up_direction(self, up: Any) -> None:
            recorded_up.append(up)

        def remove_by_name(self, name: str) -> None:
            del name

        def add_point_cloud(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            if should_fail_scene["flag"]:
                raise RuntimeError("scene upload exploded")

    class FakeViserServer:
        def __init__(self, *, host: str, port: int, verbose: bool = False) -> None:
            del verbose
            recorded_hosts.append(host)
            self._port = port
            self.scene = FakeScene()
            self.initial_camera = types.SimpleNamespace()

        def get_port(self) -> int:
            return self._port

        def stop(self) -> None:
            stopped.append(True)

    fake_viser = types.ModuleType("viser")
    setattr(fake_viser, "ViserServer", FakeViserServer)
    monkeypatch.setitem(sys.modules, "viser", fake_viser)
    monkeypatch.setattr(viser_mod, "viser_importable", lambda: True)
    monkeypatch.setattr(viser_mod, "_geometry_imports_available", lambda: True)

    svc = viser_mod.StudioViserService()
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_VISER_COORDINATE_PRESET", "opencv-colmap")
    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_VISER_UP_DIRECTION", raising=False)
    res = svc.present_geometry(run_id="r1", geometry_path=cloud_path, model_id="vggt", host="::1", port=18601)
    assert res.url == "http://[::1]:18601/"
    assert recorded_up[-1] == "-y"
    assert recorded_hosts[-1] == "::1"

    # Pool exhaustion returns a clean fallback instead of raising.
    monkeypatch.setattr(viser_mod, "_pick_pool_port", MagicMock(side_effect=OSError("pool exhausted")))
    exhausted = svc.present_geometry(run_id="r2", geometry_path=cloud_path, host="127.0.0.1", port=None)
    assert exhausted.url == ""
    assert "pool exhausted" in exhausted.caption

    # Scene population failure stops the newly started server so it does not leak.
    stopped.clear()
    should_fail_scene["flag"] = True
    failed = svc.present_geometry(run_id="r3", geometry_path=cloud_path, host="127.0.0.1", port=18602)
    assert failed.url == ""
    assert "scene upload exploded" in failed.caption
    assert stopped == [True]
    assert svc._server is None


def test_serve_rerun_frontend_rejects_non_loopback_binds_loopback_and_propagates_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from worldfoundry.studio.visualization.backends import frontends
    from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig

    rrd_path = tmp_path / "scene.rrd"
    rrd_path.write_bytes(b"rrd")
    entry = _make_catalog_entry(model_id="vggt")

    with pytest.raises(SystemExit, match="refuses to bind non-loopback"):
        frontends.serve_rerun_frontend(
            entry,
            StudioLaunchConfig(model_id="vggt", host="0.0.0.0", port=9876, asset_path=str(rrd_path)),
        )

    spawned: list[list[str]] = []

    class FakeProc:
        def __init__(self, rc: int) -> None:
            self._rc = rc

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self._rc

        def terminate(self) -> None:
            pass

    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_RERUN_COMMAND", raising=False)
    monkeypatch.delenv("WORLDFOUNDRY_STUDIO_RERUN_URL", raising=False)
    monkeypatch.setattr(frontends.subprocess, "Popen", lambda argv: (spawned.append(argv), FakeProc(7))[1])

    with pytest.raises(SystemExit) as exc_info:
        frontends.serve_rerun_frontend(
            entry,
            StudioLaunchConfig(model_id="vggt", host="localhost", port=9876, asset_path=str(rrd_path)),
        )

    assert exc_info.value.code == 7
    assert "--serve-web" in spawned[0]
    assert "--bind" in spawned[0]
    assert spawned[0][spawned[0].index("--bind") + 1] == "127.0.0.1"
    out = capsys.readouterr().out
    assert "http://127.0.0.1:9876/?url=ws://127.0.0.1:9878" in out
    assert "-L 9876:127.0.0.1:9876 -L 9878:127.0.0.1:9878" in out


def test_maybe_build_rerun_rrd_uses_isolated_recording_and_skips_bad_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging
    import sys
    import types

    import numpy as np

    from worldfoundry.studio.visualization import presentation

    bad_path = tmp_path / "a_bad.ply"
    good_path = tmp_path / "b_good.ply"
    bad_path.write_text("corrupt", encoding="utf-8")
    good_path.write_text("valid", encoding="utf-8")

    events: list[tuple[str, Any]] = []

    class FakeRecording:
        def __init__(self, application_id: str, recording_id: str | None = None) -> None:
            events.append(("new", (application_id, recording_id)))

        def save(self, path: str, default_blueprint: Any = None) -> None:
            events.append(("save", path))
            Path(path).write_bytes(b"rrd")

        def log(self, entity_path: str, entity: Any) -> None:
            events.append(("log", entity_path))

        def flush(self) -> None:
            events.append(("flush", None))

        def disconnect(self) -> None:
            events.append(("disconnect", None))

    fake_rr = types.ModuleType("rerun")
    setattr(
        fake_rr, "new_recording", lambda application_id, recording_id=None: FakeRecording(application_id, recording_id)
    )
    setattr(fake_rr, "Points3D", lambda points, colors=None: (points, colors))
    setattr(fake_rr, "init", MagicMock(side_effect=AssertionError("global rr.init must not be called")))
    fake_rrb = types.ModuleType("rerun.blueprint")
    setattr(fake_rrb, "Spatial3DView", lambda origin="scene": ("view", origin))
    monkeypatch.setitem(sys.modules, "rerun", fake_rr)
    monkeypatch.setitem(sys.modules, "rerun.blueprint", fake_rrb)

    class FakeGeom:
        vertices = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)

    fake_trimesh = types.SimpleNamespace(
        load=lambda path: (_ for _ in ()).throw(ValueError("bad ply"))
        if str(path).endswith("a_bad.ply")
        else FakeGeom()
    )
    monkeypatch.setattr(presentation, "_load_trimesh", lambda: fake_trimesh)
    monkeypatch.setattr(presentation, "collect_artifact_paths", lambda _output_dir: [str(bad_path), str(good_path)])

    with caplog.at_level(logging.WARNING, logger=presentation.__name__):
        rrd = presentation.maybe_build_rerun_rrd(str(tmp_path))

    assert rrd == str(tmp_path / "scene.rrd")
    assert Path(rrd).exists()
    assert [name for name, _ in events] == ["new", "save", "log", "flush", "disconnect"]
    assert "Skipping unreadable geometry" in caplog.text


def test_media_viewer_authenticated_html_avoids_unauthenticated_src_and_has_mime_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mimetypes
    import threading
    from functools import partial
    from urllib.request import Request, urlopen

    from worldfoundry.studio.serving import StudioServiceTelemetry, StudioThreadingHTTPServer
    from worldfoundry.studio.visualization.backends.frontends import (
        MediaViewerHandler,
        _media_mime_type,
        media_viewer_html,
    )

    monkeypatch.setattr(mimetypes, "guess_type", lambda name: (None, None))
    assert _media_mime_type(Path("photo.webp")) == "image/webp"
    assert _media_mime_type(Path("track.flac")) == "audio/flac"
    assert _media_mime_type(Path("clip.ogg")) == "audio/ogg"
    assert _media_mime_type(Path("movie.mkv")) == "video/x-matroska"
    assert '<img class="asset asset-image"' in media_viewer_html(title="WebP", asset_path=Path("photo.webp"))
    assert '<audio class="asset asset-audio"' in media_viewer_html(title="FLAC", asset_path=Path("track.flac"))
    assert '<audio class="asset asset-audio"' in media_viewer_html(title="OGG", asset_path=Path("clip.ogg"))
    assert '<video class="asset asset-video"' in media_viewer_html(title="MKV", asset_path=Path("movie.mkv"))

    asset = tmp_path / "photo.webp"
    asset.write_bytes(b"RIFFwebp")
    handler = partial(
        MediaViewerHandler,
        asset_path=asset,
        title="Auth Media",
        allowed_roots=(tmp_path.resolve(),),
        telemetry=StudioServiceTelemetry("media:test"),
        auth_token="secret-token",
    )
    server = StudioThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/?token=secret-token") as resp:
            body = resp.read().decode("utf-8")
            assert 'src="/__worldfoundry__/asset?token=secret-token"' in body
            assert ' src="/__worldfoundry__/asset"' not in body

        req = Request(f"http://127.0.0.1:{port}/", headers={"Authorization": "Bearer secret-token"})
        with urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            assert 'data-src="/__worldfoundry__/asset"' in body
            assert ' src="/__worldfoundry__/asset"' not in body

        with urlopen(f"http://127.0.0.1:{port}/__worldfoundry__/asset?token=secret-token") as resp:
            assert resp.headers.get_content_type() == "image/webp"
            assert resp.read() == b"RIFFwebp"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_launch_config_port_precedence_does_not_shadow_mode_specific_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.studio.visualization.backends.frontends import _explicit_points_port, port_for_frontend
    from worldfoundry.studio.visualization.launch_config import parse_launch_config

    entry = _make_catalog_entry(model_id="test-model")
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_PORT", "7777")
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_SPARK_PORT", "8888")
    monkeypatch.setenv("WORLDFOUNDRY_STUDIO_POINTS_PORT", "19901")

    with patch("worldfoundry.studio.visualization.launch_config.find_entry", return_value=entry):
        cfg = parse_launch_config(
            ["test-model"],
            entries=(entry,),
            studio_catalog=lambda _: (entry,),
            resolve_cli_variant_id=lambda _entry, variant: variant,
        )
        assert cfg.port is None
        assert port_for_frontend(cfg, "spark") == 8888
        assert port_for_frontend(cfg, "media") == 7777
        assert _explicit_points_port(cfg) == 19901

        monkeypatch.delenv("WORLDFOUNDRY_STUDIO_POINTS_PORT", raising=False)
        monkeypatch.delenv("WORLDFOUNDRY_STUDIO_VISER_PORT", raising=False)
        assert _explicit_points_port(cfg) == 7777

        explicit_cfg = parse_launch_config(
            ["test-model", "--port", "9123"],
            entries=(entry,),
            studio_catalog=lambda _: (entry,),
            resolve_cli_variant_id=lambda _entry, variant: variant,
        )
        assert explicit_cfg.port == 9123
        assert port_for_frontend(explicit_cfg, "spark") == 9123
        assert _explicit_points_port(explicit_cfg) == 9123


def test_dedupe_paths_survives_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldfoundry.studio.inference.paths import _dedupe_paths

    blocked = Path("/root/unreadable-worldfoundry-cache")
    original_exists = Path.exists

    def _guarded_exists(self: Path) -> bool:
        if self == blocked:
            raise PermissionError("Permission denied")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", _guarded_exists)
    result = _dedupe_paths((blocked, blocked))
    assert result == (blocked,)


def test_run_workspace_script_env_file_and_flag_precedence(tmp_path: Path) -> None:
    import subprocess

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "workspace" / "run_workspace.sh"

    # Explicit missing --env-file fails clearly.
    missing_proc = subprocess.run(
        ["bash", str(script), "--env-file", str(tmp_path / "does_not_exist.sh")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_proc.returncode != 0
    assert "Env file does not exist" in missing_proc.stderr

    # Stub python that satisfies validation checks and prints received args/env instead of starting uvicorn.
    stub_py = tmp_path / "stub_python.sh"
    stub_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-" ]]; then\n'
        "  echo stub-py\n"
        "  exit 0\n"
        "fi\n"
        "printf 'ARGS=%s\\n' \"$*\"\n"
        "printf 'MAX_JOBS=%s\\n' \"${WORLDFOUNDRY_WORKSPACE_MAX_JOBS:-}\"\n"
        "printf 'HFD_ROOT=%s\\n' \"${WORLDFOUNDRY_HFD_ROOT:-}\"\n"
        "printf 'BENCH_ROOT=%s\\n' \"${WORLDFOUNDRY_BENCHMARK_DATA_ROOT:-}\"\n"
        "printf 'HFD_DATASET_ROOT=%s\\n' \"${WORLDFOUNDRY_HFD_DATASET_ROOT:-}\"\n"
        "printf 'PYTHON=%s\\n' \"${PYTHON:-}\"\n",
        encoding="utf-8",
    )
    stub_py.chmod(0o755)

    custom_env = tmp_path / "custom_env.sh"
    custom_env.write_text(
        f'export WORLDFOUNDRY_WORKSPACE_HOST="127.0.0.2"\n'
        f'export WORLDFOUNDRY_WORKSPACE_PORT="8123"\n'
        f'export WORLDFOUNDRY_WORKSPACE_MAX_JOBS="3"\n'
        f'export WORLDFOUNDRY_HFD_ROOT="{tmp_path / "hfd_from_env"}"\n'
        f'export WORLDFOUNDRY_BENCHMARK_DATA_ROOT="{tmp_path / "bench_from_env"}"\n'
        f'export WORLDFOUNDRY_HFD_DATASET_ROOT="{tmp_path / "ds_from_env"}"\n'
        f'export PYTHON="{stub_py}"\n',
        encoding="utf-8",
    )

    # Even with WORLDFOUNDRY_UNIFIED_ENV_PREFIX already exported, explicit --env-file is sourced.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "WORLDFOUNDRY_UNIFIED_ENV_PREFIX": str(tmp_path / "preexisting"),
    }
    run_env_file = subprocess.run(
        ["bash", str(script), "--env-file", str(custom_env)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_env_file.returncode == 0, run_env_file.stderr
    assert "ARGS=-m worldfoundry.studio.workspace.app --host 127.0.0.2 --port 8123" in run_env_file.stdout
    assert "MAX_JOBS=3" in run_env_file.stdout
    assert f"HFD_ROOT={tmp_path / 'hfd_from_env'}" in run_env_file.stdout
    assert f"BENCH_ROOT={tmp_path / 'bench_from_env'}" in run_env_file.stdout
    assert f"HFD_DATASET_ROOT={tmp_path / 'ds_from_env'}" in run_env_file.stdout

    # CLI flags still take highest precedence over --env-file values.
    run_cli = subprocess.run(
        [
            "bash",
            str(script),
            "--env-file",
            str(custom_env),
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--max-jobs",
            "5",
            "--python",
            str(stub_py),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_cli.returncode == 0, run_cli.stderr
    assert "ARGS=-m worldfoundry.studio.workspace.app --host 127.0.0.1 --port 9001" in run_cli.stdout
    assert "MAX_JOBS=5" in run_cli.stdout


def test_unified_requirements_pins_pyarrow_once() -> None:
    req_path = Path(__file__).resolve().parents[2] / "requirements" / "worldfoundry-unified.txt"
    pyarrow_lines = [
        line.strip() for line in req_path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("pyarrow")
    ]
    assert pyarrow_lines == ["pyarrow>=14.0.2,<20"]
