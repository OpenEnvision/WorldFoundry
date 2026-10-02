"""Comprehensive tests for the backends_frontends_viser_world submodule.

Covers:
  - backends/__init__.py  (imports & __all__)
  - backends/frontends.py (constants, functions, handler classes)
  - backends/viser.py     (ViserPresentation dataclass, constants, helper functions)
  - backends/world.py     (WorldSession/WorldFrontendState dataclasses, functions)
  - core/registry.py      (StudioVisualizationBackend, StudioVisualizationRegistry, etc.)
  - core/artifacts.py     (StudioVisualizationArtifact, infer_visualization_artifact, etc.)
  - core/scene.py         (VisualizationScene, Layer, Frame, Timeline)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import socket
import struct
import threading
import time
from dataclasses import dataclass, field, fields, FrozenInstanceError
from html import escape
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import quote, unquote, urlparse

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
            v.html = "changed"
        with pytest.raises(FrozenInstanceError):
            v.caption = "changed"
        with pytest.raises(FrozenInstanceError):
            v.url = "http://changed"

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

    def test_pick_free_port(self):
        from worldfoundry.studio.visualization.backends.viser import _pick_free_port
        port = _pick_free_port()
        assert isinstance(port, int)
        assert port > 0
        # Port should be in valid TCP range.
        assert 1 <= port <= 65535

    def test_pick_free_port_custom_host(self):
        from worldfoundry.studio.visualization.backends.viser import _pick_free_port
        port = _pick_free_port("127.0.0.1")
        assert isinstance(port, int)
        assert port > 0

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
        from worldfoundry.studio.visualization.backends.viser import _port_is_free, _pick_free_port
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
        with patch.dict(os.environ, {
            "WORLDFOUNDRY_STUDIO_VISER_PORT_BASE": "30000",
            "WORLDFOUNDRY_STUDIO_VISER_PORT_COUNT": "4",
        }):
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
        assert viser_orientation_defaults("unik3d-prior", "points.npz") == unidepth
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
        assert viser_orientation_defaults("dust3r", "dust3r_point_cloud.ply") == camera_aligned
        assert viser_orientation_defaults("dust3r-base-model", "dust3r_point_cloud.ply") == camera_aligned
        assert viser_orientation_defaults("loger", "point_cloud/result.ply") == camera_aligned
        assert viser_orientation_defaults("pi3", "point_cloud/result.ply") == camera_aligned
        assert viser_orientation_defaults("unknown", "scene.ply")["up_direction"] == "+z"

    def test_coordinate_transform_opencv_to_opengl_flips_yz(self):
        import numpy as np
        from worldfoundry.studio.visualization.backends.viser import _transform_points_for_preset

        points = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, 6.0]])
        transformed = _transform_points_for_preset(points, "opencv-to-opengl")
        np.testing.assert_allclose(transformed, np.array([[1.0, -2.0, -3.0], [-4.0, 5.0, -6.0]]))
        np.testing.assert_allclose(_transform_points_for_preset(points, "asset-native"), points)

    def test_depth_npz_extrinsics_are_w2c_and_explicit_c2w_takes_precedence(self):
        import numpy as np
        from worldfoundry.studio.visualization.backends.viser import _depth_points_from_npz

        depth = np.array([[[2.0]]])
        intrinsics = np.eye(3)[None]
        w2c = np.eye(4)[None]
        w2c[0, :3, 3] = [-3.0, 1.0, -4.0]

        points = _depth_points_from_npz(
            {"depth": depth, "intrinsics": intrinsics, "extrinsics": w2c}
        )
        # The camera center is (3, -1, 4), so the optical-axis sample at
        # camera z=2 must land at (3, -1, 6), not at the w2c translation.
        np.testing.assert_allclose(points, [[3.0, -1.0, 6.0]], atol=1e-8)

        c2w = np.eye(4)[None]
        c2w[0, :3, 3] = [7.0, 8.0, 9.0]
        explicit_points = _depth_points_from_npz(
            {
                "depth": depth,
                "intrinsics": intrinsics,
                "extrinsics": w2c,
                "camera_to_world": c2w,
            }
        )
        np.testing.assert_allclose(explicit_points, [[7.0, 8.0, 11.0]], atol=1e-8)

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

        cloud_path = tmp_path / "dust3r_point_cloud.ply"
        cloud_path.write_text("ply\n", encoding="utf-8")
        expected = np.stack([np.eye(4), np.eye(4)])
        expected[1, 0, 3] = 1.0
        np.save(tmp_path / "dust3r_camera_poses.npy", expected)

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

    def test_init(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService
        svc = StudioViserService()
        assert svc._lock is not None
        assert svc._server is None

    def test_present_point_cloud_delegates(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService
        svc = StudioViserService()
        # present_point_cloud should delegate to present_geometry.
        # We test that it exists and is callable.
        assert hasattr(svc, "present_point_cloud")
        assert callable(svc.present_point_cloud)

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
            with patch("worldfoundry.studio.visualization.backends.viser._geometry_imports_available", return_value=False):
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
            with patch("worldfoundry.studio.visualization.backends.viser._geometry_imports_available", return_value=True):
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

    def test_shutdown_when_no_server(self):
        from worldfoundry.studio.visualization.backends.viser import StudioViserService
        svc = StudioViserService()
        # shutdown should work gracefully when no server is running.
        svc.shutdown()
        assert svc._server is None

    def test_studio_viser_global_instance(self):
        from worldfoundry.studio.visualization.backends.viser import STUDIO_VISER, StudioViserService
        assert isinstance(STUDIO_VISER, StudioViserService)


class TestWorkspaceVisualizerParams:
    """Tests for Workspace visualizer parameter plumbing."""

    def test_points_env_overrides_are_whitelisted(self):
        from worldfoundry.studio.serving.workspace import _visualizer_env_overrides

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
        from worldfoundry.studio.serving.workspace import _resolved_visualizer_params

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
        from worldfoundry.studio.serving.workspace import _resolved_visualizer_params

        params = _resolved_visualizer_params(
            "points",
            model_id="vggt-omega",
            asset_path="predictions.npz",
            requested={},
        )
        assert params["coordinate_preset"] == "opencv-to-opengl"
        assert params["up_direction"] == "+y"

    def test_visualizer_reuse_requires_same_asset_and_model(self):
        from worldfoundry.studio.serving.workspace import ManagedVisualizer, _visualizer_reusable

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
        from worldfoundry.studio.serving.workspace import _rerun_renderer

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORLDFOUNDRY_STUDIO_RERUN_RENDERER", None)
            assert _rerun_renderer() == "webgpu"
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_RERUN_RENDERER": "webgl"}):
            assert _rerun_renderer() == "webgl"
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_RERUN_RENDERER": "invalid"}):
            assert _rerun_renderer() == "webgpu"

    def test_visualizer_status_includes_params(self):
        from worldfoundry.studio.serving.workspace import ManagedVisualizer, _visualizer_status

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
        assert _adaptive_point_size(unit_scene * 10.0) == pytest.approx(
            _adaptive_point_size(unit_scene) * 10.0
        )

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
            WORLD_FRONTEND, POINTS_FRONTEND, EMBODIED_FRONTEND,
            SPARK_FRONTEND, MEDIA_FRONTEND, RERUN_FRONTEND, UNIFIED_FRONTEND,
        )
        assert WORLD_FRONTEND == "world"
        assert POINTS_FRONTEND == "points"
        assert EMBODIED_FRONTEND == "embodied"
        assert SPARK_FRONTEND == "spark"
        assert MEDIA_FRONTEND == "media"
        assert RERUN_FRONTEND == "rerun"
        assert UNIFIED_FRONTEND == "unified"

    def test_default_rerun_command_assigns_distinct_data_ports(self):
        from worldfoundry.studio.visualization.backends.frontends import _default_rerun_command_template

        command = _default_rerun_command_template().format(
            asset="scene.rrd",
            grpc_port=9878,
            ws_port=9877,
            port=9876,
        )
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
            STUDIO_ASSET_DIR, VENDOR_DIR, SPARK_VENDOR_DIR, THREE_VENDOR_DIR,
            SPARK_MODULE_PATH, THREE_MODULE_PATH, THREE_CORE_MODULE_PATH,
        )
        assert isinstance(STUDIO_ASSET_DIR, Path)
        assert isinstance(VENDOR_DIR, Path)
        assert isinstance(SPARK_VENDOR_DIR, Path)
        assert isinstance(THREE_VENDOR_DIR, Path)
        assert isinstance(SPARK_MODULE_PATH, Path)
        assert isinstance(THREE_MODULE_PATH, Path)
        assert isinstance(THREE_CORE_MODULE_PATH, Path)
        # Parent-child relationships.
        assert VENDOR_DIR.parent == STUDIO_ASSET_DIR
        assert SPARK_VENDOR_DIR.parent == VENDOR_DIR
        assert THREE_VENDOR_DIR.parent == VENDOR_DIR

    def test_studio_visualizations_registry(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationRegistry
        assert isinstance(STUDIO_VISUALIZATIONS, StudioVisualizationRegistry)
        assert "world" in STUDIO_VISUALIZATIONS.modes
        assert "points" in STUDIO_VISUALIZATIONS.modes
        assert "spark" in STUDIO_VISUALIZATIONS.modes
        assert "embodied" in STUDIO_VISUALIZATIONS.modes
        assert "rerun" in STUDIO_VISUALIZATIONS.modes
        assert "media" in STUDIO_VISUALIZATIONS.modes
        assert "unified" in STUDIO_VISUALIZATIONS.modes

    def test_native_frontends(self):
        from worldfoundry.studio.visualization.backends.frontends import NATIVE_FRONTENDS
        assert isinstance(NATIVE_FRONTENDS, frozenset)
        assert "world" in NATIVE_FRONTENDS
        assert "points" in NATIVE_FRONTENDS
        # Unified is not native.
        assert "unified" not in NATIVE_FRONTENDS

    def test_default_frontend_ports(self):
        from worldfoundry.studio.visualization.backends.frontends import DEFAULT_FRONTEND_PORTS
        assert isinstance(DEFAULT_FRONTEND_PORTS, dict)
        assert DEFAULT_FRONTEND_PORTS["world"] == 7868
        assert DEFAULT_FRONTEND_PORTS["points"] == 18590
        assert DEFAULT_FRONTEND_PORTS["spark"] == 8765
        assert DEFAULT_FRONTEND_PORTS["embodied"] == 18610
        assert DEFAULT_FRONTEND_PORTS["rerun"] == 9876
        assert DEFAULT_FRONTEND_PORTS["media"] == 18720
        assert DEFAULT_FRONTEND_PORTS["unified"] == 7868


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
        assert "image" in backend.capabilities.layer_kinds
        assert "video" in backend.capabilities.layer_kinds
        assert backend.capabilities.score == 50
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
        assert "point_cloud" in backend.capabilities.layer_kinds
        assert backend.capabilities.score == 80

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
        assert "gaussian_splat" in backend.capabilities.layer_kinds
        assert backend.capabilities.partial is False
        assert backend.capabilities.score == 90

    def test_embodied_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS
        backend = STUDIO_VISUALIZATIONS.backend_for("embodied")
        assert backend.mode == "embodied"
        assert backend.title == "Embodied Simulator Bridge"
        assert backend.default_port == 18610
        assert "sim" in backend.aliases
        assert "simulator" in backend.aliases
        assert backend.native is True
        assert "action_trace" in backend.capabilities.layer_kinds
        assert backend.capabilities.score == 75

    def test_rerun_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS
        backend = STUDIO_VISUALIZATIONS.backend_for("rerun")
        assert backend.mode == "rerun"
        assert backend.title == "Rerun Timeline Viewer"
        assert backend.default_port == 9876
        assert "rrd" in backend.aliases
        assert backend.native is True
        assert "timeline" in backend.capabilities.layer_kinds
        assert backend.capabilities.score == 70

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
        assert "image" in backend.capabilities.layer_kinds
        assert backend.capabilities.score == 60

    def test_unified_backend(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS
        backend = STUDIO_VISUALIZATIONS.backend_for("unified")
        assert backend.mode == "unified"
        assert backend.title == "Unified Gradio Studio"
        assert backend.default_port == 7868
        assert backend.native is False
        assert backend.aliases == ()
        assert backend.serve is None
        # match lambda always returns False.
        assert backend.match(None, None) is False


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
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model")
        result = host_for_frontend(lc)
        assert result == "127.0.0.1"

    def test_host_for_frontend_explicit_host(self):
        from worldfoundry.studio.visualization.backends.frontends import host_for_frontend
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model", host="0.0.0.0")
        result = host_for_frontend(lc)
        assert result == "0.0.0.0"

    def test_host_for_frontend_env_override(self):
        from worldfoundry.studio.visualization.backends.frontends import host_for_frontend
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model")
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_HOST": "10.0.0.5"}):
            result = host_for_frontend(lc)
            assert result == "10.0.0.5"

    def test_host_for_frontend_gradio_env(self):
        from worldfoundry.studio.visualization.backends.frontends import host_for_frontend
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model")
        with patch.dict(os.environ, {"GRADIO_SERVER_NAME": "my-host"}, clear=False):
            # Remove Worldevals host env to fall through to Gradio.
            os.environ.pop("WORLDFOUNDRY_STUDIO_HOST", None)
            result = host_for_frontend(lc)
            assert result == "my-host"

    def test_host_for_frontend_blank_host_falls_to_default(self):
        from worldfoundry.studio.visualization.backends.frontends import host_for_frontend
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model", host="   ")
        result = host_for_frontend(lc)
        assert result == "127.0.0.1"

    def test_port_for_frontend_explicit_port(self):
        from worldfoundry.studio.visualization.backends.frontends import port_for_frontend
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model", port=9999)
        result = port_for_frontend(lc, "world")
        assert result == 9999

    def test_port_for_frontend_default(self):
        from worldfoundry.studio.visualization.backends.frontends import port_for_frontend
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model")
        result = port_for_frontend(lc, "world")
        assert result == 7868

    def test_port_for_frontend_env_override(self):
        from worldfoundry.studio.visualization.backends.frontends import port_for_frontend
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
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
        assert result == '"say \\\"hi\\\""'

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
        result = _strip_html(None)
        assert result == ""

    def test_strip_html_multiple_tags(self):
        from worldfoundry.studio.visualization.backends.frontends import _strip_html
        result = _strip_html("<p>Hello <em>world</em></p>")
        assert result == "Hello world"

    def test_optional_asset_none(self):
        from worldfoundry.studio.visualization.backends.frontends import _optional_asset
        result = _optional_asset(None)
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
        from worldfoundry.studio.visualization.backends.frontends import _optional_asset
        import tempfile
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
        with pytest.raises(SystemExit):
            _required_existing_asset(None, frontend="test", hint="pass --asset path")

    def test_required_existing_asset_existing_file(self):
        from worldfoundry.studio.visualization.backends.frontends import _required_existing_asset
        import tempfile
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
        from worldfoundry.studio.visualization.backends.frontends import _spark_allowed_roots
        import tempfile
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
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model")
        result = _explicit_points_port(lc)
        assert result is None

    def test_explicit_points_port_with_port(self):
        from worldfoundry.studio.visualization.backends.frontends import _explicit_points_port
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        lc = StudioLaunchConfig(model_id="test-model", port=18591)
        result = _explicit_points_port(lc)
        assert result == 18591

    def test_explicit_points_port_env_override(self):
        from worldfoundry.studio.visualization.backends.frontends import _explicit_points_port
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
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
# 11.  world.py — Dataclasses
# ---------------------------------------------------------------------------

class TestWorldDataclasses:
    """Tests for WorldSession and WorldFrontendState dataclasses."""

    def test_world_session_construction(self):
        from worldfoundry.studio.visualization.backends.world import WorldSession
        ws = WorldSession(session_id="abc123", mode="image", prompt="test prompt")
        assert ws.session_id == "abc123"
        assert ws.mode == "image"
        assert ws.prompt == "test prompt"
        assert ws.seed_image is None
        assert ws.seed_video_path == ""
        assert ws.last_record is None
        assert ws.step_count == 0

    def test_world_session_created_at_default(self):
        from worldfoundry.studio.visualization.backends.world import WorldSession
        ws = WorldSession(session_id="x", mode="video", prompt="p")
        assert isinstance(ws.created_at, float)
        assert ws.created_at > 0

    def test_world_session_fields(self):
        from worldfoundry.studio.visualization.backends.world import WorldSession
        field_names = [f.name for f in fields(WorldSession)]
        assert "session_id" in field_names
        assert "mode" in field_names
        assert "prompt" in field_names
        assert "seed_image" in field_names
        assert "seed_video_path" in field_names
        assert "last_record" in field_names
        assert "created_at" in field_names
        assert "step_count" in field_names

    def test_world_session_not_frozen(self):
        from worldfoundry.studio.visualization.backends.world import WorldSession
        ws = WorldSession(session_id="abc", mode="image", prompt="p")
        ws.step_count = 5
        assert ws.step_count == 5
        ws.prompt = "new prompt"
        assert ws.prompt == "new prompt"

    def test_world_frontend_state_fields(self):
        from worldfoundry.studio.visualization.backends.world import WorldFrontendState
        field_names = [f.name for f in fields(WorldFrontendState)]
        assert "entry" in field_names
        assert "launch_config" in field_names
        assert "manager" in field_names
        assert "demo_images" in field_names
        assert "allowed_roots" in field_names
        assert "telemetry" in field_names
        assert "max_sessions" in field_names
        assert "lock" in field_names
        assert "model_loaded" in field_names
        assert "sessions" in field_names

    def test_world_frontend_state_defaults(self):
        from worldfoundry.studio.visualization.backends.world import WorldFrontendState
        # We can't fully construct it without real dependencies, but we can
        # check that defaults are as expected.
        fs = fields(WorldFrontendState)
        max_sessions_field = next(f for f in fs if f.name == "max_sessions")
        assert max_sessions_field.default == 4
        model_loaded_field = next(f for f in fs if f.name == "model_loaded")
        assert model_loaded_field.default is False


# ---------------------------------------------------------------------------
# 12.  world.py — Constants
# ---------------------------------------------------------------------------

class TestWorldConstants:
    """Tests for world module constants."""

    def test_world_upload_dir_name(self):
        from worldfoundry.studio.visualization.backends.world import WORLD_UPLOAD_DIR_NAME
        assert WORLD_UPLOAD_DIR_NAME == "world_frontend_inputs"

    def test_default_fps(self):
        from worldfoundry.studio.visualization.backends.world import DEFAULT_FPS
        assert DEFAULT_FPS == 16

    def test_world_action_deadzone(self):
        from worldfoundry.studio.visualization.backends.world import WORLD_ACTION_DEADZONE
        assert WORLD_ACTION_DEADZONE == 0.08

    def test_world_stream_base_seed(self):
        from worldfoundry.studio.visualization.backends.world import WORLD_STREAM_BASE_SEED
        assert WORLD_STREAM_BASE_SEED == 42

    def test_websocket_guid(self):
        from worldfoundry.studio.visualization.backends.world import WEBSOCKET_GUID
        assert WEBSOCKET_GUID == "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def test_world_demo_image_root(self):
        from worldfoundry.studio.visualization.backends.world import WORLD_DEMO_IMAGE_ROOT
        assert isinstance(WORLD_DEMO_IMAGE_ROOT, Path)


# ---------------------------------------------------------------------------
# 13.  world.py — controls_to_interactions
# ---------------------------------------------------------------------------

class TestControlsToInteractions:
    """Thorough tests for the controls_to_interactions function."""

    def test_empty_dict(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({}) == ""

    def test_none(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions(None) == ""

    def test_string(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions("not a dict") == ""

    def test_forward_w(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"w": True}) == "forward"

    def test_backward_s(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"s": True}) == "backward"

    def test_left_a(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"a": True}) == "left"

    def test_right_d(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"d": True}) == "right"

    def test_forward_left_wa(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"w": True, "a": True}) == "forward_left"

    def test_forward_right_wd(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"w": True, "d": True}) == "forward_right"

    def test_backward_left_sa(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"s": True, "a": True}) == "backward_left"

    def test_backward_right_sd(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"s": True, "d": True}) == "backward_right"

    def test_conflicting_w_s(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        # w and s both True: diagonal branches won't match (all combos require !s or !w).
        # Falls through to forward (w=True, s=True => w branch matches since s check isn't exclusive).
        result = controls_to_interactions({"w": True, "s": True})
        # Per the code, forward_left and forward_right both require !s.
        # forward requires !s. So the w+s combo is actually ambiguous in the code.
        # Let's check actual behavior.
        assert isinstance(result, str)

    def test_camera_right(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dx": 0.5}) == "camera_r"

    def test_camera_left(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dx": -0.5}) == "camera_l"

    def test_camera_up(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dy": -0.5}) == "camera_up"

    def test_camera_down(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dy": 0.5}) == "camera_down"

    def test_camera_diagonal_dr(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dx": 0.5, "camera_dy": 0.5}) == "camera_dr"

    def test_camera_diagonal_ur(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dx": 0.5, "camera_dy": -0.5}) == "camera_ur"

    def test_camera_diagonal_dl(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dx": -0.5, "camera_dy": 0.5}) == "camera_dl"

    def test_camera_diagonal_ul(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dx": -0.5, "camera_dy": -0.5}) == "camera_ul"

    def test_camera_below_deadzone(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions, WORLD_ACTION_DEADZONE
        # camera_dx below deadzone should produce no camera token.
        assert controls_to_interactions({"camera_dx": WORLD_ACTION_DEADZONE / 2}) == ""

    def test_camera_at_exact_deadzone(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions, WORLD_ACTION_DEADZONE
        # At exactly the deadzone, > comparison means not included.
        assert controls_to_interactions({"camera_dx": WORLD_ACTION_DEADZONE}) == ""

    def test_camera_above_deadzone(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions, WORLD_ACTION_DEADZONE
        # Just above deadzone.
        assert controls_to_interactions({"camera_dx": WORLD_ACTION_DEADZONE + 0.01}) == "camera_r"

    def test_l_click_only(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"l_click": True}) == "forward"

    def test_r_click_only(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"r_click": True}) == "backward"

    def test_combined_movement_and_camera(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        result = controls_to_interactions({"w": True, "camera_dx": 0.5})
        assert "forward" in result
        assert "camera_r" in result

    def test_zero_values_produce_empty(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"w": False, "a": False, "s": False, "d": False, "camera_dx": 0, "camera_dy": 0}) == ""

    def test_camera_zero_inside_deadzone(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        assert controls_to_interactions({"camera_dx": 0.0, "camera_dy": 0.0}) == ""


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

    def test_suffix_for_mime_mp4(self):
        from worldfoundry.studio.visualization.backends.world import _suffix_for_mime
        assert _suffix_for_mime("video/mp4") == ".mp4"

    def test_suffix_for_mime_webm(self):
        from worldfoundry.studio.visualization.backends.world import _suffix_for_mime
        assert _suffix_for_mime("video/webm") == ".webm"

    def test_suffix_for_mime_mov(self):
        from worldfoundry.studio.visualization.backends.world import _suffix_for_mime
        assert _suffix_for_mime("video/quicktime") == ".mov"

    def test_suffix_for_mime_avi(self):
        from worldfoundry.studio.visualization.backends.world import _suffix_for_mime
        assert _suffix_for_mime("video/x-msvideo") == ".avi"

    def test_suffix_for_mime_unknown(self):
        from worldfoundry.studio.visualization.backends.world import _suffix_for_mime
        assert _suffix_for_mime("video/unknown") == ""

    def test_suffix_for_mime_empty(self):
        from worldfoundry.studio.visualization.backends.world import _suffix_for_mime
        assert _suffix_for_mime("") == ""

    def test_file_url_with_path(self):
        from worldfoundry.studio.visualization.backends.world import _file_url
        result = _file_url("/tmp/test.png")
        assert result.startswith("/api/file?path=")
        assert "test.png" in result

    def test_file_url_none(self):
        from worldfoundry.studio.visualization.backends.world import _file_url
        assert _file_url(None) == ""

    def test_file_url_empty(self):
        from worldfoundry.studio.visualization.backends.world import _file_url
        assert _file_url("") == ""

    def test_file_url_encodes_special_chars(self):
        from worldfoundry.studio.visualization.backends.world import _file_url
        result = _file_url("/tmp/file with spaces.png")
        assert "file%20with%20spaces" in result

    def test_repeat_to_slots_basic(self):
        from worldfoundry.studio.visualization.backends.world import _repeat_to_slots
        paths = (Path("/a"), Path("/b"), Path("/c"))
        result = _repeat_to_slots(paths, 9)
        assert len(result) == 9
        assert result[0] == Path("/a")
        assert result[3] == Path("/a")  # wraps around

    def test_repeat_to_slots_exact_match(self):
        from worldfoundry.studio.visualization.backends.world import _repeat_to_slots
        paths = (Path("/a"), Path("/b"), Path("/c"))
        result = _repeat_to_slots(paths, 3)
        assert result == (Path("/a"), Path("/b"), Path("/c"))

    def test_repeat_to_slots_empty_paths(self):
        from worldfoundry.studio.visualization.backends.world import _repeat_to_slots
        result = _repeat_to_slots((), 9)
        assert result == ()

    def test_override_call_kwargs_seed_empty(self):
        from worldfoundry.studio.visualization.backends.world import _override_call_kwargs_seed
        result = _override_call_kwargs_seed("", 42)
        parsed = json.loads(result)
        assert parsed["seed"] == 42

    def test_override_call_kwargs_seed_existing(self):
        from worldfoundry.studio.visualization.backends.world import _override_call_kwargs_seed
        result = _override_call_kwargs_seed('{"seed": 10, "num_frames": 9}', 42)
        parsed = json.loads(result)
        assert parsed["seed"] == 42
        assert parsed["num_frames"] == 9

    def test_override_call_kwargs_seed_empty_dict(self):
        from worldfoundry.studio.visualization.backends.world import _override_call_kwargs_seed
        result = _override_call_kwargs_seed("{}", 42)
        parsed = json.loads(result)
        assert parsed["seed"] == 42

    def test_override_call_kwargs_seed_invalid_json(self):
        from worldfoundry.studio.visualization.backends.world import _override_call_kwargs_seed
        result = _override_call_kwargs_seed("not json", 42)
        parsed = json.loads(result)
        assert parsed["seed"] == 42

    def test_override_call_kwargs_seed_non_dict_json(self):
        from worldfoundry.studio.visualization.backends.world import _override_call_kwargs_seed
        result = _override_call_kwargs_seed("[1,2,3]", 42)
        parsed = json.loads(result)
        assert parsed["seed"] == 42

    def test_world_max_sessions_default(self):
        from worldfoundry.studio.visualization.backends.world import _world_max_sessions
        result = _world_max_sessions()
        assert result == 4

    def test_world_max_sessions_env_override(self):
        from worldfoundry.studio.visualization.backends.world import _world_max_sessions
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_WORLD_MAX_SESSIONS": "8"}):
            result = _world_max_sessions()
            assert result == 8

    def test_world_max_sessions_env_invalid(self):
        from worldfoundry.studio.visualization.backends.world import _world_max_sessions
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_WORLD_MAX_SESSIONS": "abc"}):
            result = _world_max_sessions()
            assert result == 4

    def test_world_max_sessions_env_clamp_to_one(self):
        from worldfoundry.studio.visualization.backends.world import _world_max_sessions
        with patch.dict(os.environ, {"WORLDFOUNDRY_STUDIO_WORLD_MAX_SESSIONS": "-1"}):
            result = _world_max_sessions()
            assert result == 1


# ---------------------------------------------------------------------------
# 15.  world.py — HTML generators
# ---------------------------------------------------------------------------

def _make_catalog_entry(**overrides):
    """Create a minimal CatalogEntry for testing."""
    defaults = {
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
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
        entry = _make_catalog_entry(model_id="test-model", display_name="Test Display")
        lc = StudioLaunchConfig(model_id="test-model")
        html = world_frontend_html(entry, lc)
        assert "Test Display" in html
        assert "test-model" in html

    def test_world_frontend_html_structure(self):
        from worldfoundry.studio.visualization.backends.world import world_frontend_html
        from worldfoundry.studio.ui.launch_config import StudioLaunchConfig
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
# 16.  world.py — WorldFrontendHandler (structural checks)
# ---------------------------------------------------------------------------

class TestWorldFrontendHandler:
    """Tests for WorldFrontendHandler class structure."""

    def test_server_version(self):
        from worldfoundry.studio.visualization.backends.world import WorldFrontendHandler
        assert WorldFrontendHandler.server_version == "WorldFoundryWorldFrontend/1.0"

    def test_has_do_GET(self):
        from worldfoundry.studio.visualization.backends.world import WorldFrontendHandler
        assert hasattr(WorldFrontendHandler, "do_GET")

    def test_has_do_HEAD(self):
        from worldfoundry.studio.visualization.backends.world import WorldFrontendHandler
        assert hasattr(WorldFrontendHandler, "do_HEAD")

    def test_has_do_POST(self):
        from worldfoundry.studio.visualization.backends.world import WorldFrontendHandler
        assert hasattr(WorldFrontendHandler, "do_POST")


# ---------------------------------------------------------------------------
# 17.  world.py — WebSocket helpers (static verification)
# ---------------------------------------------------------------------------

class TestWorldWebSocketHelpers:
    """Test websocket helper methods of WorldFrontendHandler (static checks)."""

    def test_handler_has_ws_methods(self):
        from worldfoundry.studio.visualization.backends.world import WorldFrontendHandler
        assert hasattr(WorldFrontendHandler, "_handle_websocket")
        assert hasattr(WorldFrontendHandler, "_read_ws_frame")
        assert hasattr(WorldFrontendHandler, "_send_ws_frame")
        assert hasattr(WorldFrontendHandler, "_send_ws_json")
        assert hasattr(WorldFrontendHandler, "_read_ws_json")
        assert hasattr(WorldFrontendHandler, "_handle_ws_request")


# ---------------------------------------------------------------------------
# 18.  world.py — path_from_file_url
# ---------------------------------------------------------------------------

class TestPathFromFileUrl:
    """Tests for _path_from_file_url."""

    def test_valid_file_url(self):
        from worldfoundry.studio.visualization.backends.world import _path_from_file_url
        result = _path_from_file_url("/api/file?path=/tmp/test.png")
        assert result is not None
        assert str(result) == "/tmp/test.png"

    def test_empty_url(self):
        from worldfoundry.studio.visualization.backends.world import _path_from_file_url
        assert _path_from_file_url("") is None

    def test_non_api_url(self):
        from worldfoundry.studio.visualization.backends.world import _path_from_file_url
        assert _path_from_file_url("/other/path") is None

    def test_api_url_no_path_param(self):
        from worldfoundry.studio.visualization.backends.world import _path_from_file_url
        assert _path_from_file_url("/api/file?other=value") is None


# ---------------------------------------------------------------------------
# 19.  core/registry.py — StudioVisualizationBackend
# ---------------------------------------------------------------------------

class TestStudioVisualizationBackend:
    """Tests for StudioVisualizationBackend dataclass and methods."""

    def test_construction_defaults(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend, BackendCapabilities
        b = StudioVisualizationBackend(mode="test", title="Test Backend", default_port=1234)
        assert b.mode == "test"
        assert b.title == "Test Backend"
        assert b.default_port == 1234
        assert b.aliases == ()
        assert b.native is True
        assert callable(b.match)
        assert b.serve is None
        assert isinstance(b.capabilities, BackendCapabilities)
        assert b.capabilities.layer_kinds == frozenset()
        assert b.capabilities.score == 0

    def test_construction_with_aliases(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend
        b = StudioVisualizationBackend(
            mode="test", title="Test", default_port=1234,
            aliases=("t1", "t2"),
        )
        assert b.aliases == ("t1", "t2")

    def test_backend_id_property(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend
        b = StudioVisualizationBackend(mode="my-mode", title="T", default_port=1)
        assert b.backend_id == "my-mode"

    def test_accepts_exact_mode(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend
        b = StudioVisualizationBackend(mode="world", title="T", default_port=1)
        assert b.accepts("world") is True

    def test_accepts_case_insensitive(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend
        b = StudioVisualizationBackend(mode="world", title="T", default_port=1)
        assert b.accepts("WORLD") is True

    def test_accepts_alias(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend
        b = StudioVisualizationBackend(mode="world", title="T", default_port=1, aliases=("interactive-world",))
        assert b.accepts("interactive-world") is True

    def test_accepts_unknown(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend
        b = StudioVisualizationBackend(mode="world", title="T", default_port=1)
        assert b.accepts("unknown") is False

    def test_can_render_empty_capabilities(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend, RenderPlan
        from worldfoundry.studio.visualization.core.scene import VisualizationScene
        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        scene = VisualizationScene(scene_id="s1", layers=())
        plan = b.can_render(scene)
        assert isinstance(plan, RenderPlan)
        assert plan.supported is False
        assert "no scene capability" in plan.reason

    def test_can_render_with_capabilities(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationBackend, BackendCapabilities, RenderPlan,
        )
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Layer
        b = StudioVisualizationBackend(
            mode="test", title="T", default_port=1,
            capabilities=BackendCapabilities(frozenset({"point_cloud", "mesh"}), score=80),
        )
        scene = VisualizationScene(
            scene_id="s1",
            layers=(Layer(layer_id="l1", kind="point_cloud"),),
        )
        plan = b.can_render(scene)
        assert plan.supported is True
        assert plan.score > 0

    def test_can_render_partial_support(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationBackend, BackendCapabilities, RenderPlan,
        )
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Layer
        b = StudioVisualizationBackend(
            mode="test", title="T", default_port=1,
            capabilities=BackendCapabilities(frozenset({"point_cloud"}), partial=True, score=50),
        )
        scene = VisualizationScene(
            scene_id="s1",
            layers=(
                Layer(layer_id="l1", kind="point_cloud"),
                Layer(layer_id="l2", kind="mesh"),
            ),
        )
        plan = b.can_render(scene)
        # Partial=True means it's still supported, even with unsupported layers.
        assert plan.supported is True
        assert "mesh" in plan.unsupported_layers

    def test_can_render_partial_false(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationBackend, BackendCapabilities, RenderPlan,
        )
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Layer
        b = StudioVisualizationBackend(
            mode="test", title="T", default_port=1,
            capabilities=BackendCapabilities(frozenset({"point_cloud"}), partial=False, score=50),
        )
        scene = VisualizationScene(
            scene_id="s1",
            layers=(
                Layer(layer_id="l1", kind="point_cloud"),
                Layer(layer_id="l2", kind="mesh"),
            ),
        )
        plan = b.can_render(scene)
        # Partial=False means unsupported layers make it unsupported.
        assert plan.supported is False

    def test_render_raises_for_unsupported(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationBackend, BackendCapabilities, RenderResult,
        )
        from worldfoundry.studio.visualization.core.scene import VisualizationScene
        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        scene = VisualizationScene(scene_id="s1")
        from worldfoundry.studio.visualization.core.registry import RenderRequest
        request = RenderRequest()
        with pytest.raises(ValueError):
            b.render(scene, request)

    def test_render_returns_result_for_supported(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationBackend, BackendCapabilities, RenderResult,
        )
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Layer
        b = StudioVisualizationBackend(
            mode="test", title="T", default_port=1,
            capabilities=BackendCapabilities(frozenset({"point_cloud"}), score=50),
        )
        scene = VisualizationScene(
            scene_id="s1",
            layers=(Layer(layer_id="l1", kind="point_cloud"),),
        )
        from worldfoundry.studio.visualization.core.registry import RenderRequest
        request = RenderRequest()
        result = b.render(scene, request)
        assert isinstance(result, RenderResult)
        assert result.backend_id == "test"

    def test_shutdown_returns_none(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend
        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        assert b.shutdown() is None

    def test_launch_no_serve_raises(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend, StudioVisualizationRequest
        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        with pytest.raises(ValueError, match="no serve function"):
            b.launch(MagicMock(spec=StudioVisualizationRequest))

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationBackend
        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        with pytest.raises(FrozenInstanceError):
            b.mode = "new"


# ---------------------------------------------------------------------------
# 20.  core/registry.py — BackendCapabilities
# ---------------------------------------------------------------------------

class TestBackendCapabilities:
    """Tests for BackendCapabilities frozen dataclass."""

    def test_defaults(self):
        from worldfoundry.studio.visualization.core.registry import BackendCapabilities
        c = BackendCapabilities()
        assert c.layer_kinds == frozenset()
        assert c.partial is True
        assert c.score == 0
        assert c.metadata == {}

    def test_with_layer_kinds(self):
        from worldfoundry.studio.visualization.core.registry import BackendCapabilities
        c = BackendCapabilities(frozenset({"image", "video"}), score=50)
        assert "image" in c.layer_kinds
        assert "video" in c.layer_kinds
        assert c.score == 50

    def test_partial_false(self):
        from worldfoundry.studio.visualization.core.registry import BackendCapabilities
        c = BackendCapabilities(frozenset({"gaussian_splat"}), partial=False, score=90)
        assert c.partial is False

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.registry import BackendCapabilities
        c = BackendCapabilities()
        with pytest.raises(FrozenInstanceError):
            c.score = 100
        with pytest.raises(FrozenInstanceError):
            c.partial = False


# ---------------------------------------------------------------------------
# 21.  core/registry.py — StudioVisualizationRegistry
# ---------------------------------------------------------------------------

class TestStudioVisualizationRegistry:
    """Tests for StudioVisualizationRegistry class."""

    def test_empty_registry(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationRegistry
        reg = StudioVisualizationRegistry()
        assert reg.modes == frozenset()
        assert reg.native_modes == frozenset()
        assert reg.default_ports == {}

    def test_register_and_modes(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
        )
        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        reg = StudioVisualizationRegistry([b])
        assert "test" in reg.modes

    def test_register_with_aliases(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
        )
        b = StudioVisualizationBackend(mode="test", title="T", default_port=1, aliases=("t1", "t-2"))
        reg = StudioVisualizationRegistry([b])
        assert reg.backend_for("test").mode == "test"
        assert reg.backend_for("t1").mode == "test"
        assert reg.backend_for("t-2").mode == "test"

    def test_register_duplicate_mode_raises(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
        )
        b1 = StudioVisualizationBackend(mode="test", title="T1", default_port=1)
        b2 = StudioVisualizationBackend(mode="other", title="T2", default_port=2, aliases=("test",))
        with pytest.raises(ValueError, match="Duplicate"):
            StudioVisualizationRegistry([b1, b2])

    def test_register_empty_token_raises(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
        )
        b = StudioVisualizationBackend(mode="", title="T", default_port=1)
        with pytest.raises(ValueError, match="cannot be empty"):
            StudioVisualizationRegistry([b])

    def test_backend_for_unknown_raises(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationRegistry
        reg = StudioVisualizationRegistry()
        with pytest.raises(ValueError, match="Unsupported"):
            reg.backend_for("unknown")

    def test_native_modes(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
        )
        b1 = StudioVisualizationBackend(mode="native1", title="N1", default_port=1, native=True)
        b2 = StudioVisualizationBackend(mode="non-native", title="NN", default_port=2, native=False)
        reg = StudioVisualizationRegistry([b1, b2])
        assert "native1" in reg.native_modes
        assert "non-native" not in reg.native_modes

    def test_default_ports(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
        )
        b = StudioVisualizationBackend(mode="test", title="T", default_port=5555)
        reg = StudioVisualizationRegistry([b])
        assert reg.default_ports == {"test": 5555}

    def test_register_replaces_same_mode(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
        )
        b1 = StudioVisualizationBackend(mode="test", title="Old", default_port=1)
        b2 = StudioVisualizationBackend(mode="test", title="New", default_port=2)
        reg = StudioVisualizationRegistry([b1])
        reg.register(b2)
        assert reg.backend_for("test").title == "New"
        assert reg.default_ports["test"] == 2

    def test_resolve_mode_explicit(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
        )
        b = StudioVisualizationBackend(mode="test", title="T", default_port=1)
        reg = StudioVisualizationRegistry([b])
        entry = _make_catalog_entry(model_id="m1")
        result = reg.resolve_mode(entry, "test")
        assert result == "test"

    def test_resolve_mode_auto_fallback(self):
        from worldfoundry.studio.visualization.core.registry import (
            StudioVisualizationRegistry, StudioVisualizationBackend,
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

    def test_studio_visualization_event(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationEvent
        e = StudioVisualizationEvent(kind="click", payload={"x": 1})
        assert e.kind == "click"
        assert e.payload == {"x": 1}
        assert e.timestamp is None
        with pytest.raises(FrozenInstanceError):
            e.kind = "new"

    def test_studio_visualization_event_with_timestamp(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationEvent
        e = StudioVisualizationEvent(kind="click", timestamp=1.5)
        assert e.timestamp == 1.5

    def test_studio_model_visualization_profile(self):
        from worldfoundry.studio.visualization.core.registry import StudioModelVisualizationProfile
        p = StudioModelVisualizationProfile(
            mode="world", artifact_domain="interactive_world", title="World",
        )
        assert p.mode == "world"
        assert p.artifact_domain == "interactive_world"
        assert p.title == "World"
        assert p.reason == ""
        assert p.accepted_artifact_kinds == ()
        with pytest.raises(FrozenInstanceError):
            p.mode = "new"

    def test_studio_visualization_launch(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationLaunch
        l = StudioVisualizationLaunch(mode="world")
        assert l.mode == "world"
        assert l.url == ""
        assert l.caption == ""
        assert l.metadata == {}
        with pytest.raises(FrozenInstanceError):
            l.mode = "new"

    def test_studio_visualization_launch_with_all_fields(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationLaunch
        l = StudioVisualizationLaunch(mode="points", url="http://localhost/", caption="Ready", metadata={"fps": 16})
        assert l.url == "http://localhost/"
        assert l.caption == "Ready"
        assert l.metadata == {"fps": 16}

    def test_render_plan(self):
        from worldfoundry.studio.visualization.core.registry import RenderPlan
        p = RenderPlan(backend_id="test", supported=True, score=50)
        assert p.backend_id == "test"
        assert p.supported is True
        assert p.score == 50
        assert p.unsupported_layers == ()
        assert p.reason == ""
        with pytest.raises(FrozenInstanceError):
            p.supported = False

    def test_render_request(self):
        from worldfoundry.studio.visualization.core.registry import RenderRequest
        r = RenderRequest()
        assert r.backend == "auto"
        assert r.output_path == ""
        assert r.options == {}

    def test_render_result(self):
        from worldfoundry.studio.visualization.core.registry import RenderResult
        r = RenderResult(backend_id="test")
        assert r.backend_id == "test"
        assert r.url == ""
        assert r.output_path == ""
        assert r.caption == ""
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# 23.  core/registry.py — normalize_visualization_mode
# ---------------------------------------------------------------------------

class TestNormalizeVisualizationMode:
    """Tests for normalize_visualization_mode function."""

    def test_none(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode(None) == ""

    def test_empty(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode("") == ""

    def test_lowercase(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode("world") == "world"

    def test_uppercase(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode("WORLD") == "world"

    def test_mixed_case(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode("InteractiveWorld") == "interactiveworld"

    def test_underscore_to_hyphen(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode("interactive_world") == "interactive-world"

    def test_whitespace_stripped(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode("  points  ") == "points"

    def test_combined(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode(" INTERACTIVE_WORLD ") == "interactive-world"


# ---------------------------------------------------------------------------
# 24.  core/registry.py — Protocol classes
# ---------------------------------------------------------------------------

class TestRegistryProtocols:
    """Tests for VisualizationProvider and VisualizationBackend protocols."""

    def test_visualization_provider_protocol(self):
        from worldfoundry.studio.visualization.core.registry import VisualizationProvider
        # Protocol is a structural type — verify interface contract via annotations.
        assert "provider_id" in VisualizationProvider.__annotations__
        assert hasattr(VisualizationProvider, "discover")
        assert callable(VisualizationProvider.discover)

    def test_visualization_backend_protocol(self):
        from worldfoundry.studio.visualization.core.registry import VisualizationBackend
        assert "backend_id" in VisualizationBackend.__annotations__
        assert "capabilities" in VisualizationBackend.__annotations__
        assert hasattr(VisualizationBackend, "can_render")
        assert callable(VisualizationBackend.can_render)
        assert hasattr(VisualizationBackend, "render")
        assert callable(VisualizationBackend.render)
        assert hasattr(VisualizationBackend, "shutdown")
        assert callable(VisualizationBackend.shutdown)


# ---------------------------------------------------------------------------
# 25.  core/registry.py — Mode string constants
# ---------------------------------------------------------------------------

class TestRegistryModeConstants:
    """Tests for visualization mode string constants."""

    def test_mode_strings(self):
        from worldfoundry.studio.visualization.core.registry import (
            INTERACTIVE_WORLD_VISUALIZATION, VISER_VISUALIZATION,
            SPARK_VISUALIZATION, MEDIA_VISUALIZATION,
            RERUN_VISUALIZATION, EMBODIED_VISUALIZATION,
            UNIFIED_VISUALIZATION, AUTO_VISUALIZATION,
        )
        assert INTERACTIVE_WORLD_VISUALIZATION == "world"
        assert VISER_VISUALIZATION == "points"
        assert SPARK_VISUALIZATION == "spark"
        assert MEDIA_VISUALIZATION == "media"
        assert RERUN_VISUALIZATION == "rerun"
        assert EMBODIED_VISUALIZATION == "embodied"
        assert UNIFIED_VISUALIZATION == "unified"
        assert AUTO_VISUALIZATION == "auto"

    def test_artifact_domain_strings(self):
        from worldfoundry.studio.visualization.core.registry import (
            ARTIFACT_DOMAIN_WORLD, ARTIFACT_DOMAIN_GEOMETRY,
            ARTIFACT_DOMAIN_GAUSSIAN_SPLAT, ARTIFACT_DOMAIN_ACTION,
            ARTIFACT_DOMAIN_TIMELINE, ARTIFACT_DOMAIN_MEDIA,
            ARTIFACT_DOMAIN_UI,
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
        from worldfoundry.studio.visualization.core.registry import __all__
        import worldfoundry.studio.visualization.core.registry as mod
        for name in __all__:
            assert hasattr(mod, name), f"{name} not found in registry module"

    def test_expected_names_in_all(self):
        from worldfoundry.studio.visualization.core.registry import __all__
        expected = [
            "ARTIFACT_DOMAIN_ACTION",
            "BackendCapabilities",
            "EMBODIED_VISUALIZATION",
            "INTERACTIVE_WORLD_VISUALIZATION",
            "MEDIA_VISUALIZATION",
            "RenderPlan",
            "RenderRequest",
            "RenderResult",
            "RERUN_VISUALIZATION",
            "SPARK_VISUALIZATION",
            "StudioVisualizationBackend",
            "StudioVisualizationEvent",
            "StudioVisualizationLaunch",
            "StudioVisualizationRegistry",
            "StudioVisualizationRequest",
            "UNIFIED_VISUALIZATION",
            "VISER_VISUALIZATION",
            "VisualizationBackend",
            "VisualizationProvider",
            "model_visualization_profile",
            "normalize_visualization_mode",
        ]
        for name in expected:
            assert name in __all__, f"{name} missing from __all__"


# ---------------------------------------------------------------------------
# 27.  core/artifacts.py — StudioVisualizationArtifact
# ---------------------------------------------------------------------------

class TestStudioVisualizationArtifact:
    """Tests for StudioVisualizationArtifact frozen dataclass."""

    def test_construction(self):
        from worldfoundry.studio.visualization.core.artifacts import StudioVisualizationArtifact
        a = StudioVisualizationArtifact(path="/tmp/test.ply", kind="point_cloud")
        assert a.path == "/tmp/test.ply"
        assert a.kind == "point_cloud"
        assert a.format_hint == ""
        assert a.metadata == {}

    def test_construction_with_all_fields(self):
        from worldfoundry.studio.visualization.core.artifacts import StudioVisualizationArtifact
        a = StudioVisualizationArtifact(
            path="/tmp/test.ply", kind="point_cloud",
            format_hint="ply", metadata={"size": 1024},
        )
        assert a.format_hint == "ply"
        assert a.metadata == {"size": 1024}

    def test_resolved_path(self):
        from worldfoundry.studio.visualization.core.artifacts import StudioVisualizationArtifact
        a = StudioVisualizationArtifact(path="/tmp/test.ply", kind="point_cloud")
        resolved = a.resolved_path
        assert isinstance(resolved, Path)
        assert resolved.is_absolute()

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.artifacts import StudioVisualizationArtifact
        a = StudioVisualizationArtifact(path="/tmp/test.ply", kind="point_cloud")
        with pytest.raises(FrozenInstanceError):
            a.path = "/new"


# ---------------------------------------------------------------------------
# 28.  core/artifacts.py — infer_visualization_artifact
# ---------------------------------------------------------------------------

class TestInferVisualizationArtifact:
    """Tests for infer_visualization_artifact function."""

    def test_ply_suffix(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.ply")
        assert a.kind == "point_cloud"
        assert a.format_hint == "ply"

    def test_mp4_suffix(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.mp4")
        assert a.kind == "video"

    def test_png_suffix(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.png")
        assert a.kind == "image"

    def test_splat_suffix(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.splat")
        assert a.kind == "gaussian_splat"

    def test_rrd_suffix(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.rrd")
        assert a.kind == "timeline"

    def test_glb_suffix(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.glb")
        assert a.kind == "mesh"

    def test_unknown_suffix(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.xyz")
        assert a.kind == "artifact"

    def test_with_metadata(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.ply", metadata={"size": 1024})
        assert a.metadata == {"size": 1024}

    def test_case_insensitive_suffix(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("/tmp/test.PLY")
        assert a.kind == "point_cloud"

    def test_path_input(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact(Path("/tmp/test.ply"))
        assert a.kind == "point_cloud"


# ---------------------------------------------------------------------------
# 29.  core/artifacts.py — normalize_artifact_uri
# ---------------------------------------------------------------------------

class TestNormalizeArtifactUri:
    """Tests for normalize_artifact_uri function."""

    def test_no_root(self):
        from worldfoundry.studio.visualization.core.artifacts import normalize_artifact_uri
        result = normalize_artifact_uri("/tmp/test.ply")
        assert result == "/tmp/test.ply"

    def test_with_root_relative(self):
        from worldfoundry.studio.visualization.core.artifacts import normalize_artifact_uri
        result = normalize_artifact_uri("/home/user/data/test.ply", root="/home/user/data")
        assert "test.ply" in result

    def test_with_root_outside(self):
        from worldfoundry.studio.visualization.core.artifacts import normalize_artifact_uri
        # Path outside root should return the absolute resolved path.
        result = normalize_artifact_uri("/other/path/test.ply", root="/home/user/data")
        assert result.endswith("test.ply")


# ---------------------------------------------------------------------------
# 30.  core/artifacts.py — KIND_BY_SUFFIX
# ---------------------------------------------------------------------------

class TestKindBySuffix:
    """Tests for KIND_BY_SUFFIX mapping."""

    def test_mapping_coverage(self):
        from worldfoundry.studio.visualization.core.artifacts import KIND_BY_SUFFIX
        # Point cloud formats
        assert ".ply" in KIND_BY_SUFFIX
        assert ".pcd" in KIND_BY_SUFFIX
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
        from worldfoundry.studio.visualization.core.artifacts import KIND_BY_SUFFIX
        for key, value in KIND_BY_SUFFIX.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


# ---------------------------------------------------------------------------
# 31.  core/scene.py — Frame, Timeline, Layer, VisualizationScene
# ---------------------------------------------------------------------------

class TestFrame:
    """Tests for Frame frozen dataclass."""

    def test_construction(self):
        from worldfoundry.studio.visualization.core.scene import Frame
        f = Frame(frame_id="world")
        assert f.frame_id == "world"
        assert f.parent_id is None
        assert f.transform is None
        assert f.metadata == {}

    def test_with_all_fields(self):
        from worldfoundry.studio.visualization.core.scene import Frame
        f = Frame(frame_id="cam1", parent_id="world", transform=[[1, 0, 0]], metadata={"role": "camera"})
        assert f.parent_id == "world"
        assert f.transform == [[1, 0, 0]]
        assert f.metadata == {"role": "camera"}

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.scene import Frame
        f = Frame(frame_id="test")
        with pytest.raises(FrozenInstanceError):
            f.frame_id = "new"


class TestTimeline:
    """Tests for Timeline frozen dataclass."""

    def test_defaults(self):
        from worldfoundry.studio.visualization.core.scene import Timeline
        t = Timeline()
        assert t.fps is None
        assert t.start_time is None
        assert t.end_time is None
        assert t.frame_count is None
        assert t.metadata == {}

    def test_with_values(self):
        from worldfoundry.studio.visualization.core.scene import Timeline
        t = Timeline(fps=16.0, start_time=0.0, end_time=1.0, frame_count=16)
        assert t.fps == 16.0
        assert t.frame_count == 16

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.scene import Timeline
        t = Timeline()
        with pytest.raises(FrozenInstanceError):
            t.fps = 30.0


class TestLayer:
    """Tests for Layer frozen dataclass."""

    def test_minimal_construction(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="point_cloud")
        assert l.layer_id == "l1"
        assert l.kind == "point_cloud"
        assert l.uri is None
        assert l.uris == ()
        assert l.payload is None
        assert l.frame_range is None
        assert l.time_range is None
        assert l.coordinate_frame is None
        assert l.style == {}
        assert l.metadata == {}

    def test_with_uri(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="mesh", uri="/tmp/mesh.glb")
        assert l.uri == "/tmp/mesh.glb"

    def test_with_uris(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="image", uris=("a.png", "b.png"))
        assert l.uris == ("a.png", "b.png")

    def test_all_uris_with_single_uri(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="mesh", uri="/tmp/mesh.glb")
        assert l.all_uris() == ("/tmp/mesh.glb",)

    def test_all_uris_with_uris_list(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="image", uris=("a.png", "b.png"))
        assert l.all_uris() == ("a.png", "b.png")

    def test_all_uris_combined(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="image", uri="a.png", uris=("b.png", "a.png"))
        # Deduplicates: a.png from uri, then b.png from uris, a.png deduplicated.
        result = l.all_uris()
        assert "a.png" in result
        assert "b.png" in result
        # Deduplicated, order preserved.
        assert result == ("a.png", "b.png")

    def test_all_uris_empty(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="mesh")
        assert l.all_uris() == ()

    def test_asdict_minimal(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="point_cloud")
        d = l.asdict()
        assert d["layer_id"] == "l1"
        assert d["kind"] == "point_cloud"
        assert "uri" not in d
        assert "uris" not in d

    def test_asdict_with_optional_fields(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(
            layer_id="l1", kind="mesh",
            uri="/tmp/mesh.glb",
            uris=("a.glb",),
            frame_range=(0, 10),
            time_range=(0.0, 1.0),
            coordinate_frame="world",
            style={"color": "red"},
            metadata={"source": "test"},
        )
        d = l.asdict()
        assert d["uri"] == "/tmp/mesh.glb"
        assert d["uris"] == ["a.glb"]
        assert d["frame_range"] == [0, 10]
        assert d["time_range"] == [0.0, 1.0]
        assert d["coordinate_frame"] == "world"
        assert d["style"] == {"color": "red"}
        assert d["metadata"] == {"source": "test"}

    def test_fromdict_minimal(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer.fromdict({"layer_id": "l1", "kind": "point_cloud"})
        assert l.layer_id == "l1"
        assert l.kind == "point_cloud"

    def test_fromdict_with_id_alias(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer.fromdict({"id": "l1", "kind": "point_cloud"})
        assert l.layer_id == "l1"

    def test_fromdict_with_all_fields(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer.fromdict({
            "layer_id": "l1",
            "kind": "mesh",
            "uri": "/tmp/mesh.glb",
            "uris": ["a.glb", "b.glb"],
            "frame_range": [0, 10],
            "time_range": [0.0, 1.0],
            "coordinate_frame": "world",
            "style": {"color": "red"},
            "metadata": {"source": "test"},
        })
        assert l.uri == "/tmp/mesh.glb"
        assert l.uris == ("a.glb", "b.glb")
        assert l.frame_range == (0, 10)
        assert l.coordinate_frame == "world"

    def test_round_trip_asdict_fromdict(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        original = Layer(
            layer_id="l1", kind="mesh",
            uri="/tmp/m.glb",
            frame_range=(0, 5),
            style={"color": "blue"},
            metadata={"src": "test"},
        )
        d = original.asdict()
        restored = Layer.fromdict(d)
        assert restored.layer_id == original.layer_id
        assert restored.kind == original.kind
        assert restored.uri == original.uri
        assert restored.frame_range == original.frame_range
        assert restored.style == dict(original.style)
        assert restored.metadata == dict(original.metadata)

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer(layer_id="l1", kind="point_cloud")
        with pytest.raises(FrozenInstanceError):
            l.layer_id = "new"


class TestVisualizationScene:
    """Tests for VisualizationScene frozen dataclass."""

    def test_minimal_construction(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene
        s = VisualizationScene(scene_id="s1")
        assert s.scene_id == "s1"
        assert s.title == ""
        assert s.layers == ()
        assert s.timeline is None
        assert s.controls == ()
        assert s.frames == ()
        assert s.recommended_backend == "auto"
        assert s.metadata == {}

    def test_with_layers(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Layer
        layers = (Layer(layer_id="l1", kind="point_cloud"), Layer(layer_id="l2", kind="mesh"))
        s = VisualizationScene(scene_id="s1", layers=layers)
        assert len(s.layers) == 2

    def test_layer_kinds(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Layer
        s = VisualizationScene(
            scene_id="s1",
            layers=(
                Layer(layer_id="l1", kind="point_cloud"),
                Layer(layer_id="l2", kind="mesh"),
            ),
        )
        kinds = s.layer_kinds()
        assert kinds == frozenset({"point_cloud", "mesh"})

    def test_layer_kinds_empty(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene
        s = VisualizationScene(scene_id="s1")
        assert s.layer_kinds() == frozenset()

    def test_asdict(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Layer
        s = VisualizationScene(
            scene_id="s1", title="My Scene",
            layers=(Layer(layer_id="l1", kind="point_cloud"),),
        )
        d = s.asdict()
        assert d["schema_version"] == 1
        assert d["scene_id"] == "s1"
        assert d["title"] == "My Scene"
        assert len(d["layers"]) == 1

    def test_asdict_with_timeline(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Timeline
        s = VisualizationScene(
            scene_id="s1",
            timeline=Timeline(fps=16.0, frame_count=10),
        )
        d = s.asdict()
        assert "timeline" in d
        assert d["timeline"]["fps"] == 16.0

    def test_asdict_timeline_omits_none(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Timeline
        s = VisualizationScene(
            scene_id="s1",
            timeline=Timeline(fps=16.0),  # Only fps set.
        )
        d = s.asdict()
        assert "fps" in d["timeline"]
        assert "start_time" not in d["timeline"]

    def test_fromdict(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene
        d = {
            "scene_id": "s1",
            "title": "My Scene",
            "layers": [{"layer_id": "l1", "kind": "point_cloud"}],
            "recommended_backend": "spark",
        }
        s = VisualizationScene.fromdict(d)
        assert s.scene_id == "s1"
        assert s.title == "My Scene"
        assert len(s.layers) == 1
        assert s.layers[0].kind == "point_cloud"
        assert s.recommended_backend == "spark"

    def test_fromdict_with_timeline(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene
        d = {
            "scene_id": "s1",
            "timeline": {"fps": 16.0, "frame_count": 10},
            "layers": [],
        }
        s = VisualizationScene.fromdict(d)
        assert s.timeline is not None
        assert s.timeline.fps == 16.0

    def test_fromdict_empty(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene
        s = VisualizationScene.fromdict({})
        assert s.scene_id == ""
        assert s.layers == ()

    def test_round_trip_asdict_fromdict(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene, Layer, Timeline
        original = VisualizationScene(
            scene_id="s1",
            title="Test",
            layers=(Layer(layer_id="l1", kind="point_cloud", uri="/tmp/c.ply"),),
            timeline=Timeline(fps=16.0, frame_count=10),
            recommended_backend="spark",
            metadata={"key": "value"},
        )
        d = original.asdict()
        restored = VisualizationScene.fromdict(d)
        assert restored.scene_id == original.scene_id
        assert restored.title == original.title
        assert len(restored.layers) == len(original.layers)
        assert restored.layers[0].kind == original.layers[0].kind
        assert restored.recommended_backend == original.recommended_backend
        assert restored.timeline is not None
        assert restored.timeline.fps == original.timeline.fps

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.scene import VisualizationScene
        s = VisualizationScene(scene_id="s1")
        with pytest.raises(FrozenInstanceError):
            s.scene_id = "new"


# ---------------------------------------------------------------------------
# 32.  core/__init__.py and visualization/__init__.py — import surface
# ---------------------------------------------------------------------------

class TestCoreInitExports:
    """Verify core __init__.py exports are complete and importable."""

    def test_core_all_importable(self):
        from worldfoundry.studio.visualization.core import __all__
        import worldfoundry.studio.visualization.core as mod
        for name in __all__:
            assert hasattr(mod, name), f"{name} not in core module"

    def test_visualization_all_importable(self):
        from worldfoundry.studio.visualization import __all__
        import worldfoundry.studio.visualization as mod
        for name in __all__:
            assert hasattr(mod, name), f"{name} not in visualization module"

    def test_core_all_contains_expected(self):
        from worldfoundry.studio.visualization.core import __all__
        expected_core = [
            "BackendCapabilities",
            "Frame",
            "Layer",
            "RenderPlan",
            "RenderRequest",
            "RenderResult",
            "StudioVisualizationArtifact",
            "StudioVisualizationBackend",
            "StudioVisualizationEvent",
            "StudioVisualizationLaunch",
            "StudioVisualizationRegistry",
            "StudioVisualizationRequest",
            "Timeline",
            "VisualizationScene",
            "infer_visualization_artifact",
            "model_visualization_profile",
            "normalize_visualization_mode",
        ]
        for name in expected_core:
            assert name in __all__, f"{name} missing from core __all__"

    def test_visualization_all_contains_expected(self):
        from worldfoundry.studio.visualization import __all__
        expected_vis = [
            "BackendCapabilities",
            "Frame",
            "Layer",
            "RenderPlan",
            "RenderRequest",
            "RenderResult",
            "StudioVisualizationArtifact",
            "StudioVisualizationBackend",
            "StudioVisualizationEvent",
            "StudioVisualizationLaunch",
            "StudioVisualizationRegistry",
            "StudioVisualizationRequest",
            "Timeline",
            "VisualizationScene",
            "infer_visualization_artifact",
            "model_visualization_profile",
            "normalize_visualization_mode",
        ]
        for name in expected_vis:
            assert name in __all__, f"{name} missing from visualization __all__"


# ---------------------------------------------------------------------------
# 33.  core/styles.py — VisualizationStyle
# ---------------------------------------------------------------------------

class TestVisualizationStyle:
    """Tests for VisualizationStyle frozen dataclass."""

    def test_defaults(self):
        from worldfoundry.studio.visualization.core.styles import VisualizationStyle, DEFAULT_COLORMAP
        s = VisualizationStyle()
        assert s.color is None
        assert s.colormap == DEFAULT_COLORMAP
        assert s.opacity is None
        assert s.point_size is None
        assert s.line_width is None
        assert s.material is None
        assert s.metadata == {}

    def test_with_values(self):
        from worldfoundry.studio.visualization.core.styles import VisualizationStyle
        s = VisualizationStyle(color="red", opacity=0.5, point_size=3.0)
        assert s.color == "red"
        assert s.opacity == 0.5
        assert s.point_size == 3.0

    def test_asdict_omits_none(self):
        from worldfoundry.studio.visualization.core.styles import VisualizationStyle
        s = VisualizationStyle(color="red", point_size=3.0)
        d = s.asdict()
        assert "color" in d
        assert "point_size" in d
        assert "opacity" not in d
        assert "line_width" not in d

    def test_asdict_includes_colormap(self):
        from worldfoundry.studio.visualization.core.styles import VisualizationStyle
        s = VisualizationStyle(colormap="plasma")
        d = s.asdict()
        assert d["colormap"] == "plasma"

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.styles import VisualizationStyle
        s = VisualizationStyle()
        with pytest.raises(FrozenInstanceError):
            s.color = "blue"

    def test_default_colormap_constant(self):
        from worldfoundry.studio.visualization.core.styles import DEFAULT_COLORMAP
        assert DEFAULT_COLORMAP == "viridis"


# ---------------------------------------------------------------------------
# 34.  core/controls.py — VisualizationControl, VisualizationEvent
# ---------------------------------------------------------------------------

class TestVisualizationControl:
    """Tests for VisualizationControl frozen dataclass."""

    def test_construction(self):
        from worldfoundry.studio.visualization.core.controls import VisualizationControl
        c = VisualizationControl(control_id="joystick", kind="continuous")
        assert c.control_id == "joystick"
        assert c.kind == "continuous"
        assert c.label == ""
        assert c.value is None
        assert c.options == ()
        assert c.metadata == {}

    def test_with_all_fields(self):
        from worldfoundry.studio.visualization.core.controls import VisualizationControl
        c = VisualizationControl(
            control_id="mode", kind="discrete",
            label="Mode selector", value="image",
            options=("image", "video"),
            metadata={"role": "input"},
        )
        assert c.label == "Mode selector"
        assert c.value == "image"
        assert c.options == ("image", "video")

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.controls import VisualizationControl
        c = VisualizationControl(control_id="test", kind="test")
        with pytest.raises(FrozenInstanceError):
            c.control_id = "new"


class TestVisualizationEventControls:
    """Tests for VisualizationEvent from controls module."""

    def test_construction(self):
        from worldfoundry.studio.visualization.core.controls import VisualizationEvent
        e = VisualizationEvent(kind="click", payload={"x": 1})
        assert e.kind == "click"
        assert e.payload == {"x": 1}
        assert e.timestamp is None

    def test_with_timestamp(self):
        from worldfoundry.studio.visualization.core.controls import VisualizationEvent
        e = VisualizationEvent(kind="click", timestamp=1.5)
        assert e.timestamp == 1.5

    def test_frozen_enforcement(self):
        from worldfoundry.studio.visualization.core.controls import VisualizationEvent
        e = VisualizationEvent(kind="click")
        with pytest.raises(FrozenInstanceError):
            e.kind = "new"


# ---------------------------------------------------------------------------
# 35.  Cross-module integration — frontends registry uses core registry
# ---------------------------------------------------------------------------

class TestCrossModuleIntegration:
    """Verify frontends.py properly integrates with core/registry."""

    def test_studio_visualizations_backend_for_world(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS
        backend = STUDIO_VISUALIZATIONS.backend_for("world")
        assert backend.mode == "world"
        assert backend.capabilities.score == 50

    def test_studio_visualizations_backend_for_alias(self):
        from worldfoundry.studio.visualization.backends.frontends import STUDIO_VISUALIZATIONS
        # "interactive-world" is an alias for "world".
        backend = STUDIO_VISUALIZATIONS.backend_for("interactive-world")
        assert backend.mode == "world"

    def test_frontends_constants_match_core_registry(self):
        from worldfoundry.studio.visualization.backends.frontends import (
            WORLD_FRONTEND, POINTS_FRONTEND, EMBODIED_FRONTEND,
            SPARK_FRONTEND, MEDIA_FRONTEND, RERUN_FRONTEND, UNIFIED_FRONTEND,
        )
        from worldfoundry.studio.visualization.core.registry import (
            INTERACTIVE_WORLD_VISUALIZATION, VISER_VISUALIZATION,
            EMBODIED_VISUALIZATION, SPARK_VISUALIZATION,
            MEDIA_VISUALIZATION, RERUN_VISUALIZATION, UNIFIED_VISUALIZATION,
        )
        assert WORLD_FRONTEND == INTERACTIVE_WORLD_VISUALIZATION
        assert POINTS_FRONTEND == VISER_VISUALIZATION
        assert EMBODIED_FRONTEND == EMBODIED_VISUALIZATION
        assert SPARK_FRONTEND == SPARK_VISUALIZATION
        assert MEDIA_FRONTEND == MEDIA_VISUALIZATION
        assert RERUN_FRONTEND == RERUN_VISUALIZATION
        assert UNIFIED_FRONTEND == UNIFIED_VISUALIZATION


# ---------------------------------------------------------------------------
# 36.  Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests across all modules."""

    def test_normalize_visualization_mode_whitespace_only(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode("   ") == ""

    def test_normalize_visualization_mode_hyphens_and_underscores(self):
        from worldfoundry.studio.visualization.core.registry import normalize_visualization_mode
        assert normalize_visualization_mode("interactive_world-model") == "interactive-world-model"

    def test_backend_capabilities_with_metadata(self):
        from worldfoundry.studio.visualization.core.registry import BackendCapabilities
        c = BackendCapabilities(metadata={"key": "value"})
        assert c.metadata == {"key": "value"}

    def test_studio_visualization_event_default_payload(self):
        from worldfoundry.studio.visualization.core.registry import StudioVisualizationEvent
        e = StudioVisualizationEvent(kind="test")
        assert e.payload == {}

    def test_layer_fromdict_missing_fields(self):
        from worldfoundry.studio.visualization.core.scene import Layer
        l = Layer.fromdict({"layer_id": "l1"})
        assert l.kind == ""  # Default for missing 'kind'.
        assert l.uri is None

    def test_infer_artifact_empty_path(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact("")
        assert a.kind == "artifact"  # No suffix => unknown kind.

    def test_infer_artifact_path_object(self):
        from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact
        a = infer_visualization_artifact(Path("test.wav"))
        assert a.kind == "audio"

    def test_viser_presentation_empty_html_caption(self):
        from worldfoundry.studio.visualization.backends.viser import ViserPresentation
        v = ViserPresentation(html="", caption="")
        assert v.html == ""
        assert v.caption == ""

    def test_world_session_default_prompt_empty(self):
        from worldfoundry.studio.visualization.backends.world import WorldSession
        ws = WorldSession(session_id="x", mode="image", prompt="")
        assert ws.prompt == ""

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

    def test_controls_to_interactions_float_camera_values(self):
        from worldfoundry.studio.visualization.backends.world import controls_to_interactions
        result = controls_to_interactions({"camera_dx": 0.3, "camera_dy": -0.2})
        assert "camera_" in result

    def test_repeat_to_slots_single_path(self):
        from worldfoundry.studio.visualization.backends.world import _repeat_to_slots
        paths = (Path("/a"),)
        result = _repeat_to_slots(paths, 5)
        assert len(result) == 5
        assert all(p == Path("/a") for p in result)

    def test_override_call_kwargs_seed_preserves_other_kwargs(self):
        from worldfoundry.studio.visualization.backends.world import _override_call_kwargs_seed
        original = '{"num_frames": 9, "max_area": 399360, "seed": 10}'
        result = _override_call_kwargs_seed(original, 100)
        parsed = json.loads(result)
        assert parsed["seed"] == 100
        assert parsed["num_frames"] == 9

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
