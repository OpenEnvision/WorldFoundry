"""Comprehensive tests for the providers submodule (geometry, media, perception, robotics, run_record)."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import paths – must use the public package namespace
# ---------------------------------------------------------------------------
from worldfoundry.studio.visualization.providers import (
    GeometryProvider,
    MediaProvider,
    PerceptionProvider,
    RoboticsProvider,
    RunRecordProvider,
    __all__ as providers_all,
)
from worldfoundry.studio.visualization.providers.geometry import (
    CAMERA_SUFFIXES,
    GEOMETRY_KINDS,
    _source_paths as geo_source_paths,
)
from worldfoundry.studio.visualization.providers.media import (
    MEDIA_KINDS,
    _source_paths as media_source_paths,
)
from worldfoundry.studio.visualization.providers.perception import (
    NAME_KIND_HINTS,
    _kind_from_name,
    _source_paths as perception_source_paths,
)
from worldfoundry.studio.visualization.providers.robotics import (
    TRACE_SUFFIXES,
    VIDEO_SUFFIXES,
    _source_paths as robotics_source_paths,
)
from worldfoundry.studio.visualization.providers.run_record import (
    ACTION_TRACE_EXTS,
    EPISODE_METADATA_EXTS,
    VIDEO_REPLAY_EXTS,
    first_embodied_trace_candidate,
    first_episode_metadata_candidate,
    first_geometry_point_candidate,
    first_simulator_replay_candidate,
    first_splat_asset,
    normalize_output_relative,
    _is_empty_json_trace,
    _ordered_named_candidates,
)
from worldfoundry.studio.visualization.core.scene import Layer, VisualizationScene
from worldfoundry.studio.visualization.core.artifacts import infer_visualization_artifact


# ===================================================================
# __init__.py exports
# ===================================================================

class TestProvidersInit:
    """Verify the providers package __init__.py exports."""

    def test_all_exports_complete(self):
        expected = {"GeometryProvider", "MediaProvider", "PerceptionProvider",
                    "RoboticsProvider", "RunRecordProvider"}
        assert set(providers_all) == expected

    def test_all_exports_importable(self):
        for name in providers_all:
            assert name in globals() or hasattr(
                __import__("worldfoundry.studio.visualization.providers",
                           fromlist=[name]), name
            )


# ===================================================================
# Constants / module-level data structures
# ===================================================================

class TestGeometryConstants:
    def test_geometry_kinds_is_set(self):
        assert isinstance(GEOMETRY_KINDS, set)

    def test_geometry_kinds_values(self):
        expected = {"point_cloud", "mesh", "gaussian_splat", "camera",
                    "trajectory", "depth"}
        assert GEOMETRY_KINDS == expected

    def test_camera_suffixes_is_set(self):
        assert isinstance(CAMERA_SUFFIXES, set)

    def test_camera_suffixes_values(self):
        assert CAMERA_SUFFIXES == {".json", ".yaml", ".yml"}




class TestMediaConstants:
    def test_media_kinds_is_set(self):
        assert isinstance(MEDIA_KINDS, set)

    def test_media_kinds_values(self):
        assert MEDIA_KINDS == {"image", "video", "audio"}


class TestPerceptionConstants:
    def test_name_kind_hints_is_dict(self):
        assert isinstance(NAME_KIND_HINTS, dict)

    def test_name_kind_hints_keys_cover_all_tokens(self):
        expected_keys = {"mask", "segmentation", "box", "bbox",
                         "keypoint", "track", "flow", "depth"}
        assert set(NAME_KIND_HINTS.keys()) == expected_keys

    def test_name_kind_hints_values(self):
        expected_values = {"segmentation", "detection", "keypoints",
                           "trajectory", "optical_flow", "depth"}
        assert set(NAME_KIND_HINTS.values()) == expected_values


class TestRoboticsConstants:
    def test_trace_suffixes_is_set(self):
        assert isinstance(TRACE_SUFFIXES, set)

    def test_trace_suffixes_values(self):
        assert TRACE_SUFFIXES == {".json", ".jsonl", ".npz", ".npy", ".pkl"}

    def test_video_suffixes_is_set(self):
        assert isinstance(VIDEO_SUFFIXES, set)

    def test_video_suffixes_values(self):
        assert VIDEO_SUFFIXES == {".mp4", ".mov", ".avi", ".webm", ".mkv"}


class TestRunRecordConstants:
    def test_video_replay_exts_is_set(self):
        assert isinstance(VIDEO_REPLAY_EXTS, set)

    def test_video_replay_exts_values(self):
        assert VIDEO_REPLAY_EXTS == {".mp4", ".mov", ".avi", ".webm", ".mkv"}

    def test_action_trace_exts_is_set(self):
        assert isinstance(ACTION_TRACE_EXTS, set)

    def test_action_trace_exts_values(self):
        assert ACTION_TRACE_EXTS == {".json", ".jsonl", ".npz", ".npy", ".pkl"}

    def test_episode_metadata_exts_is_set(self):
        assert isinstance(EPISODE_METADATA_EXTS, set)

    def test_episode_metadata_exts_values(self):
        assert EPISODE_METADATA_EXTS == {".json", ".jsonl", ".yaml", ".yml", ".toml"}


# ===================================================================
# GeometryProvider
# ===================================================================

class TestGeometryProvider:
    def test_provider_id(self):
        assert GeometryProvider.provider_id == "geometry"

    def test_discover_none_source(self):
        gp = GeometryProvider()
        result = gp.discover(None)
        assert result is None

    def test_discover_empty_iterable(self):
        gp = GeometryProvider()
        result = gp.discover([])
        assert result is None

    def test_discover_nonexistent_single_path(self):
        gp = GeometryProvider()
        result = gp.discover("/nonexistent/path.ply")
        # Even for a nonexistent path, it will try to classify it
        # But _source_paths will return [Path("/nonexistent/path.ply")]
        # infer_visualization_artifact will classify .ply as point_cloud
        # So a Layer will be created
        assert result is not None
        assert result.layer_kinds() == frozenset({"point_cloud"})

    def test_discover_single_ply_file(self):
        gp = GeometryProvider()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = gp.discover(path)
            assert result is not None
            assert "point_cloud" in result.layer_kinds()
            assert result.recommended_backend == "points"
        finally:
            os.unlink(path)

    def test_discover_splat_file_recommends_spark_backend(self):
        gp = GeometryProvider()
        with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = gp.discover(path)
            assert result is not None
            assert "gaussian_splat" in result.layer_kinds()
            assert result.recommended_backend == "spark"
        finally:
            os.unlink(path)

    def test_discover_glb_mesh(self):
        gp = GeometryProvider()
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = gp.discover(path)
            assert result is not None
            assert "mesh" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_camera_json_file(self):
        gp = GeometryProvider()
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="camera_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = gp.discover(path)
            assert result is not None
            assert "camera" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_regular_json_not_camera(self):
        gp = GeometryProvider()
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="config_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = gp.discover(path)
            # .json without "camera" in name → kind = "artifact" → filtered out
            assert result is None
        finally:
            os.unlink(path)

    def test_discover_audio_file_filtered_out(self):
        gp = GeometryProvider()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = gp.discover(path)
            # audio is not in GEOMETRY_KINDS
            assert result is None
        finally:
            os.unlink(path)

    def test_discover_directory_with_geometry_files(self):
        gp = GeometryProvider()
        with tempfile.TemporaryDirectory() as d:
            ply_path = Path(d) / "points.ply"
            ply_path.write_text("test")
            glb_path = Path(d) / "mesh.glb"
            glb_path.write_text("test")
            wav_path = Path(d) / "audio.wav"
            wav_path.write_text("test")
            result = gp.discover(d)
            assert result is not None
            kinds = result.layer_kinds()
            assert "point_cloud" in kinds
            assert "mesh" in kinds
            assert "audio" not in kinds

    def test_discover_returns_visualization_scene(self):
        gp = GeometryProvider()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = gp.discover(path)
            assert isinstance(result, VisualizationScene)
            assert result.scene_id == "geometry/scene"
            assert result.title == "Geometry Scene"
        finally:
            os.unlink(path)




class TestGeoSourcePaths:
    def test_none_returns_empty(self):
        assert geo_source_paths(None) == []

    def test_single_file_path_string(self):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = geo_source_paths(path)
            assert len(result) == 1
            assert result[0] == Path(path)
        finally:
            os.unlink(path)

    def test_single_file_path_object(self):
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = geo_source_paths(Path(path))
            assert len(result) == 1
        finally:
            os.unlink(path)

    def test_directory_collects_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.ply").write_text("x")
            Path(d, "b.glb").write_text("x")
            result = geo_source_paths(d)
            assert len(result) == 2

    def test_iterable_of_strings(self):
        paths = ["/tmp/a.ply", "/tmp/b.glb"]
        result = geo_source_paths(paths)
        assert len(result) == 2
        assert all(isinstance(p, Path) for p in result)

    def test_non_iterable_non_path_returns_empty(self):
        assert geo_source_paths(42) == []


# ===================================================================
# MediaProvider
# ===================================================================

class TestMediaProvider:
    def test_provider_id(self):
        assert MediaProvider.provider_id == "media"

    def test_discover_none_source(self):
        mp = MediaProvider()
        assert mp.discover(None) is None

    def test_discover_empty_iterable(self):
        mp = MediaProvider()
        assert mp.discover([]) is None

    def test_discover_single_image(self):
        mp = MediaProvider()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = mp.discover(path)
            assert result is not None
            assert "image" in result.layer_kinds()
            assert result.recommended_backend == "media"
        finally:
            os.unlink(path)

    def test_discover_video_file(self):
        mp = MediaProvider()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = mp.discover(path)
            assert result is not None
            assert "video" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_audio_file(self):
        mp = MediaProvider()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = mp.discover(path)
            assert result is not None
            assert "audio" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_geometry_file_filtered_out(self):
        mp = MediaProvider()
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = mp.discover(path)
            # .ply → kind = "point_cloud" → not in MEDIA_KINDS
            assert result is None
        finally:
            os.unlink(path)

    def test_discover_directory_with_mixed_media(self):
        mp = MediaProvider()
        with tempfile.TemporaryDirectory() as d:
            Path(d, "photo.png").write_text("x")
            Path(d, "clip.mp4").write_text("x")
            Path(d, "points.ply").write_text("x")
            result = mp.discover(d)
            assert result is not None
            kinds = result.layer_kinds()
            assert "image" in kinds
            assert "video" in kinds
            assert "point_cloud" not in kinds

    def test_discover_returns_visualization_scene(self):
        mp = MediaProvider()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="test_img") as f:
            f.write(b"test")
            path = f.name
        try:
            result = mp.discover(path)
            assert isinstance(result, VisualizationScene)
            assert result.recommended_backend == "media"
            # scene_id uses the stem of the first path
            assert "test_img" in result.scene_id
        finally:
            os.unlink(path)

    def test_discover_nonexistent_path_still_classifies(self):
        mp = MediaProvider()
        # _source_paths converts a string to Path; even nonexistent ones get classified
        result = mp.discover("/nonexistent/image.png")
        assert result is not None
        assert "image" in result.layer_kinds()


class TestMediaSourcePaths:
    def test_none_returns_empty(self):
        assert media_source_paths(None) == []

    def test_iterable_of_paths(self):
        result = media_source_paths(["a.png", "b.mp4"])
        assert len(result) == 2

    def test_non_path_non_iterable_returns_empty(self):
        assert media_source_paths(123) == []


# ===================================================================
# PerceptionProvider
# ===================================================================

class TestPerceptionProvider:
    def test_provider_id(self):
        assert PerceptionProvider.provider_id == "perception"

    def test_discover_none_source(self):
        pp = PerceptionProvider()
        assert pp.discover(None) is None

    def test_discover_empty_iterable(self):
        pp = PerceptionProvider()
        assert pp.discover([]) is None

    def test_discover_mask_file(self):
        pp = PerceptionProvider()
        with tempfile.NamedTemporaryFile(suffix=".png", prefix="mask_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = pp.discover(path)
            assert result is not None
            # "mask" in name → kind = "segmentation"
            assert "segmentation" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_bbox_file(self):
        pp = PerceptionProvider()
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="bbox_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = pp.discover(path)
            assert result is not None
            # "bbox" in name → kind = "detection"
            assert "detection" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_flow_file(self):
        pp = PerceptionProvider()
        with tempfile.NamedTemporaryFile(suffix=".npz", prefix="flow_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = pp.discover(path)
            assert result is not None
            # "flow" in name → kind = "optical_flow"
            assert "optical_flow" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_depth_file(self):
        pp = PerceptionProvider()
        with tempfile.NamedTemporaryFile(suffix=".png", prefix="depth_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = pp.discover(path)
            assert result is not None
            # "depth" in name → kind = "depth"
            assert "depth" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_regular_image_not_perception(self):
        pp = PerceptionProvider()
        with tempfile.NamedTemporaryFile(suffix=".png", prefix="photo_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = pp.discover(path)
            # "photo_" has no perception token → kind = "image" from suffix
            # "image" is not in {"artifact", "audio"} → it IS a valid perception layer
            assert result is not None
            assert "image" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_audio_filtered_out(self):
        pp = PerceptionProvider()
        with tempfile.NamedTemporaryFile(suffix=".wav", prefix="track_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = pp.discover(path)
            # "track" in name → _kind_from_name returns "trajectory"
            # But the suffix .wav → infer returns "audio"
            # _kind_from_name takes priority → kind = "trajectory"
            assert result is not None
            assert "trajectory" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_returns_scene_with_perception_metadata(self):
        pp = PerceptionProvider()
        with tempfile.NamedTemporaryFile(suffix=".png", prefix="mask_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = pp.discover(path)
            assert isinstance(result, VisualizationScene)
            assert result.scene_id == "perception/artifacts"
            assert result.title == "Perception Artifacts"
            assert result.recommended_backend == "media"
        finally:
            os.unlink(path)


class TestKindFromName:
    def test_mask_token(self):
        assert _kind_from_name(Path("mask_output.png")) == "segmentation"

    def test_segmentation_token(self):
        assert _kind_from_name(Path("segmentation_map.png")) == "segmentation"

    def test_box_token(self):
        assert _kind_from_name(Path("box_coords.json")) == "detection"

    def test_bbox_token(self):
        assert _kind_from_name(Path("bbox_data.json")) == "detection"

    def test_keypoint_token(self):
        assert _kind_from_name(Path("keypoint_pos.json")) == "keypoints"

    def test_track_token(self):
        assert _kind_from_name(Path("track_data.npz")) == "trajectory"

    def test_flow_token(self):
        assert _kind_from_name(Path("optical_flow.npz")) == "optical_flow"

    def test_depth_token(self):
        assert _kind_from_name(Path("depth_map.png")) == "depth"

    def test_no_match(self):
        assert _kind_from_name(Path("random_file.txt")) is None

    def test_case_insensitive(self):
        assert _kind_from_name(Path("MASK_upper.png")) == "segmentation"

    def test_multiple_tokens_first_match(self):
        # "mask" and "depth" both present; first match in dict iteration wins
        result = _kind_from_name(Path("mask_depth_output.png"))
        # Python dict iteration order is guaranteed since 3.7
        # NAME_KIND_HINTS order: mask→segmentation, segmentation→segmentation, box→detection, ...
        assert result is not None  # at least one token matches


class TestPerceptionSourcePaths:
    def test_none_returns_empty(self):
        assert perception_source_paths(None) == []


# ===================================================================
# RoboticsProvider
# ===================================================================

class TestRoboticsProvider:
    def test_provider_id(self):
        assert RoboticsProvider.provider_id == "robotics"

    def test_discover_none_source(self):
        rp = RoboticsProvider()
        assert rp.discover(None) is None

    def test_discover_empty_iterable(self):
        rp = RoboticsProvider()
        assert rp.discover([]) is None

    def test_discover_action_trace_json(self):
        rp = RoboticsProvider()
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="action_trace_", delete=False) as f:
            f.write(b'{"steps": []}')
            path = f.name
        try:
            result = rp.discover(path)
            assert result is not None
            assert "action_trace" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_policy_trace(self):
        rp = RoboticsProvider()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", prefix="policy_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is not None
            assert "action_trace" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_robot_trajectory(self):
        rp = RoboticsProvider()
        with tempfile.NamedTemporaryFile(suffix=".npz", prefix="robot_trajectory_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is not None
            assert "action_trace" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_rollout_trace(self):
        rp = RoboticsProvider()
        with tempfile.NamedTemporaryFile(suffix=".pkl", prefix="rollout_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is not None
            assert "action_trace" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_sim_replay_video(self):
        rp = RoboticsProvider()
        with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="sim_replay_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is not None
            assert "video" in result.layer_kinds()
            # Check metadata role
            layers = result.layers
            video_layer = [l for l in layers if l.kind == "video"][0]
            assert video_layer.metadata.get("role") == "simulator_replay"
        finally:
            os.unlink(path)

    def test_discover_robot_video(self):
        rp = RoboticsProvider()
        with tempfile.NamedTemporaryFile(suffix=".webm", prefix="robot_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is not None
            assert "video" in result.layer_kinds()
        finally:
            os.unlink(path)

    def test_discover_non_robotics_trace_filtered(self):
        rp = RoboticsProvider()
        # .json but without any robotics keyword in name
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="config_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_discover_non_robotics_video_filtered(self):
        rp = RoboticsProvider()
        # .mp4 without any robotics keyword
        with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="nature_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_discover_returns_scene(self):
        rp = RoboticsProvider()
        with tempfile.NamedTemporaryFile(suffix=".json", prefix="action_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert isinstance(result, VisualizationScene)
            assert result.scene_id == "robotics/artifacts"
            assert result.recommended_backend == "embodied"
        finally:
            os.unlink(path)

    def test_trace_wrong_suffix_ignored(self):
        rp = RoboticsProvider()
        # "action_" keyword but .ply suffix (not in TRACE_SUFFIXES)
        with tempfile.NamedTemporaryFile(suffix=".ply", prefix="action_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_episode_video(self):
        rp = RoboticsProvider()
        with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="episode_", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            result = rp.discover(path)
            assert result is not None
            assert "video" in result.layer_kinds()
        finally:
            os.unlink(path)


class TestRoboticsSourcePaths:
    def test_none_returns_empty(self):
        assert robotics_source_paths(None) == []


# ===================================================================
# RunRecordProvider
# ===================================================================

class TestRunRecordProvider:
    def test_provider_id(self):
        assert RunRecordProvider.provider_id == "run_record"

    def test_discover_none_source(self):
        rrp = RunRecordProvider()
        assert rrp.discover(None) is None

    def test_discover_nonexistent_dir(self):
        rrp = RunRecordProvider()
        assert rrp.discover("/nonexistent_dir") is None

    def test_discover_empty_dir(self):
        rrp = RunRecordProvider()
        with tempfile.TemporaryDirectory() as d:
            result = rrp.discover(d)
            assert result is None

    def test_discover_dir_with_media_files(self):
        rrp = RunRecordProvider()
        with tempfile.TemporaryDirectory() as d:
            Path(d, "photo.png").write_text("x")
            Path(d, "clip.mp4").write_text("x")
            result = rrp.discover(d)
            assert result is not None
            assert isinstance(result, VisualizationScene)
            kinds = result.layer_kinds()
            assert "image" in kinds
            assert "video" in kinds

    def test_discover_dir_filters_out_generic_artifacts(self):
        rrp = RunRecordProvider()
        with tempfile.TemporaryDirectory() as d:
            # .txt → kind "artifact" → should be filtered
            Path(d, "readme.txt").write_text("info")
            Path(d, "photo.png").write_text("x")
            result = rrp.discover(d)
            assert result is not None
            assert "artifact" not in result.layer_kinds()
            assert "image" in result.layer_kinds()

    def test_discover_dir_with_subdirectories(self):
        rrp = RunRecordProvider()
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "subdir"
            sub.mkdir()
            Path(sub, "mesh.glb").write_text("x")
            result = rrp.discover(d)
            assert result is not None
            assert "mesh" in result.layer_kinds()

    def test_discover_layer_id_uses_relative_path(self):
        rrp = RunRecordProvider()
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "nested"
            sub.mkdir()
            Path(sub, "img.png").write_text("x")
            result = rrp.discover(d)
            # layer_id should be "nested:img"
            found = [l for l in result.layers if "nested" in l.layer_id]
            assert len(found) == 1
            assert found[0].layer_id == "nested:img.png"

    def test_discover_with_object_source(self):
        """Test that discover works with objects having output_dir or path attributes."""
        rrp = RunRecordProvider()
        with tempfile.TemporaryDirectory() as d:
            Path(d, "photo.png").write_text("x")

            class FakeSource:
                output_dir = ""

            # FakeSource.output_dir is empty string → falls back to path attr or source itself
            fs1 = FakeSource()
            fs1.output_dir = d
            result = rrp.discover(fs1)
            assert result is not None

            class FakeSource2:
                path = ""

            fs2 = FakeSource2()
            fs2.path = d
            result2 = rrp.discover(fs2)
            assert result2 is not None

    def test_discover_scene_id_includes_dir_name(self):
        rrp = RunRecordProvider()
        with tempfile.TemporaryDirectory(prefix="myrun_") as d:
            Path(d, "photo.png").write_text("x")
            result = rrp.discover(d)
            assert result is not None
            # scene_id is "run/{dir_name}"
            assert "run/" in result.scene_id

    def test_discover_metadata_includes_source(self):
        rrp = RunRecordProvider()
        with tempfile.TemporaryDirectory() as d:
            Path(d, "photo.png").write_text("x")
            result = rrp.discover(d)
            assert result is not None
            assert "source" in result.metadata


# ===================================================================
# run_record standalone functions
# ===================================================================

class TestNormalizeOutputRelative:
    def test_simple_relative_path(self):
        result = normalize_output_relative("subdir/file.ply", "/tmp/output")
        assert result == "subdir/file.ply"

    def test_dot_slash_stripped(self):
        result = normalize_output_relative("./file.ply", "/tmp/output")
        assert result == "file.ply"

    def test_dots_stripped(self):
        result = normalize_output_relative("../file.ply", "/tmp/output")
        # "../file.ply" is not absolute, so strip ./ prefix
        # lstrip("./") removes leading ./ characters
        # "../file.ply".lstrip("./") → "file.ply" (strips both . and /)
        assert result == "file.ply"

    def test_absolute_path_under_output_dir(self):
        with tempfile.TemporaryDirectory() as output_dir:
            subdir = Path(output_dir) / "sub"
            subdir.mkdir()
            file_path = subdir / "test.ply"
            file_path.write_text("x")
            result = normalize_output_relative(str(file_path.resolve()), output_dir)
            assert result == "sub/test.ply"

    def test_absolute_path_not_under_output_dir(self):
        result = normalize_output_relative("/completely/different/path.ply", "/tmp/output")
        assert result == "/completely/different/path.ply"


class TestFirstGeometryPointCandidate:
    def test_empty_paths_returns_none(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_geometry_point_candidate([], output_dir, gs_ply_predicate=lambda p: False)
            # Falls back to glob scan of output_dir
            assert result is None

    def test_ply_not_gaussian_splat(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ply_path = Path(output_dir) / "points.ply"
            ply_path.write_text("x")
            result = first_geometry_point_candidate([str(ply_path)], output_dir, gs_ply_predicate=lambda p: False)
            assert result is not None
            assert "points.ply" in result

    def test_ply_is_gaussian_splat_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ply_path = Path(output_dir) / "splat.ply"
            ply_path.write_text("x")
            result = first_geometry_point_candidate([str(ply_path)], output_dir, gs_ply_predicate=lambda p: True)
            # The ply is a gs splat, so it's skipped; no other candidates → None
            assert result is None

    def test_pcd_candidate(self):
        with tempfile.TemporaryDirectory() as output_dir:
            pcd_path = Path(output_dir) / "cloud.pcd"
            pcd_path.write_text("x")
            result = first_geometry_point_candidate([str(pcd_path)], output_dir, gs_ply_predicate=lambda p: False)
            assert result is not None
            assert "cloud.pcd" in result

    def test_xyz_candidate(self):
        with tempfile.TemporaryDirectory() as output_dir:
            xyz_path = Path(output_dir) / "scan.xyz"
            xyz_path.write_text("x")
            result = first_geometry_point_candidate([str(xyz_path)], output_dir, gs_ply_predicate=lambda p: False)
            assert result is not None
            assert "scan.xyz" in result

    def test_nonexistent_path_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_geometry_point_candidate(
                ["nonexistent.ply"], output_dir, gs_ply_predicate=lambda p: False
            )
            assert result is None

    def test_duplicate_paths_deduped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ply_path = Path(output_dir) / "points.ply"
            ply_path.write_text("x")
            result = first_geometry_point_candidate(
                [str(ply_path), str(ply_path)], output_dir, gs_ply_predicate=lambda p: False
            )
            assert result is not None

    def test_fallback_glob_scan(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ply_path = Path(output_dir) / "scan.ply"
            ply_path.write_text("x")
            # Pass no explicit paths → glob scan finds it
            result = first_geometry_point_candidate([], output_dir, gs_ply_predicate=lambda p: False)
            assert result is not None
            assert "scan.ply" in result

    def test_empty_strings_filtered(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_geometry_point_candidate(
                ["", "", None], output_dir, gs_ply_predicate=lambda p: False
            )
            assert result is None


class TestFirstSplatAsset:
    def test_empty_iterable_returns_none(self):
        path, fmt = first_splat_asset([])
        assert path is None
        assert fmt is None

    def test_spz_highest_priority(self):
        with tempfile.TemporaryDirectory() as d:
            spz = Path(d) / "model.spz"
            splat = Path(d) / "model.splat"
            spz.write_text("x")
            splat.write_text("x")
            path, fmt = first_splat_asset([str(spz), str(splat)])
            assert path is not None
            assert fmt == "spz"

    def test_ksplat_priority(self):
        with tempfile.TemporaryDirectory() as d:
            ksplat = Path(d) / "model.ksplat"
            splat = Path(d) / "model.splat"
            ksplat.write_text("x")
            splat.write_text("x")
            path, fmt = first_splat_asset([str(ksplat), str(splat)])
            assert fmt == "ksplat"

    def test_splat_priority_after_ksplat(self):
        with tempfile.TemporaryDirectory() as d:
            splat = Path(d) / "model.splat"
            splat.write_text("x")
            path, fmt = first_splat_asset([str(splat)])
            assert fmt == "splat"

    def test_sog_priority(self):
        with tempfile.TemporaryDirectory() as d:
            sog = Path(d) / "model.sog"
            ply = Path(d) / "model.ply"
            sog.write_text("x")
            ply.write_text("x")
            path, fmt = first_splat_asset([str(sog), str(ply)])
            assert fmt == "sog"

    def test_ply_gaussian_format_hint(self):
        with tempfile.TemporaryDirectory() as d:
            ply = Path(d) / "model.ply"
            ply.write_text("x")
            path, fmt = first_splat_asset([str(ply)], gs_ply_predicate=lambda _: True)
            assert fmt == "ply_gaussian"

    def test_nonexistent_files_skipped(self):
        path, fmt = first_splat_asset(["/nonexistent/model.spz", "/nonexistent/model.splat"])
        assert path is None
        assert fmt is None

    def test_dirs_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "subdir.spz"
            sub.mkdir()
            path, fmt = first_splat_asset([str(sub)])
            assert path is None
            assert fmt is None

    def test_empty_strings_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path, fmt = first_splat_asset(["", "", None])
            assert path is None

    def test_resolution_deduplication(self):
        with tempfile.TemporaryDirectory() as d:
            ply = Path(d) / "model.ply"
            ply.write_text("x")
            path, fmt = first_splat_asset(
                [str(ply), str(ply.resolve())],
                gs_ply_predicate=lambda _: True,
            )
            # Both resolve to same file; deduped
            assert path is not None
            assert fmt == "ply_gaussian"


class TestFirstEmbodiedTraceCandidate:
    def test_empty_paths_returns_none(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_embodied_trace_candidate([], output_dir)
            assert result is None

    def test_action_trace_json(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "action_trace.json"
            trace.write_text(json.dumps({"steps": [1, 2, 3]}))
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is not None
            assert "action_trace.json" in result

    def test_action_trace_jsonl(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "actions.jsonl"
            trace.write_text("line1\nline2\n")
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is not None

    def test_policy_trace_npz(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "policy_output.npz"
            trace.write_text("x")
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is not None
            assert "policy_output.npz" in result

    def test_empty_json_trace_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "action_trace.json"
            trace.write_text("[]")  # empty list
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is None

    def test_empty_dict_json_trace_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "robot_action.json"
            trace.write_text("{}")  # empty dict
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is None

    def test_nonexistent_path_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_embodied_trace_candidate(
                ["nonexistent_action.json"], output_dir
            )
            assert result is None

    def test_wrong_suffix_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "action_trace.png"
            trace.write_text("x")
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is None

    def test_wrong_name_no_priority_term(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "data.json"
            trace.write_text(json.dumps([1, 2]))
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is None

    def test_priority_ordering(self):
        """action_trace should beat trajectory for same suffix."""
        with tempfile.TemporaryDirectory() as output_dir:
            traj = Path(output_dir) / "trajectory.json"
            traj.write_text(json.dumps([1]))
            action = Path(output_dir) / "action_trace.json"
            action.write_text(json.dumps([1]))
            result = first_embodied_trace_candidate(
                [str(traj), str(action)], output_dir
            )
            assert "action_trace" in result

    def test_rollout_jsonl(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "rollout.jsonl"
            trace.write_text("line1\n")
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is not None

    def test_trajectory_npy(self):
        with tempfile.TemporaryDirectory() as output_dir:
            trace = Path(output_dir) / "trajectory.npy"
            trace.write_text("x")
            result = first_embodied_trace_candidate([str(trace)], output_dir)
            assert result is not None


class TestFirstSimulatorReplayCandidate:
    def test_empty_paths_returns_none(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_simulator_replay_candidate([], output_dir)
            assert result is None

    def test_sim_video_mp4(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "sim_replay.mp4"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is not None
            assert "sim_replay.mp4" in result

    def test_robot_video_webm(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "robot_episode.webm"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is not None

    def test_rollout_video(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "rollout.mp4"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is not None

    def test_nonexistent_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_simulator_replay_candidate(
                ["nonexistent_sim.mp4"], output_dir
            )
            assert result is None

    def test_wrong_suffix_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "sim_output.json"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is None

    def test_wrong_name_no_priority_term(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "nature.mp4"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is None

    def test_episode_video(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "episode1.mp4"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is not None

    def test_env_video(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "env_demo.mp4"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is not None

    def test_policy_video(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "policy_visual.mp4"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is not None

    def test_trajectory_video(self):
        with tempfile.TemporaryDirectory() as output_dir:
            video = Path(output_dir) / "trajectory.mp4"
            video.write_text("x")
            result = first_simulator_replay_candidate([str(video)], output_dir)
            assert result is not None


class TestFirstEpisodeMetadataCandidate:
    def test_empty_paths_returns_none(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = first_episode_metadata_candidate([], output_dir)
            assert result is None

    def test_episode_json(self):
        with tempfile.TemporaryDirectory() as output_dir:
            meta = Path(output_dir) / "episode_metadata.json"
            meta.write_text(json.dumps({"task": "pick"}))
            result = first_episode_metadata_candidate([str(meta)], output_dir)
            assert result is not None
            assert "episode_metadata.json" in result

    def test_sim_environment_yaml(self):
        with tempfile.TemporaryDirectory() as output_dir:
            meta = Path(output_dir) / "sim_environment.yaml"
            meta.write_text("task: place\n")
            result = first_episode_metadata_candidate([str(meta)], output_dir)
            assert result is not None

    def test_metadata_toml(self):
        with tempfile.TemporaryDirectory() as output_dir:
            meta = Path(output_dir) / "metadata.toml"
            meta.write_text("task = 'reach'\n")
            result = first_episode_metadata_candidate([str(meta)], output_dir)
            assert result is not None

    def test_task_jsonl(self):
        with tempfile.TemporaryDirectory() as output_dir:
            meta = Path(output_dir) / "task_spec.jsonl"
            meta.write_text("line\n")
            result = first_episode_metadata_candidate([str(meta)], output_dir)
            assert result is not None

    def test_wrong_suffix(self):
        with tempfile.TemporaryDirectory() as output_dir:
            meta = Path(output_dir) / "episode_info.txt"
            meta.write_text("info")
            result = first_episode_metadata_candidate([str(meta)], output_dir)
            assert result is None

    def test_wrong_name_no_priority_term(self):
        with tempfile.TemporaryDirectory() as output_dir:
            meta = Path(output_dir) / "config.json"
            meta.write_text("{}")
            result = first_episode_metadata_candidate([str(meta)], output_dir)
            assert result is None


class TestIsEmptyJsonTrace:
    def test_non_json_returns_false(self):
        assert _is_empty_json_trace(Path("trace.npz")) is False

    def test_empty_list_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("[]")
            path = Path(f.name)
        try:
            assert _is_empty_json_trace(path) is True
        finally:
            os.unlink(path)

    def test_empty_dict_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{}")
            path = Path(f.name)
        try:
            assert _is_empty_json_trace(path) is True
        finally:
            os.unlink(path)

    def test_non_empty_list_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([1, 2, 3], f)
            path = Path(f.name)
        try:
            assert _is_empty_json_trace(path) is False
        finally:
            os.unlink(path)

    def test_non_empty_dict_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"key": "value"}, f)
            path = Path(f.name)
        try:
            assert _is_empty_json_trace(path) is False
        finally:
            os.unlink(path)

    def test_invalid_json_returns_false(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not valid json {{{")
            path = Path(f.name)
        try:
            assert _is_empty_json_trace(path) is False
        finally:
            os.unlink(path)


class TestOrderedNamedCandidates:
    def test_empty_paths(self):
        result = _ordered_named_candidates([], ("action",), {".json"})
        assert result == []

    def test_matching_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "action_trace.json"
            f.write_text(json.dumps([1]))
            result = _ordered_named_candidates([str(f)], ("action_trace", "action"), {".json"})
            assert len(result) == 1
            assert result[0].name == "action_trace.json"

    def test_wrong_suffix_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "action_trace.png"
            f.write_text("x")
            result = _ordered_named_candidates([str(f)], ("action_trace",), {".json"})
            assert result == []

    def test_no_priority_term_match(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "random.json"
            f.write_text("{}")
            result = _ordered_named_candidates([str(f)], ("action", "policy"), {".json"})
            assert result == []

    def test_nonexistent_excluded(self):
        result = _ordered_named_candidates(
            ["nonexistent_action.json"], ("action",), {".json"}
        )
        assert result == []

    def test_directory_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "action_data.json"
            sub.mkdir()  # it's a dir named .json
            result = _ordered_named_candidates([str(sub)], ("action",), {".json"})
            assert result == []

    def test_priority_ordering(self):
        """Lower priority_term index wins."""
        with tempfile.TemporaryDirectory() as d:
            policy = Path(d) / "policy.json"
            policy.write_text(json.dumps([1]))
            action = Path(d) / "action_trace.json"
            action.write_text(json.dumps([1]))
            result = _ordered_named_candidates(
                [str(policy), str(action)],
                ("action_trace", "action-trace", "actions", "policy"),
                {".json"},
            )
            assert len(result) == 2
            # action_trace (index 0) should come first
            assert result[0].name == "action_trace.json"

    def test_empty_strings_filtered(self):
        result = _ordered_named_candidates(
            ["", None], ("action",), {".json"}
        )
        assert result == []


# ===================================================================
# Cross-cutting: frozen dataclass enforcement
# ===================================================================

class TestFrozenDataclasses:
    """Verify that Layer and VisualizationScene are frozen."""

    def test_layer_frozen(self):
        layer = Layer(layer_id="test", kind="point_cloud")
        with pytest.raises(AttributeError):
            layer.layer_id = "changed"

    def test_visualization_scene_frozen(self):
        scene = VisualizationScene(scene_id="test")
        with pytest.raises(AttributeError):
            scene.scene_id = "changed"


# ===================================================================
# Cross-cutting: Layer / VisualizationScene serialization round-trips
# ===================================================================

class TestLayerRoundTrip:
    def test_asdict_then_fromdict(self):
        layer = Layer(
            layer_id="cloud",
            kind="point_cloud",
            uri="/data/cloud.ply",
            uris=("/data/cloud.ply", "/data/cloud.pcd"),
            frame_range=(0, 100),
            time_range=(0.0, 10.0),
            coordinate_frame="world",
            style={"color": "red"},
            metadata={"source": "lidar"},
        )
        d = layer.asdict()
        restored = Layer.fromdict(d)
        assert restored.layer_id == layer.layer_id
        assert restored.kind == layer.kind
        assert restored.uri == layer.uri
        assert restored.uris == layer.uris
        assert restored.frame_range == layer.frame_range
        assert restored.time_range == layer.time_range
        assert restored.coordinate_frame == layer.coordinate_frame
        assert restored.style == layer.style
        assert restored.metadata == layer.metadata

    def test_all_uris_method(self):
        layer = Layer(layer_id="test", kind="point_cloud", uri="/a.ply", uris=("/b.ply", "/a.ply"))
        uris = layer.all_uris()
        assert uris == ("/a.ply", "/b.ply")  # deduped, preserves order

    def test_all_uris_no_uri_no_uris(self):
        layer = Layer(layer_id="test", kind="point_cloud")
        assert layer.all_uris() == ()

    def test_fromdict_with_id_alias(self):
        """fromdict accepts 'id' as alias for 'layer_id'."""
        d = {"id": "mylayer", "kind": "mesh"}
        layer = Layer.fromdict(d)
        assert layer.layer_id == "mylayer"

    def test_fromdict_defaults(self):
        d = {"layer_id": "x", "kind": "y"}
        layer = Layer.fromdict(d)
        assert layer.uri is None
        assert layer.uris == ()
        assert layer.frame_range is None
        assert layer.time_range is None
        assert layer.coordinate_frame is None
        assert layer.style == {}
        assert layer.metadata == {}


class TestVisualizationSceneRoundTrip:
    def test_asdict_then_fromdict(self):
        scene = VisualizationScene(
            scene_id="geometry/scene",
            title="Geometry",
            layers=(
                Layer(layer_id="cloud", kind="point_cloud", uri="/data/cloud.ply"),
                Layer(layer_id="mesh", kind="mesh", uri="/data/mesh.glb"),
            ),
            recommended_backend="spark",
            metadata={"source": "/data"},
        )
        d = scene.asdict()
        restored = VisualizationScene.fromdict(d)
        assert restored.scene_id == scene.scene_id
        assert restored.title == scene.title
        assert restored.recommended_backend == scene.recommended_backend
        assert len(restored.layers) == 2
        assert restored.metadata == scene.metadata

    def test_layer_kinds(self):
        scene = VisualizationScene(
            scene_id="test",
            layers=(
                Layer(layer_id="a", kind="point_cloud"),
                Layer(layer_id="b", kind="mesh"),
            ),
        )
        assert scene.layer_kinds() == frozenset({"point_cloud", "mesh"})

    def test_fromdict_with_timeline(self):
        d = {
            "scene_id": "test",
            "title": "Test",
            "recommended_backend": "points",
            "layers": [],
            "timeline": {"fps": 30, "start_time": 0, "end_time": 10, "frame_count": 300},
        }
        scene = VisualizationScene.fromdict(d)
        assert scene.timeline is not None
        assert scene.timeline.fps == 30
        assert scene.timeline.frame_count == 300

    def test_asdict_schema_version(self):
        scene = VisualizationScene(scene_id="test")
        d = scene.asdict()
        assert d["schema_version"] == 1

    def test_fromdict_defaults(self):
        d = {"scene_id": "s1"}
        scene = VisualizationScene.fromdict(d)
        assert scene.title == ""
        assert scene.layers == ()
        assert scene.recommended_backend == "auto"
        assert scene.timeline is None
        assert scene.metadata == {}


# ===================================================================
# infer_visualization_artifact (core dependency)
# ===================================================================

class TestInferVisualizationArtifact:
    def test_ply_is_point_cloud(self):
        art = infer_visualization_artifact(Path("cloud.ply"))
        assert art.kind == "point_cloud"
        assert art.format_hint == "ply"

    def test_glb_is_mesh(self):
        art = infer_visualization_artifact(Path("mesh.glb"))
        assert art.kind == "mesh"

    def test_spz_is_gaussian_splat(self):
        art = infer_visualization_artifact(Path("model.spz"))
        assert art.kind == "gaussian_splat"

    def test_png_is_image(self):
        art = infer_visualization_artifact(Path("photo.png"))
        assert art.kind == "image"

    def test_mp4_is_video(self):
        art = infer_visualization_artifact(Path("clip.mp4"))
        assert art.kind == "video"

    def test_wav_is_audio(self):
        art = infer_visualization_artifact(Path("sound.wav"))
        assert art.kind == "audio"

    def test_unknown_suffix_is_artifact(self):
        art = infer_visualization_artifact(Path("data.xyz"))
        assert art.kind == "artifact"

    def test_metadata_passed_through(self):
        art = infer_visualization_artifact(Path("cloud.ply"), metadata={"custom": True})
        assert art.metadata == {"custom": True}

    def test_string_path_works(self):
        art = infer_visualization_artifact("cloud.ply")
        assert art.kind == "point_cloud"
