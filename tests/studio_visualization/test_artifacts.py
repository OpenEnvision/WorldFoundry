"""Tests for visualization.artifacts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Direct module imports
# ---------------------------------------------------------------------------
from worldfoundry.studio.visualization.artifacts import (
    KIND_BY_SUFFIX,
    StudioVisualizationArtifact,
    infer_visualization_artifact,
)

# ===================================================================
# Section 1 – artifacts.py
# ===================================================================


class TestStudioVisualizationArtifact:
    """Tests for StudioVisualizationArtifact dataclass."""

    def test_minimal_construction(self):
        art = StudioVisualizationArtifact(path="/tmp/model.glb", kind="mesh")
        assert art.path == "/tmp/model.glb"
        assert art.kind == "mesh"
        assert art.format_hint == ""
        assert art.metadata == {}

    def test_full_construction(self):
        art = StudioVisualizationArtifact(
            path="/tmp/pts.ply",
            kind="point_cloud",
            format_hint="ply",
            metadata={"frames": 100},
        )
        assert art.format_hint == "ply"
        assert art.metadata == {"frames": 100}

    def test_default_metadata_is_new_dict(self):
        """Each instance must get a fresh default dict."""
        a = StudioVisualizationArtifact(path="a.ply", kind="point_cloud")
        b = StudioVisualizationArtifact(path="b.ply", kind="point_cloud")
        assert isinstance(a.metadata, dict)
        a.metadata["x"] = 1  # should not affect b
        assert "x" not in b.metadata

    def test_frozen_enforcement(self):
        art = StudioVisualizationArtifact(path="/x.glb", kind="mesh")
        with pytest.raises(FrozenInstanceError):
            setattr(art, "path", "/y.glb")
        with pytest.raises(FrozenInstanceError):
            setattr(art, "kind", "point_cloud")

    def test_resolved_path_property(self):
        art = StudioVisualizationArtifact(path="/tmp/model.glb", kind="mesh")
        assert isinstance(art.resolved_path, Path)
        assert art.resolved_path == Path("/tmp/model.glb").resolve()

    def test_resolved_path_expanduser(self):
        art = StudioVisualizationArtifact(path="~/data/file.ply", kind="point_cloud")
        resolved = art.resolved_path
        assert resolved != Path("~/data/file.ply")
        assert str(resolved).startswith(str(Path.home()))


class TestKIND_BY_SUFFIX:
    """Tests for the KIND_BY_SUFFIX mapping."""

    def test_known_point_cloud_suffixes(self):
        for suffix in (".ply", ".pcd", ".xyz", ".npz"):
            assert KIND_BY_SUFFIX[suffix] == "point_cloud"

    def test_known_mesh_suffixes(self):
        for suffix in (".glb", ".gltf", ".obj"):
            assert KIND_BY_SUFFIX[suffix] == "mesh"

    def test_known_splat_suffixes(self):
        for suffix in (".spz", ".splat", ".ksplat", ".sog"):
            assert KIND_BY_SUFFIX[suffix] == "gaussian_splat"

    def test_known_image_suffixes(self):
        for suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            assert KIND_BY_SUFFIX[suffix] == "image"

    def test_known_video_suffixes(self):
        for suffix in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
            assert KIND_BY_SUFFIX[suffix] == "video"

    def test_known_audio_suffixes(self):
        for suffix in (".wav", ".mp3", ".flac", ".ogg"):
            assert KIND_BY_SUFFIX[suffix] == "audio"

    def test_timeline_suffix(self):
        assert KIND_BY_SUFFIX[".rrd"] == "timeline"

    def test_unknown_suffix_not_in_dict(self):
        assert ".unknown" not in KIND_BY_SUFFIX


class TestInferVisualizationArtifact:
    """Tests for infer_visualization_artifact()."""

    def test_known_glb(self):
        art = infer_visualization_artifact("/tmp/scene.glb")
        assert art.kind == "mesh"
        assert art.format_hint == "glb"
        assert art.path == "/tmp/scene.glb"

    def test_known_ply(self):
        art = infer_visualization_artifact("data/cloud.ply")
        assert art.kind == "point_cloud"
        assert art.format_hint == "ply"

    def test_known_xyz(self):
        art = infer_visualization_artifact("data/cloud.xyz")
        assert art.kind == "point_cloud"
        assert art.format_hint == "xyz"

    def test_unknown_suffix_defaults_to_artifact(self):
        art = infer_visualization_artifact("file.unknown")
        assert art.kind == "artifact"
        assert art.format_hint == "unknown"

    def test_path_object(self):
        art = infer_visualization_artifact(Path("/tmp/video.mp4"))
        assert art.kind == "video"
        assert art.format_hint == "mp4"

    def test_uppercase_suffix_normalized(self):
        art = infer_visualization_artifact("photo.PNG")
        assert art.kind == "image"
        assert art.format_hint == "png"

    def test_metadata_passed_through(self):
        art = infer_visualization_artifact("/x.ply", metadata={"count": 42})
        assert art.metadata == {"count": 42}

    def test_metadata_none_gives_empty_dict(self):
        art = infer_visualization_artifact("/x.ply", metadata=None)
        assert art.metadata == {}

    def test_metadata_default_is_empty(self):
        art = infer_visualization_artifact("/x.ply")
        assert art.metadata == {}

    def test_path_with_no_suffix(self):
        art = infer_visualization_artifact("README")
        assert art.kind == "artifact"
        assert art.format_hint == ""


# ===================================================================
# Section 2 – Edge cases and cross-module interactions
# ===================================================================


class TestEdgeCases:
    """Edge-case and cross-module tests."""

    def test_artifact_metadata_is_mapping_not_plain_dict(self):
        """metadata is typed as Mapping[str, Any]; a dict works."""
        art = StudioVisualizationArtifact(path="x.ply", kind="point_cloud", metadata={"k": "v"})
        assert isinstance(art.metadata, dict)

    def test_infer_artifact_preserves_path_string(self):
        p = "relative/path/to/file.ply"
        art = infer_visualization_artifact(p)
        assert art.path == p

    def test_infer_artifact_dot_path_no_suffix(self):
        art = infer_visualization_artifact(".")
        assert art.kind == "artifact"
        assert art.format_hint == ""
