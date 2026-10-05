"""Comprehensive tests for core.artifacts, core.scene, core.controls, core.styles."""

import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path

# ---------------------------------------------------------------------------
# Direct module imports
# ---------------------------------------------------------------------------
from worldfoundry.studio.visualization.core.artifacts import (
    StudioVisualizationArtifact,
    infer_visualization_artifact,
    normalize_artifact_uri,
    KIND_BY_SUFFIX,
)
from worldfoundry.studio.visualization.core.scene import (
    Frame,
    Layer,
    LayerKind,
    Timeline,
    VisualizationScene,
)
from worldfoundry.studio.visualization.core.controls import (
    VisualizationControl,
    VisualizationEvent,
)
from worldfoundry.studio.visualization.core.styles import (
    DEFAULT_COLORMAP,
    VisualizationStyle,
)

# ---------------------------------------------------------------------------
# __init__ package-level import surface
# ---------------------------------------------------------------------------
from worldfoundry.studio.visualization.core import (
    StudioVisualizationArtifact as _init_artifact,
    infer_visualization_artifact as _init_infer,
    normalize_artifact_uri as _init_normalize_uri,
    VisualizationControl as _init_control,
    VisualizationEvent as _init_event,
    Frame as _init_frame,
    Layer as _init_layer,
    Timeline as _init_timeline,
    VisualizationScene as _init_scene,
    VisualizationStyle as _init_style,
)


# ===================================================================
# Section 1 – __all__ exports & import parity
# ===================================================================

class TestInitExports:
    """Verify __init__.py exports match the direct module symbols."""

    def test_artifact_import_parity(self):
        assert _init_artifact is StudioVisualizationArtifact

    def test_infer_import_parity(self):
        assert _init_infer is infer_visualization_artifact

    def test_normalize_uri_import_parity(self):
        assert _init_normalize_uri is normalize_artifact_uri

    def test_control_import_parity(self):
        assert _init_control is VisualizationControl

    def test_event_import_parity(self):
        assert _init_event is VisualizationEvent

    def test_frame_import_parity(self):
        assert _init_frame is Frame

    def test_layer_import_parity(self):
        assert _init_layer is Layer

    def test_timeline_import_parity(self):
        assert _init_timeline is Timeline

    def test_scene_import_parity(self):
        assert _init_scene is VisualizationScene

    def test_style_import_parity(self):
        assert _init_style is VisualizationStyle

    def test_core_all_contains_artifacts_symbols(self):
        from worldfoundry.studio.visualization.core import __all__
        assert "StudioVisualizationArtifact" in __all__
        assert "infer_visualization_artifact" in __all__
        assert "normalize_artifact_uri" in __all__

    def test_core_all_contains_scene_symbols(self):
        from worldfoundry.studio.visualization.core import __all__
        assert "Frame" in __all__
        assert "Layer" in __all__
        assert "Timeline" in __all__
        assert "VisualizationScene" in __all__

    def test_core_all_contains_control_symbols(self):
        from worldfoundry.studio.visualization.core import __all__
        assert "VisualizationControl" in __all__
        assert "VisualizationEvent" in __all__

    def test_core_all_contains_style_symbols(self):
        from worldfoundry.studio.visualization.core import __all__
        assert "VisualizationStyle" in __all__


# ===================================================================
# Section 2 – artifacts.py
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
        a.metadata["x"] = 1  # should not affect b
        assert "x" not in b.metadata

    def test_frozen_enforcement(self):
        art = StudioVisualizationArtifact(path="/x.glb", kind="mesh")
        with pytest.raises(FrozenInstanceError):
            art.path = "/y.glb"
        with pytest.raises(FrozenInstanceError):
            art.kind = "point_cloud"

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
        for suffix in (".ply", ".pcd", ".npz"):
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
        assert ".xyz" not in KIND_BY_SUFFIX


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

    def test_unknown_suffix_defaults_to_artifact(self):
        art = infer_visualization_artifact("file.xyz")
        assert art.kind == "artifact"
        assert art.format_hint == "xyz"

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


class TestNormalizeArtifactURI:
    """Tests for normalize_artifact_uri()."""

    def test_path_only_no_root(self):
        result = normalize_artifact_uri("/tmp/data/file.ply")
        assert result == "/tmp/data/file.ply"

    def test_relative_path_no_root(self):
        result = normalize_artifact_uri("data/file.ply")
        assert result == "data/file.ply"

    def test_relative_under_root(self):
        """A relative path resolves against CWD, not the given root,
        so it falls back to the absolute resolved path."""
        result = normalize_artifact_uri("sub/file.ply", root="/tmp/data")
        # "sub/file.ply" resolves to CWD-based absolute path, not under /tmp/data
        assert result.startswith("/")

    def test_path_outside_root(self):
        result = normalize_artifact_uri("/other/path.ply", root="/tmp/data")
        # when path is not relative to root, falls back to absolute resolved path
        assert result.startswith("/")

    def test_expanduser(self):
        result = normalize_artifact_uri("~/data/file.ply")
        assert not result.startswith("~")

    def test_root_none(self):
        result = normalize_artifact_uri("/abs/path.ply", root=None)
        assert result == "/abs/path.ply"

    def test_pathlib_path_input(self):
        result = normalize_artifact_uri(Path("sub/file.ply"), root=Path("/tmp/data"))
        assert isinstance(result, str)


# ===================================================================
# Section 3 – scene.py
# ===================================================================

class TestFrame:
    """Tests for Frame dataclass."""

    def test_minimal_construction(self):
        f = Frame(frame_id="world")
        assert f.frame_id == "world"
        assert f.parent_id is None
        assert f.transform is None
        assert f.metadata == {}

    def test_full_construction(self):
        f = Frame(
            frame_id="camera",
            parent_id="world",
            transform=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            metadata={"type": "pinhole"},
        )
        assert f.parent_id == "world"
        assert f.transform == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_frozen_enforcement(self):
        f = Frame(frame_id="root")
        with pytest.raises(FrozenInstanceError):
            f.frame_id = "other"

    def test_default_metadata_is_new_dict(self):
        a = Frame(frame_id="a")
        b = Frame(frame_id="b")
        a.metadata["x"] = 1
        assert "x" not in b.metadata


class TestTimeline:
    """Tests for Timeline dataclass."""

    def test_default_all_none(self):
        t = Timeline()
        assert t.fps is None
        assert t.start_time is None
        assert t.end_time is None
        assert t.frame_count is None
        assert t.metadata == {}

    def test_full_construction(self):
        t = Timeline(fps=30.0, start_time=0.0, end_time=10.0, frame_count=300)
        assert t.fps == 30.0
        assert t.frame_count == 300

    def test_frozen_enforcement(self):
        t = Timeline(fps=30.0)
        with pytest.raises(FrozenInstanceError):
            t.fps = 60.0

    def test_partial_construction(self):
        t = Timeline(fps=24.0, end_time=5.0)
        assert t.start_time is None
        assert t.frame_count is None


class TestLayer:
    """Tests for Layer dataclass and its methods."""

    def test_minimal_construction(self):
        l = Layer(layer_id="pts", kind="point_cloud")
        assert l.layer_id == "pts"
        assert l.kind == "point_cloud"
        assert l.uri is None
        assert l.uris == ()
        assert l.payload is None
        assert l.frame_range is None
        assert l.time_range is None
        assert l.coordinate_frame is None
        assert l.style == {}
        assert l.metadata == {}

    def test_full_construction(self):
        l = Layer(
            layer_id="mesh1",
            kind="mesh",
            uri="/tmp/m.glb",
            uris=("/tmp/m.obj", "/tmp/m.glb"),
            payload={"data": "bytes"},
            frame_range=(0, 100),
            time_range=(0.0, 5.0),
            coordinate_frame="camera",
            style={"color": "red"},
            metadata={"source": "manual"},
        )
        assert l.uri == "/tmp/m.glb"
        assert l.uris == ("/tmp/m.obj", "/tmp/m.glb")
        assert l.frame_range == (0, 100)
        assert l.time_range == (0.0, 5.0)

    def test_frozen_enforcement(self):
        l = Layer(layer_id="x", kind="y")
        with pytest.raises(FrozenInstanceError):
            l.layer_id = "z"

    # -- all_uris() --

    def test_all_uris_uri_only(self):
        l = Layer(layer_id="a", kind="mesh", uri="/x.glb")
        assert l.all_uris() == ("/x.glb",)

    def test_all_uris_uris_only(self):
        l = Layer(layer_id="a", kind="mesh", uris=("/x.glb", "/y.obj"))
        assert l.all_uris() == ("/x.glb", "/y.obj")

    def test_all_uris_both_with_overlap(self):
        """uri + uris overlapping: deduplication preserves first occurrence."""
        l = Layer(layer_id="a", kind="mesh", uri="/x.glb", uris=("/x.glb", "/y.obj"))
        assert l.all_uris() == ("/x.glb", "/y.obj")

    def test_all_uris_no_uri(self):
        l = Layer(layer_id="a", kind="mesh")
        assert l.all_uris() == ()

    def test_all_uris_filters_empty_strings(self):
        l = Layer(layer_id="a", kind="mesh", uris=("", "/real.glb", ""))
        assert l.all_uris() == ("/real.glb",)

    def test_all_uris_uri_empty_string_filtered(self):
        l = Layer(layer_id="a", kind="mesh", uri="")
        assert l.all_uris() == ()

    # -- asdict() --

    def test_asdict_minimal(self):
        l = Layer(layer_id="pts", kind="point_cloud")
        d = l.asdict()
        assert d == {"layer_id": "pts", "kind": "point_cloud"}

    def test_asdict_full(self):
        l = Layer(
            layer_id="m",
            kind="mesh",
            uri="/m.glb",
            uris=("/m.obj",),
            frame_range=(0, 10),
            time_range=(0.0, 1.0),
            coordinate_frame="world",
            style={"color": "blue"},
            metadata={"src": "test"},
        )
        d = l.asdict()
        assert d["layer_id"] == "m"
        assert d["uri"] == "/m.glb"
        assert d["uris"] == ["/m.obj"]
        assert d["frame_range"] == [0, 10]
        assert d["time_range"] == [0.0, 1.0]
        assert d["coordinate_frame"] == "world"
        assert d["style"] == {"color": "blue"}
        assert d["metadata"] == {"src": "test"}

    def test_asdict_omits_none_fields(self):
        l = Layer(layer_id="x", kind="y")
        d = l.asdict()
        assert "uri" not in d
        assert "uris" not in d
        assert "frame_range" not in d
        assert "time_range" not in d
        assert "coordinate_frame" not in d
        assert "style" not in d
        assert "metadata" not in d

    def test_asdict_empty_collections_omitted(self):
        l = Layer(layer_id="x", kind="y", uris=(), style={}, metadata={})
        d = l.asdict()
        assert "uris" not in d
        assert "style" not in d
        assert "metadata" not in d

    # -- fromdict() --

    def test_fromdict_basic(self):
        data = {"layer_id": "pts", "kind": "point_cloud"}
        l = Layer.fromdict(data)
        assert l.layer_id == "pts"
        assert l.kind == "point_cloud"

    def test_fromdict_with_uri(self):
        data = {"layer_id": "m", "kind": "mesh", "uri": "/m.glb"}
        l = Layer.fromdict(data)
        assert l.uri == "/m.glb"

    def test_fromdict_with_uris(self):
        data = {"layer_id": "m", "kind": "mesh", "uris": ["a.glb", "b.obj"]}
        l = Layer.fromdict(data)
        assert l.uris == ("a.glb", "b.obj")

    def test_fromdict_with_ranges(self):
        data = {
            "layer_id": "m",
            "kind": "mesh",
            "frame_range": [0, 10],
            "time_range": [0.0, 5.0],
        }
        l = Layer.fromdict(data)
        assert l.frame_range == (0, 10)
        assert l.time_range == (0.0, 5.0)

    def test_fromdict_with_style_and_metadata(self):
        data = {
            "layer_id": "m",
            "kind": "mesh",
            "style": {"color": "red"},
            "metadata": {"src": "manual"},
        }
        l = Layer.fromdict(data)
        assert l.style == {"color": "red"}
        assert l.metadata == {"src": "manual"}

    def test_fromdict_alternative_id_key(self):
        """fromdict accepts 'id' as fallback for 'layer_id'."""
        data = {"id": "alt", "kind": "mesh"}
        l = Layer.fromdict(data)
        assert l.layer_id == "alt"

    def test_fromdict_missing_fields_defaults(self):
        data = {"layer_id": "x"}
        l = Layer.fromdict(data)
        assert l.kind == ""

    def test_fromdict_none_uri(self):
        data = {"layer_id": "x", "kind": "y", "uri": None}
        l = Layer.fromdict(data)
        assert l.uri is None

    def test_fromdict_none_ranges(self):
        data = {"layer_id": "x", "kind": "y", "frame_range": None, "time_range": None}
        l = Layer.fromdict(data)
        assert l.frame_range is None
        assert l.time_range is None

    # -- round-trip --

    def test_roundtrip_asdict_fromdict(self):
        original = Layer(
            layer_id="mesh1",
            kind="mesh",
            uri="/m.glb",
            uris=("/m.obj",),
            frame_range=(0, 100),
            time_range=(0.0, 5.0),
            coordinate_frame="camera",
            style={"color": "red"},
            metadata={"src": "manual"},
        )
        d = original.asdict()
        restored = Layer.fromdict(d)
        assert restored.layer_id == original.layer_id
        assert restored.kind == original.kind
        assert restored.uri == original.uri
        assert restored.uris == original.uris
        assert restored.frame_range == original.frame_range
        assert restored.time_range == original.time_range
        assert restored.coordinate_frame == original.coordinate_frame
        assert restored.style == original.style
        assert restored.metadata == original.metadata

    def test_roundtrip_minimal_layer(self):
        original = Layer(layer_id="x", kind="y")
        d = original.asdict()
        restored = Layer.fromdict(d)
        assert restored.layer_id == "x"
        assert restored.kind == "y"
        assert restored.uri is None


class TestVisualizationScene:
    """Tests for VisualizationScene dataclass and its methods."""

    def test_minimal_construction(self):
        s = VisualizationScene(scene_id="demo")
        assert s.scene_id == "demo"
        assert s.title == ""
        assert s.layers == ()
        assert s.timeline is None
        assert s.controls == ()
        assert s.frames == ()
        assert s.recommended_backend == "auto"
        assert s.metadata == {}

    def test_full_construction(self):
        layer = Layer(layer_id="pts", kind="point_cloud")
        frame = Frame(frame_id="world")
        timeline = Timeline(fps=30.0)
        ctrl = VisualizationControl(control_id="c1", kind="slider")
        s = VisualizationScene(
            scene_id="scene1",
            title="My Scene",
            layers=(layer,),
            timeline=timeline,
            controls=(ctrl,),
            frames=(frame,),
            recommended_backend="viser",
            metadata={"version": 1},
        )
        assert s.title == "My Scene"
        assert len(s.layers) == 1
        assert s.timeline.fps == 30.0
        assert len(s.frames) == 1
        assert s.recommended_backend == "viser"

    def test_frozen_enforcement(self):
        s = VisualizationScene(scene_id="x")
        with pytest.raises(FrozenInstanceError):
            s.scene_id = "y"

    # -- layer_kinds() --

    def test_layer_kinds_empty(self):
        s = VisualizationScene(scene_id="x")
        assert s.layer_kinds() == frozenset()

    def test_layer_kinds_multiple(self):
        s = VisualizationScene(
            scene_id="x",
            layers=(
                Layer(layer_id="a", kind="mesh"),
                Layer(layer_id="b", kind="point_cloud"),
                Layer(layer_id="c", kind="mesh"),
            ),
        )
        assert s.layer_kinds() == frozenset({"mesh", "point_cloud"})

    # -- asdict() --

    def test_asdict_minimal(self):
        s = VisualizationScene(scene_id="s1", title="Hello")
        d = s.asdict()
        assert d["schema_version"] == 1
        assert d["scene_id"] == "s1"
        assert d["title"] == "Hello"
        assert d["recommended_backend"] == "auto"
        assert d["layers"] == []
        assert "timeline" not in d
        assert "metadata" not in d
        assert "frames" not in d

    def test_asdict_with_timeline(self):
        s = VisualizationScene(
            scene_id="s1",
            timeline=Timeline(fps=30.0, frame_count=300),
        )
        d = s.asdict()
        assert d["timeline"]["fps"] == 30.0
        assert d["timeline"]["frame_count"] == 300
        # start_time/end_time are None so omitted
        assert "start_time" not in d["timeline"]
        assert "end_time" not in d["timeline"]

    def test_asdict_timeline_omits_empty_metadata(self):
        s = VisualizationScene(
            scene_id="s1",
            timeline=Timeline(metadata={}),
        )
        d = s.asdict()
        # timeline dict should exist (fps is None, metadata is empty)
        # only keys with non-None, non-empty values should remain
        if "timeline" in d:
            assert "metadata" not in d["timeline"]

    def test_asdict_with_frames(self):
        s = VisualizationScene(
            scene_id="s1",
            frames=(Frame(frame_id="world", parent_id=None),),
        )
        d = s.asdict()
        assert len(d["frames"]) == 1
        assert d["frames"][0]["frame_id"] == "world"
        assert d["frames"][0]["parent_id"] is None

    def test_asdict_with_metadata(self):
        s = VisualizationScene(
            scene_id="s1",
            metadata={"key": "val"},
        )
        d = s.asdict()
        assert d["metadata"] == {"key": "val"}

    def test_asdict_layers_serialized(self):
        layer = Layer(layer_id="p", kind="point_cloud", uri="/p.ply")
        s = VisualizationScene(scene_id="s1", layers=(layer,))
        d = s.asdict()
        assert len(d["layers"]) == 1
        assert d["layers"][0]["layer_id"] == "p"
        assert d["layers"][0]["uri"] == "/p.ply"

    # -- fromdict() --

    def test_fromdict_basic(self):
        data = {"scene_id": "s1", "title": "Test", "recommended_backend": "viser"}
        s = VisualizationScene.fromdict(data)
        assert s.scene_id == "s1"
        assert s.title == "Test"
        assert s.recommended_backend == "viser"

    def test_fromdict_with_layers(self):
        data = {
            "scene_id": "s1",
            "layers": [
                {"layer_id": "p", "kind": "point_cloud"},
                {"layer_id": "m", "kind": "mesh", "uri": "/m.glb"},
            ],
        }
        s = VisualizationScene.fromdict(data)
        assert len(s.layers) == 2
        assert s.layers[0].layer_id == "p"
        assert s.layers[1].uri == "/m.glb"

    def test_fromdict_with_timeline(self):
        data = {
            "scene_id": "s1",
            "timeline": {"fps": 30.0, "frame_count": 300},
        }
        s = VisualizationScene.fromdict(data)
        assert s.timeline is not None
        assert s.timeline.fps == 30.0

    def test_fromdict_timeline_none(self):
        data = {"scene_id": "s1"}
        s = VisualizationScene.fromdict(data)
        assert s.timeline is None

    def test_fromdict_empty_string_defaults(self):
        data = {"scene_id": "", "title": None, "recommended_backend": None}
        s = VisualizationScene.fromdict(data)
        assert s.scene_id == ""
        assert s.title == ""
        assert s.recommended_backend == "auto"

    def test_fromdict_layers_none_gives_empty(self):
        data = {"scene_id": "s1", "layers": None}
        s = VisualizationScene.fromdict(data)
        assert s.layers == ()

    # -- round-trip --

    def test_roundtrip_asdict_fromdict(self):
        layer = Layer(layer_id="pts", kind="point_cloud", uri="/p.ply")
        timeline = Timeline(fps=24.0, end_time=5.0)
        frame = Frame(frame_id="camera", parent_id="world", transform=[[1, 0, 0]])
        original = VisualizationScene(
            scene_id="demo",
            title="Demo Scene",
            layers=(layer,),
            timeline=timeline,
            frames=(frame,),
            recommended_backend="viser",
            metadata={"version": 2},
        )
        d = original.asdict()
        restored = VisualizationScene.fromdict(d)
        assert restored.scene_id == original.scene_id
        assert restored.title == original.title
        assert restored.recommended_backend == original.recommended_backend
        assert len(restored.layers) == 1
        assert restored.layers[0].layer_id == "pts"
        assert restored.timeline.fps == 24.0
        assert restored.timeline.end_time == 5.0
        assert restored.metadata == {"version": 2}

    def test_roundtrip_minimal_scene(self):
        original = VisualizationScene(scene_id="minimal")
        d = original.asdict()
        restored = VisualizationScene.fromdict(d)
        assert restored.scene_id == "minimal"
        assert restored.layers == ()
        assert restored.timeline is None


class TestLayerKind:
    """Test the LayerKind type alias."""

    def test_layer_kind_is_str(self):
        assert LayerKind is str


# ===================================================================
# Section 4 – controls.py
# ===================================================================

class TestVisualizationControl:
    """Tests for VisualizationControl dataclass."""

    def test_minimal_construction(self):
        c = VisualizationControl(control_id="c1", kind="slider")
        assert c.control_id == "c1"
        assert c.kind == "slider"
        assert c.label == ""
        assert c.value is None
        assert c.options == ()
        assert c.metadata == {}

    def test_full_construction(self):
        c = VisualizationControl(
            control_id="speed",
            kind="slider",
            label="Speed",
            value=0.5,
            options=(0.1, 0.5, 1.0),
            metadata={"min": 0.0, "max": 2.0},
        )
        assert c.label == "Speed"
        assert c.value == 0.5
        assert c.options == (0.1, 0.5, 1.0)

    def test_frozen_enforcement(self):
        c = VisualizationControl(control_id="c", kind="slider")
        with pytest.raises(FrozenInstanceError):
            c.control_id = "other"

    def test_default_metadata_is_new_dict(self):
        a = VisualizationControl(control_id="a", kind="x")
        b = VisualizationControl(control_id="b", kind="y")
        a.metadata["k"] = "v"
        assert "k" not in b.metadata


class TestVisualizationEvent:
    """Tests for VisualizationEvent dataclass."""

    def test_minimal_construction(self):
        e = VisualizationEvent(kind="click")
        assert e.kind == "click"
        assert e.payload == {}
        assert e.timestamp is None

    def test_full_construction(self):
        e = VisualizationEvent(
            kind="select",
            payload={"object_id": "mesh1"},
            timestamp=1.5,
        )
        assert e.payload == {"object_id": "mesh1"}
        assert e.timestamp == 1.5

    def test_frozen_enforcement(self):
        e = VisualizationEvent(kind="click")
        with pytest.raises(FrozenInstanceError):
            e.kind = "other"

    def test_default_payload_is_new_dict(self):
        a = VisualizationEvent(kind="a")
        b = VisualizationEvent(kind="b")
        a.payload["x"] = 1
        assert "x" not in b.payload


# ===================================================================
# Section 5 – styles.py
# ===================================================================

class TestDefaultColormap:
    """Test the DEFAULT_COLORMAP constant."""

    def test_default_colormap_value(self):
        assert DEFAULT_COLORMAP == "viridis"

    def test_default_colormap_is_string(self):
        assert isinstance(DEFAULT_COLORMAP, str)


class TestVisualizationStyle:
    """Tests for VisualizationStyle dataclass."""

    def test_minimal_construction(self):
        s = VisualizationStyle()
        assert s.color is None
        assert s.colormap == DEFAULT_COLORMAP
        assert s.opacity is None
        assert s.point_size is None
        assert s.line_width is None
        assert s.material is None
        assert s.metadata == {}

    def test_color_as_string(self):
        s = VisualizationStyle(color="red")
        assert s.color == "red"

    def test_color_as_rgb_tuple(self):
        s = VisualizationStyle(color=(1.0, 0.0, 0.0))
        assert s.color == (1.0, 0.0, 0.0)

    def test_color_as_rgba_tuple(self):
        s = VisualizationStyle(color=(1.0, 0.0, 0.0, 0.5))
        assert s.color == (1.0, 0.0, 0.0, 0.5)

    def test_color_none(self):
        s = VisualizationStyle(color=None)
        assert s.color is None

    def test_full_construction(self):
        s = VisualizationStyle(
            color="blue",
            colormap="plasma",
            opacity=0.8,
            point_size=2.0,
            line_width=1.5,
            material="metallic",
            metadata={"glow": True},
        )
        assert s.colormap == "plasma"
        assert s.opacity == 0.8
        assert s.point_size == 2.0
        assert s.line_width == 1.5
        assert s.material == "metallic"

    def test_frozen_enforcement(self):
        s = VisualizationStyle(color="red")
        with pytest.raises(FrozenInstanceError):
            s.color = "blue"

    def test_default_metadata_is_new_dict(self):
        a = VisualizationStyle()
        b = VisualizationStyle()
        a.metadata["x"] = 1
        assert "x" not in b.metadata

    # -- asdict() --

    def test_asdict_minimal(self):
        s = VisualizationStyle()
        d = s.asdict()
        # colormap is not None, it has default 'viridis'
        assert d == {"colormap": DEFAULT_COLORMAP}

    def test_asdict_all_fields(self):
        s = VisualizationStyle(
            color="red",
            colormap="plasma",
            opacity=0.5,
            point_size=3.0,
            line_width=1.0,
            material="glass",
            metadata={"key": "val"},
        )
        d = s.asdict()
        assert d["color"] == "red"
        assert d["colormap"] == "plasma"
        assert d["opacity"] == 0.5
        assert d["point_size"] == 3.0
        assert d["line_width"] == 1.0
        assert d["material"] == "glass"
        assert d["metadata"] == {"key": "val"}

    def test_asdict_omits_none_values(self):
        s = VisualizationStyle(colormap="cool")
        d = s.asdict()
        assert "color" not in d
        assert "opacity" not in d
        assert "point_size" not in d
        assert "line_width" not in d
        assert "material" not in d

    def test_asdict_omits_empty_metadata(self):
        s = VisualizationStyle(metadata={})
        d = s.asdict()
        assert "metadata" not in d

    def test_asdict_omits_none_color(self):
        s = VisualizationStyle(color=None)
        d = s.asdict()
        assert "color" not in d

    def test_asdict_keeps_non_none_color(self):
        s = VisualizationStyle(color=(0.5, 0.5, 0.5))
        d = s.asdict()
        assert d["color"] == (0.5, 0.5, 0.5)

    def test_asdict_colormap_always_present(self):
        """colormap has a default so it should always appear."""
        s = VisualizationStyle()
        d = s.asdict()
        assert "colormap" in d

    def test_asdict_only_none_and_empty_omitted(self):
        """The filter removes None and empty {}, but keeps other values."""
        s = VisualizationStyle(opacity=0.0)
        d = s.asdict()
        assert d["opacity"] == 0.0


# ===================================================================
# Section 6 – Edge cases and cross-module interactions
# ===================================================================

class TestEdgeCases:
    """Edge-case and cross-module tests."""

    def test_artifact_metadata_is_mapping_not_plain_dict(self):
        """metadata is typed as Mapping[str, Any]; a dict works."""
        art = StudioVisualizationArtifact(
            path="x.ply", kind="point_cloud", metadata={"k": "v"}
        )
        assert isinstance(art.metadata, dict)

    def test_scene_with_no_layers_layer_kinds_empty_frozenset(self):
        s = VisualizationScene(scene_id="empty")
        assert s.layer_kinds() == frozenset()

    def test_scene_fromdict_missing_schema_version(self):
        """fromdict does not require schema_version in input."""
        data = {"scene_id": "s1", "title": "T"}
        s = VisualizationScene.fromdict(data)
        assert s.scene_id == "s1"

    def test_layer_fromdict_with_all_none_optional_fields(self):
        data = {
            "layer_id": "x",
            "kind": "y",
            "uri": None,
            "uris": None,
            "frame_range": None,
            "time_range": None,
            "coordinate_frame": None,
            "style": None,
            "metadata": None,
        }
        l = Layer.fromdict(data)
        assert l.uri is None
        assert l.uris == ()
        assert l.frame_range is None
        assert l.time_range is None
        assert l.coordinate_frame is None
        assert l.style == {}
        assert l.metadata == {}

    def test_layer_fromdict_empty_dict(self):
        data = {}
        l = Layer.fromdict(data)
        assert l.layer_id == ""
        assert l.kind == ""

    def test_scene_fromdict_empty_dict(self):
        data = {}
        s = VisualizationScene.fromdict(data)
        assert s.scene_id == ""
        assert s.title == ""

    def test_scene_asdict_timeline_with_all_none_fields(self):
        """Timeline with all None/empty fields: the code adds the timeline key even
        when the inner dict is empty {}. This is a known source-level quirk — the
        timeline key is always present if self.timeline is not None, regardless
        of whether all its values are filtered out."""
        s = VisualizationScene(
            scene_id="s1",
            timeline=Timeline(),
        )
        d = s.asdict()
        # NOTE: Source code always adds timeline key if self.timeline is not None,
        # even when all inner values are None/{} and get filtered.
        assert "timeline" in d
        assert d["timeline"] == {}

    def test_scene_asdict_timeline_with_metadata(self):
        s = VisualizationScene(
            scene_id="s1",
            timeline=Timeline(metadata={"desc": "test"}),
        )
        d = s.asdict()
        assert "timeline" in d
        assert d["timeline"]["metadata"] == {"desc": "test"}

    def test_normalize_artifact_uri_same_as_root(self):
        """Path == root should give '.'."""
        result = normalize_artifact_uri("/tmp/data", root="/tmp/data")
        assert result == "."

    def test_infer_artifact_preserves_path_string(self):
        p = "relative/path/to/file.ply"
        art = infer_visualization_artifact(p)
        assert art.path == p

    def test_style_asdict_zero_opacity_not_omitted(self):
        s = VisualizationStyle(opacity=0.0)
        d = s.asdict()
        assert d["opacity"] == 0.0

    def test_style_asdict_zero_point_size_not_omitted(self):
        s = VisualizationStyle(point_size=0.0)
        d = s.asdict()
        assert d["point_size"] == 0.0

    def test_layer_all_uris_preserves_order(self):
        l = Layer(layer_id="a", kind="x", uris=("b", "c", "d"))
        assert l.all_uris() == ("b", "c", "d")

    def test_layer_all_uris_dedup_preserves_first(self):
        l = Layer(layer_id="a", kind="x", uris=("b", "c", "b", "d"))
        assert l.all_uris() == ("b", "c", "d")

    def test_scene_layer_tuple_is_immutable(self):
        """layers is a tuple, not a list."""
        s = VisualizationScene(scene_id="s", layers=(Layer(layer_id="a", kind="x"),))
        assert isinstance(s.layers, tuple)

    def test_scene_controls_tuple_is_immutable(self):
        s = VisualizationScene(scene_id="s", controls=(VisualizationControl(control_id="c", kind="slider"),))
        assert isinstance(s.controls, tuple)

    def test_scene_frames_tuple_is_immutable(self):
        s = VisualizationScene(scene_id="s", frames=(Frame(frame_id="f"),))
        assert isinstance(s.frames, tuple)

    def test_infer_artifact_dot_path_no_suffix(self):
        art = infer_visualization_artifact(".")
        assert art.kind == "artifact"
        assert art.format_hint == ""

    def test_normalize_artifact_uri_empty_path(self):
        result = normalize_artifact_uri("")
        assert isinstance(result, str)

    def test_event_timestamp_zero(self):
        e = VisualizationEvent(kind="tick", timestamp=0.0)
        assert e.timestamp == 0.0

    def test_control_options_with_strings(self):
        c = VisualizationControl(
            control_id="mode",
            kind="dropdown",
            options=("fast", "slow", "auto"),
        )
        assert c.options == ("fast", "slow", "auto")
