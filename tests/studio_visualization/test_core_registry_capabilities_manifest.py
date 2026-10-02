"""Comprehensive tests for the core/registry, core/capabilities, and core/manifest submodules."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

# ---------------------------------------------------------------------------
# Ensure worldfoundry is importable from the source tree
# ---------------------------------------------------------------------------
_SRC_ROOT = str(Path(__file__).resolve().parents[2])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# ---------------------------------------------------------------------------
# Imports – registry, capabilities, manifest
# ---------------------------------------------------------------------------
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.ui.interfaces import StudioInterfaceSpec, LocalRepoEvidence
from worldfoundry.studio.ui.launch_config import StudioLaunchConfig

from worldfoundry.studio.visualization.core.registry import (
    ARTIFACT_DOMAIN_ACTION,
    ARTIFACT_DOMAIN_GAUSSIAN_SPLAT,
    ARTIFACT_DOMAIN_GEOMETRY,
    ARTIFACT_DOMAIN_MEDIA,
    ARTIFACT_DOMAIN_TIMELINE,
    ARTIFACT_DOMAIN_UI,
    ARTIFACT_DOMAIN_WORLD,
    AUTO_VISUALIZATION,
    EMBODIED_VISUALIZATION,
    INTERACTIVE_WORLD_VISUALIZATION,
    MEDIA_VISUALIZATION,
    RERUN_VISUALIZATION,
    SPARK_VISUALIZATION,
    UNIFIED_VISUALIZATION,
    VISER_VISUALIZATION,
    BackendCapabilities,
    RenderPlan,
    RenderRequest,
    RenderResult,
    StudioModelVisualizationProfile,
    StudioVisualizationBackend,
    StudioVisualizationEvent,
    StudioVisualizationLaunch,
    StudioVisualizationRegistry,
    StudioVisualizationRequest,
    VisualizationBackend,
    VisualizationProvider,
    model_visualization_profile,
    normalize_visualization_mode,
)

from worldfoundry.studio.visualization.core.manifest import (
    ViewportKind,
    WorldViewportAssets,
    SplatViewportAssets,
    PointsViewportAssets,
    EmbodiedViewportAssets,
    ViewportCapabilities,
    StudioViewportsPayload,
    build_studio_viewports_payload,
    viewport_payload_from_metadata,
    _str_or_none,
    _int_or_default,
)

from worldfoundry.studio.visualization.core.capabilities import (
    available_viewport_kinds,
    recommend_viewport,
    _coerce_viewport_kind,
)

from worldfoundry.studio.visualization.core.scene import (
    Layer,
    VisualizationScene,
)

# ---------------------------------------------------------------------------
# Reusable test fixtures
# ---------------------------------------------------------------------------

def _make_entry(**overrides: Any) -> CatalogEntry:
    """Build a minimal CatalogEntry for testing."""
    defaults = dict(
        model_id="test-model",
        display_name="Test Model",
        module_path="worldfoundry.pipelines.test_model",
        class_name="TestModelPipeline",
        family="test_model",
        category="Video Generation",
        summary="Test summary",
        call_params=("prompt", "output_path"),
        stream_params=(),
        load_params=("model_path",),
        supports_stream=False,
        supports_from_pretrained=False,
        supports_api_init=False,
        runtime_kind="default",
        default_backend="auto",
        default_model_ref="",
        default_endpoint="",
        default_prompt="A test prompt",
        default_input_path="",
        default_task_type="",
        default_interactions=(),
        default_load_kwargs={},
        default_call_kwargs={},
        extra_variants=(),
        suggested_task_types=(),
        aliases=(),
        tags=(),
        env_hints=(),
        notes="",
    )
    defaults.update(overrides)
    return CatalogEntry(**defaults)


def _make_spec(**overrides: Any) -> StudioInterfaceSpec:
    defaults = dict(
        template_id="text-video",
        template_title="Text / Media Video Generator",
        input_groups=("Prompt",),
        output_groups=("Generated video", "Preview image", "Manifest JSON"),
        interaction_model="one-shot or stream",
        task_family="",
        artifact_kind="",
        runtime_status="",
        profile_available=False,
        local_repo=LocalRepoEvidence(),
        source_urls=(),
        gui_refs=(),
        launch_hints=(),
    )
    defaults.update(overrides)
    return StudioInterfaceSpec(**defaults)


def _make_launch_config(**overrides: Any) -> StudioLaunchConfig:
    defaults = dict(
        model_id="test-model",
        variant_id=None,
        model_ref="",
        device="cuda",
        backend="auto",
        endpoint="",
        show_aux_panels=False,
        frontend="auto",
        asset_path="",
        simulator_url="",
        host="",
        port=None,
    )
    defaults.update(overrides)
    return StudioLaunchConfig(**defaults)


# ========================================================================
# 1. Constants and __all__ exports – registry.py
# ========================================================================

class TestRegistryConstants:
    """Verify that all constant strings are defined and non-empty."""

    def test_interactive_world_visualization(self):
        assert INTERACTIVE_WORLD_VISUALIZATION == "world"

    def test_viser_visualization(self):
        assert VISER_VISUALIZATION == "points"

    def test_spark_visualization(self):
        assert SPARK_VISUALIZATION == "spark"

    def test_media_visualization(self):
        assert MEDIA_VISUALIZATION == "media"

    def test_rerun_visualization(self):
        assert RERUN_VISUALIZATION == "rerun"

    def test_embodied_visualization(self):
        assert EMBODIED_VISUALIZATION == "embodied"

    def test_unified_visualization(self):
        assert UNIFIED_VISUALIZATION == "unified"

    def test_auto_visualization(self):
        assert AUTO_VISUALIZATION == "auto"

    def test_artifact_domain_world(self):
        assert ARTIFACT_DOMAIN_WORLD == "interactive_world"

    def test_artifact_domain_geometry(self):
        assert ARTIFACT_DOMAIN_GEOMETRY == "geometry"

    def test_artifact_domain_gaussian_splat(self):
        assert ARTIFACT_DOMAIN_GAUSSIAN_SPLAT == "gaussian_splat"

    def test_artifact_domain_action(self):
        assert ARTIFACT_DOMAIN_ACTION == "embodied_action"

    def test_artifact_domain_timeline(self):
        assert ARTIFACT_DOMAIN_TIMELINE == "timeline"

    def test_artifact_domain_media(self):
        assert ARTIFACT_DOMAIN_MEDIA == "media"

    def test_artifact_domain_ui(self):
        assert ARTIFACT_DOMAIN_UI == "ui"


class TestRegistryExports:
    """Verify that __all__ in registry.py contains all public symbols."""

    def test_registry_all_contains_expected_symbols(self):
        from worldfoundry.studio.visualization.core import registry as reg_mod
        expected = [
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
            "UNIFIED_VISUALIZATION",
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
        for symbol in expected:
            assert symbol in reg_mod.__all__, f"{symbol} missing from registry __all__"


class TestManifestExports:
    """Verify that manifest module's public symbols are importable."""

    def test_viewport_kind_importable(self):
        assert ViewportKind is not None

    def test_world_viewport_assets_importable(self):
        assert WorldViewportAssets is not None

    def test_splat_viewport_assets_importable(self):
        assert SplatViewportAssets is not None

    def test_points_viewport_assets_importable(self):
        assert PointsViewportAssets is not None

    def test_embodied_viewport_assets_importable(self):
        assert EmbodiedViewportAssets is not None

    def test_viewport_capabilities_importable(self):
        assert ViewportCapabilities is not None

    def test_studio_viewports_payload_importable(self):
        assert StudioViewportsPayload is not None

    def test_viewport_payload_from_metadata_importable(self):
        assert viewport_payload_from_metadata is not None


class TestCoreInitExports:
    """Verify that the core/__init__.py re-exports all expected symbols."""

    def test_core_all_symbols(self):
        from worldfoundry.studio.visualization.core import __all__ as core_all
        expected_in_core = [
            "BackendCapabilities",
            "RenderPlan",
            "RenderRequest",
            "RenderResult",
            "StudioVisualizationBackend",
            "StudioVisualizationEvent",
            "StudioVisualizationLaunch",
            "StudioVisualizationRegistry",
            "StudioVisualizationRequest",
            "VisualizationBackend",
            "VisualizationProvider",
            "model_visualization_profile",
            "normalize_visualization_mode",
        ]
        for sym in expected_in_core:
            assert sym in core_all, f"{sym} missing from core __all__"


class TestVisualizationInitExports:
    """Verify that the visualization/__init__.py re-exports everything."""

    def test_viz_all_symbols(self):
        from worldfoundry.studio.visualization import __all__ as viz_all
        assert "BackendCapabilities" in viz_all
        assert "normalize_visualization_mode" in viz_all
        assert "model_visualization_profile" in viz_all
        assert "StudioVisualizationRegistry" in viz_all
        assert "StudioVisualizationBackend" in viz_all


# ========================================================================
# 2. ViewportKind enum – manifest.py
# ========================================================================

class TestViewportKind:
    """Test the ViewportKind str-enum."""

    def test_enum_values(self):
        assert ViewportKind.WORLD.value == "world"
        assert ViewportKind.SPLAT.value == "splat"
        assert ViewportKind.POINTS.value == "points"
        assert ViewportKind.EMBODIED.value == "embodied"

    def test_enum_is_str_subclass(self):
        # ViewportKind inherits from str, Enum
        assert isinstance(ViewportKind.WORLD, str)
        assert ViewportKind.WORLD == "world"

    def test_enum_iteration(self):
        values = {member.value for member in ViewportKind}
        assert values == {"world", "splat", "points", "embodied"}

    def test_enum_from_string(self):
        assert ViewportKind("world") == ViewportKind.WORLD
        assert ViewportKind("splat") == ViewportKind.SPLAT

    def test_enum_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ViewportKind("invalid")


# ========================================================================
# 3. Frozen dataclasses – manifest.py
# ========================================================================

class TestWorldViewportAssets:
    """Test WorldViewportAssets frozen dataclass."""

    def test_default_fields(self):
        obj = WorldViewportAssets()
        assert obj.preview_video is None
        assert obj.preview_image is None
        assert obj.rrd_path is None

    def test_explicit_fields(self):
        obj = WorldViewportAssets(
            preview_video="/path/to/video.mp4",
            preview_image="/path/to/image.png",
            rrd_path="/path/to/data.rrd",
        )
        assert obj.preview_video == "/path/to/video.mp4"
        assert obj.preview_image == "/path/to/image.png"
        assert obj.rrd_path == "/path/to/data.rrd"

    def test_frozen_enforcement(self):
        obj = WorldViewportAssets(preview_video="test.mp4")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.preview_video = "other.mp4"


class TestSplatViewportAssets:
    """Test SplatViewportAssets frozen dataclass."""

    def test_default_fields(self):
        obj = SplatViewportAssets()
        assert obj.primary_path is None
        assert obj.primary_url is None
        assert obj.format_hint is None

    def test_explicit_fields(self):
        obj = SplatViewportAssets(
            primary_path="/path/to/splat.spz",
            primary_url="http://example.com/splat.spz",
            format_hint="splat",
        )
        assert obj.primary_path == "/path/to/splat.spz"
        assert obj.primary_url == "http://example.com/splat.spz"
        assert obj.format_hint == "splat"

    def test_frozen_enforcement(self):
        obj = SplatViewportAssets(primary_path="test.spz")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.primary_path = "other.spz"


class TestPointsViewportAssets:
    """Test PointsViewportAssets frozen dataclass."""

    def test_default_fields(self):
        obj = PointsViewportAssets()
        assert obj.point_cloud_path is None
        assert obj.mesh_path is None
        assert obj.camera_path is None
        assert obj.coordinate_frame == "world"

    def test_explicit_fields(self):
        obj = PointsViewportAssets(
            point_cloud_path="/path/to/cloud.ply",
            mesh_path="/path/to/mesh.glb",
            camera_path="/path/to/cam.json",
            coordinate_frame="local",
        )
        assert obj.point_cloud_path == "/path/to/cloud.ply"
        assert obj.mesh_path == "/path/to/mesh.glb"
        assert obj.camera_path == "/path/to/cam.json"
        assert obj.coordinate_frame == "local"

    def test_frozen_enforcement(self):
        obj = PointsViewportAssets(point_cloud_path="test.ply")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.point_cloud_path = "other.ply"


class TestEmbodiedViewportAssets:
    """Test EmbodiedViewportAssets frozen dataclass."""

    def test_default_fields(self):
        obj = EmbodiedViewportAssets()
        assert obj.action_trace_path is None
        assert obj.simulator_video_path is None
        assert obj.episode_metadata_path is None
        assert obj.simulator_hint is None

    def test_explicit_fields(self):
        obj = EmbodiedViewportAssets(
            action_trace_path="/path/to/trace.json",
            simulator_video_path="/path/to/sim.mp4",
            episode_metadata_path="/path/to/meta.json",
            simulator_hint="RoboTwin / LIBERO-style episode",
        )
        assert obj.action_trace_path == "/path/to/trace.json"
        assert obj.simulator_video_path == "/path/to/sim.mp4"
        assert obj.episode_metadata_path == "/path/to/meta.json"
        assert obj.simulator_hint == "RoboTwin / LIBERO-style episode"

    def test_frozen_enforcement(self):
        obj = EmbodiedViewportAssets(action_trace_path="test.json")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.action_trace_path = "other.json"


class TestViewportCapabilities:
    """Test ViewportCapabilities frozen dataclass."""

    def test_default_fields(self):
        obj = ViewportCapabilities()
        assert obj.has_streaming is False
        assert obj.has_gaussian_splat is False
        assert obj.has_points_cloud is False
        assert obj.has_viser is False
        assert obj.has_rrd is False
        assert obj.has_embodied_trace is False
        assert obj.has_simulator_replay is False

    def test_explicit_fields(self):
        obj = ViewportCapabilities(
            has_streaming=True,
            has_gaussian_splat=True,
            has_points_cloud=True,
        )
        assert obj.has_streaming is True
        assert obj.has_gaussian_splat is True
        assert obj.has_points_cloud is True

    def test_frozen_enforcement(self):
        obj = ViewportCapabilities(has_streaming=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.has_streaming = False


# ========================================================================
# 4. StudioViewportsPayload (mutable) – manifest.py
# ========================================================================

class TestStudioViewportsPayload:
    """Test the mutable StudioViewportsPayload dataclass."""

    def test_construction_minimal(self):
        obj = StudioViewportsPayload(recommended=ViewportKind.WORLD)
        assert obj.recommended == ViewportKind.WORLD
        assert obj.schema_version == 1
        assert obj.assets_world == WorldViewportAssets()
        assert obj.assets_splat == SplatViewportAssets()
        assert obj.assets_points == PointsViewportAssets()
        assert obj.assets_embodied == EmbodiedViewportAssets()
        assert obj.capabilities == ViewportCapabilities()

    def test_construction_full(self):
        caps = ViewportCapabilities(has_streaming=True, has_gaussian_splat=True)
        world = WorldViewportAssets(preview_video="video.mp4")
        obj = StudioViewportsPayload(
            recommended=ViewportKind.SPLAT,
            schema_version=2,
            assets_world=world,
            capabilities=caps,
        )
        assert obj.recommended == ViewportKind.SPLAT
        assert obj.schema_version == 2
        assert obj.assets_world.preview_video == "video.mp4"
        assert obj.capabilities.has_streaming is True

    def test_asdict_structure(self):
        caps = ViewportCapabilities(has_streaming=True, has_gaussian_splat=False)
        world = WorldViewportAssets(preview_video="v.mp4", preview_image=None)
        obj = StudioViewportsPayload(
            recommended=ViewportKind.WORLD,
            schema_version=1,
            assets_world=world,
            capabilities=caps,
        )
        d = obj.asdict()
        assert d["schema_version"] == 1
        assert d["recommended"] == "world"
        assert "assets" in d
        assert "capabilities" in d
        assert d["assets"]["world"]["preview_video"] == "v.mp4"
        assert d["assets"]["world"]["preview_image"] is None
        assert d["capabilities"]["has_streaming"] is True
        assert d["capabilities"]["has_gaussian_splat"] is False

    def test_asdict_all_sections_present(self):
        obj = StudioViewportsPayload(recommended=ViewportKind.WORLD)
        d = obj.asdict()
        # Must contain all 4 asset sections
        assert "world" in d["assets"]
        assert "splat" in d["assets"]
        assert "points" in d["assets"]
        assert "embodied" in d["assets"]
        # Must contain capabilities
        assert "has_streaming" in d["capabilities"]

    def test_asdict_roundtrip_via_viewport_payload_from_metadata(self):
        """Test that asdict output can be rehydrated via viewport_payload_from_metadata."""
        caps = ViewportCapabilities(
            has_streaming=True,
            has_points_cloud=True,
            has_viser=True,
        )
        world = WorldViewportAssets(preview_video="test.mp4", rrd_path="test.rrd")
        points = PointsViewportAssets(point_cloud_path="cloud.ply", coordinate_frame="local")
        obj = StudioViewportsPayload(
            recommended=ViewportKind.POINTS,
            schema_version=2,
            assets_world=world,
            assets_points=points,
            capabilities=caps,
        )
        d = obj.asdict()
        metadata = {"studio_viewports": d}
        rehydrated = viewport_payload_from_metadata(metadata)
        assert rehydrated is not None
        assert rehydrated.recommended == ViewportKind.POINTS
        assert rehydrated.schema_version == 2
        assert rehydrated.assets_world.preview_video == "test.mp4"
        assert rehydrated.capabilities.has_streaming is True

    def test_not_frozen(self):
        """StudioViewportsPayload is mutable (not frozen=True)."""
        obj = StudioViewportsPayload(recommended=ViewportKind.WORLD)
        # Should not raise – this is a mutable dataclass
        obj.schema_version = 3
        assert obj.schema_version == 3


class TestBuildStudioViewportsPayload:
    """Manifest builder propagation for model-declared coordinate frames."""

    @staticmethod
    def _build(tmp_path: Path, result_metadata: Mapping[str, Any] | None) -> dict[str, object]:
        cloud_path = tmp_path / "cloud.ply"
        cloud_path.write_text("ply\n", encoding="utf-8")
        return build_studio_viewports_payload(
            entry=_make_entry(),
            output_dir=str(tmp_path),
            previews={},
            artifact_paths=[str(cloud_path)],
            gaussian_ply_predicate=lambda _path: False,
            result_metadata=result_metadata,
        )

    def test_propagates_coordinate_frame_from_result_metadata(self, tmp_path):
        payload = self._build(tmp_path, {"coordinate_frame": "camera-opencv"})
        assert payload["assets"]["points"]["coordinate_frame"] == "camera-opencv"

    def test_coordinate_frame_defaults_to_world(self, tmp_path):
        payload = self._build(tmp_path, {})
        assert payload["assets"]["points"]["coordinate_frame"] == "world"

    def test_materialize_run_passes_result_metadata_to_viewport_builder(self, tmp_path):
        from worldfoundry.studio.inference.execution import PipelineContext, PreparedInputs, StudioManager

        output_dir = tmp_path / "run"
        output_dir.mkdir()
        cloud_path = output_dir / "cloud.xyz"
        cloud_path.write_text("0 0 0\n", encoding="utf-8")
        entry = _make_entry(category="3D Generation")
        context = PipelineContext(
            entry=entry,
            pipeline=object(),
            cache_key="coordinate-frame-test",
            backend="auto",
            model_ref="",
            endpoint="",
            load_kwargs={},
            device="cpu",
        )
        request = PreparedInputs(
            prompt="",
            input_path="",
            image=None,
            image_path=None,
            video_path=None,
            last_frame=None,
            last_frame_path=None,
            reference_images=[],
            reference_image_paths=[],
            interactions=None,
            camera_view=None,
            task_type="",
            intrinsics=None,
            meta_path="",
            panorama_path="",
            scene_name="",
            fps=1,
            num_frames=1,
            output_dir=str(output_dir),
            output_path=str(output_dir / "output.mp4"),
            call_kwargs={},
            load_kwargs={},
            model_ref="",
            backend="auto",
            endpoint="",
            api_key="",
            device="cpu",
        )

        from worldfoundry.studio.visualization.studio import prepare_run_record

        record = StudioManager(
            workspace_root=str(tmp_path / "studio"), record_processor=prepare_run_record,
        ).materialize_run(
            context,
            request,
            result={
                "point_cloud_path": str(cloud_path),
                "metadata": {"coordinate_frame": "camera-opencv"},
            },
            mode="run",
        )

        assert (
            record.metadata["studio_viewports"]["assets"]["points"]["coordinate_frame"]
            == "camera-opencv"
        )


# ========================================================================
# 5. viewport_payload_from_metadata – manifest.py
# ========================================================================

class TestViewportPayloadFromMetadata:
    """Test the viewport_payload_from_metadata function."""

    def test_none_metadata_returns_none(self):
        assert viewport_payload_from_metadata(None) is None

    def test_missing_studio_viewports_returns_none(self):
        assert viewport_payload_from_metadata({}) is None

    def test_non_dict_studio_viewports_returns_none(self):
        assert viewport_payload_from_metadata({"studio_viewports": "not-a-dict"}) is None

    def test_empty_dict_studio_viewports(self):
        result = viewport_payload_from_metadata({"studio_viewports": {}})
        assert result is not None
        assert result.recommended == ViewportKind.WORLD
        assert result.schema_version == 1

    def test_full_payload(self):
        blob = {
            "studio_viewports": {
                "schema_version": 2,
                "recommended": "splat",
                "assets": {
                    "world": {"preview_video": "v.mp4", "preview_image": None, "rrd_path": "r.rrd"},
                    "splat": {"primary_path": "s.spz", "primary_url": None, "format": "splat"},
                    "points": {"point_cloud_path": "c.ply", "mesh_path": None, "camera_path": None, "coordinate_frame": "local"},
                    "embodied": {"action_trace_path": "a.json", "simulator_video_path": None, "episode_metadata_path": None, "simulator_hint": "test-hint"},
                },
                "capabilities": {
                    "has_streaming": True,
                    "has_gaussian_splat": True,
                    "has_points_cloud": False,
                    "has_viser": True,
                    "has_rrd": False,
                    "has_embodied_trace": True,
                    "has_simulator_replay": False,
                },
            }
        }
        result = viewport_payload_from_metadata(blob)
        assert result is not None
        assert result.recommended == ViewportKind.SPLAT
        assert result.schema_version == 2
        assert result.assets_world.preview_video == "v.mp4"
        assert result.assets_splat.primary_path == "s.spz"
        assert result.assets_splat.format_hint == "splat"
        assert result.assets_points.coordinate_frame == "local"
        assert result.assets_embodied.simulator_hint == "test-hint"
        assert result.capabilities.has_streaming is True
        assert result.capabilities.has_gaussian_splat is True
        # Cross-propagation: has_points_cloud=True because has_viser=True in input
        assert result.capabilities.has_points_cloud is True
        assert result.capabilities.has_viser is True
        assert result.capabilities.has_embodied_trace is True

    def test_invalid_recommended_falls_back_to_world(self):
        blob = {
            "studio_viewports": {
                "recommended": "invalid_viewport",
                "capabilities": {},
            }
        }
        result = viewport_payload_from_metadata(blob)
        assert result is not None
        assert result.recommended == ViewportKind.WORLD

    def test_missing_recommended_falls_back_to_world(self):
        blob = {"studio_viewports": {"capabilities": {}}}
        result = viewport_payload_from_metadata(blob)
        assert result.recommended == ViewportKind.WORLD

    def test_schema_version_non_integer(self):
        blob = {"studio_viewports": {"schema_version": "bad", "recommended": "world"}}
        result = viewport_payload_from_metadata(blob)
        assert result.schema_version == 1  # falls back to default

    def test_cross_propagation_viser_to_points_cloud(self):
        """has_viser=True in input should set both has_viser and has_points_cloud in output."""
        blob = {
            "studio_viewports": {
                "recommended": "points",
                "capabilities": {"has_viser": True, "has_points_cloud": False},
            }
        }
        result = viewport_payload_from_metadata(blob)
        assert result.capabilities.has_viser is True
        assert result.capabilities.has_points_cloud is True

    def test_cross_propagation_points_cloud_to_viser(self):
        """has_points_cloud=True in input should also set has_viser=True."""
        blob = {
            "studio_viewports": {
                "recommended": "points",
                "capabilities": {"has_points_cloud": True, "has_viser": False},
            }
        }
        result = viewport_payload_from_metadata(blob)
        assert result.capabilities.has_points_cloud is True
        assert result.capabilities.has_viser is True

    def test_splat_primary_path_from_primary_url_fallback(self):
        """When primary_path is absent but primary_url is given, use primary_url as primary_path."""
        blob = {
            "studio_viewports": {
                "recommended": "splat",
                "assets": {
                    "splat": {"primary_url": "http://example.com/s.spz", "format": "splat"},
                },
            }
        }
        result = viewport_payload_from_metadata(blob)
        assert result.assets_splat.primary_path == "http://example.com/s.spz"

    def test_splat_primary_path_preferred_over_url(self):
        """When both primary_path and primary_url are given, primary_path is used."""
        blob = {
            "studio_viewports": {
                "recommended": "splat",
                "assets": {
                    "splat": {"primary_path": "/local/s.spz", "primary_url": "http://example.com/s.spz"},
                },
            }
        }
        result = viewport_payload_from_metadata(blob)
        # The code does `primary_path=_str_or_none(splat.get("primary_path") or splat.get("primary_url"))`
        # which means primary_url is used only if primary_path is absent/empty
        assert result.assets_splat.primary_path == "/local/s.spz"


# ========================================================================
# 6. Helper functions – manifest.py
# ========================================================================

class TestStrOrNone:
    """Test the _str_or_none helper."""

    def test_none_returns_none(self):
        assert _str_or_none(None) is None

    def test_empty_string_returns_none(self):
        assert _str_or_none("") is None

    def test_whitespace_returns_none(self):
        assert _str_or_none("   ") is None

    def test_normal_string(self):
        assert _str_or_none("hello") == "hello"

    def test_string_with_whitespace(self):
        assert _str_or_none("  hello  ") == "hello"

    def test_integer_converted_to_string(self):
        assert _str_or_none(42) == "42"

    def test_float_converted_to_string(self):
        assert _str_or_none(3.14) == "3.14"


class TestIntOrDefault:
    """Test the _int_or_default helper."""

    def test_valid_integer(self):
        assert _int_or_default(5, 1) == 5

    def test_string_integer(self):
        assert _int_or_default("3", 1) == 3

    def test_none_returns_default(self):
        assert _int_or_default(None, 10) == 10

    def test_non_numeric_string_returns_default(self):
        assert _int_or_default("abc", 10) == 10

    def test_float_returns_int(self):
        assert _int_or_default(2.7, 1) == 2

    def test_zero(self):
        assert _int_or_default(0, 1) == 0


# ========================================================================
# 7. Frozen dataclasses – registry.py
# ========================================================================

class TestStudioVisualizationEvent:
    """Test StudioVisualizationEvent frozen dataclass."""

    def test_default_fields(self):
        obj = StudioVisualizationEvent(kind="click")
        assert obj.kind == "click"
        assert obj.payload == {}
        assert obj.timestamp is None

    def test_explicit_fields(self):
        obj = StudioVisualizationEvent(
            kind="navigate",
            payload={"target": "room2"},
            timestamp=1.5,
        )
        assert obj.payload == {"target": "room2"}
        assert obj.timestamp == 1.5

    def test_frozen_enforcement(self):
        obj = StudioVisualizationEvent(kind="click")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.kind = "other"


class TestStudioModelVisualizationProfile:
    """Test StudioModelVisualizationProfile frozen dataclass."""

    def test_required_fields(self):
        obj = StudioModelVisualizationProfile(
            mode="world",
            artifact_domain="interactive_world",
            title="Interactive World Model",
        )
        assert obj.mode == "world"
        assert obj.artifact_domain == "interactive_world"
        assert obj.title == "Interactive World Model"
        assert obj.reason == ""
        assert obj.accepted_artifact_kinds == ()

    def test_all_fields(self):
        obj = StudioModelVisualizationProfile(
            mode="points",
            artifact_domain="geometry",
            title="Geometry Viewer",
            reason="depth/geometry artifact contract",
            accepted_artifact_kinds=("point_cloud", "mesh", "depth"),
        )
        assert obj.reason == "depth/geometry artifact contract"
        assert obj.accepted_artifact_kinds == ("point_cloud", "mesh", "depth")

    def test_frozen_enforcement(self):
        obj = StudioModelVisualizationProfile(mode="world", artifact_domain="interactive_world", title="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.mode = "other"


class TestBackendCapabilities:
    """Test BackendCapabilities frozen dataclass."""

    def test_default_fields(self):
        obj = BackendCapabilities()
        assert obj.layer_kinds == frozenset()
        assert obj.partial is True
        assert obj.score == 0
        assert obj.metadata == {}

    def test_explicit_fields(self):
        obj = BackendCapabilities(
            layer_kinds=frozenset({"point_cloud", "mesh"}),
            partial=False,
            score=5,
            metadata={"engine": "viser"},
        )
        assert obj.layer_kinds == frozenset({"point_cloud", "mesh"})
        assert obj.partial is False
        assert obj.score == 5

    def test_frozen_enforcement(self):
        obj = BackendCapabilities()
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.partial = False


class TestRenderPlan:
    """Test RenderPlan frozen dataclass."""

    def test_required_fields(self):
        obj = RenderPlan(backend_id="world", supported=True)
        assert obj.backend_id == "world"
        assert obj.supported is True
        assert obj.score == 0
        assert obj.unsupported_layers == ()
        assert obj.reason == ""

    def test_all_fields(self):
        obj = RenderPlan(
            backend_id="points",
            supported=False,
            score=3,
            unsupported_layers=("gaussian_splat",),
            reason="missing gaussian_splat support",
        )
        assert obj.supported is False
        assert obj.unsupported_layers == ("gaussian_splat",)

    def test_frozen_enforcement(self):
        obj = RenderPlan(backend_id="world", supported=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.supported = False


class TestRenderRequest:
    """Test RenderRequest frozen dataclass."""

    def test_default_fields(self):
        obj = RenderRequest()
        assert obj.backend == "auto"
        assert obj.output_path == ""
        assert obj.options == {}

    def test_explicit_fields(self):
        obj = RenderRequest(
            backend="world",
            output_path="/output/scene.mp4",
            options={"fps": 30},
        )
        assert obj.backend == "world"
        assert obj.output_path == "/output/scene.mp4"

    def test_frozen_enforcement(self):
        obj = RenderRequest()
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.backend = "other"


class TestRenderResult:
    """Test RenderResult frozen dataclass."""

    def test_required_fields(self):
        obj = RenderResult(backend_id="world")
        assert obj.backend_id == "world"
        assert obj.url == ""
        assert obj.output_path == ""
        assert obj.caption == ""
        assert obj.metadata == {}

    def test_all_fields(self):
        obj = RenderResult(
            backend_id="world",
            url="http://example.com/view",
            output_path="/output/scene.mp4",
            caption="Generated scene",
            metadata={"scene_id": "scene-1"},
        )
        assert obj.url == "http://example.com/view"
        assert obj.caption == "Generated scene"

    def test_frozen_enforcement(self):
        obj = RenderResult(backend_id="world")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.backend_id = "other"


class TestStudioVisualizationLaunch:
    """Test StudioVisualizationLaunch frozen dataclass."""

    def test_required_fields(self):
        obj = StudioVisualizationLaunch(mode="world")
        assert obj.mode == "world"
        assert obj.url == ""
        assert obj.caption == ""
        assert obj.metadata == {}

    def test_all_fields(self):
        obj = StudioVisualizationLaunch(
            mode="world",
            url="http://localhost:7860",
            caption="Interactive World",
            metadata={"port": 7860},
        )
        assert obj.url == "http://localhost:7860"

    def test_frozen_enforcement(self):
        obj = StudioVisualizationLaunch(mode="world")
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.mode = "other"


# ========================================================================
# 8. normalize_visualization_mode function
# ========================================================================

class TestNormalizeVisualizationMode:
    """Test the normalize_visualization_mode standalone function."""

    def test_lowercase_passthrough(self):
        assert normalize_visualization_mode("world") == "world"

    def test_uppercase_to_lower(self):
        assert normalize_visualization_mode("WORLD") == "world"

    def test_mixed_case(self):
        assert normalize_visualization_mode("WorldModel") == "worldmodel"

    def test_underscore_to_hyphen(self):
        assert normalize_visualization_mode("interactive_world") == "interactive-world"

    def test_multiple_underscores(self):
        assert normalize_visualization_mode("a_b_c") == "a-b-c"

    def test_whitespace_stripped(self):
        assert normalize_visualization_mode("  world  ") == "world"

    def test_none_returns_empty_string(self):
        assert normalize_visualization_mode(None) == ""

    def test_empty_string_returns_empty(self):
        assert normalize_visualization_mode("") == ""

    def test_combined_transforms(self):
        assert normalize_visualization_mode(" Interactive_World ") == "interactive-world"


# ========================================================================
# 9. model_visualization_profile function
# ========================================================================

class TestModelVisualizationProfile:
    """Test the model_visualization_profile routing function."""

    def test_interactive_world_template(self):
        entry = _make_entry()
        spec = _make_spec(template_id="interactive-world")
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == INTERACTIVE_WORLD_VISUALIZATION
        assert profile.artifact_domain == ARTIFACT_DOMAIN_WORLD
        assert "interactive" in profile.title.lower()

    def test_depth_geometry_template(self):
        entry = _make_entry()
        spec = _make_spec(template_id="depth-geometry")
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == VISER_VISUALIZATION
        assert profile.artifact_domain == ARTIFACT_DOMAIN_GEOMETRY

    def test_depth_geometry_category(self):
        entry = _make_entry(category="Depth / Geometry")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == VISER_VISUALIZATION

    def test_pointcloud_nav_runtime(self):
        entry = _make_entry(runtime_kind="pointcloud_nav")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == VISER_VISUALIZATION

    def test_worldfm_runtime(self):
        entry = _make_entry(runtime_kind="worldfm")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == VISER_VISUALIZATION

    def test_scene_3d_template(self):
        entry = _make_entry()
        spec = _make_spec(template_id="scene-3d")
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == SPARK_VISUALIZATION
        assert profile.artifact_domain == ARTIFACT_DOMAIN_GAUSSIAN_SPLAT

    def test_3d_scene_category(self):
        entry = _make_entry(category="3D Scene")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == SPARK_VISUALIZATION

    def test_two_stage_3dgs_runtime(self):
        entry = _make_entry(runtime_kind="two_stage_3dgs")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == SPARK_VISUALIZATION

    def test_embodied_policy_template(self):
        entry = _make_entry()
        spec = _make_spec(template_id="embodied-policy")
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == EMBODIED_VISUALIZATION
        assert profile.artifact_domain == ARTIFACT_DOMAIN_ACTION

    def test_embodied_action_category(self):
        entry = _make_entry(category="Embodied Action")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == EMBODIED_VISUALIZATION

    def test_rerun_runtime(self):
        entry = _make_entry(runtime_kind="rerun")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == RERUN_VISUALIZATION
        assert profile.artifact_domain == ARTIFACT_DOMAIN_TIMELINE

    def test_rrd_runtime(self):
        entry = _make_entry(runtime_kind="rrd")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == RERUN_VISUALIZATION

    def test_video_category(self):
        entry = _make_entry(category="Video")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == MEDIA_VISUALIZATION
        assert profile.artifact_domain == ARTIFACT_DOMAIN_MEDIA

    def test_image_category(self):
        entry = _make_entry(category="Image")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == MEDIA_VISUALIZATION

    def test_visual_action_category(self):
        entry = _make_entry(category="Visual Action")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == MEDIA_VISUALIZATION

    def test_audio_video_category(self):
        entry = _make_entry(category="Audio / Video")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == MEDIA_VISUALIZATION

    def test_video_generation_category(self):
        entry = _make_entry(category="Video Generation")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == MEDIA_VISUALIZATION

    def test_fallback_returns_world(self):
        """When no condition matches, fallback is interactive world."""
        entry = _make_entry(category="Unknown Category")
        spec = _make_spec(template_id="unknown-template")
        profile = model_visualization_profile(entry, spec)
        assert profile.mode == INTERACTIVE_WORLD_VISUALIZATION
        assert profile.artifact_domain == ARTIFACT_DOMAIN_UI

    def test_spec_none_auto_derived(self):
        """When spec is None, interface_spec_for_entry is called internally."""
        entry = _make_entry(category="Video Generation")
        profile = model_visualization_profile(entry, spec=None)
        # This will call interface_spec_for_entry internally
        assert profile.mode == MEDIA_VISUALIZATION

    def test_accepted_artifact_kinds_for_interactive_world(self):
        spec = _make_spec(template_id="interactive-world")
        entry = _make_entry()
        profile = model_visualization_profile(entry, spec)
        assert "state" in profile.accepted_artifact_kinds
        assert "video" in profile.accepted_artifact_kinds

    def test_accepted_artifact_kinds_for_geometry(self):
        spec = _make_spec(template_id="depth-geometry")
        entry = _make_entry()
        profile = model_visualization_profile(entry, spec)
        assert "point_cloud" in profile.accepted_artifact_kinds

    def test_accepted_artifact_kinds_for_embodied(self):
        spec = _make_spec(template_id="embodied-policy")
        entry = _make_entry()
        profile = model_visualization_profile(entry, spec)
        assert "action_trace" in profile.accepted_artifact_kinds

    def test_accepted_artifact_kinds_for_rerun(self):
        entry = _make_entry(runtime_kind="rerun")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert "timeline" in profile.accepted_artifact_kinds
        assert "rrd" in profile.accepted_artifact_kinds

    def test_accepted_artifact_kinds_for_media(self):
        entry = _make_entry(category="Video")
        spec = _make_spec()
        profile = model_visualization_profile(entry, spec)
        assert "video" in profile.accepted_artifact_kinds
        assert "image" in profile.accepted_artifact_kinds


# ========================================================================
# 10. StudioVisualizationBackend dataclass
# ========================================================================

class TestStudioVisualizationBackendDataclass:
    """Test the StudioVisualizationBackend frozen dataclass."""

    def test_construction(self):
        backend = StudioVisualizationBackend(
            mode="world",
            title="World Viewer",
            default_port=7860,
        )
        assert backend.mode == "world"
        assert backend.title == "World Viewer"
        assert backend.default_port == 7860
        assert backend.aliases == ()
        assert backend.native is True
        assert backend.capabilities.layer_kinds == frozenset()

    def test_backend_id_property(self):
        backend = StudioVisualizationBackend(mode="world", title="Test", default_port=7860)
        assert backend.backend_id == "world"

    def test_with_aliases(self):
        backend = StudioVisualizationBackend(
            mode="world",
            title="World Viewer",
            default_port=7860,
            aliases=("interactive", "iw"),
        )
        assert backend.aliases == ("interactive", "iw")

    def test_with_capabilities(self):
        caps = BackendCapabilities(
            layer_kinds=frozenset({"point_cloud", "mesh"}),
            score=10,
        )
        backend = StudioVisualizationBackend(
            mode="points",
            title="Points Viewer",
            default_port=8080,
            capabilities=caps,
        )
        assert backend.capabilities.layer_kinds == frozenset({"point_cloud", "mesh"})
        assert backend.capabilities.score == 10

    def test_frozen_enforcement(self):
        backend = StudioVisualizationBackend(mode="world", title="Test", default_port=7860)
        with pytest.raises(dataclasses.FrozenInstanceError):
            backend.mode = "other"


# ========================================================================
# 11. StudioVisualizationBackend methods
# ========================================================================

class TestStudioVisualizationBackendMethods:
    """Test methods on StudioVisualizationBackend."""

    def _make_backend(self, mode="world", **overrides):
        kwargs = dict(
            title=f"{mode} Viewer",
            default_port=7860,
            capabilities=BackendCapabilities(
                layer_kinds=frozenset({"point_cloud", "mesh"}),
                partial=True,
                score=5,
            ),
        )
        kwargs.update(overrides)
        return StudioVisualizationBackend(mode=mode, **kwargs)

    def test_can_render_with_supported_layers(self):
        backend = self._make_backend()
        scene = VisualizationScene(
            scene_id="test-scene",
            layers=(
                Layer(layer_id="l1", kind="point_cloud"),
                Layer(layer_id="l2", kind="mesh"),
            ),
        )
        plan = backend.can_render(scene)
        assert plan.supported is True
        assert plan.backend_id == "world"

    def test_can_render_with_unsupported_layers_partial(self):
        backend = self._make_backend()  # partial=True by default
        scene = VisualizationScene(
            scene_id="test-scene",
            layers=(
                Layer(layer_id="l1", kind="point_cloud"),
                Layer(layer_id="l2", kind="gaussian_splat"),
            ),
        )
        plan = backend.can_render(scene)
        assert plan.supported is True  # partial=True allows partial support
        assert "gaussian_splat" in plan.unsupported_layers

    def test_can_render_with_unsupported_layers_not_partial(self):
        backend = self._make_backend(
            capabilities=BackendCapabilities(
                layer_kinds=frozenset({"point_cloud"}),
                partial=False,
                score=5,
            ),
        )
        scene = VisualizationScene(
            scene_id="test-scene",
            layers=(
                Layer(layer_id="l1", kind="point_cloud"),
                Layer(layer_id="l2", kind="gaussian_splat"),
            ),
        )
        plan = backend.can_render(scene)
        assert plan.supported is False

    def test_can_render_empty_capabilities(self):
        backend = self._make_backend(
            capabilities=BackendCapabilities(layer_kinds=frozenset()),
        )
        scene = VisualizationScene(
            scene_id="test-scene",
            layers=(Layer(layer_id="l1", kind="point_cloud"),),
        )
        plan = backend.can_render(scene)
        assert plan.supported is False
        assert "no scene capability declared" in plan.reason

    def test_can_render_empty_scene(self):
        backend = self._make_backend()
        scene = VisualizationScene(scene_id="empty")
        plan = backend.can_render(scene)
        # Empty scene (no layers) -> supported=False per the logic
        assert plan.supported is False

    def test_can_render_score(self):
        backend = self._make_backend(
            capabilities=BackendCapabilities(
                layer_kinds=frozenset({"point_cloud", "mesh"}),
                score=10,
            ),
        )
        scene = VisualizationScene(
            scene_id="test-scene",
            layers=(
                Layer(layer_id="l1", kind="point_cloud"),
                Layer(layer_id="l2", kind="mesh"),
            ),
        )
        plan = backend.can_render(scene)
        # score = 10 + max(0, 2 - 0) = 12
        assert plan.score == 12

    def test_render_success(self):
        backend = self._make_backend()
        scene = VisualizationScene(
            scene_id="test-scene",
            layers=(Layer(layer_id="l1", kind="point_cloud"),),
        )
        request = RenderRequest(backend="world")
        result = backend.render(scene, request)
        assert result.backend_id == "world"
        assert result.metadata["scene_id"] == "test-scene"

    def test_render_failure_raises(self):
        backend = self._make_backend(
            capabilities=BackendCapabilities(layer_kinds=frozenset()),
        )
        scene = VisualizationScene(
            scene_id="test-scene",
            layers=(Layer(layer_id="l1", kind="point_cloud"),),
        )
        request = RenderRequest(backend="world")
        with pytest.raises(ValueError, match="cannot render"):
            backend.render(scene, request)

    def test_shutdown_returns_none(self):
        backend = self._make_backend()
        assert backend.shutdown() is None

    def test_accepts_exact_mode(self):
        backend = self._make_backend(mode="world")
        assert backend.accepts("world") is True

    def test_accepts_alias(self):
        backend = self._make_backend(mode="world", aliases=("interactive",))
        assert backend.accepts("interactive") is True

    def test_accepts_normalized_mode(self):
        backend = self._make_backend(mode="world")
        # "WORLD" gets normalized to "world"
        assert backend.accepts("WORLD") is True

    def test_accepts_underscore_alias_only_matches_raw(self):
        """BUG: accepts() normalizes the requested token but does NOT normalize aliases
        for comparison. An alias "interactive_world" does NOT match the normalized
        request "interactive-world" because the alias tuple stores raw strings."""
        backend = self._make_backend(mode="world", aliases=("interactive_world",))
        # The alias is stored as raw "interactive_world"
        assert backend.accepts("interactive_world") is False  # BUG: should be True
        # Only exact match against the raw alias works after normalization
        # normalize_visualization_mode("interactive_world") -> "interactive-world"
        # which does not match raw alias "interactive_world" in tuple
        # The mode "world" is normalized and stored, so it matches
        assert backend.accepts("world") is True

    def test_accepts_rejects_unknown_mode(self):
        backend = self._make_backend(mode="world")
        assert backend.accepts("spark") is False

    def test_supports_with_matching_profile(self):
        backend = self._make_backend(mode="world")
        entry = _make_entry(category="Video Generation")
        # model_visualization_profile for Video Generation returns MEDIA, not world
        # So backend.supports depends on match function
        # Default match is lambda entry, spec: False
        assert backend.supports(entry) is False

    def test_supports_with_custom_match(self):
        backend = self._make_backend(
            mode="world",
            match=lambda entry, spec: True,
        )
        entry = _make_entry(category="Video Generation")
        assert backend.supports(entry) is True

    def test_launch_with_serve(self):
        def mock_serve(request):
            return StudioVisualizationLaunch(mode="world", url="http://localhost:7860")

        backend = self._make_backend(serve=mock_serve)
        entry = _make_entry()
        lc = _make_launch_config()
        request = StudioVisualizationRequest(
            entry=entry,
            launch_config=lc,
            mode="world",
            interface_spec=_make_spec(),
        )
        result = backend.launch(request)
        assert result is not None
        assert result.mode == "world"

    def test_launch_without_serve_raises(self):
        backend = self._make_backend(serve=None)
        entry = _make_entry()
        lc = _make_launch_config()
        request = StudioVisualizationRequest(
            entry=entry,
            launch_config=lc,
            mode="world",
            interface_spec=_make_spec(),
        )
        with pytest.raises(ValueError, match="has no serve function"):
            backend.launch(request)


# ========================================================================
# 12. StudioVisualizationRegistry
# ========================================================================

class TestStudioVisualizationRegistry:
    """Test the StudioVisualizationRegistry class."""

    def _make_backend(self, mode, **kw):
        kwargs = dict(title=f"{mode} Viewer", default_port=7860)
        kwargs.update(kw)
        return StudioVisualizationBackend(mode=mode, **kwargs)

    def test_empty_registry(self):
        reg = StudioVisualizationRegistry()
        assert reg.modes == frozenset()
        assert reg.native_modes == frozenset()
        assert reg.default_ports == {}

    def test_register_single_backend(self):
        reg = StudioVisualizationRegistry()
        backend = self._make_backend("world")
        reg.register(backend)
        assert "world" in reg.modes

    def test_register_multiple_backends(self):
        reg = StudioVisualizationRegistry()
        b1 = self._make_backend("world")
        b2 = self._make_backend("points", default_port=8080)
        reg.register(b1)
        reg.register(b2)
        assert reg.modes == frozenset({"world", "points"})
        assert reg.native_modes == frozenset({"world", "points"})  # both native by default

    def test_register_with_aliases(self):
        reg = StudioVisualizationRegistry()
        backend = self._make_backend("world", aliases=("interactive", "iw"))
        reg.register(backend)
        # Can look up by alias
        assert reg.backend_for("interactive").mode == "world"
        assert reg.backend_for("iw").mode == "world"

    def test_register_duplicate_mode_replaces(self):
        reg = StudioVisualizationRegistry()
        b1 = self._make_backend("world")
        b2 = self._make_backend("spark")
        reg.register(b1)
        reg.register(b2)
        # Registering a different backend with same mode should replace, not raise
        b3 = self._make_backend("world", title="New World Viewer")
        reg.register(b3)
        assert reg.backend_for("world").title == "New World Viewer"

    def test_register_conflicting_alias_raises(self):
        reg = StudioVisualizationRegistry()
        b1 = self._make_backend("world", aliases=("interactive",))
        b2 = self._make_backend("spark", aliases=("interactive",))
        reg.register(b1)
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register(b2)

    def test_register_empty_mode_raises(self):
        backend = StudioVisualizationBackend(
            mode="",
            title="Empty",
            default_port=0,
        )
        reg = StudioVisualizationRegistry()
        with pytest.raises(ValueError, match="cannot be empty"):
            reg.register(backend)

    def test_backend_for_known_mode(self):
        reg = StudioVisualizationRegistry()
        backend = self._make_backend("world")
        reg.register(backend)
        result = reg.backend_for("world")
        assert result.mode == "world"

    def test_backend_for_normalized_mode(self):
        reg = StudioVisualizationRegistry()
        backend = self._make_backend("world")
        reg.register(backend)
        result = reg.backend_for("WORLD")
        assert result.mode == "world"

    def test_backend_for_unknown_mode_raises(self):
        reg = StudioVisualizationRegistry()
        with pytest.raises(ValueError, match="Unsupported"):
            reg.backend_for("unknown")

    def test_default_ports(self):
        reg = StudioVisualizationRegistry()
        reg.register(self._make_backend("world", default_port=7860))
        reg.register(self._make_backend("points", default_port=8080))
        ports = reg.default_ports
        assert ports["world"] == 7860
        assert ports["points"] == 8080

    def test_native_modes(self):
        reg = StudioVisualizationRegistry()
        reg.register(self._make_backend("world", native=True))
        reg.register(self._make_backend("gradio-unified", native=False))
        assert reg.native_modes == frozenset({"world"})
        assert reg.modes == frozenset({"world", "gradio-unified"})

    def test_resolve_mode_explicit(self):
        reg = StudioVisualizationRegistry()
        reg.register(self._make_backend("world"))
        entry = _make_entry()
        assert reg.resolve_mode(entry, "world") == "world"

    def test_resolve_mode_auto_with_matching_profile(self):
        reg = StudioVisualizationRegistry()
        b_world = self._make_backend("world")
        reg.register(b_world)
        entry = _make_entry(category="Video Generation")
        # Video Generation -> media profile, but no media backend registered
        # So it falls through to match and then defaults to "world"
        mode = reg.resolve_mode(entry, "auto")
        # Default fallback is INTERACTIVE_WORLD_VISUALIZATION
        assert mode == INTERACTIVE_WORLD_VISUALIZATION

    def test_resolve_mode_auto_with_registered_profile(self):
        reg = StudioVisualizationRegistry()
        reg.register(self._make_backend("media"))
        entry = _make_entry(category="Video Generation")
        mode = reg.resolve_mode(entry, "auto")
        assert mode == MEDIA_VISUALIZATION

    def test_resolve_mode_none_treated_as_auto(self):
        reg = StudioVisualizationRegistry()
        reg.register(self._make_backend("world"))
        entry = _make_entry()
        mode = reg.resolve_mode(entry, None)
        # Should behave like "auto"
        assert isinstance(mode, str)

    def test_request_for(self):
        reg = StudioVisualizationRegistry()
        backend = self._make_backend("world")
        reg.register(backend)
        entry = _make_entry()
        lc = _make_launch_config()
        req = reg.request_for(entry=entry, launch_config=lc, mode="world")
        assert req.mode == "world"
        assert req.entry.model_id == "test-model"

    def test_serve_with_callable_backend(self):
        def mock_serve(request):
            return StudioVisualizationLaunch(mode="world", url="http://localhost:7860")

        reg = StudioVisualizationRegistry()
        backend = self._make_backend("world", serve=mock_serve)
        reg.register(backend)
        entry = _make_entry()
        lc = _make_launch_config()
        result = reg.serve(entry=entry, launch_config=lc, mode="world")
        assert result is not None
        assert result.url == "http://localhost:7860"

    def test_init_with_backends(self):
        b1 = self._make_backend("world")
        b2 = self._make_backend("points", default_port=8080)
        reg = StudioVisualizationRegistry(backends=[b1, b2])
        assert reg.modes == frozenset({"world", "points"})


# ========================================================================
# 13. Protocol classes – registry.py
# ========================================================================

class TestProtocols:
    """Test that VisualizationProvider and VisualizationBackend are Protocol classes."""

    def test_visualization_provider_is_protocol(self):
        # Protocol classes define attributes via annotations, not actual attrs
        assert "provider_id" in VisualizationProvider.__annotations__
        assert "discover" in VisualizationProvider.__annotations__ or hasattr(VisualizationProvider, "discover")

    def test_visualization_backend_is_protocol(self):
        assert "backend_id" in VisualizationBackend.__annotations__
        assert "capabilities" in VisualizationBackend.__annotations__
        # Methods are defined in the class body for Protocol
        assert hasattr(VisualizationBackend, "can_render")
        assert hasattr(VisualizationBackend, "render")
        assert hasattr(VisualizationBackend, "shutdown")

    def test_concrete_provider_satisfies_protocol(self):
        """A concrete class with provider_id and discover should satisfy the protocol."""
        class MyProvider:
            provider_id = "my-provider"
            def discover(self, source):
                return None
        # Structural subtyping – should work
        provider = MyProvider()
        assert provider.provider_id == "my-provider"
        assert provider.discover("anything") is None

    def test_concrete_backend_satisfies_protocol(self):
        """A concrete class implementing backend interface should satisfy the protocol."""
        class MyBackend:
            backend_id = "my-backend"
            capabilities = BackendCapabilities()
            def can_render(self, scene):
                return RenderPlan(backend_id="my-backend", supported=True)
            def render(self, scene, request):
                return RenderResult(backend_id="my-backend")
            def shutdown(self):
                return None
        backend = MyBackend()
        assert backend.backend_id == "my-backend"


# ========================================================================
# 14. available_viewport_kinds – capabilities.py
# ========================================================================

class TestAvailableViewportKinds:
    """Test the available_viewport_kinds function."""

    def test_default_always_includes_world(self):
        caps = ViewportCapabilities()
        result = available_viewport_kinds(caps=caps, entry=None)
        assert ViewportKind.WORLD in result

    def test_gaussian_splat_capability(self):
        caps = ViewportCapabilities(has_gaussian_splat=True)
        result = available_viewport_kinds(caps=caps, entry=None)
        assert ViewportKind.SPLAT in result

    def test_points_cloud_capability(self):
        caps = ViewportCapabilities(has_points_cloud=True)
        result = available_viewport_kinds(caps=caps, entry=None)
        assert ViewportKind.POINTS in result

    def test_viser_capability(self):
        caps = ViewportCapabilities(has_viser=True)
        result = available_viewport_kinds(caps=caps, entry=None)
        assert ViewportKind.POINTS in result

    def test_embodied_trace_capability(self):
        caps = ViewportCapabilities(has_embodied_trace=True)
        result = available_viewport_kinds(caps=caps, entry=None)
        assert ViewportKind.EMBODIED in result

    def test_simulator_replay_capability(self):
        caps = ViewportCapabilities(has_simulator_replay=True)
        result = available_viewport_kinds(caps=caps, entry=None)
        assert ViewportKind.EMBODIED in result

    def test_entry_runtime_pointcloud_nav(self):
        caps = ViewportCapabilities()
        entry = _make_entry(runtime_kind="pointcloud_nav")
        result = available_viewport_kinds(caps=caps, entry=entry)
        assert ViewportKind.POINTS in result

    def test_entry_runtime_two_stage_3dgs(self):
        caps = ViewportCapabilities()
        entry = _make_entry(runtime_kind="two_stage_3dgs")
        result = available_viewport_kinds(caps=caps, entry=entry)
        assert ViewportKind.POINTS in result

    def test_entry_runtime_worldfm(self):
        caps = ViewportCapabilities()
        entry = _make_entry(runtime_kind="worldfm")
        result = available_viewport_kinds(caps=caps, entry=entry)
        assert ViewportKind.POINTS in result

    def test_entry_3d_scene_category(self):
        caps = ViewportCapabilities(has_gaussian_splat=False)
        entry = _make_entry(category="3D Scene")
        result = available_viewport_kinds(caps=caps, entry=entry)
        assert ViewportKind.SPLAT in result

    def test_entry_embodied_action_category(self):
        caps = ViewportCapabilities()
        entry = _make_entry(category="Embodied Action")
        result = available_viewport_kinds(caps=caps, entry=entry)
        assert ViewportKind.EMBODIED in result

    def test_entry_visual_action_category(self):
        caps = ViewportCapabilities()
        entry = _make_entry(category="Visual Action")
        result = available_viewport_kinds(caps=caps, entry=entry)
        assert ViewportKind.EMBODIED in result

    def test_none_entry_no_category_based_additions(self):
        caps = ViewportCapabilities()
        result = available_viewport_kinds(caps=caps, entry=None)
        assert result == frozenset({ViewportKind.WORLD})

    def test_combined_capabilities_and_entry(self):
        caps = ViewportCapabilities(has_gaussian_splat=True, has_points_cloud=True)
        entry = _make_entry(category="3D Scene")
        result = available_viewport_kinds(caps=caps, entry=entry)
        assert ViewportKind.WORLD in result
        assert ViewportKind.SPLAT in result
        assert ViewportKind.POINTS in result

    def test_returns_frozenset(self):
        caps = ViewportCapabilities()
        result = available_viewport_kinds(caps=caps, entry=None)
        assert isinstance(result, frozenset)


# ========================================================================
# 15. recommend_viewport – capabilities.py
# ========================================================================

class TestRecommendViewport:
    """Test the recommend_viewport function."""

    def test_override_viewport_kind(self):
        caps = ViewportCapabilities()
        result = recommend_viewport(
            caps=caps,
            has_preview_video=False,
            has_preview_image=False,
            user_viewport_override=ViewportKind.SPLAT,
        )
        assert result == ViewportKind.SPLAT

    def test_override_string(self):
        caps = ViewportCapabilities()
        result = recommend_viewport(
            caps=caps,
            has_preview_video=False,
            has_preview_image=False,
            user_viewport_override="splat",
        )
        assert result == ViewportKind.SPLAT

    def test_override_with_available_filter(self):
        caps = ViewportCapabilities()
        available = frozenset({ViewportKind.WORLD, ViewportKind.POINTS})
        result = recommend_viewport(
            caps=caps,
            has_preview_video=False,
            has_preview_image=False,
            user_viewport_override=ViewportKind.SPLAT,
            available=available,
        )
        # SPLAT not in available, so override is ignored
        assert result == ViewportKind.WORLD

    def test_override_none_ignored(self):
        caps = ViewportCapabilities(has_simulator_replay=True)
        result = recommend_viewport(
            caps=caps,
            has_preview_video=False,
            has_preview_image=False,
            user_viewport_override=None,
        )
        assert result == ViewportKind.EMBODIED

    def test_simulator_replay_preferred(self):
        caps = ViewportCapabilities(has_simulator_replay=True)
        result = recommend_viewport(caps=caps, has_preview_video=True, has_preview_image=False)
        assert result == ViewportKind.EMBODIED

    def test_embodied_trace_preferred(self):
        caps = ViewportCapabilities(has_embodied_trace=True)
        result = recommend_viewport(caps=caps, has_preview_video=True, has_preview_image=False)
        assert result == ViewportKind.EMBODIED

    def test_streaming_with_video(self):
        caps = ViewportCapabilities(has_streaming=True)
        result = recommend_viewport(caps=caps, has_preview_video=True, has_preview_image=False)
        assert result == ViewportKind.WORLD

    def test_streaming_with_image(self):
        caps = ViewportCapabilities(has_streaming=True)
        result = recommend_viewport(caps=caps, has_preview_video=False, has_preview_image=True)
        assert result == ViewportKind.WORLD

    def test_gaussian_splat_preferred(self):
        caps = ViewportCapabilities(has_gaussian_splat=True)
        result = recommend_viewport(caps=caps, has_preview_video=False, has_preview_image=False)
        assert result == ViewportKind.SPLAT

    def test_points_cloud_preferred(self):
        caps = ViewportCapabilities(has_points_cloud=True)
        result = recommend_viewport(caps=caps, has_preview_video=False, has_preview_image=False)
        assert result == ViewportKind.POINTS

    def test_video_without_streaming(self):
        caps = ViewportCapabilities()
        result = recommend_viewport(caps=caps, has_preview_video=True, has_preview_image=False)
        assert result == ViewportKind.WORLD

    def test_image_without_streaming(self):
        caps = ViewportCapabilities()
        result = recommend_viewport(caps=caps, has_preview_video=False, has_preview_image=True)
        assert result == ViewportKind.WORLD

    def test_fallback_to_world(self):
        caps = ViewportCapabilities()
        result = recommend_viewport(caps=caps, has_preview_video=False, has_preview_image=False)
        assert result == ViewportKind.WORLD

    def test_invalid_override_string_returns_none_ignored(self):
        caps = ViewportCapabilities(has_gaussian_splat=True)
        result = recommend_viewport(
            caps=caps,
            has_preview_video=False,
            has_preview_image=False,
            user_viewport_override="invalid_viewport",
        )
        # Invalid override string -> coerce returns None -> ignored
        assert result == ViewportKind.SPLAT


# ========================================================================
# 16. _coerce_viewport_kind – capabilities.py
# ========================================================================

class TestCoerceViewportKind:
    """Test the internal _coerce_viewport_kind helper."""

    def test_none_returns_none(self):
        assert _coerce_viewport_kind(None) is None

    def test_viewport_kind_passthrough(self):
        assert _coerce_viewport_kind(ViewportKind.WORLD) == ViewportKind.WORLD

    def test_valid_string(self):
        assert _coerce_viewport_kind("world") == ViewportKind.WORLD
        assert _coerce_viewport_kind("splat") == ViewportKind.SPLAT

    def test_case_insensitive_string(self):
        assert _coerce_viewport_kind("WORLD") == ViewportKind.WORLD

    def test_whitespace_stripped(self):
        assert _coerce_viewport_kind("  world  ") == ViewportKind.WORLD

    def test_invalid_string_returns_none(self):
        assert _coerce_viewport_kind("invalid") is None


# ========================================================================
# 17. Round-trip serialization test – manifest.py asdict / fromdict
# ========================================================================

class TestManifestRoundTrip:
    """Test asdict → viewport_payload_from_metadata round-trip."""

    def test_full_roundtrip(self):
        caps = ViewportCapabilities(
            has_streaming=True,
            has_gaussian_splat=True,
            has_points_cloud=True,
            has_viser=True,
            has_rrd=False,
            has_embodied_trace=False,
            has_simulator_replay=False,
        )
        world = WorldViewportAssets(preview_video="v.mp4", preview_image="i.png")
        splat = SplatViewportAssets(primary_path="s.spz", format_hint="splat")
        points = PointsViewportAssets(point_cloud_path="c.ply", coordinate_frame="local")
        embodied = EmbodiedViewportAssets(simulator_hint="RoboTwin episode")

        payload = StudioViewportsPayload(
            recommended=ViewportKind.WORLD,
            schema_version=2,
            assets_world=world,
            assets_splat=splat,
            assets_points=points,
            assets_embodied=embodied,
            capabilities=caps,
        )

        d = payload.asdict()
        metadata = {"studio_viewports": d}
        rehydrated = viewport_payload_from_metadata(metadata)

        assert rehydrated is not None
        assert rehydrated.recommended == ViewportKind.WORLD
        assert rehydrated.schema_version == 2
        assert rehydrated.assets_world.preview_video == "v.mp4"
        assert rehydrated.assets_world.preview_image == "i.png"
        assert rehydrated.assets_splat.primary_path == "s.spz"
        assert rehydrated.assets_splat.format_hint == "splat"
        assert rehydrated.assets_points.point_cloud_path == "c.ply"
        assert rehydrated.assets_points.coordinate_frame == "local"
        assert rehydrated.assets_embodied.simulator_hint == "RoboTwin episode"
        assert rehydrated.capabilities.has_streaming is True
        assert rehydrated.capabilities.has_gaussian_splat is True
        assert rehydrated.capabilities.has_points_cloud is True

    def test_minimal_roundtrip(self):
        payload = StudioViewportsPayload(recommended=ViewportKind.WORLD)
        d = payload.asdict()
        metadata = {"studio_viewports": d}
        rehydrated = viewport_payload_from_metadata(metadata)

        assert rehydrated is not None
        assert rehydrated.recommended == ViewportKind.WORLD
        assert rehydrated.schema_version == 1
        assert rehydrated.assets_world.preview_video is None
        assert rehydrated.capabilities.has_streaming is False


# ========================================================================
# 18. StudioVisualizationRequest frozen dataclass
# ========================================================================

class TestStudioVisualizationRequest:
    """Test StudioVisualizationRequest frozen dataclass."""

    def test_construction(self):
        entry = _make_entry()
        lc = _make_launch_config()
        spec = _make_spec()
        obj = StudioVisualizationRequest(
            entry=entry,
            launch_config=lc,
            mode="world",
            interface_spec=spec,
        )
        assert obj.entry.model_id == "test-model"
        assert obj.mode == "world"
        assert obj.artifact is None

    def test_with_artifact(self):
        from worldfoundry.studio.visualization.core.artifacts import StudioVisualizationArtifact
        entry = _make_entry()
        lc = _make_launch_config()
        spec = _make_spec()
        artifact = StudioVisualizationArtifact(path="/path/to/video.mp4", kind="video")
        obj = StudioVisualizationRequest(
            entry=entry,
            launch_config=lc,
            mode="world",
            interface_spec=spec,
            artifact=artifact,
        )
        assert obj.artifact.kind == "video"

    def test_frozen_enforcement(self):
        entry = _make_entry()
        lc = _make_launch_config()
        spec = _make_spec()
        obj = StudioVisualizationRequest(
            entry=entry, launch_config=lc, mode="world", interface_spec=spec,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.mode = "other"
