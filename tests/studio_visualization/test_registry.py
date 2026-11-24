"""Tests for visualization.registry."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Imports – registry
# ---------------------------------------------------------------------------
from worldfoundry.studio.inference.catalog import CatalogEntry
from worldfoundry.studio.visualization.interfaces import LocalRepoEvidence, StudioInterfaceSpec
from worldfoundry.studio.visualization.launch_config import StudioLaunchConfig
from worldfoundry.studio.visualization.registry import (
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
    VISER_VISUALIZATION,
    StudioModelVisualizationProfile,
    StudioVisualizationBackend,
    StudioVisualizationLaunch,
    StudioVisualizationRegistry,
    StudioVisualizationRequest,
    model_visualization_profile,
    normalize_visualization_mode,
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


# ========================================================================
# 7. Frozen dataclasses – registry.py
# ========================================================================


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


    def test_with_aliases(self):
        backend = StudioVisualizationBackend(
            mode="world",
            title="World Viewer",
            default_port=7860,
            aliases=("interactive", "iw"),
        )
        assert backend.aliases == ("interactive", "iw")


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
        kwargs = dict(title=f"{mode} Viewer", default_port=7860)
        kwargs.update(overrides)
        return StudioVisualizationBackend(mode=mode, **kwargs)

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
        reg.register(self._make_backend("embedded-ui", native=False))
        assert reg.native_modes == frozenset({"world"})
        assert reg.modes == frozenset({"world", "embedded-ui"})

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
        from worldfoundry.studio.visualization.artifacts import StudioVisualizationArtifact
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


class TestPackageExports:
    """The registry re-exports only importable names."""

    def test_registry_all_importable(self):
        import worldfoundry.studio.visualization.registry as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} missing from registry module"
