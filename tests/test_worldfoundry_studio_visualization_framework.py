from __future__ import annotations

from pathlib import Path

from worldfoundry.studio.inference.catalog import find_entry
from worldfoundry.studio.visualization.backends.frontends import (
    DEFAULT_FRONTEND_PORTS,
    MEDIA_FRONTEND,
    NATIVE_FRONTENDS,
    POINTS_FRONTEND,
    RERUN_FRONTEND,
    SPARK_FRONTEND,
    STUDIO_VISUALIZATIONS,
    WORLD_FRONTEND,
    media_viewer_html,
    resolve_frontend_mode,
)
from worldfoundry.studio.ui.launch_config import CLI_FRONTEND_CHOICES, StudioLaunchConfig
from worldfoundry.studio.visualization.core.registry import (
    ARTIFACT_DOMAIN_ACTION,
    ARTIFACT_DOMAIN_GAUSSIAN_SPLAT,
    ARTIFACT_DOMAIN_GEOMETRY,
    ARTIFACT_DOMAIN_MEDIA,
    ARTIFACT_DOMAIN_WORLD,
    StudioVisualizationArtifact,
    StudioVisualizationBackend,
    StudioVisualizationRegistry,
    infer_visualization_artifact,
    model_visualization_profile,
)


def test_builtin_visualization_registry_routes_core_frontends() -> None:
    assert WORLD_FRONTEND in NATIVE_FRONTENDS
    assert POINTS_FRONTEND in NATIVE_FRONTENDS
    assert SPARK_FRONTEND in NATIVE_FRONTENDS
    assert MEDIA_FRONTEND in NATIVE_FRONTENDS
    assert RERUN_FRONTEND in NATIVE_FRONTENDS
    assert DEFAULT_FRONTEND_PORTS[WORLD_FRONTEND] == 7868
    assert DEFAULT_FRONTEND_PORTS[POINTS_FRONTEND] == 18590
    assert DEFAULT_FRONTEND_PORTS[SPARK_FRONTEND] == 8765
    assert DEFAULT_FRONTEND_PORTS[MEDIA_FRONTEND] == 18720
    assert DEFAULT_FRONTEND_PORTS[RERUN_FRONTEND] == 9876

    assert resolve_frontend_mode(find_entry("lingbot-world"), "auto") == WORLD_FRONTEND
    assert resolve_frontend_mode(find_entry("pi3"), "auto") == POINTS_FRONTEND
    assert resolve_frontend_mode(find_entry("vggt"), "auto") == SPARK_FRONTEND


def test_visualization_registry_accepts_public_aliases() -> None:
    entry = find_entry("lingbot-world")

    assert STUDIO_VISUALIZATIONS.resolve_mode(entry, "interactive-world") == WORLD_FRONTEND
    assert STUDIO_VISUALIZATIONS.resolve_mode(entry, "viser") == POINTS_FRONTEND
    assert STUDIO_VISUALIZATIONS.resolve_mode(entry, "3dgs") == SPARK_FRONTEND
    assert STUDIO_VISUALIZATIONS.resolve_mode(entry, "preview") == MEDIA_FRONTEND
    assert STUDIO_VISUALIZATIONS.resolve_mode(entry, "rrd") == RERUN_FRONTEND
    assert {"viser", "interactive-world", "splat", "preview", "rrd"} <= CLI_FRONTEND_CHOICES


def test_model_visualization_profile_is_domain_based_not_folder_based() -> None:
    profiles = {
        "lingbot-world": model_visualization_profile(find_entry("lingbot-world")),
        "pi3": model_visualization_profile(find_entry("pi3")),
        "vggt": model_visualization_profile(find_entry("vggt")),
        "openvla": model_visualization_profile(find_entry("openvla")),
        "animatediff": model_visualization_profile(find_entry("animatediff")),
    }

    assert profiles["lingbot-world"].mode == WORLD_FRONTEND
    assert profiles["lingbot-world"].artifact_domain == ARTIFACT_DOMAIN_WORLD
    assert profiles["pi3"].mode == POINTS_FRONTEND
    assert profiles["pi3"].artifact_domain == ARTIFACT_DOMAIN_GEOMETRY
    assert profiles["vggt"].mode == SPARK_FRONTEND
    assert profiles["vggt"].artifact_domain == ARTIFACT_DOMAIN_GAUSSIAN_SPLAT
    assert profiles["openvla"].artifact_domain == ARTIFACT_DOMAIN_ACTION
    assert profiles["animatediff"].mode == MEDIA_FRONTEND
    assert profiles["animatediff"].artifact_domain == ARTIFACT_DOMAIN_MEDIA


def test_visualization_artifact_inference_is_shared_across_models(tmp_path: Path) -> None:
    cases = {
        "scene.ply": ("point_cloud", "ply"),
        "scene.splat": ("gaussian_splat", "splat"),
        "timeline.rrd": ("timeline", "rrd"),
        "preview.mp4": ("video", "mp4"),
        "preview.png": ("image", "png"),
        "sound.wav": ("audio", "wav"),
    }
    for name, expected in cases.items():
        artifact = infer_visualization_artifact(tmp_path / name)
        assert (artifact.kind, artifact.format_hint) == expected


def test_visualization_request_keeps_artifact_contract(tmp_path: Path) -> None:
    asset = tmp_path / "scene.splat"
    asset.write_bytes(b"stub")
    artifact = StudioVisualizationArtifact(path=str(asset), kind="gaussian_splat", format_hint="splat")
    request = STUDIO_VISUALIZATIONS.request_for(
        entry=find_entry("vggt"),
        launch_config=StudioLaunchConfig(model_id="vggt", frontend="spark", asset_path=str(asset)),
        mode="spark",
        artifact=artifact,
    )

    assert request.mode == SPARK_FRONTEND
    assert request.artifact is artifact
    assert request.artifact.resolved_path == asset.resolve()
    assert request.interface_spec.template_id


def test_custom_visualization_registry_is_lightweight_and_ordered() -> None:
    calls: list[str] = []
    registry = StudioVisualizationRegistry(
        (
            StudioVisualizationBackend(
                mode="alpha",
                title="Alpha",
                default_port=1001,
                aliases=("a",),
                match=lambda entry, spec: False,
                serve=lambda request: calls.append(request.mode),
            ),
            StudioVisualizationBackend(
                mode="beta",
                title="Beta",
                default_port=1002,
                match=lambda entry, spec: True,
            ),
        )
    )

    entry = find_entry("lingbot-world")
    assert registry.resolve_mode(entry, "a") == "alpha"
    assert registry.resolve_mode(entry, "auto") == "beta"
    registry.serve(entry=entry, launch_config=StudioLaunchConfig(model_id=entry.model_id), mode="alpha")
    assert calls == ["alpha"]


def test_media_viewer_html_selects_expected_media_tags(tmp_path: Path) -> None:
    image = tmp_path / "preview.png"
    video = tmp_path / "preview.mp4"
    audio = tmp_path / "preview.wav"

    assert '<img class="asset asset-image"' in media_viewer_html(title="Image", asset_path=image)
    assert '<video class="asset asset-video"' in media_viewer_html(title="Video", asset_path=video)
    assert '<audio class="asset asset-audio"' in media_viewer_html(title="Audio", asset_path=audio)
