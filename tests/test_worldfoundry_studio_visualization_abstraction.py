from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from worldfoundry.studio.visualization.core import (
    BackendCapabilities,
    Layer,
    RenderRequest,
    StudioVisualizationBackend,
    Timeline,
    VisualizationScene,
)
from worldfoundry.studio.visualization.providers.geometry import GeometryProvider
from worldfoundry.studio.visualization.providers.media import MediaProvider


def test_visualization_scene_round_trips_manifest_shape() -> None:
    scene = VisualizationScene(
        scene_id="run/example",
        title="Example",
        layers=(
            Layer(layer_id="preview", kind="video", uri="outputs/world.mp4"),
            Layer(layer_id="points", kind="point_cloud", uri="outputs/scene.ply", coordinate_frame="world"),
        ),
        timeline=Timeline(fps=16, frame_count=32),
        recommended_backend="auto",
        metadata={"model_id": "matrix-game-1"},
    )

    restored = VisualizationScene.fromdict(scene.asdict())

    assert restored.scene_id == scene.scene_id
    assert restored.timeline is not None
    assert restored.timeline.fps == 16
    assert restored.layer_kinds() == {"video", "point_cloud"}
    assert restored.layers[1].coordinate_frame == "world"


def test_backend_capability_matching_scores_supported_layers() -> None:
    backend = StudioVisualizationBackend(
        mode="media",
        title="Media",
        default_port=18720,
        capabilities=BackendCapabilities(frozenset({"image", "video"}), score=10),
    )
    scene = VisualizationScene(
        scene_id="scene/media",
        layers=(Layer(layer_id="preview", kind="video", uri="preview.mp4"),),
    )

    plan = backend.can_render(scene)
    result = backend.render(scene, RenderRequest(backend="media"))

    assert plan.supported is True
    assert plan.score >= 11
    assert result.backend_id == "media"
    assert result.metadata["scene_id"] == "scene/media"


def test_provider_discovery_builds_backend_neutral_layers(tmp_path: Path) -> None:
    image = tmp_path / "preview.png"
    image.write_bytes(b"placeholder")
    ply = tmp_path / "scene.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0 0 0\n", encoding="ascii")

    media_scene = MediaProvider().discover(image)
    geometry_scene = GeometryProvider().discover(tmp_path)

    assert media_scene is not None
    assert media_scene.layers[0].kind == "image"
    assert geometry_scene is not None
    assert "point_cloud" in geometry_scene.layer_kinds()






def test_legacy_visualization_root_modules_are_removed() -> None:
    removed_modules = [
        "worldfoundry.studio.visualization_optical_flow",
        "worldfoundry.studio.visualization_projection",
        "worldfoundry.studio.visualization_pointcloud_sequence",
        "worldfoundry.studio.visualization_game_controls_v2",
        "worldfoundry.studio.visualization_video_utils",
        "worldfoundry.studio.visualization_metrics",
    ]

    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None

    assert importlib.util.find_spec(
        "worldfoundry.studio.visualization.plugins.perception.render"
    ) is not None


def test_legacy_core_shims_are_removed_in_favor_of_unified_modules() -> None:
    viewport_schema = importlib.import_module("worldfoundry.studio.visualization.core.manifest")
    viewport_router = importlib.import_module("worldfoundry.studio.visualization.core.capabilities")
    studio_root = Path(importlib.import_module("worldfoundry.studio").__file__).parent

    assert viewport_schema.__name__ == "worldfoundry.studio.visualization.core.manifest"
    assert viewport_router.__name__ == "worldfoundry.studio.visualization.core.capabilities"
    assert hasattr(viewport_schema, "StudioViewportsPayload")
    assert hasattr(viewport_router, "recommend_viewport")
    for module_name in (
        "frontends.py",
        "viser_host.py",
        "world_frontend.py",
        "viewport_schema.py",
        "viewport_router.py",
        "viewport_manifest.py",
        "viewport_discovery.py",
    ):
        assert not (studio_root / module_name).exists()
    assert importlib.import_module("worldfoundry.studio.visualization.backends.frontends").__name__ == (
        "worldfoundry.studio.visualization.backends.frontends"
    )
