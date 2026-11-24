from __future__ import annotations

import importlib
import importlib.util


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

    assert importlib.util.find_spec("worldfoundry.base_models.perception_core.visualization.render") is not None
