"""Keep Echo's compact camera controls on the published per-chunk scale."""

import inspect
import math
from pathlib import Path

import pytest
import yaml

from worldfoundry.base_models.diffusion_model.models.initializers.echo_memory_actions import (
    echo_camera_trajectory_actions,
)
from worldfoundry.synthesis.visual_generation.echo_memory.runtime import EchoMemoryRuntime


def test_default_forward_action_matches_published_translation_delta():
    actions = echo_camera_trajectory_actions("w*80", frame_count=81)
    assert actions.shape == (21, 12)
    assert actions[0, :3] == pytest.approx([0.0, 0.0, 0.0])
    assert actions[-1, :3] == pytest.approx([0.0, 0.1, 0.0])


def test_default_left_action_matches_published_rotation_delta():
    actions = echo_camera_trajectory_actions("left*80", frame_count=81)
    final_rotation = actions[-1, 3:].reshape(3, 3)
    assert math.degrees(math.atan2(final_rotation[1, 0], final_rotation[0, 0])) == pytest.approx(45.0)


def test_echo_runtime_and_catalog_share_the_action_default():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "worldfoundry/data/models/runtime/configs/echo_memory/runtime_defaults.yaml").read_text()
    )["defaults"]["echo-memory-context-k1"]
    runtime_default = inspect.signature(EchoMemoryRuntime.__init__).parameters[
        "camera_translation_step"
    ].default
    helper_default = inspect.signature(echo_camera_trajectory_actions).parameters[
        "translation_step"
    ].default
    assert config["camera_translation_step"] == runtime_default == helper_default == pytest.approx(0.00125)
    runtime_rotation = inspect.signature(EchoMemoryRuntime.__init__).parameters[
        "camera_rotation_step_degrees"
    ].default
    helper_rotation = inspect.signature(echo_camera_trajectory_actions).parameters[
        "rotation_step_degrees"
    ].default
    assert config["camera_rotation_step_degrees"] == runtime_rotation == helper_rotation == pytest.approx(0.5625)
