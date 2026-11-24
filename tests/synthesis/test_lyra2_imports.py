# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.unit


def test_config_discovery_uses_installed_namespace_in_a_fresh_process() -> None:
    code = """
import sys
from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.configs.config import make_config
config = make_config()
assert config.defaults[1] == {"model": "fsdp_wan2pt1_lyra2_spatial"}
assert not any(name == "lyra_2" or name.startswith("lyra_2.") for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("direction", ["left", "right"])
def test_camera_builder_imports_packaged_helper_and_preserves_simple_trajectory(direction: str) -> None:
    from worldfoundry.synthesis.visual_generation.lyra_2.lyra_2._src.inference import camera_traj_utils, camera_utils

    assert camera_traj_utils.create_horizontal_trajectory is camera_utils.create_horizontal_trajectory
    pose, intrinsics = torch.eye(4), torch.eye(3)
    poses, cameras = camera_traj_utils.build_camera_trajectory(
        pose, intrinsics, 2.0, 5, "horizontal_simple", direction, 0.2
    )
    expected = pose.repeat(5, 1, 1)
    shift = torch.linspace(0.0, 0.2, 5) * 2.0
    expected[:, 0, 3] = -shift if direction == "right" else shift
    torch.testing.assert_close(poses, expected, rtol=0, atol=0)
    torch.testing.assert_close(cameras, intrinsics.repeat(5, 1, 1), rtol=0, atol=0)
