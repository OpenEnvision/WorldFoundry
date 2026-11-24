# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""NWM and Vid2World retain their distinct live inference input transforms."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("module_name", "resolution"),
    [
        ("worldfoundry.synthesis.visual_generation.nwm.misc", 224),
        ("worldfoundry.synthesis.visual_generation.vid2world.nvm_utils.misc", 320),
    ],
)
def test_navigation_transform_keeps_family_resolution(module_name: str, resolution: int) -> None:
    misc = importlib.import_module(module_name)
    image = Image.fromarray(np.zeros((48, 80, 3), dtype=np.uint8))
    assert misc.CenterCropAR()(image).size == (64, 48)
    output = misc.transform(image)
    assert output.shape == (3, resolution, resolution)
    assert output.dtype == torch.float32
    assert torch.equal(output, -torch.ones_like(output))
    for name in (
        "get_action_torch",
        "calculate_delta_yaw",
        "plot_images_and_actions",
        "save_planning_pred",
        "log_viz_single",
    ):
        assert not hasattr(misc, name)


def test_nwm_output_unnormalizer_is_retained() -> None:
    from worldfoundry.synthesis.visual_generation.nwm.misc import unnormalize

    x = torch.tensor([-1.0, 0.0, 1.0]).reshape(3, 1, 1)
    torch.testing.assert_close(unnormalize(x), (x + 1) / 2, rtol=0, atol=0)
