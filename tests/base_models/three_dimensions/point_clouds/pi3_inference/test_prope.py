# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.networks.sana import sana_camctrl_blocks
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm import prope as minwm_prope
from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers import prope

pytestmark = pytest.mark.unit


def test_camera_consumers_use_pi3_block_diagonal_and_se3_algebra() -> None:
    assert sana_camctrl_blocks._apply_block_diagonal is prope._apply_block_diagonal
    assert sana_camctrl_blocks._invert_SE3 is prope._invert_SE3
    assert minwm_prope._apply_block_diagonal is prope._apply_block_diagonal


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
def test_prope_block_diagonal_and_se3_preserve_dtype(dtype: torch.dtype) -> None:
    pose = torch.eye(4, dtype=dtype)[None]
    pose[:, :3, 3] = torch.tensor([2.0, -1.0, 3.0], dtype=dtype)
    inverse = prope._invert_SE3(pose)
    assert inverse.dtype == dtype
    torch.testing.assert_close(inverse[:, :3, 3], -pose[:, :3, 3], rtol=0, atol=0)
    assert torch.equal(inverse[:, 3], pose[:, 3])
    x = torch.arange(8, dtype=dtype)[None]
    actual = prope._apply_block_diagonal(x, [(lambda part: part + 1, 4), (lambda part: -part, 4)])
    torch.testing.assert_close(actual, torch.cat([x[:, :4] + 1, -x[:, 4:]], dim=-1), rtol=0, atol=0)
