# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""DPT fusion contracts preserve residual routing, normalization and released keys."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.base_models.three_dimensions.depth.midas.fusion import (
    FeatureFusionBlock,
    GroupedFeatureFusionBlock,
    GroupedResidualConvUnit,
    ResidualConvUnit,
    SlamFeatureFusionBlock,
    VideoFeatureFusionBlock,
    make_scratch,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.heads.dpt_head import DPTHead

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("batchnorm", [False, True])
def test_standard_and_slam_fusion_preserve_different_residual_bindings(batchnorm: bool) -> None:
    standard = FeatureFusionBlock(4, nn.ReLU(), bn=batchnorm)
    slam = SlamFeatureFusionBlock(4, nn.ReLU(), bn=batchnorm)
    assert isinstance(standard.resConfUnit1, ResidualConvUnit)
    assert isinstance(slam.resConfUnit1, GroupedResidualConvUnit)
    assert ("resConfUnit1.bn1.running_mean" in standard.state_dict()) is batchnorm
    assert not any("bn1" in key for key in slam.state_dict())
    assert slam.resConfUnit1.norm1 is None


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("has_residual", [False, True])
def test_grouped_fusion_keeps_parameter_registration_and_strict_reload(groups: int, has_residual: bool) -> None:
    torch.manual_seed(7)
    block = GroupedFeatureFusionBlock(4, nn.ReLU(), groups=groups, has_residual=has_residual).eval()
    copy = GroupedFeatureFusionBlock(4, nn.ReLU(), groups=groups, has_residual=has_residual).eval()
    copy.load_state_dict(block.state_dict(), strict=True)
    assert hasattr(block, "resConfUnit1") is has_residual
    assert block.resConfUnit2.conv1.groups == groups
    x = torch.randn(2, 4, 5, 7)
    xs = (x, torch.randn_like(x)) if has_residual else (x,)
    torch.testing.assert_close(copy(*xs, size=(8, 11)), block(*xs, size=(8, 11)), rtol=0, atol=0)


def test_video_fusion_makes_the_interpolation_input_contiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    contiguous: list[bool] = []
    interpolate = nn.functional.interpolate

    def record(
        x: torch.Tensor,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        contiguous.append(x.is_contiguous())
        return interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

    monkeypatch.setattr(nn.functional, "interpolate", record)
    x = torch.randn(2, 4, 5, 7).transpose(-1, -2)
    block = VideoFeatureFusionBlock(4, nn.ReLU())
    actual = block._interpolate(x, size=(9, 11))
    assert contiguous == [True]
    torch.testing.assert_close(
        actual, interpolate(x.contiguous(), size=(9, 11), mode="bilinear", align_corners=True), rtol=0, atol=0
    )


@pytest.mark.parametrize("count", [3, 4])
def test_make_scratch_keeps_released_layer_names_and_expanded_widths(count: int) -> None:
    scratch = make_scratch([4] * count, 4, groups=2, expand=True)
    assert list(scratch.state_dict()) == [f"layer{i + 1}_rn.weight" for i in range(count)]
    assert [module.out_channels for module in scratch.children()] == [4 * 2**i for i in range(count)]


def test_vggt_head_uses_shared_fusion_for_dense_predictions() -> None:
    torch.manual_seed(9)
    head = DPTHead(
        dim_in=8, features=4, out_channels=[4, 4, 8, 8], intermediate_layer_idx=[0, 1, 2, 3], pos_embed=False
    ).eval()
    assert isinstance(head.scratch.refinenet1, GroupedFeatureFusionBlock)
    assert not hasattr(head.scratch.refinenet4, "resConfUnit1")
    images = torch.randn(1, 2, 3, 28, 28)
    tokens = [torch.randn(1, 2, 4, 8) for _ in range(4)]
    result = head(tokens, images, patch_start_idx=0, frames_chunk_size=1)
    assert isinstance(result, tuple)
    assert result[0].shape == (1, 2, 28, 28, 3)
    assert result[1].shape == (1, 2, 28, 28)
