# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from MiDaS / Depth Anything (MIT / Apache-2.0) and VGGT (Apache-2.0).
# Copyright (c) Intel ISL, Depth Anything authors, Meta Platforms, Inc. and affiliates.
"""DPT fusion leaves shared by depth and point-cloud reconstruction heads."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor


def make_scratch(in_shape: Sequence[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features: int, activation: nn.Module, bn: bool) -> None:
        """
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    residual_unit: type[nn.Module] = ResidualConvUnit

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: int | tuple[int, int] | None = None,
    ) -> None:
        """
        Args:
            features (int): number of features
        """
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = self.residual_unit(features, activation, bn)
        self.resConfUnit2 = self.residual_unit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs: Tensor, size: int | tuple[int, int] | None = None) -> Tensor:
        """
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = self._interpolate(output, **modifier)

        output = self.out_conv(output)

        return output

    def _interpolate(
        self, x: Tensor, size: int | tuple[int, int] | None = None, scale_factor: float | None = None
    ) -> Tensor:
        return nn.functional.interpolate(
            x, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=self.align_corners
        )


class GroupedResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1) -> None:
        """
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class GroupedFeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: int | tuple[int, int] | None = None,
        has_residual: bool = True,
        groups: int = 1,
    ) -> None:
        """
        Args:
            features (int): number of features
        """
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:  # noqa: E712 - Keeps `==` for operands that may compare elementwise.
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = GroupedResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = GroupedResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs: Tensor, size: int | tuple[int, int] | None = None) -> Tensor:
        """
        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        scale_factor = cast(float, scale_factor)
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor: float, mode: str, align_corners: bool = False) -> None:
        """
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super().__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        return x


class SlamFeatureFusionBlock(FeatureFusionBlock):
    """LingBot fusion keeps depth-style skip routing and the VGGT residual-unit binding."""

    residual_unit = GroupedResidualConvUnit


class VideoFeatureFusionBlock(FeatureFusionBlock):
    """Video Depth Anything interpolates contiguous features."""

    def _interpolate(
        self, x: Tensor, size: int | tuple[int, int] | None = None, scale_factor: float | None = None
    ) -> Tensor:
        return super()._interpolate(x.contiguous(), size=size, scale_factor=scale_factor)


def make_fusion_block(features: int, use_bn: bool, size: int | tuple[int, int] | None = None) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


def make_grouped_fusion_block(
    features: int, size: int | None = None, has_residual: bool = True, groups: int = 1
) -> nn.Module:
    return GroupedFeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def make_video_fusion_block(features: int, use_bn: bool, size: int | tuple[int, int] | None = None) -> nn.Module:
    return VideoFeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


def make_slam_fusion_block(features: int, use_bn: bool, size: int | tuple[int, int] | None = None) -> nn.Module:
    return SlamFeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )
