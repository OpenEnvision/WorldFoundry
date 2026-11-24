# SPDX-FileCopyrightText: 2026 Physical Intelligence Team, Shanghai AI Laboratory
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Dense iterative 2D point tracking branch for Pi4.

Adapted from ``yyfz/Pi4`` revision ``32dfbea9ce6268ea17446208a804710df9475c32``
(``pi4/models/heads/tracking.py``). Upstream code and weights are licensed under
CC BY-NC 4.0 (with CoWTracker-derived components subject to the FAIR Noncommercial
Research License as recorded in upstream ``NOTICE``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..layers import Block, ConvPyramid, feature_pyramid


def _framewise(
    operation: Callable[[torch.Tensor], torch.Tensor],
    features: torch.Tensor,
    chunk_size: int = 8,
) -> torch.Tensor:
    return torch.cat([operation(chunk) for chunk in features.split(chunk_size)])


class RGBResidual(nn.Module):
    """Residual 2D conv block for the half-resolution RGB feature encoder."""

    def __init__(self, input_channels: int, output_channels: int, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=kernel // 2),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
        )
        self.skip_conv: nn.Module = (
            nn.Conv2d(input_channels, output_channels, 1, stride=stride)
            if input_channels != output_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features) + self.skip_conv(features)


class RGBEncoder(nn.Module):
    """U-Net RGB encoder producing half-resolution 128-channel tracking features."""

    def __init__(self) -> None:
        super().__init__()
        self.ds1 = RGBResidual(3, 64, kernel=7, stride=2)
        self.conv1 = RGBResidual(64, 64)
        self.conv2 = RGBResidual(64, 128, stride=2)
        self.conv3 = RGBResidual(128, 256, stride=2)
        self.conv4 = RGBResidual(256, 512, stride=2)
        self.up_4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.proj_3 = RGBResidual(256, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.proj_2 = RGBResidual(128, 128)
        self.up_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.proj_1 = RGBResidual(64, 128)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        first = self.conv1(self.ds1(images))
        second = self.conv2(first)
        third = self.conv3(second)
        fourth = self.conv4(third)
        third = self.proj_3(third + self.up_4(fourth))
        second = self.proj_2(second + self.up_3(third))
        return self.proj_1(first + self.up_2(second))


class FusionResidual(nn.Module):
    """DPT-style residual conv unit."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.conv2(F.relu(self.conv1(F.relu(features))))


class FusionBlock(nn.Module):
    """DPT multi-scale feature fusion block."""

    def __init__(self, skip: bool = True) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(64, 64, 1)
        if skip:
            self.resConfUnit1 = FusionResidual()
        self.resConfUnit2 = FusionResidual()

    def forward(
        self,
        features: torch.Tensor,
        skip: torch.Tensor | None = None,
        size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            features = features + self.resConfUnit1(skip)
        features = self.resConfUnit2(features)
        features = F.interpolate(
            features,
            size=size,
            scale_factor=2 if size is None else None,
            mode="bilinear",
            align_corners=True,
        )
        return self.out_conv(features)


class _TrackingScratch(nn.Module):
    """Typed container matching the upstream ``scratch.*`` parameter namespace and registration order."""

    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.layer1_rn = nn.Conv2d(channels[0], 64, 3, padding=1, bias=False)
        self.refinenet1 = FusionBlock(skip=True)
        self.layer2_rn = nn.Conv2d(channels[1], 64, 3, padding=1, bias=False)
        self.refinenet2 = FusionBlock(skip=True)
        self.layer3_rn = nn.Conv2d(channels[2], 64, 3, padding=1, bias=False)
        self.refinenet3 = FusionBlock(skip=True)
        self.layer4_rn = nn.Conv2d(channels[3], 64, 3, padding=1, bias=False)
        self.refinenet4 = FusionBlock(skip=False)
        self.output_conv1 = nn.Conv2d(64, 32, 3, padding=1)


class TrackingPyramid(nn.Module):
    """DPT-style pyramid head decoding cross-view refinement tokens."""

    def __init__(self) -> None:
        super().__init__()
        channels = (48, 96, 192, 384)
        self.projects = nn.ModuleList(nn.Conv2d(384, dimension, 1) for dimension in channels)
        self.resize_layers = nn.ModuleList(
            (
                nn.ConvTranspose2d(48, 48, 4, stride=4),
                nn.ConvTranspose2d(96, 96, 2, stride=2),
                nn.Identity(),
                nn.Conv2d(384, 384, 3, stride=2, padding=1),
            )
        )
        self.scratch = _TrackingScratch(channels)

    def forward(self, tokens: Sequence[torch.Tensor], height: int, width: int) -> torch.Tensor:
        rn_layers = (
            self.scratch.layer1_rn,
            self.scratch.layer2_rn,
            self.scratch.layer3_rn,
            self.scratch.layer4_rn,
        )
        pyramid = []
        for index, features in enumerate(tokens):
            features = features.transpose(1, 2).reshape(features.shape[0], 384, height, width)
            features = self.resize_layers[index](self.projects[index](features))
            pyramid.append(rn_layers[index](features))
        features = self.scratch.refinenet4(pyramid[3], size=pyramid[2].shape[-2:])
        features = self.scratch.refinenet3(features, pyramid[2], size=pyramid[1].shape[-2:])
        features = self.scratch.refinenet2(features, pyramid[1], size=pyramid[0].shape[-2:])
        features = self.scratch.refinenet1(features, pyramid[0])
        return self.scratch.output_conv1(features)


class CrossViewBlock(nn.Module):
    """Per-patch temporal self-attention block across sequence frames."""

    def __init__(self) -> None:
        super().__init__()
        self.self_attn_block = Block(384, 6, epsilon=1e-5)

    def forward(self, features: torch.Tensor, batch_size: int, frames: int) -> torch.Tensor:
        if frames == 1:
            return features
        patches = features.shape[1]
        features = (
            features.reshape(batch_size, frames, patches, 384)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * patches, frames, 384)
        )
        features = _framewise(self.self_attn_block, features, chunk_size=64)
        return (
            features.reshape(batch_size, patches, frames, 384)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * frames, patches, 384)
            .contiguous()
        )


class TrackingRefinement(nn.Module):
    """Spatiotemporal ViT + DPT refinement network for iterative dense flow updates."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(64, 384, 4, stride=4)
        self.pos_embed = nn.Parameter(torch.empty(1, 64, 384))
        self.blks = nn.ModuleList(Block(384, 6) for _ in range(12))
        self.temporal_blocks = nn.ModuleList(CrossViewBlock() for _ in range(6))
        self.dpt_head = TrackingPyramid()

    def encode(self, features: torch.Tensor) -> list[torch.Tensor]:
        batch_size, frames, _, height, width = features.shape
        features = (
            _framewise(self.patch_embed, features.reshape(batch_size * frames, 64, height, width))
            .flatten(2)
            .transpose(1, 2)
        )
        if features.shape[1] == 64 and height == width:
            position = self.pos_embed
        else:
            position = F.interpolate(
                self.pos_embed.float().reshape(1, 8, 8, 384).permute(0, 3, 1, 2),
                scale_factor=(height / 32, width / 32),
                mode="bicubic",
                antialias=False,
            )
            position = position.permute(0, 2, 3, 1).reshape(1, -1, 384).to(features.dtype)
        features = features + position
        pyramid = []
        for index, block in enumerate(self.blks):
            features = _framewise(block, features)
            if index % 2 == 1:
                features = self.temporal_blocks[index // 2](features, batch_size, frames)
            if index in (2, 5, 8, 11):
                pyramid.append(features)
        return pyramid

    def decode(self, pyramid: Sequence[torch.Tensor], height: int, width: int) -> torch.Tensor:
        features = self.dpt_head(pyramid, height // 4, width // 4)
        return F.interpolate(features.contiguous(), size=(height, width), mode="bilinear", align_corners=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, frames, _, height, width = features.shape
        pyramid = self.encode(features)
        decoded = torch.cat(
            [
                self.decode([level[start : start + 8] for level in pyramid], height, width)
                for start in range(0, batch_size * frames, 8)
            ]
        )
        return decoded.reshape(batch_size, frames, 32, height, width)


def coordinate_grid(batch_size: int, height: int, width: int, device: torch.device | str) -> torch.Tensor:
    """Create ``[B, 2, H, W]`` (x, y) pixel coordinate grid."""
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return torch.stack((columns, rows), dim=0).float().unsqueeze(0).expand(batch_size, -1, -1, -1)


def warp(features: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``features`` at pixel ``coordinates`` with ``align_corners=True``."""
    height, width = features.shape[-2:]
    grid = coordinates.permute(0, 2, 3, 1)
    horizontal, vertical = grid.unbind(-1)
    grid = torch.stack((2 * horizontal / (width - 1) - 1, 2 * vertical / (height - 1) - 1), dim=-1)
    return F.grid_sample(features.float(), grid.float(), align_corners=True)


def convex_upsample(
    flow: torch.Tensor,
    information: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2x RAFT-style convex upsampling of flow and visibility/confidence maps."""
    batch_size, _, height, width = flow.shape
    mask = weights.reshape(batch_size, 1, 9, 2, 2, height, width).softmax(dim=2)
    up_flow = F.unfold(2 * flow, 3, padding=1).reshape(batch_size, 2, 9, 1, 1, height, width)
    up_information = F.unfold(information, 3, padding=1).reshape(batch_size, 2, 9, 1, 1, height, width)
    up_flow = (mask * up_flow).sum(dim=2).permute(0, 1, 4, 2, 5, 3).reshape(batch_size, 2, height * 2, width * 2)
    up_information = (
        (mask * up_information).sum(dim=2).permute(0, 1, 4, 2, 5, 3).reshape(batch_size, 2, height * 2, width * 2)
    )
    return up_flow, up_information


class TrackingHead(nn.Module):
    """5-iteration dense warping and convex-upsampling tracking head."""

    def __init__(self) -> None:
        super().__init__()
        self.refine_net = TrackingRefinement()
        self.fmap_conv = nn.Conv2d(256, 64, 1)
        self.hidden_conv = nn.Conv2d(128, 64, 1)
        self.warp_linear = nn.Conv2d(194, 64, 1)
        self.refine_transform = nn.Conv2d(96, 64, 1)
        self.upsample_weight = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 36, 1))
        self.flow_head = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 4, 1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, frames, _, height, width = features.shape
        flattened_batch = batch_size * frames
        device = self.fmap_conv.weight.device
        reference = features[:, 0].to(device)
        features = features.flatten(0, 1)
        hidden = torch.empty_like(features)
        flow = torch.zeros(flattened_batch, 2, height, width)
        base = coordinate_grid(1, height, width, device)
        batches = torch.arange(flattened_batch, device=device) // frames
        for start in range(0, flattened_batch, 8):
            stop = min(start + 8, flattened_batch)
            values = features[start:stop].to(device)
            initial = self.hidden_conv(torch.cat((reference[batches[start:stop]], values), dim=1))
            hidden[start:stop].copy_(initial.cpu())
            del values, initial
        up_flow = torch.empty(flattened_batch, 2, height * 2, width * 2)
        information = torch.empty_like(up_flow)
        for iteration in range(5):
            inputs = torch.empty_like(features, device=device)
            for start in range(0, flattened_batch, 8):
                stop = min(start + 8, flattened_batch)
                values = features[start:stop].to(device)
                previous = flow[start:stop].to(device)
                sampled = warp(values, base + previous).to(values.dtype)
                projected = self.warp_linear(
                    torch.cat(
                        (
                            reference[batches[start:stop]],
                            sampled,
                            hidden[start:stop].to(device),
                            previous.to(values.dtype),
                        ),
                        dim=1,
                    )
                )
                inputs[start:stop].copy_(projected)
                del values, previous, sampled, projected
            pyramid = self.refine_net.encode(inputs.reshape(batch_size, frames, 64, height, width))
            del inputs
            for start in range(0, flattened_batch, 8):
                stop = min(start + 8, flattened_batch)
                refined = self.refine_net.decode([level[start:stop] for level in pyramid], height, width)
                updated = self.refine_transform(torch.cat((refined, hidden[start:stop].to(device)), dim=1))
                hidden[start:stop].copy_(updated.cpu())
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    update = self.flow_head(updated.float())
                    next_flow = flow[start:stop].to(device) + update[:, :2]
                    flow[start:stop].copy_(next_flow.cpu())
                    if iteration == 4:
                        weights = 0.25 * self.upsample_weight(updated.float())
                        coordinates, scores = convex_upsample(next_flow, update[:, 2:], weights)
                        up_flow[start:stop].copy_(coordinates.cpu())
                        information[start:stop].copy_(scores.cpu())
                        del weights, coordinates, scores
                del refined, updated, update, next_flow
            del pyramid
        up_flow = up_flow.reshape(batch_size, frames, 2, height * 2, width * 2)
        up_flow.sub_(up_flow[:, :1].clone())
        up_flow.add_(coordinate_grid(1, height * 2, width * 2, "cpu")[None])
        tracks = up_flow.permute(0, 1, 3, 4, 2).to(device)
        information = information.reshape(batch_size, frames, 2, height * 2, width * 2).to(device)
        return tracks, information[:, :, 0], information[:, :, 1]


class Tracker(nn.Module):
    """Dense query-view pixel tracker fusing motion decoder tokens with RGB features."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 1024))
        self.motion_features = ConvPyramid((512, 256, 128, 128), 128)
        self.rgb_encoder = RGBEncoder()
        self.head = TrackingHead()

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        query_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, frames, _, height, width = images.shape
        images = images.flatten(0, 1)
        features: torch.Tensor | None = None
        for start in range(0, batch_size * frames, 8):
            stop = min(start + 8, batch_size * frames)
            motion = []
            for frame_tokens in tokens[start:stop].split(1):
                projected = self.proj(frame_tokens.float())
                motion.append(self.motion_features(feature_pyramid(projected, height, width, 4))[0])
            mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            deviation = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            rgb = self.rgb_encoder((images[start:stop] * deviation + mean).clamp(0, 1))
            values = _framewise(
                self.head.fmap_conv,
                torch.cat((torch.cat(motion).to(rgb.dtype), rgb), dim=1),
                chunk_size=1,
            )
            if features is None:
                features = torch.empty(batch_size * frames, *values.shape[1:], dtype=values.dtype)
            features[start:stop].copy_(values.cpu())
            del motion, projected, rgb, values
        assert features is not None
        features = features.reshape(batch_size, frames, 64, height // 2, width // 2)
        indices = torch.arange(frames, device=tokens.device).expand(batch_size, -1)
        order = (indices + query_frames[:, None]) % frames
        batch_indices = torch.arange(batch_size, device=tokens.device)[:, None]
        tracks, visibility, confidence = self.head(features[batch_indices.cpu(), order.cpu()])
        restore = order.argsort(dim=1)
        return (
            tracks[batch_indices, restore],
            visibility[batch_indices, restore],
            confidence[batch_indices, restore],
        )
