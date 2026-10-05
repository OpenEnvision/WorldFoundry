# Copyright (C) 2026 Tencent. Adapted for WorldFoundry inference.
# Source revision: 7e57e4d0448e8661eb6ae078f53e19638bd07102
# Academic-use terms: THIRD-PARTY-NOTICES (WorldCrafter)
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


LOW_RANK_DIM = 64
LOW_RANK_SCALE = 32.0 / LOW_RANK_DIM
EXPECTED_LOW_RANK_MODULES = 438


class LowRankLinear(nn.Linear):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.low_rank_a = nn.Parameter(
            torch.empty(LOW_RANK_DIM, source.in_features, dtype=torch.float32),
            requires_grad=False,
        )
        self.low_rank_b = nn.Parameter(
            torch.empty(source.out_features, LOW_RANK_DIM, dtype=torch.float32),
            requires_grad=False,
        )
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = F.linear(value, self.weight, self.bias)
        update = F.linear(F.linear(value, self.low_rank_a), self.low_rank_b)
        return base + update * LOW_RANK_SCALE


class LowRankConv2d(nn.Conv2d):
    def __init__(self, source: nn.Conv2d) -> None:
        if source.groups != 1 or source.kernel_size[0] != source.kernel_size[1]:
            raise ValueError("Only square, ungrouped Conv2d layers are supported")
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        kernel = source.kernel_size[0]
        expanded_rank = LOW_RANK_DIM * kernel
        self.low_rank_a = nn.Parameter(
            torch.empty(expanded_rank, source.in_channels * kernel, dtype=torch.float32),
            requires_grad=False,
        )
        self.low_rank_b = nn.Parameter(
            torch.empty(source.out_channels * kernel, expanded_rank, dtype=torch.float32),
            requires_grad=False,
        )
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        delta = (self.low_rank_b @ self.low_rank_a).reshape_as(self.weight) * LOW_RANK_SCALE
        base = self._conv_forward(value, self.weight, self.bias)
        update = self._conv_forward(value, delta, None)
        return base + update


def _uses_low_rank_branch(name: str) -> bool:
    if name.startswith("dino_tail.blocks."):
        return True
    if name.startswith(("vggt.frame_blocks.", "vggt.global_blocks.")):
        return True
    if name.startswith("vggt.camera_mlp.") or name == "vggt.geo_feature_connector":
        return True
    if name == "repfeature.target_embedding":
        return True
    if name.startswith("repfeature.blocks."):
        return True
    if name.startswith("repfeature.final_block."):
        return True
    return False


def _parent_and_child(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_low_rank_branches(model: nn.Module) -> int:
    count = 0
    for module_name, module in list(model.named_modules()):
        if not module_name or not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if not _uses_low_rank_branch(module_name):
            continue
        parent, child = _parent_and_child(model, module_name)
        replacement: nn.Module
        if isinstance(module, nn.Linear):
            replacement = LowRankLinear(module)
        else:
            replacement = LowRankConv2d(module)
        setattr(parent, child, replacement)
        count += 1
    if count != EXPECTED_LOW_RANK_MODULES:
        raise RuntimeError(
            f"RepEncoder low-rank graph changed: {count} != {EXPECTED_LOW_RANK_MODULES}"
        )
    return count


__all__ = [
    "EXPECTED_LOW_RANK_MODULES",
    "LOW_RANK_DIM",
    "LOW_RANK_SCALE",
    "LowRankConv2d",
    "LowRankLinear",
    "inject_low_rank_branches",
]
