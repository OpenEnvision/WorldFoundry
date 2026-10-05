"""Explicit mixed Conv2d/Conv3d weight layout tuning before compilation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConvolutionLayoutReport:
    conv2d_total: int
    conv2d_converted: int
    conv3d_total: int
    conv3d_converted: int


def convert_convolution_weight_layouts(
    module: nn.Module,
    *,
    conv2d: bool = False,
    conv3d: bool = False,
) -> ConvolutionLayoutReport:
    """Preserve parameters and values; reject invalid requests before mutation.

    Conv2d uses channels_last and Conv3d uses channels_last_3d. No activation
    layout or convolution geometry is changed. Apply after loading weights,
    before compile/capture; latency improvement depends on the target hardware.
    """
    if not isinstance(conv2d, bool) or not isinstance(conv3d, bool):
        raise TypeError("convolution layout switches must be bools")
    modules_2d = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
    modules_3d = [child for child in module.modules() if isinstance(child, nn.Conv3d)]
    requested = [(child, torch.channels_last) for child in modules_2d] if conv2d else []
    requested += [(child, torch.channels_last_3d) for child in modules_3d] if conv3d else []
    if conv2d and not modules_2d or conv3d and not modules_3d:
        raise ValueError("requested convolution layout has no matching weights")
    if any(child.weight.is_meta for child, _ in requested):
        raise RuntimeError("cannot convert meta convolution weights")
    with torch.no_grad():
        for child, memory_format in requested:
            child.weight.data = child.weight.data.contiguous(memory_format=memory_format)
    converted_2d = (
        sum(child.weight.is_contiguous(memory_format=torch.channels_last) for child in modules_2d) if conv2d else 0
    )
    converted_3d = (
        sum(child.weight.is_contiguous(memory_format=torch.channels_last_3d) for child in modules_3d) if conv3d else 0
    )
    if converted_2d != (len(modules_2d) if conv2d else 0) or converted_3d != (len(modules_3d) if conv3d else 0):
        raise RuntimeError("convolution layout conversion was incomplete")
    return ConvolutionLayoutReport(len(modules_2d), converted_2d, len(modules_3d), converted_3d)


__all__ = ["ConvolutionLayoutReport", "convert_convolution_weight_layouts"]
