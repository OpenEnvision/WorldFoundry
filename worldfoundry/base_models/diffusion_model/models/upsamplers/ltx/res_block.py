"""GroupNorm + SiLU residual block for the LTX latent upsampler CNN.

Norms are invoked as modules (not fused functionals) so VRAM offload
wrappers can onload parameters before the activation runs.
"""

import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.upsamplers.ltx.triton_upsampler_ops import (
    can_use_triton_upsampler_group_norm_silu,
    triton_upsampler_group_norm_silu,
)


def _has_hooks(module: torch.nn.Module) -> bool:
    return bool(module._forward_hooks or module._forward_pre_hooks)


def _can_defer_conv_bias(conv: torch.nn.Module, x: torch.Tensor) -> bool:
    return (
        type(conv) in (torch.nn.Conv2d, torch.nn.Conv3d)
        and not _has_hooks(conv)
        and conv.padding_mode == "zeros"
        and x.is_cuda
        and not torch.is_grad_enabled()
        and conv.weight.device == x.device
        and (conv.bias is None or conv.bias.device == x.device)
    )


def _can_fuse_norm_act_module(
    norm: torch.nn.Module,
    act: torch.nn.Module,
    device: torch.device,
) -> bool:
    return (
        type(norm) is torch.nn.GroupNorm
        and type(act) is torch.nn.SiLU
        and not _has_hooks(norm)
        and not _has_hooks(act)
        and (norm.weight is None or norm.weight.device == device)
        and (norm.bias is None or norm.bias.device == device)
    )


def _conv_no_bias(conv: torch.nn.Conv2d | torch.nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if isinstance(conv, torch.nn.Conv3d):
        return F.conv3d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
    return F.conv2d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)


class ResBlock(torch.nn.Module):
    """
    Residual block with two convolutional layers, group normalization, and SiLU activation.
    Args:
        channels (int): Number of input and output channels.
        mid_channels (Optional[int]): Number of channels in the intermediate convolution layer. Defaults to `channels`
        if not specified.
        dims (int): Dimensionality of the convolution (2 for Conv2d, 3 for Conv3d). Defaults to 3.
    """

    def __init__(self, channels: int, mid_channels: int | None = None, dims: int = 3):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d

        self.conv1 = conv(channels, mid_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, mid_channels)
        self.conv2 = conv(mid_channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # Only fuse directly against unwrapped nn.Conv/nn.GroupNorm modules with
        # on-device parameters; VRAM-offload wrappers (AutoWrappedModule) or hooks
        # fall back transparently to calling the modules directly.
        if _can_defer_conv_bias(self.conv1, x) and _can_fuse_norm_act_module(self.norm1, self.activation, x.device):
            h = _conv_no_bias(self.conv1, x)
            if can_use_triton_upsampler_group_norm_silu(
                h, self.norm1.num_groups, self.norm1.weight, self.norm1.bias, pre_bias=self.conv1.bias
            ):
                x = triton_upsampler_group_norm_silu(
                    h,
                    self.norm1.num_groups,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm1.eps,
                    pre_bias=self.conv1.bias,
                    apply_silu=True,
                )
            else:
                if self.conv1.bias is not None:
                    h = h + self.conv1.bias.view((1, -1) + (1,) * (h.ndim - 2)).to(h)
                x = self.activation(self.norm1(h))
        else:
            x = self.conv1(x)
            x = self.activation(self.norm1(x))

        if _can_defer_conv_bias(self.conv2, x) and _can_fuse_norm_act_module(self.norm2, self.activation, x.device):
            h = _conv_no_bias(self.conv2, x)
            if can_use_triton_upsampler_group_norm_silu(
                h,
                self.norm2.num_groups,
                self.norm2.weight,
                self.norm2.bias,
                pre_bias=self.conv2.bias,
                residual=residual,
            ):
                return triton_upsampler_group_norm_silu(
                    h,
                    self.norm2.num_groups,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.norm2.eps,
                    pre_bias=self.conv2.bias,
                    residual=residual,
                    apply_silu=True,
                )
            if self.conv2.bias is not None:
                h = h + self.conv2.bias.view((1, -1) + (1,) * (h.ndim - 2)).to(h)
            return self.activation(self.norm2(h) + residual)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return x
