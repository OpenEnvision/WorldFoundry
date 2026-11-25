"""Space-to-depth downsample and depth-to-space upsample for LTX video.

:class:`SpaceToDepthDownsample` and :class:`DepthToSpaceUpsample`
implement the LTX video scale changes (official 8× time /
32× space overall).  They operate on BCTHW feature maps
inside :mod:`.video_vae`.

Not the same as audio :mod:`~..audio.downsample` /
:mod:`~..audio.upsample` (those are causal 2-D).

This is LTX video graph math, not spatial tiling
(:mod:`.tiling`).
"""

import math

import torch
from einops import rearrange
from torch import nn

from .convolution import CausalConv3d, make_conv_nd
from .enums import PaddingModeType
from .triton_vae_ops import (
    can_use_triton_ltx_pad_3d,
    can_use_triton_space_to_depth_3d_add,
    triton_depth_to_space_3d,
    triton_ltx_pad_3d,
    triton_space_to_depth_3d_add,
)


class SpaceToDepthDownsample(nn.Module):
    """Pack spatial (and optional temporal) pixels into channels."""

    def __init__(
        self,
        dims: int | tuple[int, int],
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels // math.prod(stride),
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        dup_first_t = self.stride[0] == 2
        if (
            x.is_cuda
            and x.ndim == 5
            and not torch.is_grad_enabled()
            and type(self.conv) is CausalConv3d
            and self.conv._can_use_fast_conv_path(x)
        ):
            pad, s_mode, conv_pad = self.conv._fused_pad_spec(causal, extra_pt_l=1 if dup_first_t else 0)
            if all(p == 0 for p in pad) or can_use_triton_ltx_pad_3d(
                x, pad, spatial_mode=s_mode, temporal_mode="replicate"
            ):
                x_pad = (
                    x
                    if all(p == 0 for p in pad)
                    else triton_ltx_pad_3d(x, pad, spatial_mode=s_mode, temporal_mode="replicate")
                )
                h, h_bias = self.conv._forward_conv_only(x_pad, conv_pad, defer_bias=True)
                if can_use_triton_space_to_depth_3d_add(
                    h, x, self.stride, self.group_size, dup_first_t=dup_first_t, h_bias=h_bias
                ):
                    return triton_space_to_depth_3d_add(
                        h, x, self.stride, self.group_size, dup_first_t=dup_first_t, h_bias=h_bias
                    )
                if h_bias is not None:
                    h = h + h_bias.view(1, -1, 1, 1, 1).to(h)
                x_dup = torch.cat([x[:, :, :1, :, :], x], dim=2) if dup_first_t else x
                x_in = rearrange(
                    x_dup,
                    "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
                    p1=self.stride[0],
                    p2=self.stride[1],
                    p3=self.stride[2],
                )
                x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size).mean(dim=2)
                h_out = rearrange(
                    h,
                    "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
                    p1=self.stride[0],
                    p2=self.stride[1],
                    p3=self.stride[2],
                )
                return h_out + x_in

        if dup_first_t:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)  # duplicate first frames for padding

        # skip connection
        x_in = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
        x_in = x_in.mean(dim=2)

        # conv
        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )

        x = x + x_in

        return x


class DepthToSpaceUpsample(nn.Module):
    """Unpack channels back to spatial/temporal resolution."""

    def __init__(
        self,
        dims: int | tuple[int, int],
        in_channels: int,
        stride: tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        if x.is_cuda and x.ndim == 5:
            h_bias: torch.Tensor | None = None
            if not torch.is_grad_enabled() and type(self.conv) is CausalConv3d and self.conv._can_use_fast_conv_path(x):
                pad, s_mode, conv_pad = self.conv._fused_pad_spec(causal)
                if all(p == 0 for p in pad) or can_use_triton_ltx_pad_3d(
                    x, pad, spatial_mode=s_mode, temporal_mode="replicate"
                ):
                    x_pad = (
                        x
                        if all(p == 0 for p in pad)
                        else triton_ltx_pad_3d(x, pad, spatial_mode=s_mode, temporal_mode="replicate")
                    )
                    x_conv, h_bias = self.conv._forward_conv_only(x_pad, conv_pad, defer_bias=True)
                else:
                    x_conv = self.conv(x, causal=causal)
            else:
                x_conv = self.conv(x, causal=causal)
            return triton_depth_to_space_3d(
                x_conv,
                self.stride,
                residual=x if self.residual else None,
                out_channels_reduction_factor=self.out_channels_reduction_factor,
                h_bias=h_bias,
            )

        if self.residual:
            # Reshape and duplicate the input to match the output shape
            x_in = rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.stride[0],
                p2=self.stride[1],
                p3=self.stride[2],
            )
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]
        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]
        if self.residual:
            x = x + x_in
        return x
