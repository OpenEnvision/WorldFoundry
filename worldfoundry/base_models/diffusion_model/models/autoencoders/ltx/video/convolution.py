"""N-D conv factory (2-D / 3-D / causal) for the LTX video VAE.

:func:`make_conv_nd` / :func:`make_linear_nd` build the
convolutions used by residual and mid blocks.
:class:`CausalConv3d` pads time so future frames do not leak;
:class:`DualConv3d` is the 2-D+1-D factorization some
checkpoints use.

Tensors are BCTHW video maps.  Audio uses
:mod:`~..audio.causal_conv_2d` instead.

This is LTX video graph math consumed by :mod:`.resnet`.
"""

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.enums import PaddingModeType
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.triton_vae_ops import (
    _is_channels_last_3d,
    can_use_triton_ltx_pad_3d,
    can_use_triton_ltx_pad_3d_align_c,
    can_use_triton_ltx_residual_bias_add,
    triton_ltx_pad_3d,
    triton_ltx_pad_3d_align_c,
    triton_ltx_residual_bias_add,
)


def _has_hooks(module: nn.Module) -> bool:
    return bool(module._forward_hooks or module._forward_pre_hooks)


def make_conv_nd(
    dims: int | tuple[int, int],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causal: bool = False,
    spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    temporal_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
) -> nn.Module:
    """Build a 2-D, 3-D, or causal convolution for the LTX video VAE."""
    if not (spatial_padding_mode == temporal_padding_mode or causal):
        raise NotImplementedError("spatial and temporal padding modes must be equal")
    if dims == 2:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == 3:
        if causal:
            return CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                spatial_padding_mode=spatial_padding_mode,
            )
        return nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    elif dims == (2, 1):
        return DualConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=spatial_padding_mode.value,
        )
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def make_linear_nd(
    dims: int,
    in_channels: int,
    out_channels: int,
    bias: bool = True,
) -> nn.Module:
    """Build a 1x1 N-D projection used as a channel mixer."""
    if dims == 2:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    elif dims in (3, (2, 1)):
        return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


class DualConv3d(nn.Module):
    """Factorized 2-D + 1-D convolution used by some LTX checkpoints."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        # Ensure kernel_size, stride, padding, and dilation are tuples of length 3
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if kernel_size == (1, 1, 1):
            raise ValueError("kernel_size must be greater than 1. Use make_linear_nd instead.")
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        # Set parameters for convolutions
        self.groups = groups
        self.bias = bias

        # Define the size of the channels after the first convolution
        intermediate_channels = out_channels if in_channels < out_channels else in_channels

        # Define parameters for the first convolution
        self.weight1 = nn.Parameter(
            torch.Tensor(
                intermediate_channels,
                in_channels // groups,
                1,
                kernel_size[1],
                kernel_size[2],
            )
        )
        self.stride1 = (1, stride[1], stride[2])
        self.padding1 = (0, padding[1], padding[2])
        self.dilation1 = (1, dilation[1], dilation[2])
        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(intermediate_channels))
        else:
            self.register_parameter("bias1", None)

        # Define parameters for the second convolution
        self.weight2 = nn.Parameter(torch.Tensor(out_channels, intermediate_channels // groups, kernel_size[0], 1, 1))
        self.stride2 = (stride[0], 1, 1)
        self.padding2 = (padding[0], 0, 0)
        self.dilation2 = (dilation[0], 1, 1)
        if bias:
            self.bias2 = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias2", None)

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight1, a=torch.sqrt(5))
        nn.init.kaiming_uniform_(self.weight2, a=torch.sqrt(5))
        if self.bias:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
            bound1 = 1 / torch.sqrt(fan_in1)
            nn.init.uniform_(self.bias1, -bound1, bound1)
            fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
            bound2 = 1 / torch.sqrt(fan_in2)
            nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(
        self,
        x: torch.Tensor,
        use_conv3d: bool = False,
        skip_time_conv: bool = False,
    ) -> torch.Tensor:
        if use_conv3d:
            return self.forward_with_3d(x=x, skip_time_conv=skip_time_conv)
        else:
            return self.forward_with_2d(x=x, skip_time_conv=skip_time_conv)

    def forward_with_3d(self, x: torch.Tensor, skip_time_conv: bool = False) -> torch.Tensor:
        # First convolution
        x = F.conv3d(
            x,
            self.weight1,
            self.bias1,
            self.stride1,
            self.padding1,
            self.dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        if skip_time_conv:
            return x

        # Second convolution
        x = F.conv3d(
            x,
            self.weight2,
            self.bias2,
            self.stride2,
            self.padding2,
            self.dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )

        return x

    def forward_with_2d(self, x: torch.Tensor, skip_time_conv: bool = False) -> torch.Tensor:
        b, _, _, h, w = x.shape

        # First 2D convolution
        x = rearrange(x, "b c d h w -> (b d) c h w")
        # Squeeze the depth dimension out of weight1 since it's 1
        weight1 = self.weight1.squeeze(2)
        # Select stride, padding, and dilation for the 2D convolution
        stride1 = (self.stride1[1], self.stride1[2])
        padding1 = (self.padding1[1], self.padding1[2])
        dilation1 = (self.dilation1[1], self.dilation1[2])
        x = F.conv2d(
            x,
            weight1,
            self.bias1,
            stride1,
            padding1,
            dilation1,
            self.groups,
            padding_mode=self.padding_mode,
        )

        _, _, h, w = x.shape

        if skip_time_conv:
            x = rearrange(x, "(b d) c h w -> b c d h w", b=b)
            return x

        # Second convolution which is essentially treated as a 1D convolution across the 'd' dimension
        x = rearrange(x, "(b d) c h w -> (b h w) c d", b=b)

        # Reshape weight2 to match the expected dimensions for conv1d
        weight2 = self.weight2.squeeze(-1).squeeze(-1)
        # Use only the relevant dimension for stride, padding, and dilation for the 1D convolution
        stride2 = self.stride2[0]
        padding2 = self.padding2[0]
        dilation2 = self.dilation2[0]
        x = F.conv1d(
            x,
            weight2,
            self.bias2,
            stride2,
            padding2,
            dilation2,
            self.groups,
            padding_mode=self.padding_mode,
        )
        x = rearrange(x, "(b h w) c d -> b c d h w", b=b, h=h, w=w)

        return x

    @property
    def weight(self) -> torch.Tensor:
        return self.weight2


class CausalConv3d(nn.Module):
    """3-D convolution that pads time so future frames do not leak."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int] = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]

        dilation = (dilation, 1, 1)

        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=spatial_padding_mode.value,
            groups=groups,
            bias=bias,
        )

    def _temporal_pad(self, causal: bool) -> tuple[int, int]:
        if causal:
            return (self.time_kernel_size - 1, 0)
        half = (self.time_kernel_size - 1) // 2
        return (half, half)

    def _fused_pad_spec(
        self,
        causal: bool,
        *,
        extra_pt_l: int = 0,
    ) -> tuple[tuple[int, int, int, int, int, int], str, tuple[int, int, int]]:
        pt_l, pt_r = self._temporal_pad(causal)
        pt_l += extra_pt_l
        pmode = self.conv.padding_mode
        pad_cfg = self.conv.padding
        if isinstance(pad_cfg, str):
            return (0, 0, 0, 0, pt_l, pt_r), "constant", (0, 0, 0)
        p0, p1, p2 = int(pad_cfg[0]), int(pad_cfg[1]), int(pad_cfg[2])
        if pmode == "zeros":
            return (0, 0, 0, 0, pt_l, pt_r), "constant", (p0, p1, p2)
        return (p2, p2, p1, p1, pt_l, pt_r), pmode, (0, 0, 0)

    def _can_use_fast_conv_path(self, x: torch.Tensor) -> bool:
        conv = self.conv
        return (
            type(self) is CausalConv3d
            and type(conv) is nn.Conv3d
            and not _has_hooks(self)
            and not _has_hooks(conv)
            and x.is_cuda
            and x.ndim == 5
            and not torch.is_grad_enabled()
            and conv.weight.device == x.device
            and (conv.bias is None or conv.bias.device == x.device)
            and conv.padding_mode in ("zeros", "reflect", "replicate")
        )

    def _forward_conv_only(
        self,
        x_pad: torch.Tensor,
        conv_padding: tuple[int, int, int],
        *,
        defer_bias: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        conv = self.conv
        out = F.conv3d(x_pad, conv.weight, None, conv.stride, conv_padding, conv.dilation, conv.groups)
        if conv.bias is None:
            return out, None
        if defer_bias:
            return out, conv.bias
        if can_use_triton_ltx_residual_bias_add(None, out, conv.bias):
            return triton_ltx_residual_bias_add(None, out, conv.bias), None
        return out + conv.bias.view(1, -1, 1, 1, 1).to(out), None

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if self._can_use_fast_conv_path(x):
            conv = self.conv
            pad, s_mode, conv_padding = self._fused_pad_spec(causal)
            if (
                conv.groups == 1
                and x.shape[1] < 8
                and conv.weight.shape[1] == x.shape[1]
                and (_is_channels_last_3d(x) or _is_channels_last_3d(conv.weight))
                and can_use_triton_ltx_pad_3d_align_c(x, pad, spatial_mode=s_mode, temporal_mode="replicate", align_c=8)
            ):
                x_pad8 = triton_ltx_pad_3d_align_c(x, pad, spatial_mode=s_mode, temporal_mode="replicate", align_c=8)
                w = conv.weight
                w8 = torch.empty(
                    (w.shape[0], 8, w.shape[2], w.shape[3], w.shape[4]),
                    device=w.device,
                    dtype=w.dtype,
                    memory_format=torch.channels_last_3d,
                ).zero_()
                w8[:, : w.shape[1]] = w
                out = F.conv3d(x_pad8, w8, None, conv.stride, conv_padding, conv.dilation, conv.groups)
                if conv.bias is not None:
                    return triton_ltx_residual_bias_add(None, out, conv.bias)
                return out
            if all(p == 0 for p in pad) or can_use_triton_ltx_pad_3d(
                x, pad, spatial_mode=s_mode, temporal_mode="replicate"
            ):
                x_pad = (
                    x
                    if all(p == 0 for p in pad)
                    else triton_ltx_pad_3d(x, pad, spatial_mode=s_mode, temporal_mode="replicate")
                )
                out, _ = self._forward_conv_only(x_pad, conv_padding, defer_bias=False)
                return out
        if causal:
            first_frame_pad = x[:, :, :1, :, :].expand(-1, -1, self.time_kernel_size - 1, -1, -1)
            x = torch.concatenate((first_frame_pad, x), dim=2)
        else:
            first_frame_pad = x[:, :, :1, :, :].expand(-1, -1, (self.time_kernel_size - 1) // 2, -1, -1)
            last_frame_pad = x[:, :, -1:, :, :].expand(-1, -1, (self.time_kernel_size - 1) // 2, -1, -1)
            x = torch.concatenate((first_frame_pad, x, last_frame_pad), dim=2)
        x = self.conv(x)
        return x

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight
