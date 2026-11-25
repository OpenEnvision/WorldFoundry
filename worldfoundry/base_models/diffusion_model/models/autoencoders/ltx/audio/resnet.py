"""Residual blocks for the LTX audio VAE.

:class:`ResnetBlock` is the main encoder/decoder residual.
:class:`ResBlock1` / :class:`ResBlock2` are the vocoder AMP
residuals.  All use causal 2-D convs from :mod:`.causal_conv_2d`.

Tensors are ``N C H W`` spectrogram maps.  Header-level docs
only — do not comment every inner conv.

Consumed by :mod:`.audio_vae` and :mod:`.vocoder`.
"""

import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.causal_conv_2d import (
    CausalConv2d,
    make_conv2d,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.causality_axis import CausalityAxis
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.audio.triton_audio_ops import (
    can_use_triton_pixel_norm_silu_pad_2d,
    triton_pixel_norm_silu_pad_2d,
    triton_residual_bias_add_2d,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.common.normalization import (
    NormType,
    PixelNorm,
    build_normalization_layer,
)


def _has_hooks(module: torch.nn.Module) -> bool:
    return bool(module._forward_hooks or module._forward_pre_hooks)


def _can_fuse_audio_norm_silu_conv(
    norm: torch.nn.Module,
    act: torch.nn.Module,
    conv: torch.nn.Module,
    x: torch.Tensor,
    pre_bias: torch.Tensor | None = None,
) -> bool:
    if (
        type(norm) is not PixelNorm
        or norm.dim not in (1, 1 - x.ndim)
        or type(act) is not torch.nn.SiLU
        or type(conv) is not CausalConv2d
        or type(conv.conv) is not torch.nn.Conv2d
        or _has_hooks(norm)
        or _has_hooks(act)
        or _has_hooks(conv)
        or _has_hooks(conv.conv)
        or not x.is_cuda
        or torch.is_grad_enabled()
        or conv.conv.weight.device != x.device
        or (conv.conv.bias is not None and conv.conv.bias.device != x.device)
    ):
        return False
    return can_use_triton_pixel_norm_silu_pad_2d(x, conv.padding, pre_bias=pre_bias)


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    """Vocoder AMP residual with dilated causal convolutions (variant 1)."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding="same",
                ),
            ]
        )

        self.convs2 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = torch.nn.functional.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    """Vocoder AMP residual with dilated causal convolutions (variant 2)."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int] = (1, 3)):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding="same",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x


class ResnetBlock(torch.nn.Module):
    """Main residual unit used by the LTX audio encoder/decoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and norm_type == NormType.GROUP:
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.non_linearity = torch.nn.SiLU()
        self.conv1 = make_conv2d(in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = make_conv2d(out_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, causality_axis=causality_axis
                )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        dropout_is_noop = (
            type(self.dropout) is torch.nn.Dropout
            and not _has_hooks(self.dropout)
            and (not self.training or self.dropout.p == 0.0)
        )
        can_fuse_2 = (
            temb is None
            and dropout_is_noop
            and type(self.norm2) is PixelNorm
            and type(self.conv2) is CausalConv2d
            and not _has_hooks(self.norm2)
            and not _has_hooks(self.conv2)
        )

        bias1: torch.Tensor | None = None
        if (
            isinstance(self.norm1, PixelNorm)
            and isinstance(self.conv1, CausalConv2d)
            and isinstance(self.conv1.conv, torch.nn.Conv2d)
            and _can_fuse_audio_norm_silu_conv(self.norm1, self.non_linearity, self.conv1, h)
        ):
            h_pad = triton_pixel_norm_silu_pad_2d(h, self.norm1.eps, self.conv1.padding, apply_silu=True)
            c1 = self.conv1.conv
            h = torch.nn.functional.conv2d(h_pad, c1.weight, None, c1.stride, c1.padding, c1.dilation, c1.groups)
            if c1.bias is not None:
                if can_fuse_2:
                    bias1 = c1.bias
                else:
                    h = triton_residual_bias_add_2d(None, h, c1.bias)
        else:
            h = self.norm1(h)
            h = self.non_linearity(h)
            h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        bias2: torch.Tensor | None = None
        if (
            dropout_is_noop
            and isinstance(self.norm2, PixelNorm)
            and isinstance(self.conv2, CausalConv2d)
            and isinstance(self.conv2.conv, torch.nn.Conv2d)
            and _can_fuse_audio_norm_silu_conv(self.norm2, self.non_linearity, self.conv2, h, pre_bias=bias1)
        ):
            h_pad = triton_pixel_norm_silu_pad_2d(
                h, self.norm2.eps, self.conv2.padding, apply_silu=True, pre_bias=bias1
            )
            c2 = self.conv2.conv
            h = torch.nn.functional.conv2d(h_pad, c2.weight, None, c2.stride, c2.padding, c2.dilation, c2.groups)
            bias2 = c2.bias
        else:
            if bias1 is not None:
                h = triton_residual_bias_add_2d(None, h, bias1)
            h = self.norm2(h)
            h = self.non_linearity(h)
            h = self.dropout(h)
            h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)

        if bias2 is not None or (x.is_cuda and h.is_cuda and not torch.is_grad_enabled()):
            return triton_residual_bias_add_2d(x, h, bias2)
        return x + h
