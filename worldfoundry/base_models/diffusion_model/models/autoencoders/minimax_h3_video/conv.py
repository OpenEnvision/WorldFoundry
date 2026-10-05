# SPDX-License-Identifier: Apache-2.0
"""Causal 3-D convolution primitives for the MiniMax H3 video encoder.

:class:`BaseConv3d` is the padded 3-D conv used by
:mod:`.vae_cnn` so the encoder does not see future frames.
Decoder is ViT-based (:mod:`.vae_vit`) and does not use this.

Tensors are BCTHW video feature maps.  Parallel causal convs
exist in Wan / Hunyuan / LTX but those graphs are not shared.

This is H3 video encoder graph math only.
"""

import torch.nn as nn
import torch.nn.functional as F


class BaseConv3d(nn.Conv3d):
    """Padded 3-D convolution primitive for the causal H3 video encoder."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="zeros",
        padding_mode_t=None,
        causal=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        padding_mode_t = "constant" if padding_mode_t == "zeros" else padding_mode_t
        self.pad_mode = padding_mode
        self.pad_mode_t = padding_mode_t or ("constant" if causal else "replicate")
        self.causal = causal

    def _apply_temporal_padding(self, x):
        B, C, D, H, W = x.shape
        if D > 1:
            pad_size = (
                0,
                0,
                0,
                0,
                self.padding[0] * 2 if self.causal else self.padding[0],
                0 if self.causal else self.padding[0],
            )
            return F.pad(x, pad_size, mode=self.pad_mode_t)
        else:
            if self.pad_mode_t == "constant":
                assert self.causal, "Zeros padding is only supported for causal mode"
                return F.pad(
                    x,
                    (0, 0, 0, 0, self.kernel_size[0] - 1, 0),
                    mode="constant",
                )
            else:
                return x.expand(-1, -1, self.kernel_size[0], -1, -1)

    def _apply_padding(self, x):
        if sum(self.padding) == 0:
            return x

        x = F.pad(
            x,
            (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0),
            mode=self.pad_mode,
        )

        x = self._apply_temporal_padding(x)
        return x

    def forward(self, x):
        if sum(self.padding) == 0:
            return super().forward(x)

        x = self._apply_padding(x)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
        )
