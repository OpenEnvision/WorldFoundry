from abc import abstractmethod
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vwm.modules.video_attention import SpatialVideoTransformer

from .util import avg_pool_nd, conv_nd, linear, normalization, timestep_embedding, zero_module


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to 'x' given 'emb' timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that support it as an extra input.
    """

    def forward(
            self,
            x: torch.Tensor,
            emb: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            time_context: Optional[int] = None,
            num_frames: Optional[int] = None
    ):
        from .video_model import VideoResBlock

        for layer in self:
            if isinstance(layer, VideoResBlock):
                x = layer(x, emb, num_frames)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialVideoTransformer):
                x = layer(x, context, time_context, num_frames)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: Channels in the inputs and outputs.
    :param use_conv: A bool determining if a convolution is applied.
    :param dims: Determines if the signal is 1D, 2D, or 3D. If 3D, then upsampling occurs in the inner-two dimensions.
    """

    def __init__(
            self,
            channels: int,
            use_conv: bool,
            dims: int = 2,
            out_channels: Optional[int] = None,
            third_up: bool = False
    ):
        super(Upsample, self).__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.third_up = third_up
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            t_factor = 2 if self.third_up else 1
            x = F.interpolate(
                x,
                (t_factor * x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode="bicubic"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="bicubic")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: Channels in the inputs and outputs.
    :param use_conv: A bool determining if a convolution is applied.
    :param dims: Determines if the signal is 1D, 2D, or 3D. If 3D, then downsampling occurs in the inner-two dimensions.
    """

    def __init__(
            self,
            channels: int,
            use_conv: bool,
            dims: int = 2,
            out_channels: Optional[int] = None,
            third_down: bool = False
    ):
        super(Downsample, self).__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else ((2, 2, 2) if third_down else (1, 2, 2))
        if use_conv:
            print(f"Building a Downsample layer with {dims} dims")
            if dims == 3:
                print(f"Downsampling third axis (time): {third_down}")
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: The number of input channels.
    :param emb_channels: The number of timestep embedding channels.
    :param dropout: The rate of dropout.
    :param out_channels: If specified, the number of out channels.
    :param use_conv: If True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: Determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: If True, use gradient checkpointing on this module.
    :param up: If True, use this block for upsampling.
    :param down: If True, use this block for downsampling.
    """

    def __init__(
            self,
            channels: int,
            emb_channels: int,
            dropout: float,
            out_channels: Optional[int] = None,
            use_conv: bool = False,
            use_scale_shift_norm: bool = False,
            dims: int = 2,
            use_checkpoint: bool = False,
            up: bool = False,
            down: bool = False,
            kernel_size: int = 3,
            exchange_temb_dims: bool = False,
            skip_t_emb: bool = False
    ):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        del use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if isinstance(kernel_size, Iterable):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(emb_channels, self.emb_out_channels)
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)
            )
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: An [N x C x ...] Tensor of features.
        :param emb: An [N x emb_channels] Tensor of timestep embeddings.
        :return: An [N x C x ...] Tensor of outputs.
        """

        return self._forward(x, emb)

    def _forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = emb_out.chunk(2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Timestep(nn.Module):
    def __init__(self, dim: int):
        super(Timestep, self).__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(t, self.dim)
