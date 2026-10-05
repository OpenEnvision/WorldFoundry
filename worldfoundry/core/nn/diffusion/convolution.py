# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!

"""Classic convolutional diffusion layers shared by latent video models.

ResNet / attention blocks from U-Net era codecs (not DiT). Kept for
VAE and legacy denoisers so they do not import Hugging Face diffusers
just for a residual block.

Not this module:
    DiT AdaLN / modulation lives in
    :mod:`worldfoundry.core.nn.transformer.dit`. Shared ViT layers
    live in :mod:`worldfoundry.core.nn.blocks.layers`. Schedules live in
    :mod:`worldfoundry.core.nn.diffusion.schedulers`.

Public surface:

- :class:`CausalConv1d` / :class:`CausalConv2d` / :class:`CausalConv3d`
  — pad-then-crop so the first axis never sees the future.
- :func:`conv_nd` / :func:`avg_pool_nd` / :func:`linear` — dimension
  factories used by U-Net configs.
- :func:`zero_module` / :func:`scale_module` / :func:`disabled_train`.
- :func:`normalization` / :class:`GroupNormSpecific` / :class:`HybridConditioner`.
"""

import torch.nn as nn
import torch
from worldfoundry.core.model_loading.factory import instantiate_from_config


# ──────────────────────────────────────────────────────────────────────────
# Causal convolutions — pad then crop the leading axis so t cannot see t+1
# ──────────────────────────────────────────────────────────────────────────


class CausalConv1d(torch.nn.Module):
    """1D causal convolution compatible with conv_nd(..., causal=True)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        """Ignore caller ``padding``; causal width is ``(k-1)*dilation`` only."""

        super().__init__()
        del padding
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        """``[B, C, T]`` → same layout; trailing pad is dropped so output length is T."""

        x = self.conv1d(x)
        if self.padding > 0:
            x = x[:, :, : -self.padding]
        return x


class CausalConv2d(torch.nn.Module):
    """2D causal convolution over the first spatial axis."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        """Causal pad on axis 0, symmetric half-pad on axis 1 (U-Net video layout)."""

        super().__init__()
        del padding
        self.padding1 = (kernel_size - 1) * dilation
        self.padding2 = (kernel_size - 1) * dilation // 2
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(self.padding1, self.padding2),
            dilation=dilation,
        )

    def forward(self, x):
        """``[B, C, T, W]`` (or H) → crop only the causal axis after the conv."""

        x = self.conv2d(x)
        if self.padding1 > 0:
            x = x[:, :, : -self.padding1, :]
        return x


class CausalConv3d(torch.nn.Module):
    """3D causal convolution over the temporal axis."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=None):
        """Causal pad on T; symmetric pad on H/W. ``kernel_size`` may be an int or 3-tuple.

        Raises:
            AssertionError: a tuple ``kernel_size`` is not length 3.
        """

        super().__init__()
        del padding
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 3, "kernel_size must be a tuple of length 3"
        self.padding1 = (kernel_size[0] - 1) * dilation
        self.padding2 = (kernel_size[1] - 1) * dilation // 2
        self.padding3 = (kernel_size[2] - 1) * dilation // 2
        self.conv3d = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(self.padding1, self.padding2, self.padding3),
            dilation=dilation,
        )

    @property
    def weight(self):
        """Expose the inner conv weight so callers can zero-init or copy state."""

        return self.conv3d.weight

    @property
    def bias(self):
        """Expose the inner conv bias for the same checkpoint / init hooks."""

        return self.conv3d.bias

    def forward(self, x):
        """``[B, C, T, H, W]`` → crop only T after the padded conv."""

        x = self.conv3d(x)
        if self.padding1 > 0:
            x = x[:, :, : -self.padding1, :, :]
        return x


# ──────────────────────────────────────────────────────────────────────────
# Train-mode lock and parameter mutators — identity residuals at init
# ──────────────────────────────────────────────────────────────────────────


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


# ──────────────────────────────────────────────────────────────────────────
# N-D factories — U-Net configs pick 1/2/3 plus optional causal convs
# ──────────────────────────────────────────────────────────────────────────


def conv_nd(dims, *args, causal=False, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if causal:
        if dims == 1:
            return CausalConv1d(*args, **kwargs)
        elif dims == 2:
            return CausalConv2d(*args, **kwargs)
        elif dims == 3:
            return CausalConv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def nonlinearity(type='silu'):
    """Return SiLU or LeakyReLU; unknown ``type`` yields ``None`` (legacy).

    Callers historically treated a missing activation as optional, so this
    does not raise on an unrecognized name.
    """
    if type == 'silu':
        return nn.SiLU()
    elif type == 'leaky_relu':
        return nn.LeakyReLU()


class GroupNormSpecific(nn.GroupNorm):
    """GroupNorm subclass some U-Net checkpoints address by type name."""
    def forward(self, x):
        """Same as :class:`torch.nn.GroupNorm`; kept for checkpoint type identity."""
        return super().forward(x)


def normalization(channels, num_groups=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNormSpecific(num_groups, channels)


# ──────────────────────────────────────────────────────────────────────────
# Hybrid conditioner — concat stream plus cross-attn stream from configs
# ──────────────────────────────────────────────────────────────────────────


class HybridConditioner(nn.Module):
    """Instantiate concat and cross-attention conditioners from LazyConfig dicts."""

    def __init__(self, c_concat_config, c_crossattn_config):
        """Both configs are resolved through :func:`instantiate_from_config`."""

        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        """Return the U-Net dict ``{'c_concat': [...], 'c_crossattn': [...]}``."""

        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}
