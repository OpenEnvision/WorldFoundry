"""Causal 3-D CNN for Wan 2.1 (16-ch) and Wan 2.2 VAE38 (48-ch) video latents.

Checkpoint-compatible encoder / decoder with causal temporal convolutions
and optional spatial tiling.  Public surface: :class:`WanVideoVAE` /
:class:`WanVideoVAE38` ``encode`` / ``decode`` (BCTHW, official mean/std
scale).  Residual / mid / upsample blocks are implementation detail;
do not comment every conv.  Runner recipes use
:class:`~.component.WanVideoDecoder` rather than this module directly.
"""

import contextlib
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from worldfoundry.core.attention import scaled_dot_product_attention as _worldfoundry_scaled_dot_product_attention
from worldfoundry.core.kernels.triton_wan_vae import (
    can_use_wan_vae_avg_down_3d_add,
    can_use_wan_vae_causal_pad_cache,
    can_use_wan_vae_dup_up_3d_add,
    can_use_wan_vae_residual_bias_add,
    can_use_wan_vae_rms_norm,
    is_channels_last_2d,
    is_channels_last_3d,
    wan_vae_avg_down_3d,
    wan_vae_avg_down_3d_add,
    wan_vae_causal_pad_cache,
    wan_vae_dup_up_3d,
    wan_vae_dup_up_3d_add,
    wan_vae_nearest_upsample_2x,
    wan_vae_patchify_2x,
    wan_vae_residual_bias_add,
    wan_vae_rms_norm,
    wan_vae_rms_norm_silu_causal_pad,
    wan_vae_time_interleave_upsample_2x,
    wan_vae_unpatchify_2x,
    wan_vae_zero_pad_2d_rb,
)

CACHE_T = 2


def check_is_instance(model, module_class):
    if isinstance(model, module_class):
        return True
    return bool(hasattr(model, "module") and isinstance(model.module, module_class))


def block_causal_mask(x, block_size):
    # params
    b, n, s, _, device = *x.size(), x.device
    assert s % block_size == 0
    num_blocks = s // block_size

    # build mask
    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        mask[:, :, i * block_size : (i + 1) * block_size, : (i + 1) * block_size] = 1
    return mask


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        p0, p1, p2 = int(self.padding[0]), int(self.padding[1]), int(self.padding[2])
        self._padding: tuple[int, int, int, int, int, int] = (p2, p2, p1, p1, 2 * p0, 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        if (
            not torch.is_grad_enabled()
            and x.is_cuda
            and x.ndim == 5
            and type(self) is CausalConv3d
            and not _has_hooks(self)
            and self.weight.device == x.device
            and (self.bias is None or self.bias.device == x.device)
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            pad_w_l, pad_w_r, pad_h_t, pad_h_b, pad_t_l, pad_t_r = self._padding
            if pad_w_l == pad_w_r and pad_h_t == pad_h_b and pad_t_r == 0:
                if pad_w_l == 0 and pad_h_t == 0 and pad_t_l == 0 and cache_x is None:
                    out = F.conv3d(x, self.weight, None, self.stride, (0, 0, 0), self.dilation, self.groups)
                    if self.bias is not None and can_use_wan_vae_residual_bias_add(None, out, self.bias):
                        return wan_vae_residual_bias_add(None, out, self.bias)
                    return out + self.bias.view(1, -1, 1, 1, 1) if self.bias is not None else out
                align_c = (
                    32
                    if (
                        self.groups == 1
                        and x.shape[1] < 8
                        and self.weight.shape[1] == x.shape[1]
                        and (is_channels_last_3d(x) or is_channels_last_3d(self.weight))
                    )
                    else 0
                )
                if can_use_wan_vae_causal_pad_cache(
                    x, pad_w=pad_w_l, pad_h=pad_h_t, pad_t_left=pad_t_l, align_c=align_c
                ):
                    x_pad, _ = wan_vae_causal_pad_cache(
                        x,
                        cache_in=cache_x,
                        pad_w=pad_w_l,
                        pad_h=pad_h_t,
                        pad_t_left=pad_t_l,
                        write_cache=False,
                        cache_t=CACHE_T,
                        align_c=align_c,
                    )
                    weight = self.weight
                    if align_c > weight.shape[1]:
                        weight = F.pad(weight, (0, 0, 0, 0, 0, 0, 0, align_c - weight.shape[1])).contiguous(
                            memory_format=torch.channels_last_3d
                        )
                    out = F.conv3d(x_pad, weight, None, self.stride, (0, 0, 0), self.dilation, self.groups)
                    if self.bias is not None and can_use_wan_vae_residual_bias_add(None, out, self.bias):
                        return wan_vae_residual_bias_add(None, out, self.bias)
                    return out + self.bias.view(1, -1, 1, 1, 1) if self.bias is not None else out
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


def _has_hooks(module: nn.Module) -> bool:
    return bool(module._forward_hooks or module._forward_pre_hooks)


def _any_submodule_has_hooks(module: nn.Module) -> bool:
    return any(_has_hooks(m) for m in module.modules())


def _can_use_internal_cl3d(module: nn.Module, x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.ndim == 5
        and not module.training
        and not torch.is_grad_enabled()
        and x.dtype in (torch.bfloat16, torch.float16)
        and not torch.compiler.is_compiling()
        and not _any_submodule_has_hooks(module)
    )


def _ndhwc_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c).permute(0, 3, 1, 2)


def _nhwc_to_ndhwc(x: torch.Tensor, b: int, t: int) -> torch.Tensor:
    _, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)


def _5d_to_4d(x: torch.Tensor) -> torch.Tensor:
    if is_channels_last_3d(x):
        return _ndhwc_to_nhwc(x)
    if x.shape[2] == 1:
        return x.squeeze(2)
    return rearrange(x, "b c t h w -> (b t) c h w")


def _4d_to_5d(x: torch.Tensor, b: int, t: int) -> torch.Tensor:
    if is_channels_last_2d(x):
        return _nhwc_to_ndhwc(x, b, t)
    if t == 1:
        return x.unsqueeze(2)
    return rearrange(x, "(b t) c h w -> b c t h w", t=t)


def _causal_conv3d_cached(
    x: torch.Tensor,
    conv: nn.Module,
    feat_cache: list | None,
    feat_idx: list[int],
    *,
    align_c_lt8: bool = False,
) -> torch.Tensor | None:
    """Single-pass cached ``CausalConv3d`` (optional ``C: <8 -> 8`` CL3D align + coalesced bias)."""
    if (
        torch.is_grad_enabled()
        or not x.is_cuda
        or x.ndim != 5
        or type(conv) is not CausalConv3d
        or _has_hooks(conv)
        or conv.weight.device != x.device
        or (conv.bias is not None and conv.bias.device != x.device)
    ):
        return None
    pad_w_l, pad_w_r, pad_h_t, pad_h_b, pad_t_l, pad_t_r = conv._padding
    if pad_w_l != pad_w_r or pad_h_t != pad_h_b or pad_t_r != 0:
        return None
    align_c = (
        32
        if (
            align_c_lt8
            and conv.groups == 1
            and x.shape[1] < 8
            and conv.weight.shape[1] == x.shape[1]
            and (is_channels_last_3d(x) or is_channels_last_3d(conv.weight))
        )
        else 0
    )
    if not can_use_wan_vae_causal_pad_cache(x, pad_w=pad_w_l, pad_h=pad_h_t, pad_t_left=pad_t_l, align_c=align_c):
        return None
    write_cache = feat_cache is not None
    prev_cache = None
    if write_cache:
        idx = feat_idx[0]
        raw_prev = feat_cache[idx]
        if isinstance(raw_prev, torch.Tensor):
            prev_cache = raw_prev
    x_pad, new_cache = wan_vae_causal_pad_cache(
        x,
        cache_in=prev_cache,
        pad_w=pad_w_l,
        pad_h=pad_h_t,
        pad_t_left=pad_t_l,
        write_cache=write_cache,
        cache_t=CACHE_T,
        align_c=align_c,
    )
    if write_cache:
        feat_cache[feat_idx[0]] = new_cache
        feat_idx[0] += 1
    weight = conv.weight
    if align_c > weight.shape[1]:
        weight = F.pad(weight, (0, 0, 0, 0, 0, 0, 0, align_c - weight.shape[1])).contiguous(
            memory_format=torch.channels_last_3d
        )
    out = F.conv3d(x_pad, weight, None, conv.stride, (0, 0, 0), conv.dilation, conv.groups)
    if conv.bias is not None:
        return wan_vae_residual_bias_add(None, out, conv.bias)
    return out


def _try_fused_rms_silu_causal_conv3d_deferred(
    x: torch.Tensor,
    rms: nn.Module,
    silu: nn.Module,
    conv: nn.Module,
    feat_cache: list | None,
    feat_idx: list[int],
    *,
    pre_bias: torch.Tensor | None = None,
    defer_bias: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Super-fuse ``(pre_bias +) RMS_norm -> SiLU -> CausalConv3d`` with optional deferred bias."""
    if (
        torch.is_grad_enabled()
        or not x.is_cuda
        or x.ndim != 5
        or type(rms) is not RMS_norm
        or not rms.channel_first
        or type(silu) is not nn.SiLU
        or type(conv) is not CausalConv3d
        or _has_hooks(rms)
        or _has_hooks(silu)
        or _has_hooks(conv)
        or rms.gamma.device != x.device
        or conv.weight.device != x.device
        or (conv.bias is not None and conv.bias.device != x.device)
        or (pre_bias is not None and pre_bias.device != x.device)
    ):
        return None
    if not can_use_wan_vae_rms_norm(x, channel_first=True):
        return None
    pad_w_l, pad_w_r, pad_h_t, pad_h_b, pad_t_l, pad_t_r = conv._padding
    if pad_w_l != pad_w_r or pad_h_t != pad_h_b or pad_t_r != 0:
        return None
    bias_t = rms.bias if isinstance(rms.bias, torch.Tensor) else None
    if bias_t is not None and bias_t.device != x.device:
        return None

    write_cache = feat_cache is not None
    prev_cache = None
    if write_cache:
        idx = feat_idx[0]
        raw_prev = feat_cache[idx]
        if isinstance(raw_prev, torch.Tensor):
            prev_cache = raw_prev
    x_pad, new_cache = wan_vae_rms_norm_silu_causal_pad(
        x,
        rms.gamma,
        bias_t,
        cache_in=prev_cache,
        pad_w=pad_w_l,
        pad_h=pad_h_t,
        pad_t_left=pad_t_l,
        scale=rms.scale,
        write_cache=write_cache,
        cache_t=CACHE_T,
        pre_bias=pre_bias,
    )
    if write_cache:
        feat_cache[feat_idx[0]] = new_cache
        feat_idx[0] += 1
    out = F.conv3d(x_pad, conv.weight, None, conv.stride, (0, 0, 0), conv.dilation, conv.groups)
    if defer_bias:
        return out, conv.bias
    if conv.bias is not None:
        return wan_vae_residual_bias_add(None, out, conv.bias), None
    return out, None


def _try_fused_rms_silu_causal_conv3d(
    x: torch.Tensor,
    rms: nn.Module,
    silu: nn.Module,
    conv: nn.Module,
    feat_cache: list | None,
    feat_idx: list[int],
) -> torch.Tensor | None:
    """Super-fuse ``RMS_norm -> SiLU -> CausalConv3d`` (with cache + pad) on CUDA inference."""
    res = _try_fused_rms_silu_causal_conv3d_deferred(x, rms, silu, conv, feat_cache, feat_idx, defer_bias=False)
    return None if res is None else res[0]


class RMS_norm(nn.Module):
    """Norm implementation."""

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        if x.is_cuda and not torch.is_grad_enabled() and self.channel_first and self.gamma.device == x.device:
            bias_t = self.bias if isinstance(self.bias, torch.Tensor) else None
            if (bias_t is None or bias_t.device == x.device) and can_use_wan_vae_rms_norm(
                x, channel_first=self.channel_first
            ):
                return wan_vae_rms_norm(x, self.gamma, bias_t, scale=self.scale, apply_silu=False)
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        if (
            x.is_cuda
            and not torch.is_grad_enabled()
            and x.ndim == 4
            and self.mode in ("nearest", "nearest-exact")
            and self.scale_factor == (2.0, 2.0)
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return wan_vae_nearest_upsample_2x(x)
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def _can_use_fused_resample(self, x: torch.Tensor) -> bool:
        if (
            self.training
            or torch.is_grad_enabled()
            or not x.is_cuda
            or x.ndim != 5
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or _has_hooks(self)
            or _has_hooks(self.resample)
            or type(self.resample) is not nn.Sequential
            or len(self.resample) != 2
        ):
            return False
        first, conv2d = self.resample[0], self.resample[1]
        if (
            _has_hooks(first)
            or _has_hooks(conv2d)
            or type(conv2d) is not nn.Conv2d
            or conv2d.weight.device != x.device
            or (conv2d.bias is not None and conv2d.bias.device != x.device)
        ):
            return False
        if self.mode in ("upsample2d", "upsample3d"):
            if (
                type(first) is not Upsample
                or first.scale_factor != (2.0, 2.0)
                or first.mode not in ("nearest", "nearest-exact")
            ):
                return False
            if self.mode == "upsample3d":
                tc = self.time_conv
                if (
                    type(tc) is not CausalConv3d
                    or _has_hooks(tc)
                    or tc.weight.device != x.device
                    or (tc.bias is not None and tc.bias.device != x.device)
                    or tc._padding != (0, 0, 0, 0, 2, 0)
                ):
                    return False
            return True
        if self.mode in ("downsample2d", "downsample3d"):
            if type(first) is not nn.ZeroPad2d or first.padding != (0, 1, 0, 1):
                return False
            if self.mode == "downsample3d":
                tc = self.time_conv
                if (
                    type(tc) is not CausalConv3d
                    or _has_hooks(tc)
                    or tc.weight.device != x.device
                    or (tc.bias is not None and tc.bias.device != x.device)
                    or tc._padding != (0, 0, 0, 0, 0, 0)
                ):
                    return False
            return True
        return False

    def _forward_fused_deferred(
        self,
        x: torch.Tensor,
        feat_cache: list | None,
        feat_idx: list[int],
        *,
        defer_out_bias: bool = False,
        allow_4d_out: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list | None, list[int]]:
        b, _, t, _, _ = x.size()
        assert isinstance(self.resample, nn.Sequential)
        conv2d = self.resample[1]
        assert isinstance(conv2d, nn.Conv2d)
        if self.mode in ("upsample2d", "upsample3d"):
            if self.mode == "upsample3d" and feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                    x_up = wan_vae_nearest_upsample_2x(_5d_to_4d(x))
                    t_out = t
                else:
                    is_rep = isinstance(feat_cache[idx], str)
                    prev_c = None if is_rep else feat_cache[idx]
                    x_pad, new_cache = wan_vae_causal_pad_cache(
                        x,
                        cache_in=prev_c,
                        pad_w=0,
                        pad_h=0,
                        pad_t_left=2,
                        write_cache=True,
                        cache_t=CACHE_T,
                        pad_cache_zero_left=is_rep,
                    )
                    feat_cache[idx] = new_cache
                    feat_idx[0] += 1
                    tc = self.time_conv
                    h_time = F.conv3d(x_pad, tc.weight, None, tc.stride, (0, 0, 0), tc.dilation, tc.groups)
                    x_up = wan_vae_time_interleave_upsample_2x(h_time, tc.bias)
                    t_out = t * 2
            else:
                x_up = wan_vae_nearest_upsample_2x(_5d_to_4d(x))
                t_out = t
            h_2d = F.conv2d(x_up, conv2d.weight, None, conv2d.stride, conv2d.padding, conv2d.dilation, conv2d.groups)
            if defer_out_bias:
                out_t = h_2d if allow_4d_out else _4d_to_5d(h_2d, b, t_out)
                return out_t, conv2d.bias, feat_cache, feat_idx
            h_5d = _4d_to_5d(h_2d, b, t_out)
            out_t = wan_vae_residual_bias_add(None, h_5d, conv2d.bias) if conv2d.bias is not None else h_5d
            return out_t, None, feat_cache, feat_idx

        # downsample2d / downsample3d
        x_pad2d = wan_vae_zero_pad_2d_rb(_5d_to_4d(x))
        h_2d = F.conv2d(x_pad2d, conv2d.weight, None, conv2d.stride, conv2d.padding, conv2d.dilation, conv2d.groups)
        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                h_5d = _4d_to_5d(h_2d, b, t)
                h_5d = wan_vae_residual_bias_add(None, h_5d, conv2d.bias) if conv2d.bias is not None else h_5d
                feat_cache[idx] = h_5d.clone()
                feat_idx[0] += 1
                return h_5d, None, feat_cache, feat_idx
            h_5d = _4d_to_5d(h_2d, b, t)
            prev_last = feat_cache[idx][:, :, -1:, :, :]
            x_cat, new_cache = wan_vae_causal_pad_cache(
                h_5d,
                cache_in=prev_last,
                pad_w=0,
                pad_h=0,
                pad_t_left=1,
                write_cache=True,
                cache_t=1,
                pre_bias=conv2d.bias,
            )
            feat_cache[idx] = new_cache
            feat_idx[0] += 1
            tc = self.time_conv
            h_t = F.conv3d(x_cat, tc.weight, None, tc.stride, (0, 0, 0), tc.dilation, tc.groups)
            if defer_out_bias:
                return h_t, tc.bias, feat_cache, feat_idx
            out_t = wan_vae_residual_bias_add(None, h_t, tc.bias) if tc.bias is not None else h_t
            return out_t, None, feat_cache, feat_idx

        if defer_out_bias:
            out_t = h_2d if allow_4d_out else _4d_to_5d(h_2d, b, t)
            return out_t, conv2d.bias, feat_cache, feat_idx
        h_5d = _4d_to_5d(h_2d, b, t)
        out_t = wan_vae_residual_bias_add(None, h_5d, conv2d.bias) if conv2d.bias is not None else h_5d
        return out_t, None, feat_cache, feat_idx

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        if self._can_use_fused_resample(x):
            out, _, feat_cache, feat_idx = self._forward_fused_deferred(
                x, feat_cache, feat_idx, defer_out_bias=False, allow_4d_out=False
            )
            return out, feat_cache, feat_idx
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and not isinstance(feat_cache[idx], str):
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                if cache_x.shape[2] < 2 and isinstance(feat_cache[idx], str):
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                x = self.time_conv(x) if isinstance(feat_cache[idx], str) else self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x, feat_cache, feat_idx

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        if (
            patch_size == 2
            and x.is_cuda
            and not torch.is_grad_enabled()
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return wan_vae_patchify_2x(x)
        x = rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=patch_size, r=patch_size)
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        if (
            patch_size == 2
            and x.is_cuda
            and not torch.is_grad_enabled()
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return wan_vae_unpatchify_2x(x)
        x = rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=patch_size, r=patch_size)
    return x


class Resample38(Resample):
    """Resample implementation."""

    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super(Resample, self).__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        layers = self.residual
        if (
            not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.ndim == 5
            and len(layers) == 7
            and type(layers[5]) is nn.Dropout
            and not _has_hooks(layers[5])
        ):
            can_fuse_second = (
                type(layers[3]) is RMS_norm
                and layers[3].channel_first
                and type(layers[4]) is nn.SiLU
                and type(layers[6]) is CausalConv3d
                and not _has_hooks(layers[3])
                and not _has_hooks(layers[4])
                and not _has_hooks(layers[6])
            )
            res1 = _try_fused_rms_silu_causal_conv3d_deferred(
                x, layers[0], layers[1], layers[2], feat_cache, feat_idx, defer_bias=can_fuse_second
            )
            if res1 is not None:
                out1, bias1 = res1
                res2 = _try_fused_rms_silu_causal_conv3d_deferred(
                    out1, layers[3], layers[4], layers[6], feat_cache, feat_idx, pre_bias=bias1, defer_bias=True
                )
                if res2 is not None:
                    out2, bias2 = res2
                    sc = self.shortcut
                    h_bias: torch.Tensor | None = None
                    if (
                        type(sc) is CausalConv3d
                        and not _has_hooks(sc)
                        and sc._padding == (0, 0, 0, 0, 0, 0)
                        and sc.weight.device == x.device
                        and (sc.bias is None or sc.bias.device == x.device)
                    ):
                        h = F.conv3d(x, sc.weight, None, sc.stride, (0, 0, 0), sc.dilation, sc.groups)
                        h_bias = sc.bias
                    else:
                        h = sc(x)
                    if can_use_wan_vae_residual_bias_add(h, out2, bias2, h_bias):
                        return wan_vae_residual_bias_add(h, out2, bias2, h_bias), feat_cache, feat_idx
                    if bias2 is not None:
                        out2 = out2 + bias2.view(1, -1, 1, 1, 1)
                    if h_bias is not None:
                        h = h + h_bias.view(1, -1, 1, 1, 1)
                    return out2 + h, feat_cache, feat_idx
                # Fall through for second half only if conv2 was hooked/wrapped.
                h = self.shortcut(x)
                x = out1 + bias1.view(1, -1, 1, 1, 1) if bias1 is not None else out1
                for layer in (layers[3], layers[4], layers[5], layers[6]):
                    if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                        idx = feat_idx[0]
                        cache_x = x[:, :, -CACHE_T:, :, :].clone()
                        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                            cache_x = torch.cat(
                                [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                            )
                        x = layer(x, feat_cache[idx])
                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1
                    else:
                        x = layer(x)
                return x + h, feat_cache, feat_idx
        h = self.shortcut(x)
        for layer in layers:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h, feat_cache, feat_idx


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        is_cl3d = is_channels_last_3d(x)
        x_2d = _5d_to_4d(x)
        x_2d = self.norm(x_2d)

        # compute query, key, value
        if (
            not torch.is_grad_enabled()
            and x_2d.is_cuda
            and type(self.to_qkv) is nn.Conv2d
            and not _has_hooks(self.to_qkv)
            and self.to_qkv.weight.device == x_2d.device
            and (self.to_qkv.bias is None or self.to_qkv.bias.device == x_2d.device)
        ):
            qkv = F.conv2d(
                x_2d,
                self.to_qkv.weight,
                None,
                self.to_qkv.stride,
                self.to_qkv.padding,
                self.to_qkv.dilation,
                self.to_qkv.groups,
            )
            if self.to_qkv.bias is not None and can_use_wan_vae_residual_bias_add(None, qkv, self.to_qkv.bias):
                qkv = wan_vae_residual_bias_add(None, qkv, self.to_qkv.bias)
            elif self.to_qkv.bias is not None:
                qkv = qkv + self.to_qkv.bias.view(1, -1, 1, 1)
        else:
            qkv = self.to_qkv(x_2d)

        if is_channels_last_2d(qkv):
            q, k, v = qkv.permute(0, 2, 3, 1).reshape(b * t, 1, h * w, c * 3).chunk(3, dim=-1)
        else:
            q, k, v = qkv.reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x_attn = _worldfoundry_scaled_dot_product_attention(
            q,
            k,
            v,
            # attn_mask=block_causal_mask(q, block_size=h * w)
        )
        if is_cl3d:
            x_2d = x_attn.squeeze(1).reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        else:
            x_2d = x_attn.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w).contiguous()

        # output (+ fused proj.bias + identity)
        if (
            not torch.is_grad_enabled()
            and x_2d.is_cuda
            and type(self.proj) is nn.Conv2d
            and not _has_hooks(self.proj)
            and self.proj.weight.device == x_2d.device
            and (self.proj.bias is None or self.proj.bias.device == x_2d.device)
        ):
            proj_2d = F.conv2d(
                x_2d,
                self.proj.weight,
                None,
                self.proj.stride,
                self.proj.padding,
                self.proj.dilation,
                self.proj.groups,
            )
            proj_5d = (
                _nhwc_to_ndhwc(proj_2d, b, t)
                if (is_cl3d and is_channels_last_2d(proj_2d))
                else rearrange(proj_2d, "(b t) c h w -> b c t h w", t=t)
            )
            if can_use_wan_vae_residual_bias_add(identity, proj_5d, self.proj.bias):
                return wan_vae_residual_bias_add(identity, proj_5d, self.proj.bias)
            if self.proj.bias is not None:
                proj_5d = proj_5d + self.proj.bias.view(1, -1, 1, 1, 1)
            return proj_5d + identity

        x_2d = self.proj(x_2d)
        x_5d = (
            _nhwc_to_ndhwc(x_2d, b, t)
            if (is_cl3d and is_channels_last_2d(x_2d))
            else rearrange(x_2d, "(b t) c h w-> b c t h w", t=t)
        )
        return x_5d + identity


class AvgDown3D(nn.Module):
    """Avg down d implementation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and not torch.is_grad_enabled()
            and x.ndim == 5
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            if self.factor_t == 1 and self.factor_s == 1 and self.in_channels == self.out_channels:
                return x
            return wan_vae_avg_down_3d(
                x,
                out_channels=self.out_channels,
                factor_t=self.factor_t,
                factor_s=self.factor_s,
            )
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    """Dup up d implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        if (
            x.is_cuda
            and not torch.is_grad_enabled()
            and x.ndim == 5
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return wan_vae_dup_up_3d(
                x,
                out_channels=self.out_channels,
                factor_t=self.factor_t,
                factor_s=self.factor_s,
                first_chunk=bool(first_chunk),
            )
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample38(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        infer_fast = not self.training and not torch.is_grad_enabled() and x.is_cuda and x.ndim == 5
        x_orig = x if infer_fast else x.clone()
        sc = self.avg_shortcut
        if infer_fast and type(sc) is AvgDown3D and not _has_hooks(sc) and len(self.downsamples) > 0:
            down_modules = list(self.downsamples)
            last = down_modules[-1]
            if isinstance(last, Resample) and last._can_use_fused_resample(x):
                for module in down_modules[:-1]:
                    x, feat_cache, feat_idx = module(x, feat_cache, feat_idx)
                h_out, h_bias, feat_cache, feat_idx = last._forward_fused_deferred(
                    x, feat_cache, feat_idx, defer_out_bias=True, allow_4d_out=True
                )
                if can_use_wan_vae_avg_down_3d_add(
                    h_out,
                    x_orig,
                    h_bias,
                    out_channels=sc.out_channels,
                    factor_t=sc.factor_t,
                    factor_s=sc.factor_s,
                ):
                    out = wan_vae_avg_down_3d_add(
                        h_out,
                        x_orig,
                        h_bias,
                        out_channels=sc.out_channels,
                        factor_t=sc.factor_t,
                        factor_s=sc.factor_s,
                    )
                    return out, feat_cache, feat_idx
                if h_out.ndim == 4:
                    h_out = _4d_to_5d(h_out, x_orig.shape[0], h_out.shape[0] // x_orig.shape[0])
                if h_bias is not None:
                    h_out = wan_vae_residual_bias_add(None, h_out, h_bias)
                return h_out + sc(x_orig), feat_cache, feat_idx
            for module in self.downsamples:
                x, feat_cache, feat_idx = module(x, feat_cache, feat_idx)
            if sc.factor_t == 1 and sc.factor_s == 1 and sc.in_channels == sc.out_channels:
                if can_use_wan_vae_residual_bias_add(x_orig, x, None):
                    return wan_vae_residual_bias_add(x_orig, x, None), feat_cache, feat_idx
                return x + x_orig, feat_cache, feat_idx
            if can_use_wan_vae_avg_down_3d_add(
                x,
                x_orig,
                None,
                out_channels=sc.out_channels,
                factor_t=sc.factor_t,
                factor_s=sc.factor_s,
            ):
                return (
                    wan_vae_avg_down_3d_add(
                        x,
                        x_orig,
                        None,
                        out_channels=sc.out_channels,
                        factor_t=sc.factor_t,
                        factor_s=sc.factor_s,
                    ),
                    feat_cache,
                    feat_idx,
                )
            return x + sc(x_orig), feat_cache, feat_idx

        for module in self.downsamples:
            x, feat_cache, feat_idx = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_orig), feat_cache, feat_idx


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample38(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):  # noqa: B006 - Keeps the upstream default object and signature.
        infer_fast = not self.training and not torch.is_grad_enabled() and x.is_cuda and x.ndim == 5
        x_main = x if infer_fast else x.clone()
        sc = self.avg_shortcut
        if infer_fast and sc is not None and type(sc) is DupUp3D and not _has_hooks(sc) and len(self.upsamples) > 0:
            up_modules = list(self.upsamples)
            last = up_modules[-1]
            if isinstance(last, Resample) and last._can_use_fused_resample(x_main):
                for module in up_modules[:-1]:
                    x_main, feat_cache, feat_idx = module(x_main, feat_cache, feat_idx)
                h_out, h_bias, feat_cache, feat_idx = last._forward_fused_deferred(
                    x_main, feat_cache, feat_idx, defer_out_bias=True, allow_4d_out=True
                )
                if can_use_wan_vae_dup_up_3d_add(
                    h_out,
                    x,
                    h_bias,
                    out_channels=sc.out_channels,
                    factor_t=sc.factor_t,
                    factor_s=sc.factor_s,
                    first_chunk=bool(first_chunk),
                ):
                    out = wan_vae_dup_up_3d_add(
                        h_out,
                        x,
                        h_bias,
                        out_channels=sc.out_channels,
                        factor_t=sc.factor_t,
                        factor_s=sc.factor_s,
                        first_chunk=bool(first_chunk),
                    )
                    return out, feat_cache, feat_idx
                if h_out.ndim == 4:
                    h_out = _4d_to_5d(h_out, x.shape[0], h_out.shape[0] // x.shape[0])
                if h_bias is not None:
                    h_out = wan_vae_residual_bias_add(None, h_out, h_bias)
                return h_out + sc(x, first_chunk), feat_cache, feat_idx

        for module in self.upsamples:
            x_main, feat_cache, feat_idx = module(x_main, feat_cache, feat_idx)
        if sc is not None:
            if (
                infer_fast
                and type(sc) is DupUp3D
                and not _has_hooks(sc)
                and can_use_wan_vae_dup_up_3d_add(
                    x_main,
                    x,
                    None,
                    out_channels=sc.out_channels,
                    factor_t=sc.factor_t,
                    factor_s=sc.factor_s,
                    first_chunk=bool(first_chunk),
                )
            ):
                return (
                    wan_vae_dup_up_3d_add(
                        x_main,
                        x,
                        None,
                        out_channels=sc.out_channels,
                        factor_t=sc.factor_t,
                        factor_s=sc.factor_s,
                        first_chunk=bool(first_chunk),
                    ),
                    feat_cache,
                    feat_idx,
                )
            x_shortcut = sc(x, first_chunk)
            return x_main + x_shortcut, feat_cache, feat_idx
        return x_main, feat_cache, feat_idx


class Encoder3d(nn.Module):
    """Encoder d implementation."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temperal_downsample=[True, True, False],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout)
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        conv1_out = (
            _causal_conv3d_cached(x, self.conv1, feat_cache, feat_idx, align_c_lt8=True) if not self.training else None
        )
        if conv1_out is not None:
            x = conv1_out
        elif feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        head_out = (
            _try_fused_rms_silu_causal_conv3d(x, self.head[0], self.head[1], self.head[2], feat_cache, feat_idx)
            if (not self.training and len(self.head) == 3)
            else None
        )
        if head_out is not None:
            return head_out, feat_cache, feat_idx
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache, feat_idx


class Encoder3d_38(nn.Module):
    """Encoder d implementation."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temperal_downsample=[False, True, True],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        conv1_out = _causal_conv3d_cached(x, self.conv1, feat_cache, feat_idx) if not self.training else None
        if conv1_out is not None:
            x = conv1_out
        elif feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        head_out = (
            _try_fused_rms_silu_causal_conv3d(x, self.head[0], self.head[1], self.head[2], feat_cache, feat_idx)
            if (not self.training and len(self.head) == 3)
            else None
        )
        if head_out is not None:
            return head_out, feat_cache, feat_idx
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x, feat_cache, feat_idx


class Decoder3d(nn.Module):
    """Decoder d implementation."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temperal_upsample=[False, True, True],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout)
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):  # noqa: B006 - Keeps the upstream default object and signature.
        ## conv1
        conv1_out = _causal_conv3d_cached(x, self.conv1, feat_cache, feat_idx) if not self.training else None
        if conv1_out is not None:
            x = conv1_out
        elif feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        head_out = (
            _try_fused_rms_silu_causal_conv3d(x, self.head[0], self.head[1], self.head[2], feat_cache, feat_idx)
            if (not self.training and len(self.head) == 3)
            else None
        )
        if head_out is not None:
            return head_out, feat_cache, feat_idx
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache, feat_idx


class Decoder3d_38(nn.Module):
    """Decoder d implementation."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temperal_upsample=[False, True, True],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]  # noqa: RUF005 - Keeps `+` semantics for the operand's runtime type.
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout)
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 12, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):  # noqa: B006 - Keeps the upstream default object and signature.
        conv1_out = _causal_conv3d_cached(x, self.conv1, feat_cache, feat_idx) if not self.training else None
        if conv1_out is not None:
            x = conv1_out
        elif feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        ## head
        head_out = (
            _try_fused_rms_silu_causal_conv3d(x, self.head[0], self.head[1], self.head[2], feat_cache, feat_idx)
            if (not self.training and len(self.head) == 3)
            else None
        )
        if head_out is not None:
            return head_out, feat_cache, feat_idx
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x, feat_cache, feat_idx


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class VideoVAE_(nn.Module):
    """Video implementation."""

    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temperal_downsample=[False, True, True],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        ## cache
        in_was_cl3d = is_channels_last_3d(x)
        use_internal_cl3d = _can_use_internal_cl3d(self, x)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        outs = []
        for i in range(iter_):
            self._enc_conv_idx = [0]
            chunk = x[:, :, :1, :, :] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
            if use_internal_cl3d and not is_channels_last_3d(chunk):
                chunk = chunk.contiguous(memory_format=torch.channels_last_3d)
            out_, self._enc_feat_map, self._enc_conv_idx = self.encoder(
                chunk,
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            outs.append(out_)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        if use_internal_cl3d and not in_was_cl3d and is_channels_last_3d(mu):
            mu = mu.contiguous()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        in_was_cl3d = is_channels_last_3d(z)
        use_internal_cl3d = _can_use_internal_cl3d(self, z)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        if use_internal_cl3d and not is_channels_last_3d(z):
            z = z.contiguous(memory_format=torch.channels_last_3d)
        iter_ = z.shape[2]
        x = self.conv2(z)
        outs = []
        for i in range(iter_):
            self._conv_idx = [0]
            out_, self._feat_map, self._conv_idx = self.decoder(
                x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx
            )
            outs.append(out_)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        if use_internal_cl3d and not in_was_cl3d and is_channels_last_3d(out):
            out = out.contiguous()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


class WanVideoVAE(nn.Module):
    def __init__(self, z_dim=16, vae_pretrained_path: str | None = None, *, base_dim: int = 96):
        super().__init__()

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE_(dim=base_dim, z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.z_dim = z_dim
        self.vae_pretrained_path = vae_pretrained_path
        self.decode_autocast_dtype: torch.dtype | None = None

    def _decode_autocast_is_redundant(
        self,
        input_dtype: torch.dtype | None,
    ) -> bool:
        """Whether resident weights and input already satisfy the cast target."""

        dtype = getattr(self, "decode_autocast_dtype", None)
        if dtype is None or input_dtype != dtype:
            return False
        model = getattr(self, "model", None)
        if not isinstance(model, nn.Module):
            return False
        parameter = next(model.parameters(), None)
        return parameter is not None and parameter.dtype == dtype

    def _decode_autocast(self, input_dtype: torch.dtype | None = None):
        """Return the opt-in CUDA autocast context used only for VAE decode.

        Autocast is useful when FP32-resident weights should execute in BF16 or
        FP16.  It adds material dispatcher overhead to this convolution-heavy
        codec when weights and inputs already use the requested dtype, so that
        provably redundant case is elided.
        """

        dtype = getattr(self, "decode_autocast_dtype", None)
        if dtype is None or self._decode_autocast_is_redundant(input_dtype):
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    def build_1d_mask(self, length, left_bound, right_bound, border_width, *, device=None, dtype=None):
        x = torch.ones((length,), dtype=torch.float32, device=device)
        if not left_bound and border_width > 0:
            x[:border_width] = (torch.arange(border_width, dtype=torch.float32, device=device) + 1) / border_width
        if not right_bound and border_width > 0:
            x[-border_width:] = torch.flip(
                (torch.arange(border_width, dtype=torch.float32, device=device) + 1) / border_width,
                dims=(0,),
            )
        return x.to(dtype=dtype) if dtype is not None else x

    def build_mask(self, data, is_bound, border_width, *, device=None, dtype=None):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0], device=device)
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1], device=device)
        mask = torch.minimum(h[:, None], w[None, :]).reshape(1, 1, 1, H, W)
        return mask.to(dtype=dtype) if dtype is not None else mask

    @staticmethod
    def _spatial_tile_tasks(height, width, tile_size, tile_stride):
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = []
        for h in range(0, height, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= height:
                continue
            for w in range(0, width, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= width:
                    continue
                tasks.append((h, h + size_h, w, w + size_w))
        return tasks

    @staticmethod
    def parallel_tile_config_for_world_size(
        height: int,
        width: int,
        world_size: int,
        *,
        min_overlap: int = 4,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Suggest balanced ``(tile_size, tile_stride)`` yielding ``world_size`` spatial tiles.

        Callers must opt in explicitly; ``parallel_tiled_decode`` never silently
        overrides user-specified ``tile_size`` / ``tile_stride`` so parallel
        output remains bit-exact with single-GPU ``tiled_decode``.
        """
        height = int(height)
        width = int(width)
        world_size = int(world_size)
        if height <= 0 or width <= 0 or world_size <= 0:
            raise ValueError("height, width, and world_size must be positive")
        if min_overlap < 0:
            raise ValueError("min_overlap cannot be negative")
        if world_size == 1:
            return (height, width), (height, width)

        best_pair: tuple[int, int] | None = None
        best_score = float("inf")
        for rows in range(1, world_size + 1):
            if world_size % rows != 0:
                continue
            cols = world_size // rows
            if rows > height or cols > width:
                continue
            tile_h = height / rows
            tile_w = width / cols
            aspect_penalty = max(tile_h / tile_w, tile_w / tile_h)
            if aspect_penalty < best_score:
                best_score = aspect_penalty
                best_pair = (rows, cols)

        if best_pair is None:
            rows = min(height, max(1, math.isqrt(world_size)))
            cols = min(width, max(1, math.ceil(world_size / rows)))
            best_pair = (rows, cols)

        def _axis_size_stride(length: int, count: int) -> tuple[int, int]:
            if count <= 1 or length <= 1:
                return length, length
            count = min(count, length)
            max_overlap = max(0, (length - count) // max(count - 1, 1))
            overlap = min(min_overlap, max_overlap)
            while overlap >= 0:
                stride = max(1, math.ceil((length - overlap) / count))
                size = stride + overlap
                if (count - 1) * stride + size >= length and (count - 2) * stride + size < length:
                    return size, stride
                overlap -= 1
            stride = max(1, math.ceil(length / count))
            return stride, stride

        size_h, stride_h = _axis_size_stride(height, best_pair[0])
        size_w, stride_w = _axis_size_stride(width, best_pair[1])
        return (size_h, size_w), (stride_h, stride_w)

    @staticmethod
    def _resolve_merge_device(
        decoded_tiles,
        merge_device: torch.device | str | None = None,
    ) -> torch.device:
        if merge_device is not None:
            return torch.device(merge_device)
        if decoded_tiles and all(tile.device.type == "cuda" for tile in decoded_tiles):
            return decoded_tiles[0].device
        return torch.device("cpu")

    def _accumulate_spatial_decode_tile(
        self,
        values: torch.Tensor,
        weight: torch.Tensor,
        decoded_tile: torch.Tensor,
        task: tuple[int, int, int, int],
        *,
        height: int,
        width: int,
        border_width: tuple[int, int],
        data_device: torch.device,
        target_dtype: torch.dtype,
        mask_cache: dict[tuple[int, int, bool, bool, bool, bool], torch.Tensor],
    ) -> None:
        h, h_, w, w_ = task
        hidden_states_batch = decoded_tile.to(device=data_device, dtype=target_dtype)
        tile_h = int(hidden_states_batch.shape[3])
        tile_w = int(hidden_states_batch.shape[4])
        is_bound = (h == 0, h_ >= height, w == 0, w_ >= width)
        cache_key = (tile_h, tile_w, *is_bound)
        mask = mask_cache.get(cache_key)
        if mask is None:
            mask = self.build_mask(
                hidden_states_batch,
                is_bound=is_bound,
                border_width=border_width,
                device=data_device,
                dtype=target_dtype,
            )
            mask_cache[cache_key] = mask
        target_h = h * self.upsampling_factor
        target_w = w * self.upsampling_factor
        values[
            :,
            :,
            :,
            target_h : target_h + tile_h,
            target_w : target_w + tile_w,
        ].addcmul_(hidden_states_batch, mask)
        weight[
            :,
            :,
            :,
            target_h : target_h + tile_h,
            target_w : target_w + tile_w,
        ] += mask

    def _merge_spatial_decode_tiles(
        self,
        decoded_tiles,
        tasks,
        *,
        hidden_states,
        tile_size,
        tile_stride,
        merge_device: torch.device | str | None = None,
    ):
        _, _, latent_frames, height, width = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        output_frames = latent_frames * 4 - 3
        if len(decoded_tiles) != len(tasks):
            raise RuntimeError(f"decoded {len(decoded_tiles)} VAE tiles for {len(tasks)} spatial tasks")
        data_device = self._resolve_merge_device(decoded_tiles, merge_device=merge_device)
        out_h = height * self.upsampling_factor
        out_w = width * self.upsampling_factor
        border_width = (
            (size_h - stride_h) * self.upsampling_factor,
            (size_w - stride_w) * self.upsampling_factor,
        )
        # Spatial blending weights are time-invariant: shape (1, 1, 1, H, W)
        # broadcasts across T and C, cutting weight memory/traffic by T x.
        try:
            weight = torch.zeros(
                (1, 1, 1, out_h, out_w),
                dtype=hidden_states.dtype,
                device=data_device,
            )
            values = torch.zeros(
                (1, 3, output_frames, out_h, out_w),
                dtype=hidden_states.dtype,
                device=data_device,
            )
        except torch.OutOfMemoryError:
            if data_device.type != "cuda":
                raise
            data_device = torch.device("cpu")
            torch.cuda.empty_cache()
            weight = torch.zeros((1, 1, 1, out_h, out_w), dtype=hidden_states.dtype, device=data_device)
            values = torch.zeros((1, 3, output_frames, out_h, out_w), dtype=hidden_states.dtype, device=data_device)
        mask_cache: dict[tuple[int, int, bool, bool, bool, bool], torch.Tensor] = {}
        for index, task in enumerate(tasks):
            self._accumulate_spatial_decode_tile(
                values,
                weight,
                decoded_tiles[index],
                task,
                height=height,
                width=width,
                border_width=border_width,
                data_device=data_device,
                target_dtype=hidden_states.dtype,
                mask_cache=mask_cache,
            )
        if bool((weight == 0).any()):
            raise RuntimeError("Wan VAE spatial tiles did not cover the complete output")
        return values.div_(weight).clamp_(-1, 1)

    def _decode_tile(self, hidden_states, device, temporal_chunk_size=0):
        if temporal_chunk_size:
            return self.temporal_chunked_decode(
                hidden_states,
                device,
                chunk_size=temporal_chunk_size,
            )
        return self.model.decode(hidden_states, self.scale)

    def tiled_decode(
        self,
        hidden_states,
        device,
        tile_size,
        tile_stride,
        temporal_chunk_size=0,
    ):
        _, _, latent_frames, height, width = hidden_states.shape
        tasks = self._spatial_tile_tasks(height, width, tile_size, tile_stride)
        target_device = torch.device(device)
        merge_device = (
            target_device if (target_device.type == "cuda" and not temporal_chunk_size) else torch.device("cpu")
        )
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        output_frames = latent_frames * 4 - 3
        out_h = height * self.upsampling_factor
        out_w = width * self.upsampling_factor
        border_width = (
            (size_h - stride_h) * self.upsampling_factor,
            (size_w - stride_w) * self.upsampling_factor,
        )
        try:
            weight = torch.zeros((1, 1, 1, out_h, out_w), dtype=hidden_states.dtype, device=merge_device)
            values = torch.zeros((1, 3, output_frames, out_h, out_w), dtype=hidden_states.dtype, device=merge_device)
        except torch.OutOfMemoryError:
            if merge_device.type != "cuda":
                raise
            merge_device = torch.device("cpu")
            torch.cuda.empty_cache()
            weight = torch.zeros((1, 1, 1, out_h, out_w), dtype=hidden_states.dtype, device=merge_device)
            values = torch.zeros((1, 3, output_frames, out_h, out_w), dtype=hidden_states.dtype, device=merge_device)
        mask_cache: dict[tuple[int, int, bool, bool, bool, bool], torch.Tensor] = {}
        for task in tqdm(tasks, desc="VAE decoding"):
            h, h_, w, w_ = task
            tile = hidden_states[:, :, :, h:h_, w:w_]
            if tile.device != target_device:
                tile = tile.to(target_device)
            try:
                decoded = self._decode_tile(tile, target_device, temporal_chunk_size)
            except torch.OutOfMemoryError:
                if merge_device.type != "cuda":
                    raise
                merge_device = torch.device("cpu")
                values = values.to(merge_device)
                weight = weight.to(merge_device)
                mask_cache.clear()
                torch.cuda.empty_cache()
                decoded = self._decode_tile(tile, target_device, temporal_chunk_size)
            self._accumulate_spatial_decode_tile(
                values,
                weight,
                decoded,
                task,
                height=height,
                width=width,
                border_width=border_width,
                data_device=merge_device,
                target_dtype=hidden_states.dtype,
                mask_cache=mask_cache,
            )
            del decoded
        if bool((weight == 0).any()):
            raise RuntimeError("Wan VAE spatial tiles did not cover the complete output")
        return values.div_(weight).clamp_(-1, 1)

    def parallel_tiled_decode(
        self,
        hidden_states,
        device,
        tile_size,
        tile_stride,
        *,
        temporal_chunk_size=0,
        group=None,
    ):
        """Shard spatial VAE tiles across ranks and gather decoded tiles on-device.

        Every rank reconstructs the same output in exact task order so results
        remain bit-exact with single-GPU ``tiled_decode`` and downstream runner
        control flow stays rank-consistent. On CUDA without temporal chunking,
        the entire decode -> contiguous ``all_gather_into_tensor`` -> merge
        pipeline stays 100% GPU-resident with zero CPU round-trips.
        """

        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("parallel Wan VAE decode requires an initialized process group")
        world_size = int(dist.get_world_size(group))
        rank = int(dist.get_rank(group))
        if world_size <= 1:
            raise RuntimeError("parallel Wan VAE decode requires more than one rank")
        target_device = torch.device(device)
        _, _, latent_frames, height, width = hidden_states.shape
        output_frames = latent_frames * 4 - 3
        tasks = self._spatial_tile_tasks(height, width, tile_size, tile_stride)

        def _expected_tile_shape(task: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
            h, h_, w, w_ = task
            return (
                1,
                3,
                output_frames,
                (min(h_, height) - h) * self.upsampling_factor,
                (min(w_, width) - w) * self.upsampling_factor,
            )

        local_metadata = []
        local_tiles = []
        local_error = None
        nonstandard_shape = False
        try:
            for task_index in range(rank, len(tasks), world_size):
                h, h_, w, w_ = tasks[task_index]
                tile = hidden_states[:, :, :, h:h_, w:w_]
                if tile.device != target_device:
                    tile = tile.to(target_device)
                decoded = self._decode_tile(tile, target_device, temporal_chunk_size).to(
                    device=target_device,
                    dtype=hidden_states.dtype,
                )
                decoded_shape = tuple(int(dim) for dim in decoded.shape)
                if decoded_shape != _expected_tile_shape(tasks[task_index]):
                    nonstandard_shape = True
                local_metadata.append((task_index, decoded_shape, decoded.numel()))
                local_tiles.append(decoded)
        except Exception as error:  # all ranks must reach the status collective
            local_error = f"rank {rank}: {type(error).__name__}: {error}"

        status = torch.tensor(
            [1 if local_error is not None else 0, 1 if nonstandard_shape else 0],
            dtype=torch.int32,
            device=target_device,
        )
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
        if int(status[0].item()):
            errors = [None] * world_size
            dist.all_gather_object(errors, local_error, group=group)
            failures = [error for error in errors if error]
            raise RuntimeError("parallel Wan VAE tile decode failed: " + "; ".join(failures))

        if int(status[1].item()):
            metadata_by_rank = [None] * world_size
            dist.all_gather_object(metadata_by_rank, local_metadata, group=group)
        else:
            metadata_by_rank = [
                [
                    (
                        task_index,
                        (shape := _expected_tile_shape(tasks[task_index])),
                        math.prod(shape),
                    )
                    for task_index in range(rank_index, len(tasks), world_size)
                ]
                for rank_index in range(world_size)
            ]

        rank_sizes = [sum(int(item[2]) for item in (metadata or [])) for metadata in metadata_by_rank]
        max_size = max(rank_sizes, default=0)
        if max_size == 0:
            raise RuntimeError("parallel Wan VAE decode produced no tile payloads")

        payload = torch.empty(max_size, dtype=hidden_states.dtype, device=target_device)
        offset = 0
        for decoded in local_tiles:
            flat = decoded.contiguous().reshape(-1)
            numel = flat.numel()
            payload[offset : offset + numel].copy_(flat)
            offset += numel
        if offset < max_size:
            payload[offset:].zero_()
        del local_tiles

        gathered_flat = torch.empty((world_size, max_size), dtype=hidden_states.dtype, device=target_device)
        if target_device.type == "cuda" and hasattr(dist, "all_gather_into_tensor"):
            dist.all_gather_into_tensor(gathered_flat, payload, group=group)
        else:
            dist.all_gather(list(gathered_flat.unbind(0)), payload, group=group)
        del payload

        merge_device = (
            target_device if (target_device.type == "cuda" and not temporal_chunk_size) else torch.device("cpu")
        )
        decoded_tiles: list[torch.Tensor | None] = [None] * len(tasks)
        for rank_index, metadata in enumerate(metadata_by_rank):
            rank_offset = 0
            rank_buffer = gathered_flat[rank_index]
            for task_index, shape, numel in metadata or []:
                tile_view = rank_buffer[rank_offset : rank_offset + int(numel)].reshape(tuple(shape))
                decoded_tiles[int(task_index)] = (
                    tile_view if merge_device == target_device else tile_view.to(merge_device)
                )
                rank_offset += int(numel)
        if any(tile is None for tile in decoded_tiles):
            raise RuntimeError("parallel Wan VAE gather did not return every spatial tile")
        return self._merge_spatial_decode_tiles(
            decoded_tiles,
            tasks,
            hidden_states=hidden_states,
            tile_size=tile_size,
            tile_stride=tile_stride,
            merge_device=merge_device,
        )

    def temporal_chunked_decode(self, hidden_states, device, *, chunk_size):
        """Exact causal decode with bounded GPU output residency.

        The decoder's convolution cache stays live across all latent frames;
        completed output chunks are moved to CPU. This is temporal streaming,
        not independent per-chunk decoding, so no boundary context is lost.
        """

        size = int(chunk_size)
        if size <= 0:
            raise ValueError("Wan VAE temporal chunk size must be positive")
        model = self.model
        z = hidden_states.to(device)
        in_was_cl3d = is_channels_last_3d(z)
        use_internal_cl3d = isinstance(model, VideoVAE_) and _can_use_internal_cl3d(model, z)
        model.clear_cache()
        try:
            scale = [value.to(dtype=z.dtype, device=z.device) for value in self.scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1,
                self.z_dim,
                1,
                1,
                1,
            )
            if use_internal_cl3d and not is_channels_last_3d(z):
                z = z.contiguous(memory_format=torch.channels_last_3d)
            x = model.conv2(z)
            output_chunks = []
            pending = []
            for index in range(x.shape[2]):
                model._conv_idx = [0]
                kwargs = {"feat_cache": model._feat_map, "feat_idx": model._conv_idx}
                if isinstance(model, VideoVAE38_) and index == 0:
                    kwargs["first_chunk"] = True
                decoded, model._feat_map, model._conv_idx = model.decoder(
                    x[:, :, index : index + 1],
                    **kwargs,
                )
                pending.append(decoded)
                if len(pending) == size or index + 1 == x.shape[2]:
                    chunk = pending[0] if len(pending) == 1 else torch.cat(pending, dim=2)
                    if isinstance(model, VideoVAE38_):
                        chunk = (
                            wan_vae_unpatchify_2x(chunk, output_cl3d=in_was_cl3d)
                            if use_internal_cl3d
                            else unpatchify(chunk, patch_size=2)
                        )
                    elif use_internal_cl3d and not in_was_cl3d and is_channels_last_3d(chunk):
                        chunk = chunk.contiguous()
                    output_chunks.append(chunk.to("cpu"))
                    pending.clear()
            if not output_chunks:
                raise ValueError("cannot decode an empty Wan latent sequence")
            return torch.cat(output_chunks, dim=2).clamp_(-1, 1)
        finally:
            model.clear_cache()

    def tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        # Split tasks
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros(
            (1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )

        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                ),
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    def single_encode(self, video, device):
        video = video.to(device)
        x = self.model.encode(video, self.scale)
        return x

    def single_decode(self, hidden_state, device):
        hidden_state = hidden_state.to(device)
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # Spatial tiling deliberately stages the source on CPU and transfers
        # one tile at a time.  The resident dense path must keep an already
        # materialized CUDA tensor in place; forcing it through CPU here adds
        # a full video-sized D2H/H2D round trip before any encoder work.
        if tiled:
            tile_size = tuple(size * self.upsampling_factor for size in tile_size)
            tile_stride = tuple(stride * self.upsampling_factor for stride in tile_stride)
        hidden_states = []
        for video in videos:
            if tiled:
                video = video.to("cpu")
            video = video.unsqueeze(0)
            if tiled:
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(
        self,
        hidden_states,
        device,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        temporal_chunk_size=0,
    ):
        with self._decode_autocast(hidden_states.dtype):
            return self._decode(
                hidden_states,
                device,
                tiled,
                tile_size,
                tile_stride,
                temporal_chunk_size,
            )

    def _decode(
        self,
        hidden_states,
        device,
        tiled,
        tile_size,
        tile_stride,
        temporal_chunk_size,
    ):
        # Keep CUDA latents GPU-resident for spatial tiling when temporal
        # chunking is disabled; stage on CPU only when bounded-memory temporal
        # streaming is requested or target execution is non-CUDA.
        stage_on_cpu = bool(tiled and (temporal_chunk_size or torch.device(device).type != "cuda"))
        source_hidden_states = hidden_states.to("cpu") if stage_on_cpu else hidden_states
        videos = []
        for hidden_state in source_hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(
                    hidden_state,
                    device,
                    tile_size,
                    tile_stride,
                    temporal_chunk_size,
                )
            elif temporal_chunk_size:
                video = self.temporal_chunked_decode(
                    hidden_state,
                    device,
                    chunk_size=temporal_chunk_size,
                )
            else:
                video = self.single_decode(hidden_state, device)
            video = video.squeeze(0)
            videos.append(video)
        videos = torch.stack(videos)
        return videos

    def decode_framewise(self, hidden_states, device):
        """Decode each temporal latent independently for low-memory inference."""
        video = [
            self.single_decode(hidden_states[:, :, index : index + 1], device)
            for index in range(hidden_states.shape[2])
        ]
        return torch.concat(video, dim=2)

    @staticmethod
    def state_dict_converter():
        return WanVideoVAEStateDictConverter()


class WanVideoVAEStateDictConverter:
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        state_dict_ = {}
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        for name in state_dict:
            state_dict_["model." + name] = state_dict[name]
        return state_dict_


class VideoVAE38_(VideoVAE_):
    """Video implementation."""

    def __init__(
        self,
        dim=160,
        z_dim=48,
        dec_dim=256,
        dim_mult=[1, 2, 4, 4],  # noqa: B006 - Keeps the upstream default object and signature.
        num_res_blocks=2,
        attn_scales=[],  # noqa: B006 - Keeps the upstream default object and signature.
        temperal_downsample=[False, True, True],  # noqa: B006 - Keeps the upstream default object and signature.
        dropout=0.0,
    ):
        super(VideoVAE_, self).__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d_38(
            dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d_38(
            dec_dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout
        )

    def encode(self, x, scale):
        self.clear_cache()
        in_was_cl3d = is_channels_last_3d(x)
        use_internal_cl3d = _can_use_internal_cl3d(self, x)
        x = wan_vae_patchify_2x(x, output_cl3d=True) if use_internal_cl3d else patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        outs = []
        for i in range(iter_):
            self._enc_conv_idx = [0]
            chunk = x[:, :, :1, :, :] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
            out_, self._enc_feat_map, self._enc_conv_idx = self.encoder(
                chunk,
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            outs.append(out_)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        if use_internal_cl3d and not in_was_cl3d and is_channels_last_3d(mu):
            mu = mu.contiguous()
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        in_was_cl3d = is_channels_last_3d(z)
        use_internal_cl3d = _can_use_internal_cl3d(self, z)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        if use_internal_cl3d and not is_channels_last_3d(z):
            z = z.contiguous(memory_format=torch.channels_last_3d)
        iter_ = z.shape[2]
        x = self.conv2(z)
        outs = []
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out_, self._feat_map, self._conv_idx = self.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=True
                )
            else:
                out_, self._feat_map, self._conv_idx = self.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx
                )
            outs.append(out_)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
        if use_internal_cl3d:
            out = wan_vae_unpatchify_2x(out, output_cl3d=in_was_cl3d)
        else:
            out = unpatchify(out, patch_size=2)
        self.clear_cache()
        return out


class WanVideoVAE38(WanVideoVAE):
    """Wan video implementation."""

    def __init__(self, z_dim=48, dim=160, vae_pretrained_path: str | None = None):
        super(WanVideoVAE, self).__init__()

        mean = [
            -0.2289,
            -0.0052,
            -0.1323,
            -0.2339,
            -0.2799,
            0.0174,
            0.1838,
            0.1557,
            -0.1382,
            0.0542,
            0.2813,
            0.0891,
            0.1570,
            -0.0098,
            0.0375,
            -0.1825,
            -0.2246,
            -0.1207,
            -0.0698,
            0.5109,
            0.2665,
            -0.2108,
            -0.2158,
            0.2502,
            -0.2055,
            -0.0322,
            0.1109,
            0.1567,
            -0.0729,
            0.0899,
            -0.2799,
            -0.1230,
            -0.0313,
            -0.1649,
            0.0117,
            0.0723,
            -0.2839,
            -0.2083,
            -0.0520,
            0.3748,
            0.0152,
            0.1957,
            0.1433,
            -0.2944,
            0.3573,
            -0.0548,
            -0.1681,
            -0.0667,
        ]
        std = [
            0.4765,
            1.0364,
            0.4514,
            1.1677,
            0.5313,
            0.4990,
            0.4818,
            0.5013,
            0.8158,
            1.0344,
            0.5894,
            1.0901,
            0.6885,
            0.6165,
            0.8454,
            0.4978,
            0.5759,
            0.3523,
            0.7135,
            0.6804,
            0.5833,
            1.4146,
            0.8986,
            0.5659,
            0.7069,
            0.5338,
            0.4889,
            0.4917,
            0.4069,
            0.4999,
            0.6866,
            0.4093,
            0.5709,
            0.6065,
            0.6415,
            0.4944,
            0.5726,
            1.2042,
            0.5458,
            1.6887,
            0.3971,
            1.0600,
            0.3943,
            0.5537,
            0.5444,
            0.4089,
            0.7468,
            0.7744,
        ]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = VideoVAE38_(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        self.upsampling_factor = 16
        self.z_dim = z_dim
        self.vae_pretrained_path = vae_pretrained_path
        self.decode_autocast_dtype: torch.dtype | None = None
