# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""HunyuanVideo 1.5 Conv3D video autoencoder.

Checkpoint-compatible causal 3-D VAE with tiling. Public ``encode`` / ``decode`` are wrapped by :class:`~.component.HunyuanVideo15Codec`."""

import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn import Conv3d

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.model import RMS_norm as RMS_norm
from worldfoundry.core.distributed.model_parallel.sequence_mesh_state import get_parallel_state
from worldfoundry.core.kernels.triton_pad_3d import triton_pad_3d
from worldfoundry.core.kernels.triton_wan_vae import can_use_wan_vae_rms_norm, wan_vae_rms_norm
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config
from worldfoundry.core.nn.latent.distributions import AutoencoderKLOutput, DecoderOutput, DiagonalGaussianDistribution

from .triton_h15_vae import (
    _is_channels_last_3d,
    can_use_triton_h15_pad_3d_align_c,
    can_use_triton_h15_residual_bias_add,
    can_use_triton_h15_rms_norm_silu_pad_3d,
    triton_h15_downsample_add,
    triton_h15_pad_3d_align_c,
    triton_h15_repeat_add,
    triton_h15_residual_bias_add,
    triton_h15_rms_norm_silu_pad_3d,
    triton_h15_upsample_add,
)


def swish(x: Tensor, inplace=False) -> Tensor:
    """Applies the swish activation function (SiLU) with optional inplace support."""
    return F.silu(x, inplace=inplace)


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    """Forward with optional gradient checkpointing."""

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


# Optimized implementation of CogVideoXSafeConv3d
# https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L38
class PatchCausalConv3d(nn.Conv3d):
    r"""Causal Conv3d with efficient patch processing for large tensors."""

    def find_split_indices(self, seq_len, part_num):
        ideal_interval = seq_len / part_num
        possible_indices = list(range(0, seq_len, self.stride[0]))
        selected_indices = []

        for i in range(1, part_num):
            closest = min(possible_indices, key=lambda x: abs(x - round(i * ideal_interval)))
            if closest not in selected_indices:
                selected_indices.append(closest)

        merged_indices = []
        prev_idx = 0
        for idx in selected_indices:
            if idx - prev_idx >= self.kernel_size[0]:
                merged_indices.append(idx)
                prev_idx = idx

        return merged_indices

    def forward(self, input):
        T = input.shape[2]  # input: NCTHW
        memory_count = math.prod(input.shape) * 2 / 1024**3
        if self.kernel_size[0] < T and memory_count > 0.6:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            split_indices = self.find_split_indices(T, part_num)
            input_chunks = torch.tensor_split(input, split_indices, dim=2) if len(split_indices) > 0 else [input]
            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]
            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


class CausalConv3d(nn.Module):
    """Causal Conv3d with configurable padding for temporal axis."""

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        pad_mode="replicate",
        disable_causal=False,
        enable_patch_conv=False,
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
            )
        else:
            padding = (
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size - 1,
                0,
            )  # W, H, T
        self.time_causal_padding = padding

        if enable_patch_conv:
            self.conv = PatchCausalConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        else:
            self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        conv = self.conv
        if (
            type(conv) is nn.Conv3d
            and conv.groups == 1
            and not _has_hooks(conv)
            and x.shape[1] < 8
            and conv.weight.device == x.device
            and (conv.bias is None or conv.bias.device == x.device)
            and conv.weight.shape[1] == x.shape[1]
            and (_is_channels_last_3d(x) or _is_channels_last_3d(conv.weight))
            and can_use_triton_h15_pad_3d_align_c(x, self.time_causal_padding, spatial_mode=self.pad_mode, align_c=8)
        ):
            x_pad8 = triton_h15_pad_3d_align_c(x, self.time_causal_padding, spatial_mode=self.pad_mode, align_c=8)
            w = conv.weight
            w8 = torch.empty(
                (w.shape[0], 8, w.shape[2], w.shape[3], w.shape[4]),
                device=w.device,
                dtype=w.dtype,
                memory_format=torch.channels_last_3d,
            ).zero_()
            w8[:, : w.shape[1]] = w
            out = F.conv3d(x_pad8, w8, None, conv.stride, conv.padding, conv.dilation, conv.groups)
            if conv.bias is not None:
                return triton_h15_residual_bias_add(None, out, conv.bias)
            return out
        x = triton_pad_3d(x, self.time_causal_padding, spatial_mode=self.pad_mode)
        if (
            type(conv) is nn.Conv3d
            and not _has_hooks(conv)
            and conv.bias is not None
            and x.is_cuda
            and not torch.is_grad_enabled()
            and conv.weight.device == x.device
            and conv.bias.device == x.device
        ):
            out = F.conv3d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
            if can_use_triton_h15_residual_bias_add(None, out, conv.bias):
                return triton_h15_residual_bias_add(None, out, conv.bias)
            return out + conv.bias.view(1, -1, 1, 1, 1)
        return conv(x)


def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int | None = None):
    """Prepare a causal attention mask for 3D videos."""
    seq_len = n_frame * n_hw
    frame_mask = torch.triu(torch.full((n_frame, n_frame), float("-inf"), dtype=dtype, device=device), diagonal=1)
    mask = frame_mask[:, None, :, None].expand(n_frame, n_hw, n_frame, n_hw).reshape(seq_len, seq_len)
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class AttnBlock(nn.Module):
    """Self-attention block for 3D video tensors."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = RMS_norm(in_channels, images=False)

        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        is_cl3d = _is_channels_last_3d(h_)
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        is_cl3d = is_cl3d or _is_channels_last_3d(q)

        b, c, f, h, w = q.shape
        if _is_channels_last_3d(q):
            q = q.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c)
            k = k.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c)
            v = v.permute(0, 2, 3, 4, 1).reshape(b, 1, f * h * w, c)
        else:
            q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
            k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
            v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        attention_mask = prepare_causal_attention_mask(f, h * w, q.dtype, q.device, batch_size=b)
        h_ = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1))

        out = h_.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)
        if is_cl3d:
            return out if _is_channels_last_3d(out) else out.contiguous(memory_format=torch.channels_last_3d)
        return out.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


def _has_hooks(module: nn.Module) -> bool:
    return bool(module._forward_hooks or module._forward_pre_hooks)


def _rms_norm_silu_h15(x: Tensor, norm: nn.Module) -> Tensor:
    """Fuse ``RMS_norm -> SiLU`` via the shared Hopper Triton kernel on CUDA inference."""
    if (
        type(norm) is RMS_norm
        and norm.channel_first
        and x.is_cuda
        and norm.gamma.device == x.device
        and not _has_hooks(norm)
    ):
        bias_t = norm.bias if isinstance(norm.bias, torch.Tensor) else None
        if (bias_t is None or bias_t.device == x.device) and can_use_wan_vae_rms_norm(
            x, channel_first=norm.channel_first
        ):
            return wan_vae_rms_norm(x, norm.gamma, bias_t, scale=norm.scale, apply_silu=True)
    return swish(norm(x), inplace=True)


def _can_defer_conv_bias(conv: nn.Module, x: Tensor) -> bool:
    """Return True when ``CausalConv3d`` can run ``F.conv3d(..., bias=None)`` safely."""
    if type(conv) is not CausalConv3d or _has_hooks(conv):
        return False
    inner = conv.conv
    return (
        type(inner) is nn.Conv3d
        and not _has_hooks(inner)
        and inner.bias is not None
        and x.is_cuda
        and not torch.is_grad_enabled()
        and inner.weight.device == x.device
        and inner.bias.device == x.device
    )


def _rms_norm_silu_conv_h15(
    x: Tensor,
    norm: nn.Module,
    conv: nn.Module,
    *,
    pre_bias: Tensor | None = None,
    defer_bias: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Super-fuse ``(pre_bias +) RMS_norm -> SiLU -> CausalConv3d.pad`` and invoke ``conv.conv``.

    Returns ``(output, deferred_bias)`` where ``deferred_bias`` is non-None only
    when ``defer_bias=True`` and bias addition was safely deferred to the next
    fused Triton kernel (eliminating PyTorch's unvectorized 5-D Conv3d bias pass).
    """
    if (
        type(norm) is RMS_norm
        and norm.channel_first
        and type(conv) is CausalConv3d
        and x.is_cuda
        and norm.gamma.device == x.device
        and not _has_hooks(norm)
        and not _has_hooks(conv)
        and (pre_bias is None or pre_bias.device == x.device)
    ):
        bias_t = norm.bias if isinstance(norm.bias, torch.Tensor) else None
        if (bias_t is None or bias_t.device == x.device) and can_use_triton_h15_rms_norm_silu_pad_3d(
            x, conv.time_causal_padding, spatial_mode=conv.pad_mode
        ):
            x_pad = triton_h15_rms_norm_silu_pad_3d(
                x,
                norm.gamma,
                bias_t,
                conv.time_causal_padding,
                scale=norm.scale,
                spatial_mode=conv.pad_mode,
                apply_silu=True,
                pre_bias=pre_bias,
            )
            inner = conv.conv
            if (
                defer_bias
                and type(inner) is nn.Conv3d
                and not _has_hooks(inner)
                and inner.bias is not None
                and inner.weight.device == x_pad.device
                and inner.bias.device == x_pad.device
            ):
                return (
                    F.conv3d(x_pad, inner.weight, None, inner.stride, inner.padding, inner.dilation, inner.groups),
                    inner.bias,
                )
            return inner(x_pad), None
    if pre_bias is not None:
        x = x + pre_bias.view(1, -1, 1, 1, 1)
    return conv(_rms_norm_silu_h15(x, norm)), None


class ResnetBlock(nn.Module):
    """ResNet-style block for 3D video tensors."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = RMS_norm(in_channels, images=False)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = RMS_norm(out_channels, images=False)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        can_fuse_norm2 = (
            type(self.norm2) is RMS_norm
            and self.norm2.channel_first
            and type(self.conv2) is CausalConv3d
            and not _has_hooks(self.norm2)
            and not _has_hooks(self.conv2)
        )
        h, bias1 = _rms_norm_silu_conv_h15(x, self.norm1, self.conv1, defer_bias=can_fuse_norm2)
        h, bias2 = _rms_norm_silu_conv_h15(h, self.norm2, self.conv2, pre_bias=bias1, defer_bias=True)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        if bias2 is not None or (x.is_cuda and not torch.is_grad_enabled()):
            return triton_h15_residual_bias_add(x, h, bias2)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = CausalConv3d(in_channels, out_channels // factor, kernel_size=3)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        if _can_defer_conv_bias(self.conv, x):
            x_pad = triton_pad_3d(x, self.conv.time_causal_padding, spatial_mode=self.conv.pad_mode)
            inner = self.conv.conv
            h = F.conv3d(x_pad, inner.weight, None, inner.stride, inner.padding, inner.dilation, inner.groups)
            h_bias = inner.bias
        else:
            h = self.conv(x)
            h_bias = None
        return triton_h15_downsample_add(
            h,
            x,
            add_temporal_downsample=self.add_temporal_downsample,
            group_size=self.group_size,
            h_bias=h_bias,
        )


class Upsample(nn.Module):
    """Hierarchical upsampling with temporal/ spatial support."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = CausalConv3d(in_channels, out_channels * factor, kernel_size=3)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        if _can_defer_conv_bias(self.conv, x):
            x_pad = triton_pad_3d(x, self.conv.time_causal_padding, spatial_mode=self.conv.pad_mode)
            inner = self.conv.conv
            h = F.conv3d(x_pad, inner.weight, None, inner.stride, inner.padding, inner.dilation, inner.groups)
            h_bias = inner.bias
        else:
            h = self.conv(x)
            h_bias = None
        return triton_h15_upsample_add(
            h,
            x,
            add_temporal_upsample=self.add_temporal_upsample,
            repeats=self.repeats,
            h_bias=h_bias,
        )


class Encoder(nn.Module):
    """Hierarchical video encoder with temporal and spatial factorization."""

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= np.log2(ffactor_spatial // ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = Downsample(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the encoder."""
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = forward_with_checkpointing(
                    self.down[i_level].block[i_block], h, use_checkpointing=use_checkpointing
                )
            if hasattr(self.down[i_level], "downsample"):
                h = forward_with_checkpointing(self.down[i_level].downsample, h, use_checkpointing=use_checkpointing)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        if _is_channels_last_3d(h):
            b, c, f, hh, ww = h.shape
            shortcut = (
                h.permute(0, 2, 3, 4, 1)
                .reshape(b, f, hh, ww, c // group_size, group_size)
                .mean(dim=-1)
                .permute(0, 4, 1, 2, 3)
            )
        else:
            shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h, out_bias = _rms_norm_silu_conv_h15(h, self.norm_out, self.conv_out, defer_bias=True)
        if out_bias is not None:
            return triton_h15_residual_bias_add(shortcut, h, out_bias)
        h += shortcut
        return h


class Decoder(nn.Module):
    """Hierarchical video decoder with upsampling factories."""

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        block_in = block_out_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = Upsample(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass through the decoder."""
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        if _can_defer_conv_bias(self.conv_in, z):
            z_pad = triton_pad_3d(z, self.conv_in.time_causal_padding, spatial_mode=self.conv_in.pad_mode)
            inner_in = self.conv_in.conv
            h_in = F.conv3d(
                z_pad, inner_in.weight, None, inner_in.stride, inner_in.padding, inner_in.dilation, inner_in.groups
            )
            h = triton_h15_repeat_add(h_in, z, repeats, h_bias=inner_in.bias)
        else:
            h = triton_h15_repeat_add(self.conv_in(z), z, repeats)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(self.up[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
            if hasattr(self.up[i_level], "upsample"):
                h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)

        # end
        h, _ = _rms_norm_silu_conv_h15(h, self.norm_out, self.conv_out)
        return h


class AutoencoderKLConv3D(nn.Module, NativeConfigMixin):
    """KL regularized 3D Conv VAE with advanced tiling and slicing strategies."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float | None = None,
        shift_factor: float | None = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            downsample_match_channel=downsample_match_channel,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            upsample_match_channel=upsample_match_channel,
        )

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

        self._tile_parallelism_enabled = False

    def set_tile_sample_min_size(self, sample_size: int, tile_overlap_factor: float = 0.2):
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // self.ffactor_spatial
        self.tile_overlap_factor = tile_overlap_factor

        assert (self.tile_latent_min_size * self.tile_overlap_factor).is_integer(), (
            "self.tile_latent_min_size multiplied by tile_overlap_factor must be an integer"
        )

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable or disable gradient checkpointing on encoder and decoder."""
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_temporal_tiling(self, use_tiling: bool = True):
        raise RuntimeError("Temporal tiling is not supported for this VAE.")
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b horizontally into a at blend_extent region."""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b vertically into a at blend_extent region."""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b temporally into a at blend_extent region."""
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        """Tiled spatial encoding for large inputs via overlapping."""
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: torch.Tensor):
        """Tiled temporal encoding for large video sequences."""
        raise RuntimeError("Temporal tiling is not supported for this VAE.")
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])
        moments = torch.cat(result_row, dim=-3)
        return moments

    def enable_tile_parallelism(self):
        self._tile_parallelism_enabled = True

    def disable_tile_parallelism(self):
        self._tile_parallelism_enabled = False

    def tile_parallel_spatial_tiled_decode(
        self,
        z: torch.Tensor,
        *,
        group: dist.ProcessGroup | None = None,
    ):
        B, _C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        if group is not None:
            comm_group = group
            rank = int(dist.get_rank(group))
            world_size = int(dist.get_world_size(group))
        elif z.is_cuda:
            ps = get_parallel_state()
            if ps.sp > 1 and ps.sp_group is not None:
                comm_group = ps.sp_group
                rank = int(ps.sp_rank)
                world_size = int(ps.sp)
            elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                comm_group = None
                rank = int(dist.get_rank())
                world_size = int(dist.get_world_size())
            else:
                comm_group = None
                rank = 0
                world_size = 1
        elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            comm_group = None
            rank = int(dist.get_rank())
            world_size = int(dist.get_world_size())
        else:
            comm_group = None
            rank = 0
            world_size = 1

        num_rows = math.ceil(H / overlap_size)
        num_cols = math.ceil(W / overlap_size)
        tile_specs = [
            (
                ri,
                rj,
                min(self.tile_latent_min_size, H - ri * overlap_size)
                * min(self.tile_latent_min_size, W - rj * overlap_size),
            )
            for ri in range(num_rows)
            for rj in range(num_cols)
        ]
        rank_loads = [0] * world_size
        rank_bins: list[list[tuple[int, int]]] = [[] for _ in range(world_size)]
        for ri, rj, area in sorted(tile_specs, key=lambda item: (-item[2], item[0], item[1])):
            target = min(range(world_size), key=lambda r: (rank_loads[r], len(rank_bins[r]), r))
            rank_bins[target].append((ri, rj))
            rank_loads[target] += area
        tiles_per_rank = max((len(b) for b in rank_bins), default=0)

        H_out_std = max(self.tile_sample_min_size, self.tile_latent_min_size * self.ffactor_spatial)
        W_out_std = max(self.tile_sample_min_size, self.tile_latent_min_size * self.ffactor_spatial)
        out_T = (T - 1) * self.ffactor_temporal + 1
        out_channels = int(getattr(self.config, "out_channels", 3))
        if z.device.type in ("cuda", "cpu") and torch.is_autocast_enabled(z.device.type):
            out_dtype = torch.get_autocast_dtype(z.device.type)
        else:
            conv_out = getattr(self.decoder, "conv_out", None)
            inner_conv = getattr(conv_out, "conv", conv_out)
            weight = getattr(inner_conv, "weight", None)
            out_dtype = weight.dtype if isinstance(weight, torch.Tensor) else z.dtype

        local_tiles = torch.zeros(
            (tiles_per_rank, B, out_channels, out_T, H_out_std, W_out_std),
            device=z.device,
            dtype=out_dtype,
        )
        for slot, (ri, rj) in enumerate(rank_bins[rank]):
            i = ri * overlap_size
            j = rj * overlap_size
            tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
            dec = self.decoder(tile)
            if dec.dtype != local_tiles.dtype:
                local_tiles = local_tiles.to(dtype=dec.dtype)
            local_tiles[slot, :, :, :, : dec.shape[-2], : dec.shape[-1]].copy_(dec)

        if world_size == 1:
            gathered_tiles = local_tiles.unsqueeze(0)
        else:
            gathered_tiles = torch.empty(
                (world_size, *local_tiles.shape),
                device=local_tiles.device,
                dtype=local_tiles.dtype,
            )
            if local_tiles.is_cuda and hasattr(dist, "all_gather_into_tensor"):
                dist.all_gather_into_tensor(gathered_tiles, local_tiles, group=comm_group)
            else:
                dist.all_gather(list(gathered_tiles.unbind(0)), local_tiles, group=comm_group)

        if rank != 0:
            return torch.empty(0, device=z.device, dtype=local_tiles.dtype)

        rows: list[list[torch.Tensor | None]] = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(world_size):
            for k, (ri, rj) in enumerate(rank_bins[r]):
                i = ri * overlap_size
                j = rj * overlap_size
                tile_out_h = min(self.tile_latent_min_size, H - i) * self.ffactor_spatial
                tile_out_w = min(self.tile_latent_min_size, W - j) * self.ffactor_spatial
                rows[ri][rj] = gathered_tiles[r, k, :, :, :, :tile_out_h, :tile_out_w]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if tile is None:
                    continue
                if i > 0:
                    prev_row_tile = rows[i - 1][j]
                    assert prev_row_tile is not None
                    tile = self.blend_v(prev_row_tile, tile, blend_extent)
                if j > 0:
                    prev_col_tile = row[j - 1]
                    assert prev_col_tile is not None
                    tile = self.blend_h(prev_col_tile, tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        return dec

    def spatial_tiled_decode(self, z: torch.Tensor):
        if self._tile_parallelism_enabled:
            return self.tile_parallel_spatial_tiled_decode(z)

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_decode(self, z: torch.Tensor):
        """Tiled temporal decoding for long sequence latents."""
        raise RuntimeError("Temporal tiling is not supported for this VAE.")
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])
        dec = torch.cat(result_row, dim=-3)
        return dec

    def encode(self, x: Tensor, return_dict: bool = True):
        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (
                x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
            ):
                return self.spatial_tiled_encode(x)
            return self.encoder(x)

        assert len(x.shape) == 5  # (B, C, T, H, W)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, generator=None):
        def _decode(z):
            if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                return self.temporal_tiled_decode(z)
            if self.use_spatial_tiling and (
                z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_posterior: bool = True,
        return_dict: bool = True,
    ):
        """Forward autoencoder pass. Returns both reconstruction and optionally the posterior."""
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        return DecoderOutput(sample=dec, posterior=posterior) if return_dict else (dec, posterior)

    @contextmanager
    def memory_efficient_context(self):
        original_use_slicing = self.use_slicing
        original_use_spatial_tiling = self.use_spatial_tiling

        self.enable_slicing()
        self.enable_tiling()
        yield
        self.use_slicing = original_use_slicing
        self.use_spatial_tiling = original_use_spatial_tiling
