import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum

from src.modules.attention_temporal_videoae import default, exists

try:
    import xformers  # noqa: F401 - Availability probe for XFORMERS_IS_AVAILBLE.
    import xformers.ops as xops

    XFORMERS_IS_AVAILBLE = True
except BaseException:
    XFORMERS_IS_AVAILBLE = False


def silu(x):
    # swish
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return silu(x)


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == "batch":
        return torch.nn.SyncBatchNorm(in_channels)


# Does not support dilation


class SamePadConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride, strict=False)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, ())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        # print(x.dtype)
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        padding_type="replicate",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride, strict=False)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, ())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=tuple([k - 1 for k in kernel_size]),
        )

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
        padding_type="replicate",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x + h


class SpatialCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        patch_size=1,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        # print(f"query dimension is {query_dim}")

        self.patch_size = patch_size
        patch_dim = query_dim * patch_size * patch_size
        self.norm = nn.LayerNorm(patch_dim)

        self.to_q = nn.Linear(patch_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, patch_dim), nn.Dropout(dropout))
        self.attention_op: Any | None = None

    def forward(self, x, context=None, mask=None):
        b, c, t, height, width = x.shape

        # patch: [patch_size, patch_size]
        divide_factor_height = height // self.patch_size
        divide_factor_width = width // self.patch_size
        x = rearrange(
            x,
            "b c t (df1 ph) (df2 pw) -> (b t) (df1 df2) (ph pw c)",
            df1=divide_factor_height,
            df2=divide_factor_width,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = self.norm(x)

        context = default(context, x)
        context = repeat(context, "b n d -> (b t) n d", b=b, t=t)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=self.heads) for t in (q, k, v))

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b t h) () j", t=t, h=self.heads)

        if XFORMERS_IS_AVAILBLE:
            if exists(mask):
                mask = mask.to(q.dtype)
                max_neg_value = -torch.finfo(q.dtype).max

                attn_bias = torch.zeros_like(mask)
                attn_bias.masked_fill_(mask <= 0.5, max_neg_value)

                mask = mask.detach().cpu()
                attn_bias = attn_bias.expand(-1, q.shape[1], -1)

                attn_bias_expansion_q = (attn_bias.shape[1] + 7) // 8 * 8
                attn_bias_expansion_k = (attn_bias.shape[2] + 7) // 8 * 8

                attn_bias_expansion = torch.zeros(
                    (attn_bias.shape[0], attn_bias_expansion_q, attn_bias_expansion_k),
                    dtype=attn_bias.dtype,
                    device=attn_bias.device,
                )
                attn_bias_expansion[:, : attn_bias.shape[1], : attn_bias.shape[2]] = attn_bias

                attn_bias = attn_bias.detach().cpu()

                out = xops.memory_efficient_attention(
                    q,
                    k,
                    v,
                    attn_bias=attn_bias_expansion[:, : attn_bias.shape[1], : attn_bias.shape[2]],
                    scale=self.scale,
                )
            else:
                out = xops.memory_efficient_attention(q, k, v, scale=self.scale)
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                max_neg_value = -torch.finfo(sim.dtype).max
                sim.masked_fill_(~(mask > 0.5), max_neg_value)
            attn = sim.softmax(dim=-1)
            out = einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)

        ret = self.to_out(out)
        ret = rearrange(
            ret,
            "(b t) (df1 df2) (ph pw c) -> b c t (df1 ph) (df2 pw)",
            b=b,
            t=t,
            df1=divide_factor_height,
            df2=divide_factor_width,
            ph=self.patch_size,
            pw=self.patch_size,
        )
        return ret


# ---------------------------------------------------------------------------------------------------=


class EncoderTemporal1DCNN(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        attn_temporal_factor=[],  # noqa: B006 - Read-only default; retain the public signature.
        temporal_scale_factor=4,
        hidden_channel=128,
        **ignore_kwargs,
    ):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.temporal_scale_factor = temporal_scale_factor

        # conv_in + resblock + down_block + resblock + down_block + final_block
        self.conv_in = SamePadConv3d(ch, hidden_channel, kernel_size=3, padding_type="replicate")

        self.mid_blocks = nn.ModuleList()

        num_ds = int(math.log2(temporal_scale_factor))
        norm_type = "group"

        curr_temporal_factor = 1
        for i in range(num_ds):
            block = nn.Module()
            # compute in_ch, out_ch, stride
            in_channels = hidden_channel * 2**i
            out_channels = hidden_channel * 2 ** (i + 1)
            temporal_stride = 2
            curr_temporal_factor = curr_temporal_factor * 2

            block.down = SamePadConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(temporal_stride, 1, 1),
                padding_type="replicate",
            )
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)

            block.attn = nn.ModuleList()
            if curr_temporal_factor in attn_temporal_factor:
                block.attn.append(SpatialCrossAttention(query_dim=out_channels, context_dim=1024))

            self.mid_blocks.append(block)
            # n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type),
            SiLU(),
            SamePadConv3d(out_channels, out_ch * 2, kernel_size=3, padding_type="replicate"),
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) and module.weight.requires_grad_:
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x, text_embeddings=None, text_attn_mask=None):
        # x: [b c t h w]
        # x: [1, 4, 16, 32, 32]
        # timestep embedding
        h = self.conv_in(x)
        for block in self.mid_blocks:
            h = block.down(h)
            h = block.res(h)
            if len(block.attn) > 0:
                for attn in block.attn:
                    h = attn(h, context=text_embeddings, mask=text_attn_mask) + h

        h = self.final_block(h)

        return h


class TemporalUpsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class DecoderTemporal1DCNN(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        attn_temporal_factor=[],  # noqa: B006 - Read-only default; retain the public signature.
        temporal_scale_factor=4,
        hidden_channel=128,
        **ignore_kwargs,
    ):
        super().__init__()

        self.ch = ch
        self.temb_ch = 0
        self.temporal_scale_factor = temporal_scale_factor

        num_us = int(math.log2(temporal_scale_factor))
        norm_type = "group"

        # conv_in, mid_blocks, final_block
        # out channel of encoder, before the last conv layer
        enc_out_channels = hidden_channel * 2**num_us
        self.conv_in = SamePadConv3d(ch, enc_out_channels, kernel_size=3, padding_type="replicate")

        self.mid_blocks = nn.ModuleList()
        curr_temporal_factor = self.temporal_scale_factor

        for i in range(num_us):
            block = nn.Module()
            in_channels = enc_out_channels if i == 0 else hidden_channel * 2 ** (num_us - i + 1)  # max_us: 3
            out_channels = hidden_channel * 2 ** (num_us - i)
            # block.up = SamePadConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=(temporal_stride, 1, 1))
            block.up = torch.nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=(2, 1, 1),
                padding=(1, 1, 1),
                output_padding=(1, 0, 0),
            )
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.attn1 = nn.ModuleList()

            if curr_temporal_factor in attn_temporal_factor:
                block.attn1.append(SpatialCrossAttention(query_dim=out_channels, context_dim=1024))

            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)

            block.attn2 = nn.ModuleList()
            if curr_temporal_factor in attn_temporal_factor:
                block.attn2.append(SpatialCrossAttention(query_dim=out_channels, context_dim=1024))

            curr_temporal_factor = curr_temporal_factor / 2
            self.mid_blocks.append(block)

        self.conv_last = SamePadConv3d(out_channels, out_ch, kernel_size=3)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) and module.weight.requires_grad_:
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.Conv3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            if isinstance(module, nn.ConvTranspose3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x, text_embeddings=None, text_attn_mask=None):
        # x: [b c t h w]
        h = self.conv_in(x)
        for _i, block in enumerate(self.mid_blocks):
            h = block.up(h)
            h = block.res1(h)
            if len(block.attn1) > 0:
                for attn in block.attn1:
                    h = attn(h, context=text_embeddings, mask=text_attn_mask) + h

            h = block.res2(h)
            if len(block.attn2) > 0:
                for attn in block.attn2:
                    h = attn(h, context=text_embeddings, mask=text_attn_mask) + h

        h = self.conv_last(h)

        return h
