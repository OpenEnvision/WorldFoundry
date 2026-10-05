# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Linear-attention Wan 2.1 / Sana-Video DiT (official packing).

Replaces selected self-attention layers with ReLU / power linear
attention (MLLA, STConv, LePE variants).  Flash attention remains
available per layer via ``self_attn_types``.  TeaCache-style
:class:`BlockHook` / :class:`AttentionHook` can snapshot activations.

Adds linear attention and TeaCache hooks.  No VACE or causal cache.
"""
import math
from typing import Optional

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from worldfoundry.core.model_loading.model_configuration import NativeConfigMixin, register_to_config

from worldfoundry.core.nn.blocks.convolutional_mlp import GLUMBConvTemp, Mlp
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.linear_attention import flash_attention

__all__ = ["WanModel"]


class AttentionHook:
    """Stores one attention output tensor for TeaCache / debug inspection."""

    output = None

    def __init__(self, device):
        """Init.

        Args:
            device: The device.
        """
        self.device = device

    def __call__(self, attn_output):
        """Call.

        Args:
            attn_output: The attn output.
        """
        self.output = attn_output
        self.output.to(self.device)

    def clear(self):
        """Clear."""
        self.output = None


def cosine_similarity(x, y, dim=1):
    """Cosine similarity.

    Args:
        x: The x.
        y: The y.
        dim: The dim.
    """
    x_norm = F.normalize(x, p=2, dim=dim)
    y_norm = F.normalize(y, p=2, dim=dim)
    return torch.sum(x_norm * y_norm, dim=dim)


class BlockHook:
    """TeaCache-style block hook: snapshot or skip a transformer block."""

    x_in = None
    x_self_attn = None
    x_cross_attn = None
    x_ffn = None

    def __init__(self, device, detach=True, score_only="cos"):
        """Init.

        Args:
            device: The device.
            detach: The detach.
            score_only: The score only.
        """
        self.device = device
        self.detach = detach if not score_only else True  # if score_only, always detach
        self.score_only = score_only

    def __call__(self, x_in, x_self_attn, x_cross_attn, x_ffn):
        """Call.

        Args:
            x_in: The x in.
            x_self_attn: The x self attn.
            x_cross_attn: The x cross attn.
            x_ffn: The x ffn.
        """
        # input shape is B,L,C
        if self.score_only == "cos":
            # make sure all feats are not none
            assert x_in is not None
            assert x_self_attn is not None
            assert x_cross_attn is not None
            assert x_ffn is not None
            # detach and float all feats
            x_in = x_in.detach().float()
            x_self_attn = x_self_attn.detach().float()
            x_cross_attn = x_cross_attn.detach().float()
            x_ffn = x_ffn.detach().float()

            # compute cosine similarity
            self.x_in = None
            self.x_self_attn = cosine_similarity(x_in, x_self_attn, dim=-1).to(self.device)
            self.x_cross_attn = cosine_similarity(x_self_attn, x_cross_attn, dim=-1).to(self.device)
            self.x_ffn = cosine_similarity(x_cross_attn, x_ffn, dim=-1).to(self.device)
        elif self.score_only == "l2":
            # make sure all feats are not none
            assert x_in is not None
            assert x_self_attn is not None
            assert x_cross_attn is not None
            assert x_ffn is not None
            # detach and float all feats
            x_in = x_in.detach().float()
            x_self_attn = x_self_attn.detach().float()
            x_cross_attn = x_cross_attn.detach().float()
            x_ffn = x_ffn.detach().float()

            self.x_in = None
            self.x_self_attn = F.mse_loss(x_in, x_self_attn, reduction="none").mean(dim=-1).to(self.device)
            self.x_cross_attn = F.mse_loss(x_self_attn, x_cross_attn, reduction="none").mean(dim=-1).to(self.device)
            self.x_ffn = F.mse_loss(x_cross_attn, x_ffn, reduction="none").mean(dim=-1).to(self.device)  # B,L
        else:
            self.x_in = x_in.to(self.device) if x_in is not None else None
            self.x_self_attn = x_self_attn.to(self.device) if x_self_attn is not None else None
            self.x_cross_attn = x_cross_attn.to(self.device) if x_cross_attn is not None else None
            self.x_ffn = x_ffn.to(self.device) if x_ffn is not None else None
            if self.detach:
                self.x_in = self.x_in.detach() if self.x_in is not None else None
                self.x_self_attn = self.x_self_attn.detach() if self.x_self_attn is not None else None
                self.x_cross_attn = self.x_cross_attn.detach() if self.x_cross_attn is not None else None
                self.x_ffn = self.x_ffn.detach() if self.x_ffn is not None else None

    def clear(self):
        """Clear."""
        self.x_in = None
        self.x_self_attn = None
        self.x_cross_attn = None
        self.x_ffn = None

    def get_output(self):
        """Get output."""
        return {
            "x_in": self.x_in,
            "x_self_attn": self.x_self_attn,
            "x_cross_attn": self.x_cross_attn,
            "x_ffn": self.x_ffn,
        }


def sinusoidal_embedding_1d(dim, position):
    """Classic sinusoidal features for 1-D timestep / position ids."""
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """Build complex cis frequencies for one RoPE axis."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """Apply per-sample 3D RoPE to packed tokens; leave pad tokens unrotated."""
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    """RMSNorm with a learned scale; compute in fp32 when the impl upcasts."""

    def __init__(self, dim, eps=1e-5):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        """RMS-normalize the last dim of ``x``."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm that typically upcasts to fp32, then restores the activation dtype."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        """Init.

        Args:
            dim: The dim.
            eps: The eps.
            elementwise_affine: The elementwise affine.
        """
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):
    """Wan self-attention: RMSNorm Q/K, 3D RoPE, flash or variant kernel."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, **kwargs):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.qkv_store_buffer = None

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        # print(f"In Attention, x dtype {x.dtype}")
        x_dtype = x.dtype

        # query, key, value function
        def qkv_fn(x):
            """Project and reshape QKV to ``[B, L, H, D]``."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        if self.qkv_store_buffer is not None:
            self.qkv_store_buffer["q"] = q[1].cpu()  # b, n, h, h_d
            self.qkv_store_buffer["k"] = k[1].cpu()  # b, n, h, h_d
            self.qkv_store_buffer["v"] = v[1].cpu()  # b, n, h, h_d

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        x = x.flatten(2).to(x_dtype)
        x = self.o(x)
        return x


class WanLinearAttention(WanSelfAttention):
    """ReLU / power linear self-attention used in place of flash MSA."""

    PAD_VAL = 1

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, **kwargs):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        super().__init__(dim, num_heads, window_size, qk_norm, eps, **kwargs)
        self.kernel_func = nn.ReLU(inplace=False)
        self.fp32_attention = True
        self.qkv_store_buffer = None
        self.rope_after = kwargs.get("rope_after", False)
        self.power = kwargs.get("power", 1.0)

    @torch.autocast(device_type="cuda", enabled=False)
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        """Attn matmul.

        Args:
            q: The q.
            k: The k.
            v: The v.

        Returns:
            The return value.
        """
        v = F.pad(v.float(), (0, 0, 0, 1), mode="constant", value=self.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)  # b, h, h_d, n

        norm_out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return norm_out

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        x_dtype = x.dtype

        def qkv_fn(x):
            """Project and reshape QKV to ``[B, L, H, D]``."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)  # B, seq, num_heads, head_dim

        # save before rope
        if self.qkv_store_buffer is not None:
            # qkv store buffer shoud be dict
            self.qkv_store_buffer["q"] = q[1].cpu()  # b, n, h, h_d
            self.qkv_store_buffer["k"] = k[1].cpu()  # b, n, h, h_d
            self.qkv_store_buffer["v"] = v[1].cpu()  # b, n, h, h_d

        power = self.power
        rope_after = self.rope_after
        if rope_after:
            # apply kernel function
            q = self.kernel_func(q)  # B, h, h_d, N
            k = self.kernel_func(k)

            # power qk
            if power != 1.0:
                q_norm = q.norm(dim=-1, keepdim=True)
                k_norm = k.norm(dim=-1, keepdim=True)
                q = q**power
                k = k**power
                q = (q / (q.norm(dim=-1, keepdim=True) + 1e-6)) * q_norm
                k = (k / (k.norm(dim=-1, keepdim=True) + 1e-6)) * k_norm

            # apply rope after kernel function
            q_rope = rope_apply(q, grid_sizes, freqs)
            k_rope = rope_apply(k, grid_sizes, freqs)

            with torch.autocast(device_type="cuda", enabled=False):
                q_rope = q_rope.permute(0, 2, 1, 3).contiguous()  # B, seq, num_heads, head_dim -> B, h, seq, h_d
                k_rope = k_rope.permute(0, 2, 1, 3).contiguous()  # B, seq, num_heads, head_dim -> B, h, seq, h_d

                q = q.permute(0, 2, 1, 3).contiguous()  # B, seq, num_heads, head_dim -> B, h, seq, h_d
                k = k.permute(0, 2, 1, 3).contiguous()  # B, seq, num_heads, head_dim -> B, h, seq, h_d
                v = v.permute(0, 2, 1, 3).contiguous()  # B, seq, num_heads, head_dim -> B, h, seq, h_d

                z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)

                kv = (k_rope.transpose(-2, -1)) @ v.float() / s
                x = q_rope @ kv

                x = x * z

        else:
            # apply rope before kernel function
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

            # apply kernel function
            q = self.kernel_func(q)  # B, h, h_d, N
            k = self.kernel_func(k)

            # power qk
            if power != 1.0:
                q_norm = q.norm(dim=-1, keepdim=True)
                k_norm = k.norm(dim=-1, keepdim=True)
                q = q**power
                k = k**power
                q = (q / (q.norm(dim=-1, keepdim=True) + 1e-6)) * q_norm
                k = (k / (k.norm(dim=-1, keepdim=True) + 1e-6)) * k_norm

            x = self.attn_matmul(q.permute(0, 2, 3, 1), k.permute(0, 2, 1, 3), v.permute(0, 2, 3, 1))

        x = x.view(b, n * d, s).permute(0, 2, 1).to(x_dtype)  # B, C, N -> B, N, C
        x = self.o(x)
        return x


class STConv(nn.Module):
    """Space-time depthwise conv used by MLLA linear attention."""

    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        """Init.

        Args:
            in_dim: The in dim.
            out_dim: The out dim.
            kernel_size: The kernel size.
            stride: The stride.
            padding: The padding.
        """
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=in_dim)
        self.temporal_conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=in_dim)

    def forward(self, x):
        """Forward.

        Args:
            x: The x.
        """
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.spatial_conv(x)
        x = x.reshape(B, T, C, H, W).permute(0, 3, 4, 2, 1).reshape(B * H * W, C, T)  # B, T, C, H, W -> B*H*W, C, T
        x = self.temporal_conv(x)
        x = x.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2).reshape(B, C, T, H, W)  # B*H*W, C, T -> B, C, T, H, W
        return x


class MLLALinearAttention(WanLinearAttention):
    """MLLA linear attention (ReLU kernel + optional conv)."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, **kwargs):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        super().__init__(dim, num_heads, window_size, qk_norm, eps, **kwargs)
        self.kernel_func = nn.ReLU(inplace=False)
        self.fp32_attention = True
        self.qkv_store_buffer = None
        self.st_conv = STConv(dim, dim)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        F, H, W = grid_sizes[0]  # grid size should be the same for all the samples
        B, L, C = x.shape

        x = x.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)  # B, C, F, H, W
        x = self.act(self.st_conv(x)).permute(0, 2, 3, 4, 1).reshape(B, L, C)

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        x_dtype = x.dtype

        def qkv_fn(x):
            """Project and reshape QKV to ``[B, L, H, D]``."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)  # B, seq, num_heads, head_dim
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # apply kernel function
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)
        # attn matmul input: q: b, h, h_d, n
        # k: b, h, n, h_d
        # v: b, h, h_d, n
        # output: b, h, n, h_d
        x = self.attn_matmul(q.permute(0, 2, 3, 1), k.permute(0, 2, 1, 3), v.permute(0, 2, 3, 1))
        x = x.view(b, n * d, s).permute(0, 2, 1).to(x_dtype)  # B, C, N -> B, N, C

        x = self.o(x)  # B, L, C

        return x


class MLLALePEAttention(WanLinearAttention):
    """MLLA linear attention with locally-enhanced positional encoding."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, **kwargs):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        super().__init__(dim, num_heads, window_size, qk_norm, eps, **kwargs)
        self.kernel_func = nn.ELU(inplace=False)
        self.fp32_attention = True
        self.qkv_store_buffer = None
        self.st_conv = STConv(dim, dim)
        self.act = nn.SiLU(inplace=False)
        self.lepe_conv = STConv(dim, dim)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """

        F, H, W = grid_sizes[0]  # grid size should be the same for all the samples
        B, L, C = x.shape

        x = x.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)  # B, C, F, H, W
        x = self.act(self.st_conv(x)).permute(0, 2, 3, 4, 1).reshape(B, L, C)

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        x_dtype = x.dtype

        def qkv_fn(x):
            """Project and reshape QKV to ``[B, L, H, D]``."""
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)  # B, seq, num_heads, head_dim

        # apply kernel function before rope
        q = self.kernel_func(q) + 1
        k = self.kernel_func(k) + 1  # elu + 1

        # apply rope
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # attn matmul input: q: b, h, h_d, n
        # k: b, h, n, h_d
        # v: b, h, h_d, n
        # output: b, h, n, h_d
        x = self.attn_matmul(q.permute(0, 2, 3, 1), k.permute(0, 2, 1, 3), v.permute(0, 2, 3, 1))
        x = x.view(b, n * d, s).permute(0, 2, 1).to(x_dtype)  # B, C, N -> B, N, C
        # LePE conv for v
        v = v.reshape(B, F, H, W, C).permute(0, 4, 1, 2, 3)  # B, C, F, H, W
        lepe_v = self.lepe_conv(v).permute(0, 2, 3, 4, 1).reshape(B, L, C)

        x = self.o(x + lepe_v)  # B, L, C

        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Wan text or image-text cross-attention."""

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """Wan text or image-text cross-attention."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        """Init.

        Args:
            dim: The dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            eps: The eps.
        """
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanT2VCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}

WAN_SELFATTENTION_CLASSES = {
    "flash": WanSelfAttention,
    "linear": WanLinearAttention,
    "mllalinear": MLLALinearAttention,
    "mllalepe": MLLALePEAttention,
    "bsa": WanSelfAttention,
}


class WanAttentionBlock(nn.Module):
    """Wan transformer block: AdaLN self-attn, cross-attn, then AdaLN FFN."""

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        self_attn_type="flash",
        rope_after=False,
        power=1.0,
        ffn_type="mlp",
        mlp_acts=("silu", "silu", None),
    ):
        """Init.

        Args:
            cross_attn_type: The cross attn type.
            dim: The dim.
            ffn_dim: The ffn dim.
            num_heads: The num heads.
            window_size: The window size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
            self_attn_type: The self attn type.
            rope_after: The rope after.
            power: The power.
            ffn_type: The ffn type.
            mlp_acts: The mlp acts.
        """
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attn_hook: Optional[AttentionHook] = None
        self.block_hook: Optional[BlockHook] = None

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WAN_SELFATTENTION_CLASSES[self_attn_type](
            dim, num_heads, window_size, qk_norm, eps, rope_after=rope_after, power=power
        )
        self.self_attn_type = self_attn_type
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        if ffn_type == "mlp":
            self.skip_ffn = None
        elif ffn_type == "GLUMBConvTemp":
            self.skip_ffn = GLUMBConvTemp(
                in_features=dim,
                hidden_features=ffn_dim,
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                t_kernel_size=3,
            )
            nn.init.zeros_(self.skip_ffn.t_conv.weight)
            nn.init.zeros_(self.skip_ffn.point_conv.conv.weight)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        intermediate_feats = {
            "x_in": x,
            "x_self_attn": None,
            "x_cross_attn": None,
            "x_ffn": None,
        }
        # self-attention
        x_dtype = x.dtype
        x_sa_in = (self.norm1(x).float() * (1 + e[1]) + e[0]).to(x_dtype)
        self_attn_kwargs = dict(
            x=x_sa_in,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
        )

        y = self.self_attn(**self_attn_kwargs)

        # Call hook if registered
        if self.attn_hook is not None:
            self.attn_hook(y)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        intermediate_feats["x_self_attn"] = x

        # cross-attention
        x = x.to(x_dtype)
        x_ca_in = self.norm3(x)
        x = x + self.cross_attn(x_ca_in, context, context_lens)

        intermediate_feats["x_cross_attn"] = x

        # ffn
        ffn_in = (self.norm2(x).float() * (1 + e[4]) + e[3]).to(x_dtype)
        y = self.ffn(ffn_in)
        if self.skip_ffn is not None:
            y_skip = self.skip_ffn(ffn_in, HW=grid_sizes[0])  # grid_sizes[0] is three values for T,H,W
            y = y + y_skip

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]

        intermediate_feats["x_ffn"] = x
        if self.block_hook is not None:
            self.block_hook(**intermediate_feats)

        del intermediate_feats

        return x.to(x_dtype)


class Head(nn.Module):
    """AdaLN-modulated output projection from tokens to packed patch voxels."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        """Init.

        Args:
            dim: The dim.
            out_dim: The out dim.
            patch_size: The patch size.
            eps: The eps.
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        x_type = x.dtype
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x.to(x_type)


class MLPProj(torch.nn.Module):
    """CLIP image-token projector (I2V / FLF2V)."""

    def __init__(self, in_dim, out_dim):
        """Init.

        Args:
            in_dim: The in dim.
            out_dim: The out dim.
        """
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        """Forward.

        Args:
            image_embeds: The image embeds.
        """
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(nn.Module, NativeConfigMixin):
    r"""Wan 2.1 official-layout stem; subclassed by the linear-attention variant."""

    # ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    ignore_for_config = ["cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        image_dim=1280,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.gradient_checkpointing = False
        self.enable_autocast = True

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(image_dim, dim)

        # initialize weights
        self.init_weights()
        self.lr_scale = None

    def forward(self, x, timestep, context, seq_len, clip_fea=None, y=None, **kwargs):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        x = [_x.to(self.dtype) for _x in x]
        context = [_c.to(self.dtype) for _c in context]
        t = timestep
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context.to(self.dtype),
            context_lens=context_lens,
        )

        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)  # .float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

        # init zero for v_img in each block
        for block in self.blocks:
            if isinstance(block.cross_attn, WanI2VCrossAttention):
                nn.init.zeros_(block.cross_attn.v_img.weight)
                nn.init.zeros_(block.cross_attn.v_img.bias)

    def register_attn_hook(self, layers=None, device="cpu"):
        """Register attn hook.

        Args:
            layers: The layers.
            device: The device.
        """
        for i, block in enumerate(self.blocks):
            if layers is None or i in layers:
                block.attn_hook = AttentionHook(device)

    def get_attn_output(self):
        """Get attn output."""
        attn_outputs = {}
        for i, block in enumerate(self.blocks):
            if block.attn_hook is not None:
                attn_outputs[i] = block.attn_hook.output
                block.attn_hook.clear()
        return attn_outputs

    def register_block_hook(self, layers=None, device="cpu", detach=True, score_only=False):
        """Register block hook.

        Args:
            layers: The layers.
            device: The device.
            detach: The detach.
            score_only: The score only.
        """
        for i, block in enumerate(self.blocks):
            if layers is None or i in layers:
                block.block_hook = BlockHook(device, detach, score_only)

    def get_block_output(self):
        """Get block output."""
        block_outputs = {}
        for i, block in enumerate(self.blocks):
            if block.block_hook is not None:
                block_outputs[i] = block.block_hook.get_output()
                block.block_hook.clear()
        return block_outputs


class WanLinearAttentionModel(WanModel):
    """Wan 2.1 stem whose self-attn layers can be flash or linear."""

    ignore_for_config = ["cross_attn_norm", "qk_norm", "text_dim", "window_size"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        image_dim=1280,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        linear_attn_idx=None,
        attn_type="flash",  # flash, linear, mllalinear
        ffn_type="mlp",
        rope_after=False,
        power=1.0,
    ):
        """Init.

        Args:
            model_type: The model type.
            patch_size: The patch size.
            text_len: The text len.
            in_dim: The in dim.
            dim: The dim.
            ffn_dim: The ffn dim.
            freq_dim: The freq dim.
            image_dim: The image dim.
            text_dim: The text dim.
            out_dim: The out dim.
            num_heads: The num heads.
            num_layers: The num layers.
            window_size: The window size.
            qk_norm: The qk norm.
            cross_attn_norm: The cross attn norm.
            eps: The eps.
            linear_attn_idx: The linear attn idx.
            attn_type: The attn type.
            ffn_type: The ffn type.
            rope_after: The rope after.
            power: The power.
        """
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            image_dim=image_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self_attn_types = ["flash"] * num_layers
        ffn_types = ["mlp"] * num_layers
        if linear_attn_idx is not None:
            for la_idx in linear_attn_idx:
                self_attn_types[la_idx] = attn_type
                ffn_types[la_idx] = ffn_type

        self.self_attn_types = self_attn_types
        self.repo_after = rope_after
        self.power = power

        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    self_attn_types[i],
                    rope_after,
                    power,
                    ffn_types[i],
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, timestep, context, seq_len, clip_fea=None, y=None, block_mask=None, **kwargs):
        r"""
        Forward pass through the diffusion model
        Same as WanModel, but save qkv for linear attention

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        x = [_x.to(self.dtype) for _x in x]
        context = [_c.to(self.dtype) for _c in context]
        t = timestep
        self.inference_timestep = int(t[-1].item())
        if not self.training and self.inference_timestep >= 850:
            # NOTE: hard code now. Keep the first several steps using dense attention
            pass

        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        for i, block in enumerate(self.blocks):
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context.to(self.dtype),
                context_lens=context_lens,
                block_mask=None,
            )

            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        return torch.stack(x, dim=0)  # .float()


def init_model_configs(model_cfg, vae_cfg):
    """Init model configs.

    Args:
        model_cfg: The model cfg.
        vae_cfg: The vae cfg.
    """
    # 1.3B T2V
    model_name = model_cfg.model
    if "1300M" in model_name or "1.3B" in model_name:
        basic_config = {
            "model_type": "t2v",
            "dim": 1536,
            "eps": 1e-06,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "in_dim": 16,
            "num_heads": 12,
            "num_layers": 30,
            "out_dim": 16,
            "text_len": 512,
            "patch_size": (1, 2, 2),
        }

    elif "14B" in model_name:  # default is T2V
        basic_config = {
            "model_type": "t2v",
            "dim": 5120,
            "eps": 1e-06,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "in_dim": 16,
            "num_heads": 40,
            "num_layers": 40,
            "out_dim": 16,
            "text_len": 512,
            "patch_size": (1, 2, 2),
        }
    else:
        raise ValueError(f"Model {model_name} not found")

    # update basic config with specific model config
    # now all I2V are use cross attention
    if "i2v" in model_name.lower():
        basic_config["model_type"] = "i2v"
        in_dim = vae_cfg.vae_latent_dim * 2
        if model_cfg.mask is not None:
            in_dim += 4
    else:
        in_dim = vae_cfg.vae_latent_dim

    basic_config["in_dim"] = in_dim
    basic_config["out_dim"] = vae_cfg.vae_latent_dim
    basic_config["patch_size"] = model_cfg.patch_size
    basic_config["linear_attn_idx"] = model_cfg.linear_attn_idx
    basic_config["attn_type"] = model_cfg.self_attn_type
    basic_config["rope_after"] = model_cfg.rope_after
    basic_config["power"] = model_cfg.power
    basic_config["ffn_type"] = model_cfg.ffn_type

    return basic_config
