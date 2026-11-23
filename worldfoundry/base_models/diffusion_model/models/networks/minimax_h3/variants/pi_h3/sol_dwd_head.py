# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hansheng Chen.
# Adapted from Lakonik/LakonLab (Apache-2.0), revision 2ee2b334374ec91b7cee371a85a524bf4ba644d3:
#   lakonlab/models/architectures/sol_h3/model.py
#   lakonlab/models/architectures/sol_h3/sol_h3.py
#   lakonlab/models/architectures/autoencoders/color_encoders.py
# Sol-H3-VAE reference decoder (Apache-2.0):
#   https://huggingface.co/Efficient-Large-Model/Sol-H3-VAE
"""Pure-PyTorch Sol-H3 DWD pixel refiner head and RGB color codec for pi-H3."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sol-H3-VAE decoder defaults (MiniMax-H3 vae/config.json and Sol-H3-VAE reference decoder).
NUM_HEADS = 32
HEAD_DIM = 64
DIM = NUM_HEADS * HEAD_DIM
FFN_MULT = 4
NUM_LAYERS = 24
NORM_EPS = 1e-5
NUM_REGISTER_TOKENS = 4
ROPE_THETA = 100.0
ROPE_DIM = int(HEAD_DIM * 0.75)
WINDOW = 16  # spatial window in tokens; odd layers shift by half
BLOCK = 5  # native latent frames per block (17 pixel frames)
T_SHIFT = 2  # temporal window shift of odd layers
TIME_NORM = 5
PATCH = 16
PATCH_T = 4
OUT_CH = 3
FRAMES_TO_TRIM = 3  # pixel frames dropped from the first latent of every 5-latent block

PI_H3_RGB_MEAN: tuple[float, float, float] = (-0.03, -0.088, -0.188)
PI_H3_RGB_STD: tuple[float, float, float] = (0.458, 0.448, 0.45)


def _rope(
    T: int,
    H: int,
    W: int,
    s: int,
    st: int,
    device: torch.device,
    *,
    block: int = BLOCK,
    window: int = WINDOW,
    time_norm: int = TIME_NORM,
    rope_dim: int = ROPE_DIM,
    rope_theta: float = ROPE_THETA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute window-local 3D RoPE cos/sin ``(1, T*H*W, 1, rope_dim // 2)`` for shifts ``(st, s)``."""

    if rope_dim <= 0 or rope_dim % 6 != 0:
        raise ValueError(f"Sol-H3 rope_dim must be a positive multiple of 6, got {rope_dim}.")
    lt = ((torch.arange(T, device=device) + st) % block).float()
    ly = ((torch.arange(H, device=device) + s) % window).float()
    lx = ((torch.arange(W, device=device) + s) % window).float()
    t = (lt + 0.5) / time_norm * 2.0 - 1.0
    hy = (ly + 0.5) / window * 2.0 - 1.0
    wx = (lx + 0.5) / window * 2.0 - 1.0
    grid = torch.stack(torch.meshgrid(t, hy, wx, indexing="ij"), -1).reshape(1, -1, 3)
    num_freqs = rope_dim // 6
    inv_freq = 1.0 / (rope_theta ** (torch.arange(num_freqs, dtype=torch.float32, device=device) / num_freqs))
    angles = 2.0 * math.pi * grid[:, :, :, None] * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3).unsqueeze(2)
    return angles.cos(), angles.sin()


def _rotate_(x: torch.Tensor, emb: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Apply rotary embeddings in place to ``(B, L, heads, head_dim)`` matching ``x * cos + rotate_half(x) * sin``."""

    cos = emb[0].to(x.dtype)
    sin = emb[1].to(x.dtype)
    half = cos.size(-1)
    first, second = x[..., :half], x[..., half : 2 * half]
    first_sin, second_sin = first * sin, second * sin
    first.mul_(cos).sub_(second_sin)
    second.mul_(cos).add_(first_sin)


def _partition(
    x: torch.Tensor,
    s: int,
    st: int,
    *,
    block: int = BLOCK,
    window: int = WINDOW,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int, int, int, int, int, int, int]]:
    """Partition ``(B, T, H, W, ...)`` into shifted ``(B*nW, block*window*window, ...)`` windows."""

    B, T, H, W = x.shape[:4]
    ws, tb = window, block
    rest = x.shape[4:]
    Tp = ((T + st + tb - 1) // tb) * tb
    Hp = ((H + s + ws - 1) // ws) * ws
    Wp = ((W + s + ws - 1) // ws) * ws
    xp = x.new_zeros(B, Tp, Hp, Wp, *rest)
    xp[:, st : st + T, s : s + H, s : s + W] = x
    valid = torch.zeros(Tp, Hp, Wp, dtype=torch.bool, device=x.device)
    valid[st : st + T, s : s + H, s : s + W] = True
    nt, nh, nw = Tp // tb, Hp // ws, Wp // ws
    r = len(rest)
    xw = xp.view(B, nt, tb, nh, ws, nw, ws, *rest).permute(0, 1, 3, 5, 2, 4, 6, *range(7, 7 + r))
    xw = xw.reshape(B * nt * nh * nw, tb * ws * ws, *rest)
    vw = valid.view(nt, tb, nh, ws, nw, ws).permute(0, 2, 4, 1, 3, 5).reshape(nt * nh * nw, tb * ws * ws)
    return xw, vw, (B, T, H, W, Tp, Hp, Wp, nt, nh, nw)


def _pack(rows: torch.Tensor, valid: torch.Tensor, total: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the ``total`` valid entries of ``(B, n, L)`` rows and return ``(packed_rows, cu_seqlens)``."""

    valid_exp = valid.expand(rows.shape).flatten(0, 1)
    order = valid_exp.logical_not().flatten().to(torch.int8).argsort(stable=True)[:total]
    cu_seqlens = F.pad(valid_exp.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
    return rows.flatten()[order], cu_seqlens


def _window_rows(
    B: int,
    T: int,
    H: int,
    W: int,
    nb: int,
    S: int,
    s: int,
    st: int,
    device: torch.device,
    *,
    padded: bool = False,
    block: int = BLOCK,
    window: int = WINDOW,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build window-attention gather indices over token rows and per-block global-token rows."""

    N = T * H * W
    tokens, valid, info = _partition(
        torch.arange(1, N + 1, device=device).view(1, T, H, W),
        s,
        st,
        block=block,
        window=window,
    )
    Tp, Hp, Wp, nt, nh, nw = info[4:]
    blocks = [(max(j * block - st, 0) // block, (min((j + 1) * block - st, T) - 1) // block) for j in range(nt)]
    global_rows = torch.tensor([[b * S + i for b in pair for i in range(S)] for pair in blocks], device=device)
    global_valid = torch.tensor([[True] * S + [b1 != b0] * S for b0, b1 in blocks], device=device)
    global_rows, global_valid = (
        t[:, None, None].expand(nt, nh, nw, 2 * S).reshape(-1, 2 * S) for t in (global_rows, global_valid)
    )
    batch = torch.arange(B, device=device)[:, None, None]
    token_rows = batch * N + tokens - 1
    key_rows = torch.cat([token_rows, B * N + batch * nb * S + global_rows], -1)
    key_valid = torch.cat([valid, global_valid], -1)
    if padded:
        output_rows = (
            torch.arange(token_rows.numel(), device=device)
            .view(B, nt, nh, nw, block, window, window)
            .permute(0, 1, 4, 2, 5, 3, 6)
            .reshape(B, Tp, Hp, Wp)
        )
        output_rows = output_rows[:, st : st + T, s : s + H, s : s + W].reshape(-1)
        return (
            token_rows.masked_fill(~valid, 0).flatten(0, 1),
            key_rows.masked_fill(~key_valid, 0).flatten(0, 1),
            key_valid.expand_as(key_rows).flatten(0, 1).contiguous(),
            output_rows,
        )
    q_rows, cu_seqlens_q = _pack(token_rows, valid, B * N)
    k_rows, cu_seqlens_k = _pack(
        key_rows,
        key_valid,
        B * (N + nh * nw * S * sum(1 + (b1 != b0) for b0, b1 in blocks)),
    )
    return q_rows, k_rows, cu_seqlens_q, cu_seqlens_k


def _global_rows(
    B: int,
    T: int,
    H: int,
    W: int,
    nb: int,
    S: int,
    device: torch.device,
    *,
    padded: bool = False,
    block: int = BLOCK,
) -> tuple[torch.Tensor, ...]:
    """Build global-token attention gather indices over each native block's tokens and global tokens."""

    N = T * H * W
    tokens = torch.arange(nb * block * H * W, device=device).view(nb, -1)
    batch = torch.arange(B, device=device)[:, None, None]
    globals_ = torch.arange(nb * S, device=device).view(nb, S)
    rows = torch.cat([batch * N + tokens, B * N + batch * nb * S + globals_], -1)
    valid = torch.cat([tokens < N, torch.ones(nb, S, dtype=torch.bool, device=device)], -1)
    if padded:
        return rows.masked_fill(~valid, 0).flatten(0, 1), valid.expand_as(rows).flatten(0, 1).contiguous()
    packed_rows, cu_seqlens_k = _pack(rows, valid, B * (N + nb * S))
    cu_seqlens_q = torch.arange(B * nb + 1, dtype=torch.int32, device=device) * S
    return packed_rows, cu_seqlens_q, cu_seqlens_k


def _use_padded_attention(device: torch.device, dtype: torch.dtype, *, force_padded: bool = False) -> bool:
    """Use masked SDPA on CPU, fp32, or pre-SM80 GPUs; use varlen FlashAttention on SM80+ half/bf16 CUDA."""

    if force_padded:
        return True
    return not (
        device.type == "cuda"
        and dtype in (torch.bfloat16, torch.float16)
        and torch.backends.cuda.is_flash_attention_available()
        and torch.cuda.get_device_capability(device) >= (8, 0)
        and hasattr(torch.ops.aten, "_flash_attention_forward")
    )


def _fp32_rms_norm(norm: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """Run RMSNorm in fp32 and cast back to ``x.dtype`` without requiring fp32 weight storage."""

    weight = None if norm.weight is None else norm.weight.float()
    return F.rms_norm(x.float(), norm.normalized_shape, weight, norm.eps).to(x.dtype)


class SolH3Attention(nn.Module):
    """Sol-H3 window/global multi-head attention with QK RMSNorm and SDPA/FlashAttention varlen dispatch."""

    def __init__(
        self,
        *,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        norm_eps: float = NORM_EPS,
    ) -> None:
        super().__init__()
        dim = num_heads * head_dim
        self.heads = num_heads
        self.dim_head = head_dim
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(head_dim, eps=norm_eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(head_dim, eps=norm_eps, elementwise_affine=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention on packed varlen ``(total, H, D)`` or padded ``(N, L, H, D)`` tensors."""

        if attn_mask is None:
            if cu_seqlens_q is None or cu_seqlens_k is None or max_seqlen_q is None or max_seqlen_k is None:
                raise ValueError("Varlen SolH3Attention requires cu_seqlens_q/k and max_seqlen_q/k.")
            return torch.ops.aten._flash_attention_forward(
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                0.0,
                False,
                False,
            )[0]
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return attended.transpose(1, 2)


class SolH3SwiGLU(nn.Module):
    """Sol-H3 SwiGLU feed-forward block with ``[gate, up]`` projection order."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.w1(x)
        gate, up = hidden.chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class SolH3Block(nn.Module):
    """One Sol-H3 decoder layer operating on ``(B, T, H, W, D)`` tokens and ``(B, nb, S, D)`` global tokens."""

    def __init__(
        self,
        *,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        ffn_mult: int = FFN_MULT,
        norm_eps: float = NORM_EPS,
        window_size: int = WINDOW,
        block_size: int = BLOCK,
        t_shift: int = T_SHIFT,
        time_norm: int = TIME_NORM,
        rope_theta: float = ROPE_THETA,
    ) -> None:
        super().__init__()
        dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.block_size = block_size
        self.t_shift = t_shift
        self.time_norm = time_norm
        self.rope_dim = int(head_dim * 0.75)
        self.rope_theta = rope_theta

        self.norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.scale2 = nn.Parameter(torch.empty(dim))
        self.attn = SolH3Attention(num_heads=num_heads, head_dim=head_dim, norm_eps=norm_eps)
        self.ff = SolH3SwiGLU(dim, dim * ffn_mult)

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        shifted: bool,
        global_indices: tuple[torch.Tensor, ...] | None = None,
        window_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        *,
        force_padded: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, g = self._attention(x, g, shifted, global_indices, window_indices, force_padded=force_padded)
        scale2 = self.scale2.to(x.dtype)
        x = x + self.ff(_fp32_rms_norm(self.norm2, x)) * scale2
        g = g + self.ff(_fp32_rms_norm(self.norm2, g)) * scale2
        return x, g

    def _attention(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        shifted: bool,
        global_indices: tuple[torch.Tensor, ...] | None,
        window_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        *,
        force_padded: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, W, _ = x.shape
        nb, S = g.shape[1:3]
        N = B * T * H * W
        heads, dh = self.num_heads, self.head_dim
        s, st = (self.window_size // 2, self.t_shift) if shifted else (0, 0)

        packed = torch.cat([x.flatten(0, 3), g.flatten(0, 2)], dim=0)
        qkv = self.attn.to_qkv(_fp32_rms_norm(self.norm1, packed))
        qkv = qkv.view(-1, heads, 3, dh)
        q = self.attn.norm_q(qkv[:, :, 0].float()).to(x.dtype)
        k = self.attn.norm_k(qkv[:, :, 1].float()).to(x.dtype)
        v = qkv[:, :, 2]

        rope = _rope(
            T,
            H,
            W,
            s,
            st,
            x.device,
            block=self.block_size,
            window=self.window_size,
            time_norm=self.time_norm,
            rope_dim=self.rope_dim,
            rope_theta=self.rope_theta,
        )
        _rotate_(q[:N].view(B, -1, heads, dh), rope)
        _rotate_(k[:N].view(B, -1, heads, dh), rope)

        padded = _use_padded_attention(x.device, x.dtype, force_padded=force_padded)
        if global_indices is None:
            global_indices = _global_rows(B, T, H, W, nb, S, x.device, padded=padded, block=self.block_size)
        if padded:
            rows, key_mask = global_indices
            og = self.attn(
                q[N:].unflatten(0, (B * nb, S)),
                k[rows],
                v[rows],
                attn_mask=key_mask[:, None, None, :],
            ).flatten(0, 1)
        else:
            rows, cu_seqlens_q, cu_seqlens_k = global_indices
            og = self.attn(
                q[N:],
                k[rows],
                v[rows],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=S,
                max_seqlen_k=self.block_size * H * W + S,
            )

        if window_indices is None:
            window_indices = _window_rows(
                B,
                T,
                H,
                W,
                nb,
                S,
                s,
                st,
                x.device,
                padded=padded,
                block=self.block_size,
                window=self.window_size,
            )
        q_rows, k_rows, *_ = window_indices
        q_win = q[q_rows]
        k_win = k[k_rows]
        v_win = v[k_rows]
        del qkv

        if padded:
            _, _, key_mask, output_rows = window_indices
            o = self.attn(q_win, k_win, v_win, attn_mask=key_mask[:, None, None, :]).flatten(0, 1)
            o = o[output_rows]
        else:
            _, _, cu_seqlens_q, cu_seqlens_k = window_indices
            max_q = self.block_size * self.window_size * self.window_size
            o = self.attn(
                q_win,
                k_win,
                v_win,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_q + 2 * S,
            )
            o = torch.empty_like(o).index_copy_(0, q_rows, o)

        scale1 = self.scale1.to(x.dtype)
        x = x + self.attn.to_out(o.flatten(1)).view_as(x) * scale1
        g = g + self.attn.to_out(og.flatten(1)).view_as(g) * scale1
        return x, g


class SolH3DWDHead(nn.Module):
    """Last ``num_layers`` Sol-H3-VAE decoder layers, final norm, and dual pixel projections.

    Expands final-layer DiT tokens onto ``patch``-pixel Sol tokens, refines them with
    alternating shifted 3D window + per-block global attention, and projects directly to
    ImageNet-normalized RGB frames ``(B, F, 3, H_pix, W_pix)`` without running the video VAE.
    """

    patch_size = 1

    def __init__(
        self,
        in_channels: int,
        src_patch_size: Sequence[int] = (PATCH_T, 2 * PATCH, 2 * PATCH),
        first_frame_src_patch_size: Sequence[int] | None = (PATCH_T - FRAMES_TO_TRIM, 2 * PATCH, 2 * PATCH),
        first_frame_period: int = BLOCK,
        *,
        residual: bool = False,
        pixel_channels: int = OUT_CH,
        num_layers: int = 8,
        total_vae_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        head_dim: int = HEAD_DIM,
        ffn_mult: int = FFN_MULT,
        norm_eps: float = NORM_EPS,
        num_register_tokens: int = NUM_REGISTER_TOKENS,
        window_size: int = WINDOW,
        block_size: int = BLOCK,
        t_shift: int = T_SHIFT,
        time_norm: int = TIME_NORM,
        spatial_patch: int = PATCH,
        patch_t: int = PATCH_T,
        frames_to_trim: int = FRAMES_TO_TRIM,
        rope_theta: float = ROPE_THETA,
    ) -> None:
        super().__init__()
        if residual:
            raise ValueError("SolH3DWDHead requires linear_head=False (residual=False) on the DWD head.")
        src_tuple = tuple(int(v) for v in src_patch_size)
        if len(src_tuple) != 3:
            raise ValueError(f"src_patch_size must have 3 elements, got {src_tuple}.")
        _, src_h, src_w = src_tuple
        if src_tuple != (patch_t, src_h, src_w) or src_h % spatial_patch != 0 or src_w % spatial_patch != 0:
            raise ValueError(
                f"Tokens must cover one latent frame and whole Sol tokens, got src_patch_size={src_tuple} "
                f"for patch_t={patch_t}, spatial_patch={spatial_patch}."
            )
        if (
            first_frame_src_patch_size is None
            or tuple(int(v) for v in first_frame_src_patch_size) != (patch_t - frames_to_trim, src_h, src_w)
            or first_frame_period != block_size
            or pixel_channels != OUT_CH
        ):
            raise ValueError("Pixel geometry must follow Sol-H3 frame trimming at every native latent block.")

        self.dim = num_heads * head_dim
        self.expansion = (src_h // spatial_patch, src_w // spatial_patch)
        self.num_layers = num_layers
        self.total_vae_layers = total_vae_layers
        self.block_size = block_size
        self.window_size = window_size
        self.t_shift = t_shift
        self.spatial_patch = spatial_patch
        self.patch_t = patch_t
        self.frames_to_trim = frames_to_trim
        self.pixel_channels = pixel_channels

        # Preserve exact upstream checkpoint parameter names.
        self.feature_expand = nn.Linear(in_channels, self.dim * self.expansion[0] * self.expansion[1])
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, self.dim))
        self.blocks = nn.ModuleList(
            [
                SolH3Block(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_mult=ffn_mult,
                    norm_eps=norm_eps,
                    window_size=window_size,
                    block_size=block_size,
                    t_shift=t_shift,
                    time_norm=time_norm,
                    rope_theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.dim, eps=norm_eps)
        self.proj_out = nn.Linear(self.dim, pixel_channels * patch_t * spatial_patch * spatial_patch)
        self.first_frame_proj_out = nn.Linear(
            self.dim,
            pixel_channels * (patch_t - frames_to_trim) * spatial_patch * spatial_patch,
        )

    def forward(
        self,
        pixels: torch.Tensor | None,
        features: torch.Tensor,
        token_grid_size: tuple[int, int, int],
        global_tokens: torch.Tensor | None = None,
        *,
        force_padded: bool = False,
    ) -> torch.Tensor:
        """Decode ``(B, t * h * w, in_channels)`` DiT features into ``(B, F, 3, H_pix, W_pix)``."""

        if pixels is not None:
            raise ValueError("SolH3DWDHead does not take residual linear-head pixels; pass pixels=None.")
        B = features.size(0)
        t, h, w = token_grid_size
        e_h, e_w = self.expansion
        x = self.feature_expand(features.to(self.feature_expand.weight.dtype))
        x = (
            x.view(B, t, h, w, e_h, e_w, self.dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(B, t, h * e_h, w * e_w, self.dim)
        )
        return self.decode(x, global_tokens=global_tokens, force_padded=force_padded)

    def decode(
        self,
        x: torch.Tensor,
        global_tokens: torch.Tensor | None = None,
        *,
        force_padded: bool = False,
    ) -> torch.Tensor:
        """Run Sol-H3 layers on ``(B, T, H, W, D)`` activations; returns ``(B, F, 3, H_pix, W_pix)``."""

        B, T, H, W, _ = x.shape
        if global_tokens is None:
            reg = self.register_tokens.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)
            cls_tok = x.new_zeros(B, 1, self.dim)
            global_tokens = torch.cat([reg, cls_tok], dim=1)
            num_blocks = (T + self.block_size - 1) // self.block_size
            global_tokens = global_tokens.unsqueeze(1).expand(B, num_blocks, -1, -1).contiguous()
        g = global_tokens.to(device=x.device, dtype=x.dtype)

        for block, args in self.block_args(x.shape, g.shape, x.device, x.dtype, force_padded=force_padded):
            x, g = block(x, g, *args, force_padded=force_padded)

        x = self.norm_out(x.to(self.norm_out.weight.dtype)).to(self.proj_out.weight.dtype)
        num_chunks, tail = divmod(T, self.block_size)
        out: torch.Tensor | None = None
        if num_chunks:
            out = self._project_pixels(x[:, : num_chunks * self.block_size].unflatten(1, (num_chunks, self.block_size)))
            out = out.flatten(1, 2)
        if tail:
            end = self._project_pixels(x[:, num_chunks * self.block_size :].unsqueeze(1)).flatten(1, 2)
            out = torch.cat((out, end), dim=1) if out is not None else end
        assert out is not None
        return out

    def block_args(
        self,
        x_shape: Sequence[int],
        g_shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        *,
        force_padded: bool = False,
    ) -> Iterator[tuple[nn.Module, tuple[Any, ...]]]:
        """Yield each Sol block and its cached ``(shifted, global_indices, window_indices)`` arguments."""

        B, T, H, W, _ = x_shape
        nb, S = int(g_shape[1]), int(g_shape[2])
        padded = _use_padded_attention(device, dtype, force_padded=force_padded)
        global_indices = _global_rows(B, T, H, W, nb, S, device, padded=padded, block=self.block_size)
        window_indices: dict[bool, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        start_index = self.total_vae_layers - len(self.blocks)
        for index, block in enumerate(self.blocks, start_index):
            shifted = index % 2 == 1
            if shifted not in window_indices:
                s, st = (self.window_size // 2, self.t_shift) if shifted else (0, 0)
                window_indices[shifted] = _window_rows(
                    B,
                    T,
                    H,
                    W,
                    nb,
                    S,
                    s,
                    st,
                    device,
                    padded=padded,
                    block=self.block_size,
                    window=self.window_size,
                )
            yield block, (shifted, global_indices, window_indices[shifted])

    def _project_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Project chunks ``(B, n, t, H, W, D)`` into ``(B, n, F, 3, patch*H, patch*W)``."""

        B, n, t, H, W, _ = x.shape
        first_t = self.patch_t - self.frames_to_trim
        p = self.spatial_patch
        c = self.pixel_channels
        first = self.first_frame_proj_out(x[:, :, :1])
        first = (
            first.view(B, n, H, W, c, first_t, p, p)
            .permute(0, 1, 5, 4, 2, 6, 3, 7)
            .reshape(B, n, first_t, c, H * p, W * p)
        )
        if t > 1:
            rest = self.proj_out(x[:, :, 1:])
            rest = (
                rest.view(B, n, t - 1, H, W, c, self.patch_t, p, p)
                .permute(0, 1, 2, 6, 5, 3, 7, 4, 8)
                .reshape(B, n, (t - 1) * self.patch_t, c, H * p, W * p)
            )
            first = torch.cat((first, rest), dim=2)
        return first


class RGBColorEncoder(nn.Module):
    """Affine RGB color codec matching LakonLab's Sol-H3 DWD output space."""

    affine_mean: torch.Tensor
    affine_std: torch.Tensor

    def __init__(
        self,
        *,
        use_affine_norm: bool = True,
        mean: Sequence[float] = PI_H3_RGB_MEAN,
        std: Sequence[float] | float = PI_H3_RGB_STD,
    ) -> None:
        super().__init__()
        self.use_affine_norm = bool(use_affine_norm)
        if self.use_affine_norm:
            self.register_buffer("affine_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("affine_std", torch.tensor(std, dtype=torch.float32))

    @property
    def dtype(self) -> torch.dtype:
        if self.use_affine_norm:
            return self.affine_mean.dtype
        return torch.float32

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """Map ``[-1, 1]`` RGB ``(B, 3, ...)`` into affine-normalized space."""

        if self.use_affine_norm:
            n_dim = img.dim() - 2
            mean = self.affine_mean.to(device=img.device, dtype=img.dtype).reshape(-1, *([1] * n_dim))
            std = self.affine_std.to(device=img.device, dtype=img.dtype).reshape(-1, *([1] * n_dim))
            img = (img - mean) / std
        return img

    def decode(self, img: torch.Tensor) -> torch.Tensor:
        """Map affine-normalized RGB ``(B, 3, ...)`` into ``[-1, 1]`` clamped RGB."""

        if self.use_affine_norm:
            n_dim = img.dim() - 2
            mean = self.affine_mean.to(device=img.device, dtype=img.dtype).reshape(-1, *([1] * n_dim))
            std = self.affine_std.to(device=img.device, dtype=img.dtype).reshape(-1, *([1] * n_dim))
            return (img * std + mean).clamp(-1.0, 1.0)
        return img.clamp(-1.0, 1.0)

    def decode_to_unit_range(self, img: torch.Tensor) -> torch.Tensor:
        """Decode affine-normalized RGB directly to ``[0, 1]`` (``decode(x).add(1).mul(0.5)``)."""

        return self.decode(img).add(1.0).mul(0.5).clamp(0.0, 1.0)


__all__ = [
    "BLOCK",
    "DIM",
    "FFN_MULT",
    "FRAMES_TO_TRIM",
    "HEAD_DIM",
    "NORM_EPS",
    "NUM_HEADS",
    "NUM_LAYERS",
    "NUM_REGISTER_TOKENS",
    "OUT_CH",
    "PATCH",
    "PATCH_T",
    "PI_H3_RGB_MEAN",
    "PI_H3_RGB_STD",
    "ROPE_DIM",
    "ROPE_THETA",
    "TIME_NORM",
    "T_SHIFT",
    "WINDOW",
    "RGBColorEncoder",
    "SolH3Attention",
    "SolH3Block",
    "SolH3DWDHead",
    "SolH3SwiGLU",
    "_global_rows",
    "_pack",
    "_partition",
    "_rope",
    "_rotate_",
    "_use_padded_attention",
    "_window_rows",
]
