"""Unified DiffusionTransformer3D for Kandinsky 6 video (+ optional audio) generation.

Architecture decisions:
- Single class with is_multimodal flag; no separate T2VA subclass.
- FusedTransformerDecoderBlock selected at init when is_multimodal=True.
- forward() takes explicit x_video / x_audio kwargs so denoise_loop stays clean.
- No distributed primitives — prod-level TP is layered externally.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.tensors import get_freqs, apply_scale_shift_norm, apply_gate_sum, apply_rotary
from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.attention.dispatch import SelfAttentionEngine, _sdpa
from worldfoundry.base_models.diffusion_model.models.networks.kandinsky6.rope import RoPE1D, RoPE3D


# ---------------------------------------------------------------------------
# Small embedding / projection modules
# ---------------------------------------------------------------------------

class _TimestepEmbedder(nn.Module):
    """MLP after the sinusoidal timestep. Names match Diffusers ``TimestepEmbedding``."""

    def __init__(self, model_dim: int, time_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, time_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_dim, time_dim)


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0):
        super().__init__()
        assert model_dim % 2 == 0
        self.register_buffer("freqs", get_freqs(model_dim // 2, max_period), persistent=False)
        self.timestep_embedder = _TimestepEmbedder(model_dim, time_dim)

    def forward(self, time: Tensor) -> Tensor:
        # Sinusoidal + MLP in fp32 for numerical stability; emit weight dtype.
        embedder = self.timestep_embedder
        freqs = self.freqs.to(device=time.device, dtype=torch.float32)
        args = torch.outer(time.float(), freqs)
        embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        h = F.linear(
            embed,
            embedder.linear_1.weight.float(),
            None if embedder.linear_1.bias is None else embedder.linear_1.bias.float(),
        )
        out = F.linear(
            embedder.act(h),
            embedder.linear_2.weight.float(),
            None if embedder.linear_2.bias is None else embedder.linear_2.bias.float(),
        )
        return out.to(dtype=embedder.linear_2.weight.dtype)

    def reset_parameters(self) -> None:
        self.freqs = get_freqs(self.freqs.shape[0], 10000.0).to(self.freqs.device)


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim: int, model_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder outputs are often fp32; DiT Linears are bf16 (no autocast).
        wdtype = self.in_layer.weight.dtype
        return self.norm(self.in_layer(x.to(dtype=wdtype)))

    def reset_parameters(self) -> None:
        self.norm.reset_parameters()
        self.in_layer.reset_parameters()


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim: int, model_dim: int, patch_size: tuple):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        batched = x.ndim == 5
        if not batched and x.ndim != 4:
            raise ValueError("visual input must have shape (T,H,W,C) or (B,T,H,W,C)")

        offset = 1 if batched else 0
        shape = x.shape
        T, H, W, C = shape[offset:]
        pT, pH, pW = self.patch_size
        if batched:
            x = (
                x.view(shape[0], T // pT, pT, H // pH, pH, W // pW, pW, C)
                .permute(0, 1, 3, 5, 2, 4, 6, 7)
                .flatten(4, 7)
            )
        else:
            x = (
                x.view(T // pT, pT, H // pH, pH, W // pW, pW, C)
                .permute(0, 2, 4, 1, 3, 5, 6)
                .flatten(3, 6)
            )
        return self.in_layer(x.to(dtype=self.in_layer.weight.dtype))


class Modulation(nn.Module):
    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.act = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        # AdaLN params in fp32 for numerical stability.
        out = F.linear(
            self.act(x.float()),
            self.out_layer.weight.float(),
            None if self.out_layer.bias is None else self.out_layer.bias.float(),
        )
        return out.to(dtype=x.dtype)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)


class _GELUProjection(nn.Module):
    """Bias-free linear + exact GELU. The linear is ``proj``, matching Diffusers ``GELU``."""

    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, ff_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        # Dropout keeps the output linear at index 2, matching Diffusers ``FeedForward.net``.
        self.net = nn.ModuleList([
            _GELUProjection(dim, ff_dim),
            nn.Dropout(0.0),
            nn.Linear(ff_dim, dim, bias=False),
        ])

    def forward(self, x: Tensor) -> Tensor:
        for module in self.net:
            x = module(x)
        return x


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class MultiheadSelfAttentionEnc(nn.Module):
    """Self-attention for text encoder blocks (text_token_padding-aware)."""

    def __init__(self, dim: int, head_dim: int, engine: str = "auto", text_token_padding: bool = False):
        super().__init__()
        self.num_heads = dim // head_dim
        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer    = nn.Linear(dim, dim)
        self.force_sdpa = text_token_padding
        self.attn   = SelfAttentionEngine("sdpa" if text_token_padding else engine)

    def forward(self, x: Tensor, rope: Tensor, attn_mask=None) -> Tensor:
        shape = x.shape[:-1]
        q = self.to_query(x).reshape(*shape, self.num_heads, -1)
        k = self.to_key(x).reshape(*shape, self.num_heads, -1)
        v = self.to_value(x).reshape(*shape, self.num_heads, -1)
        q = self.query_norm(q)
        k = self.key_norm(k)
        q = apply_rotary(q, rope).type_as(q)
        k = apply_rotary(k, rope).type_as(k)
        query_was_batched = q.dim() == 4
        if not query_was_batched:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        args = {"q": q, "k": k, "v": v}
        if attn_mask is not None:
            args["attn_mask"] = attn_mask
        out = (_sdpa if attn_mask is not None else self.attn.get_attention())(**args)
        if not query_was_batched:
            out = out[0]
        out = out.flatten(-2, -1)
        return self.out_layer(out)


class MultiheadSelfAttentionDec(nn.Module):
    """Self-attention for visual decoder blocks."""

    def __init__(self, dim: int, head_dim: int, engine: str = "auto"):
        super().__init__()
        self.num_heads = dim // head_dim
        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.to_value = nn.Linear(dim, dim)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer    = nn.Linear(dim, dim)
        self.attn   = SelfAttentionEngine(engine)

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        shape = x.shape[:-1]
        q = self.to_query(x).reshape(*shape, self.num_heads, -1)
        k = self.to_key(x).reshape(*shape, self.num_heads, -1)
        v = self.to_value(x).reshape(*shape, self.num_heads, -1)
        q = self.query_norm(q)
        k = self.key_norm(k)
        q = apply_rotary(q, rope).type_as(q)
        k = apply_rotary(k, rope).type_as(k)

        query_was_batched = q.dim() == 4
        if not query_was_batched:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)

        out = self.attn.get_attention()(q=q, k=k, v=v)

        if not query_was_batched:
            out = out[0]
        out = out.flatten(-2, -1)

        return self.out_layer(out)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, q_dim: int, head_dim: int, kv_dim: int | None = None, engine: str = "auto", text_token_padding: bool = False):
        super().__init__()
        kv_dim = kv_dim or q_dim
        self.num_heads = q_dim // head_dim
        self.to_query = nn.Linear(q_dim, q_dim)
        self.to_key = nn.Linear(kv_dim, q_dim)
        self.to_value = nn.Linear(kv_dim, q_dim)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer    = nn.Linear(q_dim, q_dim)
        self.force_sdpa = text_token_padding
        self.attn   = SelfAttentionEngine("sdpa" if text_token_padding else engine)

    def forward(self, x: Tensor, cond: Tensor, attn_mask=None, rope_q: Tensor | None = None, rope_kv: Tensor | None = None) -> Tensor:
        sq, sk = x.shape[:-1], cond.shape[:-1]
        q = self.to_query(x).reshape(*sq, self.num_heads, -1)
        k = self.to_key(cond).reshape(*sk, self.num_heads, -1)
        v = self.to_value(cond).reshape(*sk, self.num_heads, -1)
        q = self.query_norm(q)
        k = self.key_norm(k)
        if rope_q  is not None: q = apply_rotary(q, rope_q).type_as(q)
        if rope_kv is not None: k = apply_rotary(k, rope_kv).type_as(k)
        query_was_batched = q.dim() == 4
        if not query_was_batched:
            q = q.unsqueeze(0)
        if k.dim() < q.dim():
            k, v = k.unsqueeze(0), v.unsqueeze(0)
        args = {"q": q, "k": k, "v": v}
        if attn_mask is not None: args["attn_mask"] = attn_mask
        out = (_sdpa if attn_mask is not None else self.attn.get_attention())(**args)
        if not query_was_batched:
            out = out[0]
        return self.out_layer(out.flatten(-2, -1))


# ---------------------------------------------------------------------------
# Output layers
# ---------------------------------------------------------------------------

class OutLayer(nn.Module):
    """Projects model_dim → pixel patches for the video output."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple):
        super().__init__()
        self.patch_size = patch_size
        self.modulation  = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim)

    def forward(self, visual_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        condition_shape = (scale.shape[0],) + (1,) * (visual_embed.ndim - 2) + (scale.shape[-1],)
        x = apply_scale_shift_norm(
            self.norm,
            visual_embed,
            scale.reshape(condition_shape),
            shift.reshape(condition_shape),
        ).type_as(visual_embed)
        x = self.out_layer(x)

        if x.ndim == 5:
            batch, T, H, W, _ = x.shape
            pT, pH, pW = self.patch_size
            return (
                x.view(batch, T, H, W, -1, pT, pH, pW)
                .permute(0, 1, 5, 2, 6, 3, 7, 4)
                .flatten(1, 2)
                .flatten(2, 3)
                .flatten(3, 4)
            )

        T, H, W, _ = x.shape
        pT, pH, pW = self.patch_size
        return (
            x.view(T, H, W, -1, pT, pH, pW)
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1).flatten(1, 2).flatten(2, 3)
        )

    def reset_parameters(self) -> None:
        self.modulation.reset_parameters()
        self.norm.reset_parameters()
        self.out_layer.reset_parameters()


class OutLayerAudio(nn.Module):
    """Projects model_dim_a → audio latent channels."""

    def __init__(self, model_dim: int, time_dim: int, audio_dim: int):
        super().__init__()
        self.modulation  = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, audio_dim)

    def forward(self, audio_embed: Tensor, time_embed: Tensor) -> Tensor:
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        x = apply_scale_shift_norm(self.norm, audio_embed, scale, shift).type_as(audio_embed)
        x = self.norm(x)  # matches reference training (double norm — do not remove for parity)
        return self.out_layer(x)

    def reset_parameters(self) -> None:
        self.modulation.reset_parameters()
        self.norm.reset_parameters()
        self.out_layer.reset_parameters()


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Text-only self-attention + FFN block."""

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int, engine: str = "auto", text_token_padding: bool = False):
        super().__init__()
        self.text_modulation  = Modulation(time_dim, model_dim, 6)
        self.attn_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.attn = MultiheadSelfAttentionEnc(model_dim, head_dim, engine, text_token_padding)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward      = FeedForward(model_dim, ff_dim)

    def forward(self, x: Tensor, time_embed: Tensor, rope: Tensor, attn_mask=None) -> Tensor:
        sa_p, ff_p = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        x = apply_gate_sum(x, self.attn(apply_scale_shift_norm(self.attn_norm, x, scale, shift), rope, attn_mask), gate)
        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        x = apply_gate_sum(x, self.feed_forward(apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)), gate)
        return x

    def reset_parameters(self) -> None:
        for m in self.children():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


class TransformerDecoderBlock(nn.Module):
    """Visual self-attention + cross-attention to text + FFN block."""

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int, engine: str = "auto", text_token_padding: bool = False):
        super().__init__()
        self.visual_modulation      = Modulation(time_dim, model_dim, 9)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention       = MultiheadSelfAttentionDec(model_dim, head_dim, engine)
        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention       = MultiheadCrossAttention(model_dim, head_dim, engine=engine, text_token_padding=text_token_padding)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward       = FeedForward(model_dim, ff_dim)

    def forward(self, vis: Tensor, text: Tensor, time_embed: Tensor, rope: Tensor, attn_mask=None) -> Tensor:
        sa_p, ca_p, ff_p = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        vis = apply_gate_sum(vis, self.self_attention(apply_scale_shift_norm(self.self_attention_norm, vis, scale, shift), rope), gate)
        shift, scale, gate = torch.chunk(ca_p, 3, dim=-1)
        vis = apply_gate_sum(vis, self.cross_attention(apply_scale_shift_norm(self.cross_attention_norm, vis, scale, shift), text, attn_mask), gate)
        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        vis = apply_gate_sum(vis, self.feed_forward(apply_scale_shift_norm(self.feed_forward_norm, vis, scale, shift)), gate)
        return vis

    def reset_parameters(self) -> None:
        for m in self.children():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


class FusedTransformerDecoderBlock(nn.Module):
    """Fused video + audio block with cross-modal attention (T2VA backbone)."""

    def __init__(
        self,
        model_dim:   int, time_dim:   int, ff_dim:   int, head_dim:   int,
        model_dim_a: int, time_dim_a: int, ff_dim_a: int, head_dim_a: int,
        engine: str = "auto", text_token_padding: bool = False,
        ca_rope: bool = False, cross_gates: bool = False, fix_modulation: bool = False,
    ):
        super().__init__()
        self.video_dec_block = TransformerDecoderBlock(model_dim,   time_dim,   ff_dim,   head_dim,   engine, text_token_padding)
        self.audio_dec_block = TransformerDecoderBlock(model_dim_a, time_dim_a, ff_dim_a, head_dim_a, engine, text_token_padding)

        self.va_cross_attention = MultiheadCrossAttention(model_dim,   head_dim,   model_dim_a, engine)
        self.av_cross_attention = MultiheadCrossAttention(model_dim_a, head_dim_a, model_dim,   engine)

        self.va_modulation  = Modulation(time_dim,   model_dim   if not cross_gates else model_dim * 2 + model_dim_a, 1 if cross_gates else 3)
        self.av_modulation  = Modulation(time_dim_a, model_dim_a if not cross_gates else model_dim_a * 2 + model_dim, 1 if cross_gates else 3)
        self.va_normalization = nn.LayerNorm(model_dim,   elementwise_affine=False)
        self.av_normalization = nn.LayerNorm(model_dim_a, elementwise_affine=False)

        self.ca_rope       = ca_rope
        self.cross_gates   = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim     = model_dim
        self.model_dim_a   = model_dim_a

    def forward(
        self,
        vis: Tensor,
        aud: Tensor,
        text_v: Tensor,
        text_a: Tensor,
        time_embed,           # (video_time, audio_time) tuple
        vis_rope: Tensor,
        aud_rope: Tensor,
        attn_mask=None,
        modality_mask=None,
        av_gate_scale: float = 1.0,
        va_gate_scale: float = 1.0,
    ):
        fake_audio = modality_mask[0] if modality_mask is not None else 0
        fake_video = modality_mask[1] if modality_mask is not None else 0
        t_v, t_a = time_embed

        # ---- video backbone ----
        if vis is not None:
            sa_p, ca_p, ff_p = torch.chunk(self.video_dec_block.visual_modulation(t_v), 3, dim=-1)

            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            vis = apply_gate_sum(vis, self.video_dec_block.self_attention(apply_scale_shift_norm(self.video_dec_block.self_attention_norm, vis, scale, shift), vis_rope), gate).type_as(vis)

            shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
            vis_pre_ca = apply_scale_shift_norm(self.video_dec_block.cross_attention_norm, vis, scale, shift).type_as(vis)
            vis_out_t  = self.video_dec_block.cross_attention(vis_pre_ca, text_v, attn_mask)  # text CA (saved for cross-modal)

        # ---- audio backbone ----
        if aud is not None:
            sa_p, ca_p, ff_p_a = torch.chunk(self.audio_dec_block.visual_modulation(t_a), 3, dim=-1)

            shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
            aud = apply_gate_sum(aud, self.audio_dec_block.self_attention(apply_scale_shift_norm(self.audio_dec_block.self_attention_norm, aud, scale, shift), aud_rope), gate).type_as(aud)

            shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
            aud_pre_ca = apply_scale_shift_norm(self.audio_dec_block.cross_attention_norm, aud, scale, shift).type_as(aud)
            aud_out_t  = self.audio_dec_block.cross_attention(aud_pre_ca, text_a, attn_mask)
            aud = apply_gate_sum(aud, aud_out_t, gate_a).type_as(aud)

            # ---- cross-modal attention ----
            if vis is not None:
                t_va_mod = t_a if not self.fix_modulation else t_v
                t_av_mod = t_v if not self.fix_modulation else t_a
                va_params = self.va_modulation(t_va_mod)
                av_params = self.av_modulation(t_av_mod)

                if self.cross_gates:
                    va_shift, va_scale, va_gate = torch.split(va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1)
                    av_shift, av_scale, av_gate = torch.split(av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1)
                else:
                    va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
                    av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)

                vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)
                vis_for_va = apply_scale_shift_norm(self.va_normalization, vis, va_scale, va_shift).type_as(vis)
                aud_for_av = apply_scale_shift_norm(self.av_normalization, aud, av_scale, av_shift).type_as(aud)

                # V→A and A→V attention
                rq_v = vis_rope if self.ca_rope else None
                rk_a = aud_rope if self.ca_rope else None
                vis_from_aud = self.va_cross_attention(vis_for_va, aud_pre_ca, rope_q=rq_v, rope_kv=rk_a) * (1 - fake_audio) * (1 - fake_video)
                aud_from_vis = self.av_cross_attention(aud_for_av, vis_pre_ca, rope_q=rk_a, rope_kv=rq_v) * (1 - fake_audio) * (1 - fake_video)

                va_g = (va_gate if not self.cross_gates else av_gate) * va_gate_scale
                av_g = (av_gate if not self.cross_gates else va_gate) * av_gate_scale
                vis = apply_gate_sum(vis, vis_from_aud, va_g).type_as(vis)
                aud = apply_gate_sum(aud, aud_from_vis, av_g).type_as(aud)
        else:
            vis = apply_gate_sum(vis, vis_out_t, gate_v).type_as(vis)

        # ---- FFN ----
        if vis is not None:
            shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
            vis = apply_gate_sum(vis, self.video_dec_block.feed_forward(apply_scale_shift_norm(self.video_dec_block.feed_forward_norm, vis, scale, shift)), gate).type_as(vis)

        if aud is not None:
            shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
            aud = apply_gate_sum(aud, self.audio_dec_block.feed_forward(apply_scale_shift_norm(self.audio_dec_block.feed_forward_norm, aud, scale, shift)), gate).type_as(aud)

        return vis, aud

    def reset_parameters(self) -> None:
        for m in self.children():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


# ---------------------------------------------------------------------------
# Unified DiffusionTransformer3D
# ---------------------------------------------------------------------------

class DiffusionTransformer3D(nn.Module):
    """Kandinsky 6 DiT — handles T2V (is_multimodal=False) and T2VA (is_multimodal=True).

    Constructor kwargs map directly to the YAML dit_params section.
    Audio-specific kwargs are only used when is_multimodal=True.
    """

    def __init__(
        self,
        in_visual_dim:  int   = 16,
        out_visual_dim: int   = 16,
        in_text_dim:    int   = 3584,
        in_text_dim2:   int   = 768,
        time_dim:       int   = 1024,
        patch_size:     tuple = (1, 2, 2),
        model_dim:      int   = 4096,
        ff_dim:         int   = 16384,
        num_text_blocks:   int = 4,
        num_visual_blocks: int = 60,
        axes_dims:      tuple = (32, 48, 48),
        visual_cond:    bool  = True,
        is_multimodal:  bool  = False,
        # Audio (T2VA only)
        in_audio_dim:    int   = 20,
        out_audio_dim:   int | None = None,
        model_dim_a:     int | None = None,
        time_dim_a:      int | None = None,
        ff_dim_a:        int | None = None,
        axes_dims_a:     tuple | None = None,
        audio_freqs_scaling: float = 1.0,
        # Misc
        attention_engine:    str  = "auto",
        text_token_padding:  bool = False,
        ca_rope:             bool = False,
        cross_gates:         bool = False,
        fix_modulation:      bool = False,
        # I2VA: 0 = off; 2 = generated vs reference frame (tail_cond_first_frame)
        visual_token_type_num_embeddings: int = 0,
    ):
        super().__init__()
        self.patch_size     = patch_size
        self.visual_cond    = visual_cond
        self.is_multimodal  = is_multimodal
        self.in_visual_dim  = in_visual_dim
        self.in_audio_dim   = in_audio_dim
        self.text_token_padding = text_token_padding
        self.visual_token_type_num_embeddings = int(visual_token_type_num_embeddings or 0)
        # Time-independent text/pooled projections (cleared each generation).
        self._text_proj_cache: dict[tuple, object] = {}

        head_dim = sum(axes_dims)

        # Effective audio dims (default to video dims)
        model_dim_a  = model_dim_a  or model_dim
        time_dim_a   = time_dim_a   or time_dim
        ff_dim_a     = ff_dim_a     or ff_dim
        axes_dims_a  = axes_dims_a  or axes_dims
        head_dim_a   = sum(axes_dims_a)

        # ---- visual backbone (shared) ----
        vis_in_dim = (2 * in_visual_dim + 1) if visual_cond else in_visual_dim
        self.visual_embeddings   = VisualEmbeddings(vis_in_dim, model_dim, patch_size)
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = nn.Embedding(
                self.visual_token_type_num_embeddings, model_dim
            )
        self.visual_rope         = RoPE3D(axes_dims)
        self.out_layer           = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

        if not is_multimodal:
            # T2V: single text/time embedding branch
            self.time_embeddings         = TimeEmbeddings(model_dim, time_dim)
            self.text_embeddings         = TextEmbeddings(in_text_dim, model_dim)
            self.pooled_text_embeddings  = TextEmbeddings(in_text_dim2, time_dim)
            self.text_rope               = RoPE1D(head_dim)
            self.text_blocks             = nn.ModuleList([
                TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine, text_token_padding)
                for _ in range(num_text_blocks)
            ])
            self.visual_transformer_blocks           = nn.ModuleList([
                TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine, text_token_padding)
                for _ in range(num_visual_blocks)
            ])
        else:
            # T2VA: dual (video / audio) text+time branches + fused blocks
            self.audio_embeddings       = TextEmbeddings(in_audio_dim, model_dim_a)
            self.audio_rope             = RoPE1D(head_dim_a, freqs_scaling=audio_freqs_scaling)
            self.audio_out_layer        = OutLayerAudio(
                model_dim_a, time_dim_a, out_audio_dim or in_audio_dim
            )

            for prefix, md, td, fd, hd in [
                ("video", model_dim,   time_dim,   ff_dim,   head_dim),
                ("audio", model_dim_a, time_dim_a, ff_dim_a, head_dim_a),
            ]:
                setattr(self, f"{prefix}_time_embeddings",        TimeEmbeddings(md, td))
                setattr(self, f"{prefix}_text_embeddings",        TextEmbeddings(in_text_dim, md))
                setattr(self, f"{prefix}_pooled_text_embeddings", TextEmbeddings(in_text_dim2, td))
                setattr(self, f"{prefix}_text_rope",              RoPE1D(hd))
                setattr(self, f"{prefix}_text_transformer_blocks", nn.ModuleList([
                    TransformerEncoderBlock(md, td, fd, hd, attention_engine, text_token_padding)
                    for _ in range(num_text_blocks)
                ]))

            self.visual_transformer_blocks = nn.ModuleList([
                FusedTransformerDecoderBlock(
                    model_dim, time_dim, ff_dim, head_dim,
                    model_dim_a, time_dim_a, ff_dim_a, head_dim_a,
                    attention_engine, text_token_padding,
                    ca_rope=ca_rope, cross_gates=cross_gates, fix_modulation=fix_modulation,
                )
                for _ in range(num_visual_blocks)
            ])

    # ------------------------------------------------------------------
    # Stage helpers (shared by DiffusionTransformer3D.forward / MagCache)
    # ------------------------------------------------------------------

    def clear_text_proj_cache(self) -> None:
        """Drop cached text/pooled projections (call once per generation)."""
        self._text_proj_cache.clear()

    def _project_pooled(self, prefix: str | None, pooled: Tensor) -> Tensor:
        key = ("pe", prefix, pooled.data_ptr(), tuple(pooled.shape))
        hit = self._text_proj_cache.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        if prefix is None:
            pe = self.pooled_text_embeddings(pooled)
        else:
            pe = getattr(self, f"{prefix}_pooled_text_embeddings")(pooled)
        self._text_proj_cache[key] = pe
        return pe

    def _project_text_tokens(
        self, prefix: str | None, text_embed: Tensor, pooled: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Time-independent token + pooled Linears (cached within a generation)."""
        key = (
            "te",
            prefix,
            text_embed.data_ptr(),
            pooled.data_ptr(),
            tuple(text_embed.shape),
            tuple(pooled.shape),
        )
        hit = self._text_proj_cache.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        if prefix is None:
            te = self.text_embeddings(text_embed)
        else:
            te = getattr(self, f"{prefix}_text_embeddings")(text_embed)
        pe = self._project_pooled(prefix, pooled)
        self._text_proj_cache[key] = (te, pe)
        return te, pe

    def _time_embed(self, prefix: str | None, time: Tensor, pooled_proj: Tensor) -> Tensor:
        if prefix is None:
            return self.time_embeddings(time) + pooled_proj
        return getattr(self, f"{prefix}_time_embeddings")(time) + pooled_proj

    @staticmethod
    def _normalize_attn_mask(attn_mask: Tensor | None) -> Tensor | None:
        """Normalize HF/K5 key-padding masks for ``[B,H,Q,K]`` attention."""
        if attn_mask is None:
            return None
        if attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dim() == 2:
            return attn_mask[:, None, None, :]
        return attn_mask

    def _run_text_blocks(
        self,
        prefix: str | None,
        te: Tensor,
        tm: Tensor,
        text_rope: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        blocks = self.text_blocks if prefix is None else getattr(self, f"{prefix}_text_transformer_blocks")
        for blk in blocks:
            te = blk(te, tm, text_rope, attn_mask)
        return te

    def _encode_text(
        self,
        prefix: str,
        text_embed: Tensor,
        pooled: Tensor,
        time: Tensor,
        text_rope: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        te, pe = self._project_text_tokens(prefix, text_embed, pooled)
        tm = self._time_embed(prefix, time, pe)
        te = self._run_text_blocks(prefix, te, tm, text_rope, attn_mask)
        return te, tm

    def _encode_t2v(
        self,
        text_embed: Tensor,
        pooled: Tensor,
        time: Tensor,
        text_rope: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        te, pe = self._project_text_tokens(None, text_embed, pooled)
        tm = self._time_embed(None, time, pe)
        te = self._run_text_blocks(None, te, tm, text_rope, attn_mask)
        return te, tm

    def _time_only(self, prefix: str | None, pooled: Tensor, time: Tensor) -> Tensor:
        """Pooled (cached) + time — OutLayer path when MagCache skips text+visual blocks."""
        pe = self._project_pooled(prefix, pooled)
        return self._time_embed(prefix, time, pe)

    def _embed_visual(
        self,
        x_video: Tensor,
        visual_rope: Tensor,
        *,
        visual_token_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, tuple, Tensor]:
        if x_video.ndim == 4:
            x_video = x_video.unsqueeze(0)
        vis_embed = self.visual_embeddings(x_video)
        if hasattr(self, "visual_token_type_embeddings") and visual_token_type_ids is not None:
            if visual_token_type_ids.ndim == 1:
                visual_token_type_ids = visual_token_type_ids.unsqueeze(0)
            if visual_token_type_ids.shape[:2] != vis_embed.shape[:2]:
                raise ValueError(
                    "visual_token_type_ids shape must match visual latent batch and frames: "
                    f"type_ids={tuple(visual_token_type_ids.shape)}, "
                    f"visual={tuple(vis_embed.shape)}"
                )
            vis_embed = vis_embed + self.visual_token_type_embeddings(
                visual_token_type_ids.to(device=vis_embed.device)
            )[:, :, None, None, :]
        vis_shape = vis_embed.shape[-4:-1]
        vis_rope = visual_rope.flatten(0, 2)
        vis_embed = vis_embed.flatten(1, 3)
        return vis_embed, vis_shape, vis_rope

    def _embed_audio(self, x_audio: Tensor, audio_rope: Tensor) -> tuple[Tensor, Tensor]:
        if x_audio.ndim == 2:
            x_audio = x_audio.unsqueeze(0)
        aud_embed = self.audio_embeddings(x_audio)
        return aud_embed, audio_rope

    def _run_visual_blocks_single(
        self,
        vis_embed: Tensor | None,
        aud_embed: Tensor | None,
        te: Tensor,
        tm: Tensor,
        vis_rope: Tensor | None,
        aud_rope: Tensor | None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        for blk in self.visual_transformer_blocks:
            if self.is_multimodal:
                if vis_embed is not None and aud_embed is None:
                    vis_embed, _ = blk(
                        vis_embed, None, te, te, (tm, tm), vis_rope, None, attn_mask,
                    )
                elif aud_embed is not None and vis_embed is None:
                    _, aud_embed = blk(
                        None, aud_embed, te, te, (tm, tm), None, aud_rope, attn_mask,
                    )
                else:
                    raise RuntimeError("single-modality fused path expects exactly one of video/audio")
            else:
                if vis_embed is not None:
                    vis_embed = blk(vis_embed, te, tm, vis_rope, attn_mask)
                else:
                    aud_embed = blk(aud_embed, te, tm, aud_rope, attn_mask)
        return vis_embed, aud_embed

    def _run_visual_blocks_fused(
        self,
        vis_embed: Tensor,
        aud_embed: Tensor,
        video_te: Tensor,
        audio_te: Tensor,
        video_tm: Tensor,
        audio_tm: Tensor,
        vis_rope: Tensor,
        aud_rope: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        for blk in self.visual_transformer_blocks:
            vis_embed, aud_embed = blk(
                vis_embed, aud_embed,
                video_te, audio_te,
                (video_tm, audio_tm),
                vis_rope, aud_rope,
                attn_mask,
            )
        return vis_embed, aud_embed

    def _project_video(self, vis_embed: Tensor, vis_shape: tuple, tm: Tensor) -> Tensor:
        vis_embed = vis_embed.reshape(-1, *vis_shape, vis_embed.shape[-1])
        return self.out_layer(vis_embed, tm)

    def _project_audio(self, aud_embed: Tensor, tm: Tensor) -> Tensor:
        return self.audio_out_layer(aud_embed, tm)

    def _project_fused(
        self,
        vis_embed: Tensor,
        aud_embed: Tensor,
        vis_shape: tuple,
        video_tm: Tensor,
        audio_tm: Tensor,
    ) -> tuple[Tensor, Tensor]:
        vis_embed = vis_embed.reshape(-1, *vis_shape, vis_embed.shape[-1])
        video_vel = self.out_layer(vis_embed, video_tm)
        audio_vel = self.audio_out_layer(aud_embed, audio_tm)
        return video_vel, audio_vel

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_video: Tensor | None,
        x_audio: Tensor | None,
        text_embed: Tensor | list[Tensor],
        pooled_text_embed: Tensor | list[Tensor],
        time: Tensor | list[Tensor],
        visual_rope: Tensor | None,
        audio_rope: Tensor | None,
        text_rope: Tensor | list[Tensor],
        attention_mask: Tensor | None = None,
        visual_token_type_ids: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """RoPE tensors are precomputed outside (pipeline / export); not from positions."""
        both = x_video is not None and x_audio is not None and self.is_multimodal
        attn_mask = self._normalize_attn_mask(attention_mask)

        if not both:
            te_in = text_embed[0] if isinstance(text_embed, list) else text_embed
            pe_in = pooled_text_embed[0] if isinstance(pooled_text_embed, list) else pooled_text_embed
            rope_in = text_rope[0] if isinstance(text_rope, list) else text_rope
            t_in = time[0] if isinstance(time, list) else time

            if self.is_multimodal:
                prefix = "audio" if x_audio is not None else "video"
                # Multimodal single-modality path: list text_rope is [video, audio].
                if isinstance(text_rope, list):
                    rope_in = text_rope[1] if prefix == "audio" else text_rope[0]
                te, tm = self._encode_text(prefix, te_in, pe_in, t_in, rope_in, attn_mask)
            else:
                te, tm = self._encode_t2v(te_in, pe_in, t_in, rope_in, attn_mask)

            if x_video is not None:
                vis_embed, vis_shape, vis_rope = self._embed_visual(
                    x_video, visual_rope,
                    visual_token_type_ids=visual_token_type_ids,
                )
                vis_embed, _ = self._run_visual_blocks_single(
                    vis_embed, None, te, tm, vis_rope, None, attn_mask,
                )
                return self._project_video(vis_embed, vis_shape, tm)

            aud_embed, aud_rope = self._embed_audio(x_audio, audio_rope)
            _, aud_embed = self._run_visual_blocks_single(
                None, aud_embed, te, tm, None, aud_rope, attn_mask,
            )
            return self._project_audio(aud_embed, tm)

        te_v, pe_v = (
            (text_embed[0], pooled_text_embed[0])
            if isinstance(text_embed, list)
            else (text_embed, pooled_text_embed)
        )
        te_a, pe_a = (
            (text_embed[1], pooled_text_embed[1])
            if isinstance(text_embed, list)
            else (text_embed, pooled_text_embed)
        )
        if isinstance(text_rope, list):
            rope_v, rope_a = text_rope[0], text_rope[1]
        else:
            rope_v = rope_a = text_rope
        t_v, t_a = (time[0], time[1]) if isinstance(time, list) else (time, time)

        video_te, video_tm = self._encode_text("video", te_v, pe_v, t_v, rope_v, attn_mask)
        audio_te, audio_tm = self._encode_text("audio", te_a, pe_a, t_a, rope_a, attn_mask)

        vis_embed, vis_shape, vis_rope = self._embed_visual(
            x_video, visual_rope,
            visual_token_type_ids=visual_token_type_ids,
        )
        aud_embed, aud_rope = self._embed_audio(x_audio, audio_rope)

        vis_embed, aud_embed = self._run_visual_blocks_fused(
            vis_embed, aud_embed,
            video_te, audio_te, video_tm, audio_tm,
            vis_rope, aud_rope, attn_mask,
        )
        return self._project_fused(
            vis_embed, aud_embed, vis_shape, video_tm, audio_tm,
        )

    def reset_parameters(self) -> None:
        for m in self.modules():
            if m is not self and hasattr(m, "reset_parameters"):
                m.reset_parameters()
        if hasattr(self, "visual_token_type_embeddings"):
            nn.init.zeros_(self.visual_token_type_embeddings.weight)
