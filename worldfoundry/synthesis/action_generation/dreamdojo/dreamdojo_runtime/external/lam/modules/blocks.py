import math

import torch
import torch.nn as nn
from einops import rearrange
from external.lam.modules.embeddings import RotaryEmbedding
from torch import Tensor


def patchify(videos: Tensor, size: int) -> Tensor:
    B, T, H, W, C = videos.shape
    videos = videos[:, :, :H - (H % size), :W - (W % size), :]
    x = rearrange(videos, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size)
    return x


def unpatchify(patches: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = rearrange(patches, "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c", hp=size, wp=size, hn=hn)
    return x[:, :, :h_out, :w_out]


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_enc = pe

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]].cuda()


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, rot_emb: bool = False) -> None:
        super(SelfAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

        self.rot_emb = rot_emb
        if rot_emb:
            self.rotary_embedding = RotaryEmbedding(dim=inner_dim)

    def scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False
    ) -> Tensor:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query)
        if is_causal:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(attn_bias)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        if self.rot_emb:
            q = self.rotary_embedding.rotate_queries_or_keys(q, self.rotary_embedding.freqs)
            k = self.rotary_embedding.rotate_queries_or_keys(k, self.rotary_embedding.freqs)
            q, k = map(lambda t: t.contiguous(), (q, k))
        out = self.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        t_len = x.shape[1]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTemporalBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioTemporalBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.temporal_attn = SelfAttention(model_dim, num_heads, dropout=dropout, rot_emb=True)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, causal_temporal: bool = False) -> Tensor:
        t_len, s_len = x.shape[1:3]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Temporal attention
        x = rearrange(x, "b t s e -> (b s) t e")
        x_ = self.norm2(x)
        if causal_temporal:
            x_ = self.temporal_attn(x_, is_causal=True)
        else:
            x_ = self.temporal_attn(x_)
        x = x + x_
        x = rearrange(x, "(b s) t e -> b t s e", s=s_len)

        # Feedforward
        x_ = self.norm3(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0
    ) -> None:
        super(SpatioTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.out(x)
        return x  # (B, T, E)


class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_temporal: bool = False
    ) -> None:
        super(SpatioTemporalTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    model_dim,
                    num_heads,
                    dropout
                ) for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)
        self.causal_temporal = causal_temporal

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x, self.causal_temporal)
        x = self.out(x)
        return x  # (B, T, E)
