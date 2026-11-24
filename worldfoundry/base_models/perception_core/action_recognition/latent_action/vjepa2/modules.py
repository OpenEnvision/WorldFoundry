import torch
import torch.nn.functional as F
from torch import nn


def _apply_rope(value, position):
    dimension = value.shape[-1]
    frequency = torch.arange(dimension // 2, dtype=value.dtype, device=value.device)
    frequency = 1.0 / 10000 ** (frequency / (dimension / 2.0))
    angles = torch.einsum("..., f -> ... f", position, frequency)
    sine = angles.sin().repeat_interleave(2, dim=-1)
    cosine = angles.cos().repeat_interleave(2, dim=-1)
    first, second = value.unflatten(-1, (-1, 2)).unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return value * cosine + rotated * sine


class MLP(nn.Module):
    def __init__(self, dimension, hidden_dimension, activation=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(dimension, hidden_dimension)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dimension, dimension)

    def forward(self, value):
        return self.fc2(self.act(self.fc1(value)))


class Attention(nn.Module):
    def __init__(self, dimension, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dimension, dimension * 3, bias=qkv_bias)
        self.proj = nn.Linear(dimension, dimension)

    def forward(self, value):
        batch_size, token_count, dimension = value.shape
        query, key, value = (
            self.qkv(value)
            .reshape(
                batch_size,
                token_count,
                3,
                self.num_heads,
                dimension // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        value = F.scaled_dot_product_attention(query, key, value)
        return self.proj(value.transpose(1, 2).reshape(batch_size, token_count, dimension))


class RoPEAttention(Attention):
    def __init__(self, dimension, num_heads, qkv_bias=True):
        super().__init__(dimension, num_heads, qkv_bias)
        head_dimension = dimension // num_heads
        axis_dimension = 2 * ((head_dimension // 3) // 2)
        self.axis_dimensions = (axis_dimension, axis_dimension, axis_dimension)

    def forward(self, value, frames, height, width):
        batch_size, token_count, dimension = value.shape
        query, key, value = (
            self.qkv(value)
            .reshape(
                batch_size,
                token_count,
                3,
                self.num_heads,
                dimension // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        indices = torch.arange(frames * height * width, device=value.device)
        depth = indices // (height * width)
        within_frame = indices % (height * width)
        row = within_frame // width
        column = within_frame % width

        offset = 0
        query_parts = []
        key_parts = []
        for size, position in zip(self.axis_dimensions, (depth, row, column), strict=False):
            query_parts.append(_apply_rope(query[..., offset : offset + size], position))
            key_parts.append(_apply_rope(key[..., offset : offset + size], position))
            offset += size
        if offset < query.shape[-1]:
            query_parts.append(query[..., offset:])
            key_parts.append(key[..., offset:])
        query = torch.cat(query_parts, dim=-1)
        key = torch.cat(key_parts, dim=-1)
        value = F.scaled_dot_product_attention(query, key, value)
        return self.proj(value.transpose(1, 2).reshape(batch_size, token_count, dimension))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        use_rope=False,
        act_layer=nn.GELU,
        **_,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        attention = RoPEAttention if use_rope else Attention
        self.attn = attention(dim, num_heads, qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer)

    def forward(self, value, frames=None, height=None, width=None, **_):
        normalized = self.norm1(value)
        if isinstance(self.attn, RoPEAttention):
            normalized = self.attn(normalized, frames, height, width)
        else:
            normalized = self.attn(normalized)
        value = value + normalized
        return value + self.mlp(self.norm2(value))


class CrossAttention(nn.Module):
    def __init__(self, dimension, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dimension, dimension, bias=qkv_bias)
        self.kv = nn.Linear(dimension, dimension * 2, bias=qkv_bias)

    def forward(self, query, value):
        batch_size, query_count, dimension = query.shape
        query = (
            self.q(query)
            .reshape(
                batch_size,
                query_count,
                self.num_heads,
                dimension // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )
        key, value = (
            self.kv(value)
            .reshape(
                batch_size,
                value.shape[1],
                2,
                self.num_heads,
                dimension // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        result = F.scaled_dot_product_attention(query, key, value)
        return result.transpose(1, 2).reshape(batch_size, query_count, dimension)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads, qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer)

    def forward(self, query, value):
        query = query + self.xattn(query, self.norm1(value))
        return query + self.mlp(self.norm2(query))
