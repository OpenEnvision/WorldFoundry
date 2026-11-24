# Inference-only DB-CogACT source retained in-tree.
# SPDX-License-Identifier: Apache-2.0
"""This is basically a copy from perception_models/core/vision_encoder/pe.py"""

from collections import OrderedDict
from collections.abc import Callable
from functools import partial

# SPDX-License-Identifier: Apache-2.0
from math import pi
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F

from worldfoundry.core.attention.rotary.rope import apply_rotary_embedding
from worldfoundry.core.nn.blocks.layers import LayerScale


class Rope2D(nn.Module):
    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        freqs_for: Literal["lang", "pixel", "constant"] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        theta_rescale_factor=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        self.freqs_for = freqs_for
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self._freqs_runtime_cache = OrderedDict()
        self._max_runtime_cache_entries = 4

    def _compute_inv_freq(self, base: int | float, dim: int) -> torch.Tensor:
        if self.freqs_for == "lang":
            freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif self.freqs_for == "pixel":
            freqs = torch.linspace(1.0, self.max_freq / 2, dim // 2) * pi
        elif self.freqs_for == "constant":
            freqs = torch.ones(self.num_freqs).float()
        else:
            raise ValueError(f"Unsupported freqs_for value: {self.freqs_for}")
        return freqs

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor):
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def _compute_2d_freqs(
        self,
        grid_height: int,
        grid_width: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        grid_h_range = torch.arange(grid_height, dtype=torch.float, device=device)
        grid_w_range = torch.arange(grid_width, dtype=torch.float, device=device)
        if self.use_cls_token:
            grid_h_range += 1
            grid_w_range += 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        if device is not None:
            inv_freq = inv_freq.to(device=device)
        freqs_h = self._compute_freqs(grid_h_range, inv_freq)[:, None].expand(grid_height, grid_width, -1)
        freqs_w = self._compute_freqs(grid_w_range, inv_freq)[None, :].expand(grid_height, grid_width, -1)
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(grid_height * grid_width, -1)
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1], device=device), freqs], dim=0)
        freqs = freqs[None, None, ...]
        return freqs

    def _get_runtime_freqs(self, grid_height: int, grid_width: int, device: torch.device) -> torch.Tensor:
        cache_key = (grid_height, grid_width, device.type, device.index)
        freqs = self._freqs_runtime_cache.get(cache_key)
        if freqs is None or freqs.device != device:
            with torch.no_grad():
                freqs = self._compute_2d_freqs(grid_height, grid_width, device=device)
            self._freqs_runtime_cache[cache_key] = freqs
            self._freqs_runtime_cache.move_to_end(cache_key)
            while len(self._freqs_runtime_cache) > self._max_runtime_cache_entries:
                self._freqs_runtime_cache.popitem(last=False)
        else:
            self._freqs_runtime_cache.move_to_end(cache_key)
        return freqs

    def forward(self, q: torch.Tensor, k: torch.Tensor, grid_hw: tuple[int, int]):
        # Cache frequencies outside module state so FSDP2 wrap/prepare does not touch them,
        # while still avoiding per-forward recomputation on the common same-device/same-grid path.
        freqs = self._get_runtime_freqs(grid_hw[0], grid_hw[1], q.device)
        cos, sin = freqs.cos(), freqs.sin()
        q = apply_rotary_embedding(q, cos, sin, interleaved=True)
        k = apply_rotary_embedding(k, cos, sin, interleaved=True)
        return q, k


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x


class SelfAttention(nn.Module):
    r"""
    Implements sequence packed attention and RoPe
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # To make this compatible with nn.MultiHeadAttention
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = Rope2D(
            dim=embed_dim // num_heads,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
        )
        self.scale = self.head_dim ** (-0.5)

    def forward(
        self,
        x,
        grid_hw: tuple[int, int],
    ):
        _, _, embed_dim = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)

        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = proj.unflatten(-1, (3, embed_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = proj[0], proj[1], proj[2]

        # Use "q_" so that we don't accidentally quit in pdb :)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        q, k = self.rope(q, k, grid_hw=grid_hw)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        attn = rearrange(attn, "b h s d -> b s (h d)")

        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        max_grid_height: int,
        max_grid_width: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        use_cls_token: bool = False,
    ):
        super().__init__()

        self.attn = SelfAttention(
            d_model,
            n_head,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
        )

        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: tuple[int, int],
    ):
        x = x + self.ls_1(self.attn(self.ln_1(x), grid_hw=grid_hw))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        max_grid_height: int,
        max_grid_width: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        use_cls_token: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    max_grid_height=max_grid_height,
                    max_grid_width=max_grid_width,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    use_cls_token=use_cls_token,
                )
                for _ in range(layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: tuple[int, int],
        layer_idx: int = -1,
    ):
        stop_idx = (self.layers + layer_idx) % self.layers

        for i, r in enumerate(self.resblocks):
            x = r(x, grid_hw=grid_hw)
            if i == stop_idx:
                break

        return x


class PerceptionEncoderWithDownsample(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),  # noqa: B008 - Default evaluated once at definition, as upstream.
        use_ln_pre: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float | None = None,
        drop_path: float = 0.0,
        image_size: int = 448,  # Pretrain image size only; you can pass in any image size
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        use_cls_token: bool = False,
        output_dim: int | None = 1280,
        attn_pooler_heads: int = 8,
        pool_type: Literal["attn", "tok", "avg", "none"] = "attn",
        layer_types: list[str] = [],  # noqa: B006 - Keeps the upstream default object and signature.
        sliding_window_size: int = -1,
        # config,
        # act_layer: Callable = nn.GELU,
        # norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
    ):
        super().__init__()
        # assert config.pool_type in ("attn", "tok", "avg", "none")
        self.pool_type = pool_type
        self.patch_size = patch_size

        self.output_dim = output_dim or width
        self.heads = heads
        self.width = width
        self.layers = layers

        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope2d = use_rope2d
        if not self.use_rope2d:
            raise ValueError("use_rope2d must be True")
        self.image_size = image_size

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(self.width) if use_ln_post else nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            max_grid_height=self.image_size // self.patch_size,
            max_grid_width=self.image_size // self.patch_size,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_cls_token=self.use_cls_token,
        )

        if pool_type == "attn":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=attn_pooler_heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None

        self.vit_downsampler1 = nn.Conv2d(
            width,
            width * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.vit_downsampler2 = nn.Conv2d(
            width * 2,
            width * 4,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        if self.use_cls_token:
            self.class_embedding = nn.Parameter((self.width**-0.5) * torch.randn(self.width))

        if self.use_abs_posemb:
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                (self.width**-0.5)
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2,
                    self.width,
                )
            )

    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        """Interpolates the absolute position embedding if necessary."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width).contiguous()

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]

    def pool(self, x: torch.Tensor):
        if self.pool_type == "tok":
            return x[:, 0]
        elif self.pool_type == "avg":
            return x.mean(dim=1)
        elif self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        elif self.pool_type == "none":
            return x
        else:
            raise NotImplementedError

    def forward_features(
        self,
        x: torch.Tensor,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False,
    ):
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        x = self.conv1(x).permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                dim=1,
            )

        if self.use_abs_posemb:
            posemb = self.sample_abs_posemb(grid_h, grid_w)
            x = x + posemb

        if self.transformer.resblocks[0].attn.in_proj_weight.dtype == torch.bfloat16:
            x = x.to(torch.bfloat16)

        x = self.ln_pre(x)
        x = self.transformer(x, grid_hw=(grid_h, grid_w), layer_idx=layer_idx)

        if norm:
            x = self.ln_post(x)

        if strip_cls_token and self.use_cls_token:
            x = x[:, 1:, :]

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x, norm=True, strip_cls_token=True)
        x = self.pool(x)

        B, P, C = x.shape
        T = int(P**0.5)
        # T_real = T + 1
        # B, P, C -> B, C, P -> B, C, T, T
        x = x.transpose(2, 1).contiguous()
        x = x.view(B, C, T, T)

        x = self.vit_downsampler1(x)
        x = self.vit_downsampler2(x)

        B, C, T, T = x.shape
        return x.view(B, -1, T * T).transpose(1, 2)
