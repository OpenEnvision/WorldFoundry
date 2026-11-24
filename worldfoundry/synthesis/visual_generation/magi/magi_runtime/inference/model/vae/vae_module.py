# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from timm.models.layers import to_2tuple, trunc_normal_

from worldfoundry.core.nn.blocks.layers import Mlp as Mlp

###################################################
#     modified 3D rotary embedding from timm
###################################################


def ndgrid(*tensors) -> tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def freq_bands(
    num_bands: int, temperature: float = 10000.0, step: int = 2, device: torch.device | None = None
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exp)
    return bands


def pixel_freq_bands(
    num_bands: int, max_freq: float = 224.0, linear_bands: bool = True, device: torch.device | None = None
):
    if linear_bands:
        bands = torch.linspace(1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device)
    else:
        bands = 2 ** torch.linspace(0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device)
    return bands * torch.pi


def build_fourier_pos_embed(
    feat_shape: list[int],
    bands: torch.Tensor | None = None,
    num_bands: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    include_grid: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: list[int] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    center_imgidx=True,
) -> list[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(num_bands, float(max_res), linear_bands=linear_bands, device=device)
        else:
            bands = freq_bands(num_bands, temperature=temperature, step=1, device=device)
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=torch.float32) for s in feat_shape]
    else:
        if center_imgidx:
            t = [
                torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) - (s - 1) / 2
                if len(feat_shape) == 2 or i != 0
                else torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
                for i, s in enumerate(feat_shape)
            ]
        else:
            t = [torch.arange(s, device=device, dtype=torch.int64).to(torch.float32) for s in feat_shape]

    if ref_feat_shape is not None:
        assert len(feat_shape) == len(ref_feat_shape), "shape must be in same dimension"
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape, strict=False)]

    grid = torch.stack(ndgrid(t), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands
    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    # import ipdb; ipdb.set_trace()
    return x * cos_emb + rot(x) * sin_emb


def build_rotary_pos_embed(
    feat_shape: list[int],
    bands: torch.Tensor | None = None,
    dim: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: list[int] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    center_imgidx=True,
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // (len(feat_shape) * 2),
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
        center_imgidx=center_imgidx,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    return sin_emb, cos_emb


###################################################
# Mlp
###################################################
###################################################
# ManualLayerNorm
###################################################
class ManualLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)

        x_normalized = (x - mean) / (std + self.eps)

        return x_normalized


###################################################
# Attention
###################################################
@lru_cache(maxsize=50)
def cache_rotary_emb(feat_shape, device="cuda", dim=64, dtype=torch.bfloat16, max_res=512, ref_feat_shape=(4, 16, 16)):
    return build_rotary_pos_embed(
        feat_shape=feat_shape,
        dim=dim,
        max_res=max_res,
        in_pixels=False,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ln_in_attn=False,
        use_rope=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_rate = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if ln_in_attn:
            self.qkv_norm = ManualLayerNorm(head_dim, elementwise_affine=False)
        else:
            self.qkv_norm = nn.Identity()
        self.use_rope = use_rope

    def forward(self, x, feat_shape=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = self.qkv_norm(qkv)
        q, k, v = qkv.chunk(3, dim=2)
        if self.use_rope:
            assert feat_shape is not None
            q, k, v = qkv.chunk(3, dim=2)
            rope_emb = cache_rotary_emb(feat_shape=feat_shape, dim=C // self.num_heads, device=x.device, dtype=x.dtype)
            sin_emb = rope_emb[0].unsqueeze(0).unsqueeze(2)
            cos_emb = rope_emb[1].unsqueeze(0).unsqueeze(2)
            print(q.shape, sin_emb.shape)
            q[:, 1:, :] = apply_rot_embed(q[:, 1:, :], sin_emb, cos_emb).bfloat16()
            k[:, 1:, :] = apply_rot_embed(k[:, 1:, :], sin_emb, cos_emb).bfloat16()
            x = flash_attn_func(q, k, v, dropout_p=self.attn_drop_rate)
        else:
            x = flash_attn_qkvpacked_func(qkv=qkv.bfloat16(), dropout_p=self.attn_drop_rate)
        # x = v
        x = x.reshape(B, N, C)
        # import ipdb; ipdb.set_trace()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


###################################################
# Block
###################################################
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ln_in_attn=False,
        use_rope=False,
    ):
        super().__init__()
        if not ln_in_attn:
            self.norm1 = norm_layer(dim)
        else:
            self.norm1 = nn.Identity()
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            ln_in_attn=ln_in_attn,
            use_rope=use_rope,
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, feat_shape=None):
        x = x + self.drop_path(self.attn(self.norm1(x), feat_shape=feat_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


###################################################
# PatchEmbed
###################################################
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, video_size=224, video_length=16, patch_size=16, patch_length=1, in_chans=3, embed_dim=768):
        super().__init__()
        video_size = to_2tuple(video_size)
        patch_size = to_2tuple(patch_size)

        num_patches = (
            (video_length // patch_length) * (video_size[1] // patch_size[1]) * (video_size[0] // patch_size[0])
        )

        self.video_size = video_size
        self.patch_size = patch_size

        self.video_length = video_length
        self.patch_length = patch_length

        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(patch_length, patch_size[0], patch_size[1]),
            stride=(patch_length, patch_size[0], patch_size[1]),
        )

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W), where
                B is the batch size, C is the number of channels, T is the
                number of frames, H is the height, and W is the width.

        Returns:
            torch.Tensor: Output tensor of shape (B, L, C'), where B is the
                batch size, L is the number of tokens, and C' is the number
                of output channels after flattening and transposing.
        """
        B, C, T, H, W = x.shape

        x = self.proj(x)
        return x


###################################################
# ViTEncoder
###################################################
def resize_pos_embed(posemb, src_shape, target_shape):
    posemb = posemb.reshape(1, src_shape[0], src_shape[1], src_shape[2], -1)
    posemb = posemb.permute(0, 4, 1, 2, 3)
    posemb = nn.functional.interpolate(posemb, size=target_shape, mode="trilinear", align_corners=False)
    posemb = posemb.permute(0, 2, 3, 4, 1)
    posemb = posemb.reshape(1, target_shape[0] * target_shape[1] * target_shape[2], -1)
    return posemb


class ViTEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        video_size=256,
        video_length=16,
        patch_size=8,
        patch_length=4,
        in_chans=3,
        z_chans=4,
        double_z=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        with_cls_token=True,
        norm_code=False,
        ln_in_attn=False,
        conv_last_layer=False,
        use_rope=False,
        use_final_proj=False,
    ):
        super().__init__()

        conv_last_layer = False  # noqa: F841 - Upstream assignment kept.  # duplicate argument

        # self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.latent_size = video_size // patch_size
        self.latent_length = video_length // patch_length

        self.patch_embed = PatchEmbed(
            video_size=video_size,
            video_length=video_length,
            patch_size=patch_size,
            patch_length=patch_length,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token_nums = 1
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token_nums = 0
            self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.cls_token_nums, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    ln_in_attn=ln_in_attn,
                    use_rope=use_rope,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.norm_code = norm_code

        self.out_channels = z_chans * 2 if double_z else z_chans
        self.last_layer = nn.Linear(embed_dim, self.out_channels, bias=True)

        trunc_normal_(self.pos_embed, std=0.02)

        if self.with_cls_token:
            trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, x):
        B = x.shape[0]
        # B C T H W -> B C T/pT H/pH W//pW
        x = self.patch_embed(x)
        latentT, latentH, latentW = x.shape[2], x.shape[3], x.shape[4]
        # B C T/pT H/pH W//pW -> B (T/pT H/pH W//pW) C
        x = x.flatten(2).transpose(1, 2)

        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if latentT != self.latent_length or latentH != self.latent_size or latentW != self.latent_size:
            pos_embed = resize_pos_embed(
                self.pos_embed[:, 1:, :],
                src_shape=(self.latent_length, self.latent_size, self.latent_size),
                target_shape=(latentT, latentH, latentW),
            )
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :], pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for _idx, blk in enumerate(self.blocks):
            x = blk(x, feat_shape=(latentT, latentH, latentW))

        x = self.norm(x)
        x = self.last_layer(x)

        if self.with_cls_token:
            x = x[:, 1:]  # remove cls_token

        # B L C - > B , lT, lH, lW, zC
        x = x.reshape(B, latentT, latentH, latentW, self.out_channels)

        # B , lT, lH, lW, zC -> B, zC, lT, lH, lW
        x = x.permute(0, 4, 1, 2, 3)
        if self.norm_code:
            prev_dtype = x.dtype
            x = x.float()
            x = x / torch.norm(x, dim=1, keepdim=True)
            x = x.to(prev_dtype)
        return x

    def freeze_pretrain(self):
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False


###################################################
# ViTDecoder
###################################################
class ViTDecoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        video_size=256,
        video_length=16,
        patch_size=8,
        patch_length=4,
        in_chans=3,
        z_chans=4,
        double_z=True,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        with_cls_token=True,
        norm_code=False,
        ln_in_attn=False,
        conv_last_layer=False,
        use_rope=False,
        use_final_proj=False,
    ):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.latent_size = video_size // patch_size
        self.latent_length = video_length // patch_length
        self.patch_size = patch_size
        self.patch_length = patch_length

        self.proj_in = nn.Linear(z_chans, embed_dim)

        num_patches = self.latent_size * self.latent_size * self.latent_length

        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token_nums = 1
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token_nums = 0
            self.cls_token = None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.cls_token_nums, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    ln_in_attn=ln_in_attn,
                    use_rope=use_rope,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        assert conv_last_layer == True, "Only support conv_last_layer=True"  # noqa: E712 - Keeps `==` for operands that may compare elementwise.

        self.unpatch_channels = embed_dim // (patch_size * patch_size * patch_length)
        self.final_proj = nn.Identity()
        self.final_norm = nn.Identity()

        self.use_final_proj = use_final_proj
        if self.use_final_proj:
            self.unpatch_channels = 4
            self.final_proj = nn.Linear(
                embed_dim, self.unpatch_channels * (patch_size * patch_size * patch_length), bias=True
            )
            self.final_norm = norm_layer(self.unpatch_channels * (patch_size * patch_size * patch_length))

        self.last_layer = nn.Conv3d(
            in_channels=self.unpatch_channels, out_channels=3, kernel_size=3, stride=1, padding=1
        )

        trunc_normal_(self.pos_embed, std=0.02)

        if self.with_cls_token:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, x):
        B, C, latentT, latentH, latentW = x.shape  # x: (B, C, latentT, latentH, latenW)
        x = x.permute(0, 2, 3, 4, 1)  # x: (B, latentT, latentH, latenW, C)

        x = x.reshape(B, -1, C)

        x = self.proj_in(x)

        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if latentT != self.latent_length or latentH != self.latent_size or latentW != self.latent_size:
            pos_embed = resize_pos_embed(
                self.pos_embed[:, 1:, :],
                src_shape=(self.latent_length, self.latent_size, self.latent_size),
                target_shape=(latentT, latentH, latentW),
            )
            pos_embed = torch.cat((self.pos_embed[:, 0:1, :], pos_embed), dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for _idx, blk in enumerate(self.blocks):
            x = blk(x, feat_shape=(latentT, latentH, latentW))

        x = self.norm(x)

        if self.with_cls_token:
            x = x[:, 1:]  # remove cls_token
        # B L C - > B, lT, lH, lW, pT, pH, pW, C
        if self.use_final_proj:
            x = self.final_proj(x)
            x = self.final_norm(x)
        x = x.reshape(
            B, latentT, latentH, latentW, self.patch_length, self.patch_size, self.patch_size, self.unpatch_channels
        )
        x = rearrange(x, "B lT lH lW pT pH pW C -> B C (lT pT) (lH pH) (lW pW)", C=self.unpatch_channels)

        x = self.last_layer(x)
        return x


###################################################
# DiagonalGaussianDistribution
###################################################
class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):  # noqa: B006 - Keeps the upstream default object and signature.
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
