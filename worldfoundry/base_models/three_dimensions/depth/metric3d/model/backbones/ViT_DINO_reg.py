# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from collections.abc import Callable, Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.utils.checkpoint
from timm.layers import DropPath as DropPath
from torch import Tensor
from torch.nn.init import trunc_normal_

from worldfoundry.base_models.perception_core.general_perception.dinov2.layers.construction import (
    init_weights_vit_timm as init_weights_vit_timm,
)
from worldfoundry.base_models.perception_core.general_perception.dinov2.variants.vipe_dav2.layers.block import (
    drop_add_residual_stochastic_depth_list as drop_add_residual_stochastic_depth_list,
)
from worldfoundry.base_models.perception_core.general_perception.dinov2.variants.vipe_dav2.layers.block import (
    get_attn_bias_and_cat as get_attn_bias_and_cat,
)
from worldfoundry.base_models.three_dimensions.general_3d.vipe.ext.xformers import (
    memory_efficient_attention,
)
from worldfoundry.core.nn.blocks.layers import LayerScale as LayerScale
from worldfoundry.core.nn.blocks.layers import SwiGLUFFNFused as SwiGLUFFNFused
from worldfoundry.core.nn.blocks.module_utils import named_apply
from worldfoundry.core.nn.blocks.stochastic_depth import add_residual as add_residual
from worldfoundry.core.nn.blocks.stochastic_depth import (
    get_branges_scales as get_branges_scales,
)

logger = logging.getLogger(__name__)


# SSF finetuning originally by dongzelian
def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=0.02)
    nn.init.normal_(shift, std=0.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError("the input tensor shape does not match the shape of the scale factor.")


# LoRA finetuning originally by edwardjhu
class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    # def train(self, mode: bool = True):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w
    #     nn.Linear.train(self, mode)
    #     if mode:
    #         if self.merge_weights and self.merged:
    #             # Make sure that the weights are not merged
    #             if self.r > 0:
    #                 self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
    #             self.merged = False
    #     else:
    #         if self.merge_weights and not self.merged:
    #             # Merge the weights and mark it
    #             if self.r > 0:
    #                 self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
    #             self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
        tuning_mode: str | None = None,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if tuning_mode is not None:
            self.tuning_mode = tuning_mode
            if tuning_mode == "ssf":
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim)
            else:
                pass
                # raise NotImplementedError()
        else:
            self.tuning_mode = None

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if self.tuning_mode == "ssf":
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        tuning_mode: int | None = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

        if tuning_mode is not None:
            self.tuning_mode = tuning_mode
            if tuning_mode == "ssf":
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(hidden_features)
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)
            else:
                pass
                # raise NotImplementedError()
        else:
            self.tuning_mode = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        if self.tuning_mode == "ssf":
            x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.tuning_mode == "ssf":
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)

        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,
        drop: float = 0.0,
        bias: bool = True,
        tuning_mode: int | None = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        if tuning_mode is not None:
            self.tuning_mode = tuning_mode
            if tuning_mode == "ssf":
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(2 * hidden_features)
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(out_features)
            else:
                pass
                # raise NotImplementedError()
        else:
            self.tuning_mode = None

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        if self.tuning_mode == "ssf":
            x12 = ssf_ada(x12, self.ssf_scale_1, self.ssf_shift_1)

        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        out = self.w3(hidden)

        if self.tuning_mode == "ssf":
            out = ssf_ada(out, self.ssf_scale_2, self.ssf_scale_2)

        return out


fmha = None
XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: int = 0,
        tuning_mode: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        if tuning_mode == "lora":
            self.tuning_mode = tuning_mode
            self.qkv = LoRALinear(dim, dim * 3, bias=qkv_bias, r=8)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        if tuning_mode == "lora":
            self.tuning_mode = tuning_mode
            self.proj = LoRALinear(dim, dim, bias=proj_bias, r=8)
        else:
            self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if tuning_mode is not None:
            self.tuning_mode = tuning_mode
            if tuning_mode == "ssf":
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim * 3)
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)
            else:
                pass
                # raise NotImplementedError()
        else:
            self.tuning_mode = None

        # if not self.training:
        #
        # self.attn = ScaledDotProduct()
        # self.attn = MultiHeadDispatch(dim_model=EMB, residual_dropout=DROPOUT, num_heads=HEADS, attention=attn)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        if self.tuning_mode == "ssf":
            qkv = (
                ssf_ada(self.qkv(x), self.ssf_scale_1, self.ssf_shift_1)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        if attn_bias is not None:
            attn = attn + attn_bias[:, :, :N]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        if self.tuning_mode == "ssf":
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)

        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            # if True:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x, attn_bias)

        B, N, C = x.shape
        if self.tuning_mode == "ssf":
            qkv = ssf_ada(self.qkv(x), self.ssf_scale_1, self.ssf_shift_1).reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            )
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = torch.unbind(qkv, 2)
        if attn_bias is not None:
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias[:, :, :N])
        else:
            x = memory_efficient_attention(q, k, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        if self.tuning_mode == "ssf":
            x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)

        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        tuning_mode: int | None = None,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            tuning_mode=tuning_mode,
        )

        if tuning_mode is not None:
            self.tuning_mode = tuning_mode
            if tuning_mode == "ssf":
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(dim)
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(dim)
            else:
                pass
                # raise NotImplementedError()
        else:
            self.tuning_mode = None

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        def attn_residual_func(x: Tensor, attn_bias) -> Tensor:
            if self.tuning_mode == "ssf":
                return self.ls1(
                    self.attn(
                        ssf_ada(self.norm1(x), self.ssf_scale_1, self.ssf_shift_1),
                        attn_bias,
                    )
                )
            else:
                return self.ls1(self.attn(self.norm1(x), attn_bias))

        def ffn_residual_func(x: Tensor) -> Tensor:
            if self.tuning_mode == "ssf":
                return self.ls2(self.mlp(ssf_ada(self.norm2(x), self.ssf_scale_2, self.ssf_shift_2)))
            else:
                return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                attn_bias=attn_bias,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, attn_bias))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, attn_bias)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    attn_bias=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    residual = residual_func(x_subset, attn_bias)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: list[Tensor]) -> list[Tensor]:
        """
        x_list contains a list of tensors to nest together and run
        """
        assert isinstance(self.attn, MemEffAttention)

        if self.training and self.sample_drop_ratio > 0.0:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.attn(self.norm1(x), attn_bias=attn_bias)

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.mlp(self.norm2(x))

            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls1.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            x_list = drop_add_residual_stochastic_depth_list(
                x_list,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
                scaling_vector=(self.ls2.gamma if isinstance(self.ls1, LayerScale) else None),
            )
            return x_list
        else:

            def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls1(self.attn(self.norm1(x), attn_bias=attn_bias))

            def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
                return self.ls2(self.mlp(self.norm2(x)))

            attn_bias, x = get_attn_bias_and_cat(x_list)
            x = x + attn_residual_func(x, attn_bias=attn_bias)
            x = x + ffn_residual_func(x)
            return attn_bias.split(x)

    def forward(self, x_or_x_list, attn_bias=None):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list, attn_bias)
        elif isinstance(x_or_x_list, list):
            raise NotImplementedError("Nested tensor attention requires xFormers, which is disabled in ViPE")
        else:
            raise AssertionError


class BlockChunk(nn.ModuleList):
    def forward(self, x, others=None):
        for b in self:
            x = b(x) if others is None else b(x, others)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=1e-5,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        multi_output=False,
        tuning_mode=None,
        **kwargs,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        if tuning_mode is not None:
            self.tuning_mode = tuning_mode
            if tuning_mode == "ssf":
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim)
            else:
                pass
                # raise NotImplementedError()
        else:
            self.tuning_mode = None
        tuning_mode_list = [tuning_mode] * depth

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            tuning_mode=tuning_mode,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.multi_output = multi_output
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            # logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            # logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            # logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                tuning_mode=tuning_mode_list[i],
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list, strict=False)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list, strict=False):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        B, C, H, W = x.size()
        pad_h = self.patch_size - H % self.patch_size
        pad_w = self.patch_size - W % self.patch_size
        if pad_h == self.patch_size:
            pad_h = 0
        if pad_w == self.patch_size:
            pad_w = 0
        # x = nn.functional.pad(x, (pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2))
        if pad_h + pad_w > 0:
            x = torch.nn.functional.interpolate(x, (H + pad_h, W + pad_w), mode="bilinear")

        x = self.prepare_tokens_with_masks(x, masks)

        # for blk in self.blocks:
        # x = blk(x)

        # x_norm = self.norm(x)
        # if self.tuning_mode == 'ssf':
        # x_norm = ssf_ada(x_norm, self.ssf_scale_1, self.ssf_shift_1)

        # return {
        #     "x_norm_clstoken": x_norm[:, 0],
        #     "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
        #     "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
        #     "x_prenorm": x,
        #     "masks": masks,
        # }
        # features = []
        # features.append(x_norm)
        # features.append(x_norm)
        # features.append(x_norm)
        # features.append(x_norm)
        # return [features, (B, (H+pad_h)//self.patch_size, (W+pad_w)//self.patch_size, H, W, self.num_register_tokens)]

        if not self.multi_output:
            for blk in self.blocks:
                x = blk(x)
            x_norm = self.norm(x)
            if self.tuning_mode == "ssf":
                x_norm = ssf_ada(x_norm, self.ssf_scale_1, self.ssf_shift_1)

            features = []
            features.append(x_norm)
            features.append(x_norm)
            features.append(x_norm)
            features.append(x_norm)
            return [
                features,
                (
                    B,
                    (H + pad_h) // self.patch_size,
                    (W + pad_w) // self.patch_size,
                    H,
                    W,
                    self.num_register_tokens,
                ),
            ]
        else:
            features = []
            for blk in self.blocks:
                for idx, sub_blk in enumerate(blk):
                    x = sub_blk(x)
                    if (idx + 1) % (len(blk) // 4) == 0:
                        features.append(x)

            return [
                features,
                (
                    B,
                    (H + pad_h) // self.patch_size,
                    (W + pad_w) // self.patch_size,
                    H,
                    W,
                    self.num_register_tokens,
                ),
            ]

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens, strict=False))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret
        # if is_training:
        #     return ret
        # else:
        #     return self.head(ret["x_norm_clstoken"])


def load_ckpt_dino(checkpoint, model):
    if checkpoint is not None:
        try:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
        except Exception:
            print("NO pretrained imagenet ckpt available! Check your path!")
            del model.mask_token
            return

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            new_state_dict = {}
            for key, value in state_dict.items():
                key_new = "blocks.0" + key[len("blocks") :] if "blocks" in key else key
                new_state_dict[key_new] = value

            model.load_state_dict(new_state_dict, strict=True)
        del model.mask_token
        return
    else:
        return


def vit_small(patch_size=14, num_register_tokens=0, checkpoint=None, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )

    load_ckpt_dino(checkpoint, model)

    return model


def vit_base(patch_size=14, num_register_tokens=0, checkpoint=None, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=14, num_register_tokens=0, checkpoint=None, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            new_state_dict = {}
            for key, value in state_dict.items():
                key_new = "blocks.0" + key[len("blocks") :] if "blocks" in key else key
                new_state_dict[key_new] = value

            model.load_state_dict(new_state_dict, strict=True)
        del model.mask_token
    return model


def vit_giant2(patch_size=14, num_register_tokens=0, checkpoint=None, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        ffn_layer="swiglu",
        **kwargs,
    )
    return model


def vit_small_reg(patch_size=14, num_register_tokens=4, checkpoint=None, tuning_mode=None, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        tuning_mode=tuning_mode,
        **kwargs,
    )

    load_ckpt_dino(checkpoint, model)

    return model


def vit_base_reg(patch_size=14, num_register_tokens=4, checkpoint=None, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )

    load_ckpt_dino(checkpoint, model)

    return model


def vit_large_reg(patch_size=14, num_register_tokens=4, checkpoint=None, tuning_mode=None, **kwargs):
    model = DinoVisionTransformer(
        img_size=518,
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        tuning_mode=tuning_mode,
        **kwargs,
    )

    load_ckpt_dino(checkpoint, model)

    return model


def vit_giant2_reg(patch_size=14, num_register_tokens=4, checkpoint=None, tuning_mode=None, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        ffn_layer="swiglu",
        tuning_mode=tuning_mode,
        multi_output=True,
        **kwargs,
    )

    load_ckpt_dino(checkpoint, model)

    return model


if __name__ == "__main__":
    import os

    try:
        from mmcv.utils import Config
    except Exception:
        from mmengine import Config

    # rgb = torch.rand((2, 3, 518, 518)).cuda()

    # cfg.data_basic['crop_size']['0']
    # cfg.data_basic['crop_size']['1']
    config_path = os.environ.get("WORLDFOUNDRY_METRIC3D_CONFIG", "").strip()
    if not config_path:
        raise SystemExit(
            "Set WORLDFOUNDRY_METRIC3D_CONFIG to a Metric3D config file before running this module directly."
        )
    cfg = Config.fromfile(config_path)

    # rgb = torch.arange(0, 2*3*1036*1036, 1).cuda().float().view(2, 3, 1036, 1036)
    rgb = torch.zeros(1, 3, 616, 1064).cuda()
    cfg["tuning_mode"] = "ssf"
    # model = vit_large_reg(checkpoint="${WORLDFOUNDRY_CKPT_DIR}/dinov2_vitl14_reg4_pretrain.pth", kwarg=cfg).cuda()
    model = vit_large_reg(tuning_mode="ssf").cuda()

    # import timm
    # model2 = timm.models.vision_transformer.vit_large_patch14_dinov2().cuda()
    # timm.models.load_checkpoint(model2, '${WORLDFOUNDRY_CKPT_DIR}/dinov2_vitl14_pretrain.pth', filter_fn=timm.models.vision_transformer.checkpoint_filter_fn)

    out1 = model(rgb)
    # out2 = model2(rgb)
    temp = 0
