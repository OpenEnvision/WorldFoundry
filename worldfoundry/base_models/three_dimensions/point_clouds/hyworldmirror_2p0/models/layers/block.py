# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from collections.abc import Callable

from torch import Tensor, nn

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    Block as VGGTBlock,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    NestedTensorBlock as VGGTNestedTensorBlock,
)
from worldfoundry.core.nn import Mlp
from worldfoundry.core.nn.blocks.stochastic_depth import add_residual as add_residual
from worldfoundry.core.nn.blocks.stochastic_depth import (
    drop_add_residual_stochastic_depth as drop_add_residual_stochastic_depth,
)
from worldfoundry.core.nn.blocks.stochastic_depth import (
    get_branges_scales as get_branges_scales,
)

from .attention import Attention, DistAttention

XFORMERS_AVAILABLE = False


class Block(VGGTBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
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
        qk_norm: bool = False,
        fused_attn: bool = True,  # use _worldfoundry_scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=attn_class,
            ffn_layer=ffn_layer,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )


class DistBlock(Block):
    def __init__(self, *args, attn_class: Callable[..., nn.Module] = DistAttention, **kwargs):
        super().__init__(*args, attn_class=attn_class, **kwargs)

    def forward(
        self, x: Tensor, pos=None, sp_size=1, sp_group=None, padding_tokens=0, block_type=None, token_shape=None
    ) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, sp_size=1, sp_group=None, padding_tokens=0) -> Tensor:
            return self.ls1(
                self.attn(self.norm1(x), pos=pos, sp_size=sp_size, sp_group=sp_group, padding_tokens=padding_tokens)
            )

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                sp_size=sp_size,
                sp_group=sp_group,
                padding_tokens=padding_tokens,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(
                attn_residual_func(x, pos=pos, sp_size=sp_size, sp_group=sp_group, padding_tokens=padding_tokens)
            )
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos, sp_size=sp_size, sp_group=sp_group, padding_tokens=padding_tokens)
            x = x + ffn_residual_func(x)
        return x


class NestedTensorBlock(VGGTNestedTensorBlock, Block):
    """VGGT nested interface with the family block constructor and forward."""
