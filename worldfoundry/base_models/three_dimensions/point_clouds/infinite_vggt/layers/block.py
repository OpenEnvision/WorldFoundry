from collections.abc import Callable

from torch import Tensor, nn

from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    Block as VGGTBlock,
)
from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.layers.block import (
    NestedTensorBlock as VGGTNestedTensorBlock,
)
from worldfoundry.core.nn.blocks.layers import Mlp
from worldfoundry.core.nn.blocks.stochastic_depth import add_residual as add_residual
from worldfoundry.core.nn.blocks.stochastic_depth import (
    drop_add_residual_stochastic_depth as drop_add_residual_stochastic_depth,
)
from worldfoundry.core.nn.blocks.stochastic_depth import (
    get_branges_scales as get_branges_scales,
)

from .attention import Attention

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

    def forward(
        self, x: Tensor, pos=None, attn_mask=None, past_key_values=None, use_cache=False, cache_budget=None
    ) -> Tensor | tuple[Tensor, dict]:
        def attn_residual_func(
            x: Tensor, pos=None, attn_mask=None, past_key_values=None, use_cache=False, cache_budget=None
        ) -> Tensor | tuple[Tensor, dict]:
            if use_cache:
                output, new_kv, scores = self.attn(
                    self.norm1(x), pos=pos, past_key_values=past_key_values, use_cache=True, cache_budget=cache_budget
                )
                return self.ls1(output), new_kv, scores
            else:
                if attn_mask is not None:
                    return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
                else:
                    return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if use_cache:
            attn_output, new_kv, scores = attn_residual_func(
                x, pos=pos, past_key_values=past_key_values, use_cache=True, cache_budget=cache_budget
            )
            x = x + attn_output
            x = x + ffn_residual_func(x)
            return x, new_kv, scores

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, attn_mask=attn_mask))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
            x = x + ffn_residual_func(x)
        return x


class NestedTensorBlock(VGGTNestedTensorBlock, Block):
    """VGGT nested interface with the family block constructor and forward."""
