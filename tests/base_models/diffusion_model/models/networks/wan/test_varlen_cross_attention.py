# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants import geometry, linear
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.echo_infinity import model as echo_infinity
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing import model as forcing
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm import action, prope
from worldfoundry.base_models.diffusion_model.models.networks.wan.varlen_cross_attention import (
    CachedTextCrossAttentionMixin,
    ImageTextCrossAttentionMixin,
    UncachedTextCrossAttentionMixin,
)

pytestmark = pytest.mark.unit


def test_wan_cross_forwards_are_shared_without_collapsing_native_ancestry() -> None:
    for family in (forcing, action, echo_infinity):
        assert family.WanT2VCrossAttention.forward is CachedTextCrossAttentionMixin.forward
        assert family.WanI2VCrossAttention.forward is ImageTextCrossAttentionMixin.forward
        assert issubclass(family.WanT2VCrossAttention, family.WanSelfAttention)
        assert issubclass(family.WanI2VCrossAttention, family.WanSelfAttention)
    assert forcing.WanSelfAttention is not action.WanSelfAttention
    assert forcing.WanRMSNorm is not action.WanRMSNorm


@pytest.mark.parametrize("zero_init", [False, True])
def test_minwm_prope_installer_selects_self_attention_and_excludes_native_cross_types(zero_init: bool) -> None:
    layers = nn.ModuleDict(
        {
            "self_attn": action.WanSelfAttention(32, 4),
            "text": action.WanT2VCrossAttention(32, 4),
            "image": action.WanI2VCrossAttention(32, 4),
            "gan": action.WanGanCrossAttention(32, 4),
            "forcing_self": forcing.WanSelfAttention(32, 4),
            "echo_self": echo_infinity.WanSelfAttention(32, 4),
            "echo_text": echo_infinity.WanT2VCrossAttention(32, 4),
            "echo_image": echo_infinity.WanI2VCrossAttention(32, 4),
            "linear_self": linear.WanSelfAttention(32, 4),
            "linear_text": linear.WanT2VCrossAttention(32, 4),
            "geometry_self": geometry.WanSelfAttention(32, 4),
            "geometry_text": geometry.WanT2VCrossAttention(32, 4),
        }
    )
    prope.add_prope_parameters(layers, zero_init=zero_init)
    assert [name for name, layer in layers.items() if hasattr(layer, "prope_o")] == ["self_attn"]
    parameter = layers["self_attn"].prope_o.weight
    if zero_init:
        assert torch.count_nonzero(parameter).item() == 0
    prope.add_prope_parameters(layers, zero_init=not zero_init)
    assert layers["self_attn"].prope_o.weight is parameter
    assert type(layers["text"].norm_q) is action.WanRMSNorm


@pytest.mark.parametrize("family_name", ["linear", "geometry"])
def test_uncached_text_forward_preserves_native_provider_binding(
    family_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = {"linear": linear, "geometry": geometry}[family_name]
    cls = family.WanT2VCrossAttention
    assert cls.forward is UncachedTextCrossAttentionMixin.forward
    assert cls._cross_attention is family.flash_attention
    model = cls(32, 4)
    assert isinstance(model, family.WanSelfAttention)
    assert type(model.norm_q) is family.WanRMSNorm
    calls = []

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, k_lens: torch.Tensor) -> torch.Tensor:
        calls.append((q.shape, k.shape, v.shape, k_lens))
        return q

    monkeypatch.setattr(cls, "_cross_attention", staticmethod(attention))
    x, context, lengths = torch.randn(2, 3, 32), torch.randn(2, 7, 32), torch.tensor([5, 7])
    output = model(x, context, lengths)
    torch.testing.assert_close(output, model.o(model.norm_q(model.q(x))), rtol=0, atol=0)
    assert calls == [(torch.Size([2, 3, 4, 8]), torch.Size([2, 7, 4, 8]), torch.Size([2, 7, 4, 8]), lengths)]


def test_uncached_text_forward_keeps_linear_constructor_kwargs_and_buffer_ownership() -> None:
    model = linear.WanT2VCrossAttention(32, 4, rope_scale=2.0)
    assert model.qkv_store_buffer is None
    buffer = object()
    model.qkv_store_buffer = buffer
    assert model.qkv_store_buffer is buffer
    assert not hasattr(geometry.WanT2VCrossAttention(32, 4), "qkv_store_buffer")
    with pytest.raises(TypeError, match="rope_scale"):
        geometry.WanT2VCrossAttention(32, 4, rope_scale=2.0)  # ty: ignore[unknown-argument] - Rejected keyword.


@pytest.mark.parametrize("model_type", ["t2v", "i2v"])
def test_linear_wan_init_weights_keeps_native_image_cross_attention_selection(model_type: str) -> None:
    model = linear.WanModel(
        model_type=model_type,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        image_dim=16,
        text_dim=16,
        in_dim=4,
        out_dim=4,
        num_heads=4,
        num_layers=1,
    )
    cross_attention = model.blocks[0].cross_attn
    model.init_weights()
    assert torch.count_nonzero(cross_attention.v.weight).item() > 0
    if model_type == "i2v":
        assert type(cross_attention) is linear.WanI2VCrossAttention
        assert torch.count_nonzero(cross_attention.v_img.weight).item() == 0
        assert torch.count_nonzero(cross_attention.v_img.bias).item() == 0
    else:
        assert type(cross_attention) is linear.WanT2VCrossAttention
        assert not isinstance(cross_attention, linear.WanI2VCrossAttention)
        assert not hasattr(cross_attention, "v_img")
