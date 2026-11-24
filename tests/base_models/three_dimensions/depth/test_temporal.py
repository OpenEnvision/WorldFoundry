# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Temporal depth blocks retain their state layout and family attention policies."""

import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.base_models.three_dimensions.depth.temporal import attention, motion
from worldfoundry.base_models.three_dimensions.depth.video_depth_anything_longvie.video_depth_anything.motion_module import (
    attention as longvie_attention,
)
from worldfoundry.base_models.three_dimensions.depth.video_depth_anything_longvie.video_depth_anything.motion_module import (
    motion_module as longvie,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("family", [motion, longvie], ids=["vda", "longvie"])
@pytest.mark.parametrize("position", ["ape", "rope"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_initialized_temporal_module_is_identity(family: ModuleType, position: str, dtype: torch.dtype) -> None:
    module = family.TemporalModule(
        16,
        num_attention_heads=2,
        num_transformer_block=1,
        num_attention_blocks=1,
        norm_num_groups=4,
        temporal_max_len=4,
        pos_embedding_type=position,
    ).to(dtype)
    value = torch.randn(2, 16, 3, 2, 3, dtype=dtype)
    with torch.no_grad():
        result = module(value, None)
    torch.testing.assert_close(result, value, rtol=0, atol=0)


def test_longvie_shell_constructs_its_own_attention_binding() -> None:
    module = longvie.TemporalModule(16, num_attention_heads=2, norm_num_groups=4)
    transformer = module.temporal_transformer
    block = transformer.transformer_blocks[0]
    assert type(transformer) is longvie.TemporalTransformer3DModel
    assert type(block) is longvie.TemporalTransformerBlock
    assert type(block.attention_blocks[0]) is longvie.TemporalAttention
    assert isinstance(block.attention_blocks[0], motion.TemporalAttention)
    assert isinstance(block.attention_blocks[0], longvie_attention.CrossAttention)


def test_temporal_attention_keeps_sdpa_and_optional_xformers_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    native = motion.TemporalAttention(query_dim=16, heads=2, dim_head=8, temporal_max_len=4)
    variant = longvie.TemporalAttention(query_dim=16, heads=2, dim_head=8, temporal_max_len=4)
    monkeypatch.setattr(longvie, "XFORMERS_AVAILABLE", False)
    calls: list[str] = []
    native_backend = native._memory_efficient_attention_xformers
    variant_backend = variant._attention

    def sdpa(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        calls.append("vda_sdpa")
        return native_backend(query, key, value, attention_mask)

    def manual(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        calls.append("longvie_manual")
        return variant_backend(query, key, value, attention_mask)

    monkeypatch.setattr(native, "_memory_efficient_attention_xformers", sdpa)
    monkeypatch.setattr(variant, "_attention", manual)
    value = torch.randn(6, 4, 16)
    native(value, video_length=3)
    variant(value, video_length=3)
    assert calls == ["vda_sdpa", "longvie_manual"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("activation", ["gelu", "geglu", "geglu-approximate"])
def test_feedforward_keeps_activation_arithmetic_and_native_keys(dtype: torch.dtype, activation: str) -> None:
    module = attention.FeedForward(8, 12, mult=3, activation_fn=activation).to(dtype)
    assert list(module.state_dict()) == ["net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"]
    value = torch.randn(2, 3, 8, dtype=dtype)
    projected = module.net[0].proj(value)
    if activation == "geglu":
        projected, gate = projected.chunk(2, dim=-1)
        activated = projected * F.gelu(gate)
    elif activation == "gelu":
        activated = F.gelu(projected)
    else:
        activated = projected * torch.sigmoid(1.702 * projected)
    torch.testing.assert_close(module(value), module.net[2](activated), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 3, 8), (2, 3, 4, 8)])
def test_rotary_uses_sequence_axis_one_and_flattens_paired_features(shape: tuple[int, ...]) -> None:
    value = torch.randn(shape)
    frequencies = attention.precompute_freqs_cis(8, 3)
    assert frequencies.dtype == torch.complex64
    query, key = attention.apply_rotary_emb(value, value, frequencies)
    torch.testing.assert_close(query, key, rtol=0, atol=0)
    torch.testing.assert_close(query[:, 0], value[:, 0].flatten(1), rtol=0, atol=0)
    with pytest.raises(AssertionError):
        attention.apply_rotary_emb(value, value, frequencies[:2])


def test_sdpa_batch_split_keeps_order_across_the_native_limit() -> None:
    module = attention.CrossAttention(2, heads=1, dim_head=2)
    query = torch.zeros(65536, 1, 1, 2)
    value = torch.arange(65536, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 1, 1, 2)
    actual = module._memory_efficient_attention_split(query, query, value, None)
    torch.testing.assert_close(actual, value, rtol=0, atol=0)


def test_shared_temporal_import_does_not_load_model_roots_or_resolvers() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from worldfoundry.base_models.three_dimensions.depth.temporal import motion; "
            "assert not any(name.startswith(('hydra', 'omegaconf', "
            "'worldfoundry.base_models.three_dimensions.general_3d.vipe', "
            "'worldfoundry.base_models.three_dimensions.depth.videodepthanything', "
            "'worldfoundry.base_models.three_dimensions.depth.video_depth_anything_longvie')) "
            "for name in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[4],
        check=True,
    )
