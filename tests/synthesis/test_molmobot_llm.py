# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldfoundry.synthesis.action_generation.molmobot.modeling import llm

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("attention_type", [llm.AttentionType.direct, llm.AttentionType.sdpa])
@pytest.mark.parametrize("norm_after", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sequential_block_checkpoint_forward_and_cache(
    tmp_path: Path, attention_type: llm.AttentionType, norm_after: bool, dtype: torch.dtype
) -> None:
    config = llm.LlmConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        norm_after=norm_after,
        attention_type=attention_type,
        block_type=llm.BlockType.sequential,
        rope=True,
    )
    torch.manual_seed(13)
    block = llm.OLMoBlock.build(0, config, llm.BufferCache(), device="cpu").eval().to(dtype)
    assert isinstance(block, llm.OLMoSequentialBlock)
    checkpoint = tmp_path / "model.pt"
    torch.save(block.state_dict(), checkpoint)
    restored = llm.OLMoBlock.build(0, config, llm.BufferCache(), device="cpu").eval().to(dtype)
    result = restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    inputs = torch.randn(1, 4, 32).to(dtype)
    with torch.inference_mode():
        expected, expected_cache = block(inputs, use_cache=True)
        actual, cache = restored(inputs, use_cache=True)
    assert actual.shape == (1, 4, 32)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache is not None and expected_cache is not None
    for value, reference in zip(cache, expected_cache, strict=True):
        assert value.shape == (1, 4, 4, 8)
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_moe_config_remains_parseable_but_is_not_an_inference_block(tmp_path: Path) -> None:
    source = tmp_path / "llm.yaml"
    source.write_text("block_type: moe\nmoe_num_experts: 8\n", encoding="utf-8")
    config = llm.LlmConfig.load(source)
    assert config.block_type == llm.BlockType.moe
    assert config.moe_num_experts == 8
    with pytest.raises(NotImplementedError, match="Unknown block type: 'moe'"):
        llm.OLMoBlock.build(0, config, llm.BufferCache(), device="cpu")
