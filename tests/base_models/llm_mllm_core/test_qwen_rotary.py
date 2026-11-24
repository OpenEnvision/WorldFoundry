# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen rotary tables retain FP32 arithmetic and dynamic cache resets."""

import pytest
import torch
from transformers import PretrainedConfig

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.rotary import QwenRotaryEmbedding

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rope_type", ["default", "dynamic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_qwen_rotary_embedding_preserves_tables_across_sequence_lengths(rope_type: str, dtype: torch.dtype) -> None:
    config = PretrainedConfig(
        hidden_size=32,
        num_attention_heads=4,
        max_position_embeddings=8,
        rope_theta=10000.0,
        rope_scaling=None if rope_type == "default" else {"rope_type": "dynamic", "factor": 2.0},
    )
    rotary = QwenRotaryEmbedding(config)
    original_frequency = rotary.inv_freq.clone()
    assert not rotary.state_dict()
    for length in (5, 12, 4):
        positions = torch.arange(length).expand(2, -1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            cos, sin = rotary(torch.zeros(2, length, 32, dtype=dtype), positions)
        frequencies = positions.float().unsqueeze(-1) * rotary.inv_freq.float()
        angles = torch.cat((frequencies, frequencies), dim=-1)
        torch.testing.assert_close(cos, (angles.cos() * rotary.attention_scaling).to(dtype), atol=0, rtol=0)
        torch.testing.assert_close(sin, (angles.sin() * rotary.attention_scaling).to(dtype), atol=0, rtol=0)
        assert rotary.max_seq_len_cached == (12 if rope_type == "dynamic" and length == 12 else 8)
    torch.testing.assert_close(rotary.inv_freq, original_frequency, atol=0, rtol=0)
