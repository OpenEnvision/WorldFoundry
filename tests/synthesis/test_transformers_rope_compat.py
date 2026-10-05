
import pytest

# This test module imports worldfoundry code that requires the optional
# "omegaconf" dependency at import time; skip when it is unavailable.
pytest.importorskip("omegaconf")
from types import SimpleNamespace

import torch

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.qwen2_5 import (
    Qwen2_5_VLRotaryEmbedding,
)
from worldfoundry.synthesis.visual_generation.show_o.show_o_runtime.models.phi import PhiAttention


def test_qwen_default_rope_is_available_with_transformers_5() -> None:
    config = SimpleNamespace(
        rope_scaling=None,
        max_position_embeddings=128,
        hidden_size=64,
        num_attention_heads=4,
        rope_theta=10000.0,
    )

    rotary = Qwen2_5_VLRotaryEmbedding(config)

    assert rotary.inv_freq.shape == (8,)
    assert torch.isfinite(rotary.inv_freq).all()


def test_show_o_phi_uses_standard_rope_base_when_config_omits_it() -> None:
    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        partial_rotary_factor=0.5,
        attention_dropout=0.0,
        qk_layernorm=False,
        rope_scaling={"rope_type": "default"},
    )

    attention = PhiAttention(config, layer_idx=0)

    assert attention.rope_theta == 10000.0
